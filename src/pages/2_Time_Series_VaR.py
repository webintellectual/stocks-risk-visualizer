from stocks_risk_visualizer.utils import (
    VaRCalculator,
    apply_custom_css
)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Time Series VaR Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

def plot_time_series_var(dates, portfolio_values, var_values, confidence_level):
    """
    Create a time series plot showing portfolio values and VaR over time
    """
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Portfolio Value Over Time', f'VaR Return ({confidence_level*100:.0f}% Confidence)'),
                       vertical_spacing=0.15)
    
    # Portfolio Value plot
    fig.add_trace(
        go.Scatter(x=dates, y=portfolio_values, name='Portfolio Value',
                  line=dict(color='blue')),
        row=1, col=1
    )
    
    # VaR return plot (percentage)
    fig.add_trace(
        go.Scatter(x=dates, y=[v*100 for v in var_values], name='VaR Return (%)',
                  line=dict(color='red')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Time Series VaR Analysis",
        title_x=0.5
    )
    
    # Axis labels
    fig.update_yaxes(title_text='Portfolio Value (USD)', row=1, col=1)
    fig.update_yaxes(title_text='VaR Return (%)', row=2, col=1)
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Time Series VaR Analysis</h1>', unsafe_allow_html=True)
    st.markdown("**Analyze how your portfolio's risk has evolved over time using Historical VaR methodology**")
    
    # Explanation
    with st.expander("ℹ️ About Time Series VaR Analysis"):
        st.markdown("""
        **Time Series Historical VaR**
        
        Analyzes how your portfolio's risk profile has changed over time:
        
        - **Method:** Historical VaR calculated for multiple points in time
        - **Time Window:** Analyzes portfolio evolution across dates
        - **Purpose:** "How has my portfolio's risk changed over time?"
        - **Output:** VaR estimates across different dates
        
        **Key Insights:**
        - Track risk trends over time
        - Identify periods of increased risk
        - Validate risk management strategies
        - Compare current risk levels to historical patterns
        """)
    
    # Setup sidebar parameters
    st.sidebar.header("Analysis Parameters")
    
    confidence_level = st.sidebar.slider(
        "Confidence Level",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
        format="%.2f",
        help="The confidence level for VaR calculation (e.g., 0.95 means 95% confidence)"
    )
    
    time_horizon = st.sidebar.slider(
        "Time Horizon (Days)",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
        help="The time horizon for VaR calculation in days"
    )
    
    rolling_period = st.sidebar.slider(
        "Rolling Period",
        min_value=50,
        max_value=500,
        value=252,
        step=10,
        help="Number of rolling returns used to estimate VaR at each point"
    )
    
    required_price_days = rolling_period + time_horizon
    
    st.header("📋 Historical Portfolio Input")
    st.markdown("""
    Please provide your historical portfolio holdings with dates. Each entry should include:
    - Date
    - Stock Symbol
    - Quantity held on that date
    """)
    
    # Portfolio Input Methods
    input_method = st.radio(
        "Select input method:",
        ["Sample Historical Portfolio", "Upload CSV", "Paste Text"],
        horizontal=True,
        index=0
    )

    portfolio_history_df = None
    
    if input_method == "Sample Historical Portfolio":
        st.markdown("""
        Using a sample historical portfolio data for demonstration purposes.
        You can switch to other input methods to analyze your own portfolio history.
        """)
        
        # Generate sample historical data
        np.random.seed(42)  # Seed for reproducible random sample quantities
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
        # latest_quantities = {'AAPL': 10, 'MSFT': 5, 'AMZN': 3, 'GOOGL': 2, 'META': 8}
        
        data = []
        for i, date in enumerate(dates):
            for j, symbol in enumerate(symbols):
                # if i == len(dates) - 1:
                #     # For the latest date, use the specified quantities
                #     quantity = latest_quantities[symbol]
                # else:
                # For earlier dates, use a trend and some noise
                base = 100 + 10 * j
                quantity = base + i * (j + 1) + np.random.randint(-5, 6)
                data.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Quantity': quantity
                })
        
        portfolio_history_df = pd.DataFrame(data)
        
    elif input_method == "Upload CSV":
        st.markdown("""
        Upload a CSV file with your historical portfolio holdings.
        The CSV should have three columns:
        - Date: Date of the holding (YYYY-MM-DD)
        - Symbol: Stock symbol (e.g., AAPL, MSFT)
        - Quantity: Number of shares held on that date
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                portfolio_history_df = pd.read_csv(uploaded_file)
                portfolio_history_df['Date'] = pd.to_datetime(portfolio_history_df['Date'])
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                st.stop()
                
    else:  # Paste Text
        st.markdown("""
        Paste your historical portfolio holdings in the format:
        ```
        2025-06-01,AAPL,100
        2025-06-01,MSFT,50
        2025-06-02,AAPL,120
        2025-06-02,MSFT,45
        ```
        One holding per line: date,symbol,quantity
        """)
        
        text_data = st.text_area("Paste portfolio data here:")
        if text_data:
            try:
                # Parse the text data
                rows = [line.strip().split(',') for line in text_data.strip().split('\n')]
                portfolio_history_df = pd.DataFrame(rows, columns=['Date', 'Symbol', 'Quantity'])
                portfolio_history_df['Date'] = pd.to_datetime(portfolio_history_df['Date'])
                portfolio_history_df['Quantity'] = pd.to_numeric(portfolio_history_df['Quantity'])
            except Exception as e:
                st.error(f"Error parsing text data: {str(e)}")
                st.stop()

    if portfolio_history_df is not None and not portfolio_history_df.empty:
        # Display the historical portfolio as a pivot table: dates rows, symbols as columns
        st.markdown("### Historical Portfolio Data")
        pivot_df = portfolio_history_df.pivot(index='Date', columns='Symbol', values='Quantity')
        pivot_df = pivot_df.fillna(0).astype(int).sort_index()
        st.dataframe(pivot_df, use_container_width=True)
        # Show how many unique dates are in the history
        st.markdown(f"**Number of Historical Dates:** {pivot_df.shape[0]}")
        st.markdown("---")

        run_analysis = st.button("🚀 Run Time Series VaR Analysis", type="primary", use_container_width=True)

        if run_analysis:
            with st.spinner("Calculating Time Series VaR... This may take a moment."):
                # Get unique dates
                unique_dates = sorted(portfolio_history_df['Date'].unique())
                
                # Initialize VaR calculator with all symbols (zero quantities) for data fetching
                symbols = portfolio_history_df['Symbol'].unique()
                empty_portfolio = pd.DataFrame({
                    'Symbol': symbols,
                    'Quantity': [0] * len(symbols),
                    'Current_Price': [0] * len(symbols)
                })
                # var_calculator = VaRCalculator(
                #     portfolio_data=empty_portfolio,
                #     confidence_level=confidence_level,
                #     time_horizon=time_horizon
                # )

                # Calculate VaR for each date
                var_results = []
                portfolio_values = []
                
                for date in unique_dates:
                    # Get portfolio for this date
                    date_portfolio = portfolio_history_df[portfolio_history_df['Date'] == date]
                    
                    # Fetch full historical price data for returns and current prices
                    temp_calc = VaRCalculator(
                        portfolio_data=date_portfolio[['Symbol', 'Quantity']],
                        confidence_level=confidence_level,
                        time_horizon=time_horizon
                    )

                    price_data = temp_calc.fetch_market_data(
                        number_of_days_back=required_price_days+1,
                        end_date=date,
                        suppress_warnings=True
                    )

                    # Determine current prices from last row
                    current_prices = price_data.iloc[-1]
                    # Add Current_Price to portfolio
                    date_portfolio = date_portfolio.copy()
                    date_portfolio['Current_Price'] = date_portfolio['Symbol'].map(current_prices)

                    # remove reference date from fetched price data
                    if price_data is not None and (len(price_data) != required_price_days):
                        # remove the last row if it is the reference date
                        price_data = price_data.iloc[:-1]

                    # st.write(f"Date: {date}, Price data shape: {price_data.shape}")
                    # st.write(f"len of price data: {len(price_data)}")
                    # st.write("Required price days: ", required_price_days)
                    # st.dataframe(price_data, use_container_width=True)


                    # verify if price data for correct length
                    if price_data is not None and (len(price_data) != required_price_days):
                        st.write(f"Insufficient price data for date {date}. Expected at least {required_price_days} days, got {len(price_data)}.")
                        st.dataframe(price_data, use_container_width=True)
                        st.stop()
                    # st.stop()
                    # return
                    if price_data is None or price_data.empty:
                        st.error(f"Could not fetch price data for date {date}")
                        continue

                    # Initialize calculator with complete portfolio including prices
                    calc = VaRCalculator(
                        portfolio_data=date_portfolio[['Symbol', 'Quantity', 'Current_Price']],
                        confidence_level=confidence_level,
                        time_horizon=time_horizon
                    )

                    # Calculate portfolio returns using rolling method
                    portfolio_returns = calc.calculate_portfolio_returns(price_data, rolling_period=rolling_period)
                    if portfolio_returns is None or portfolio_returns.empty:
                        st.error(f"Could not calculate returns for date {date}")
                        st.stop()

                    # check if number of returns equal to rolling period
                    if len(portfolio_returns) != rolling_period:
                        st.warning(f"Something went wrong: Expected {rolling_period} returns, got {len(portfolio_returns)} for date {date}.")
                        st.stop()
                    
                    # Now calculate VaR using these returns (period returns, not daily returns)
                    var_result = calc.calculate_var(portfolio_returns, is_period_returns=True)
                    # st.write(var_result)
                    # st.write(abs(var_result['var_percentage']*100))
                    # Record the VaR as return percentage
                    var_results.append(abs(var_result['var_percentage']*100))
                    portfolio_values.append(var_result['portfolio_value'])

                # Create visualization
                st.subheader("Time Series VaR Analysis")
                fig = plot_time_series_var(unique_dates, portfolio_values, var_results, confidence_level)
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk trend analysis
                st.subheader("Risk Trend Analysis")
                
                # Calculate key metrics
                latest_var = var_results[-1]
                # st.write(var_results)
                avg_var = np.mean(var_results)
                max_var = max(var_results)
                var_trend = np.polyfit(range(len(var_results)), var_results, 1)[0]

                # range(len(var_results)) - Creates x-coordinates: [0, 1, 2, 3, ...] representing time points
                # var_results - The y-coordinates: VaR values for each date
                # np.polyfit(..., 1) - Fits a polynomial of degree 1 (linear line) to the data
                # [0] - Extracts the slope coefficient from the polynomial
                                
                trend_description = "increasing" if var_trend > 0 else "decreasing" # because if slope is positive, VaR is increasing over time
                
                st.markdown(f"""
                #### Key Findings
                
                1. **Current Risk Level**: {latest_var:,.2f}% VaR
                2. **Average Risk**: {avg_var:,.2f}% VaR
                3. **Maximum Risk**: {max_var:,.2f}% VaR
                4. **Risk Trend**: Portfolio risk is {trend_description} over time
                
                **Risk Assessment:**
                - Your portfolio's risk profile has been {trend_description} over the analyzed period
                - The highest risk level was {max_var:,.2f}% VaR
                - Currently, your risk level is {'above' if latest_var > avg_var else 'below'} the historical average
                """)
                
                # Download results
                st.subheader("Download Results")
                results_df = pd.DataFrame({
                    'Date': unique_dates,
                    'Portfolio_Value': portfolio_values,
                    'VaR': var_results
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Time Series VaR Results as CSV",
                    data=csv,
                    file_name=f"time_series_var_results_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()