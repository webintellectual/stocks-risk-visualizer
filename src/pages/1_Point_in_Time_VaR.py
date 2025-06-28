import streamlit as st
import pandas as pd
import warnings  # used to suppress data fetching warnings
from datetime import datetime, timedelta
from stocks_risk_visualizer.utils import (
    VaRCalculator,
    apply_custom_css,
    get_sample_current_portfolio,
    plot_var_results
)

st.set_page_config(
    page_title="Point-in-Time VaR Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Point-in-Time VaR Analysis</h1>', unsafe_allow_html=True)
    st.markdown("**Analyze your current portfolio's risk using Historical VaR methodology**")
    
    # Explanation
    with st.expander("ℹ️ About Point-in-Time VaR Analysis"):
        st.markdown("""
        **Point-in-Time Historical VaR**
        
        Calculates VaR for your portfolio **as of today** using recent historical market data:
        
        - **Method:** Historical VaR (non-parametric, uses actual returns)
        - **Reference Date:** Current date (today)
        - **Data Window:** Rolling window of recent trading days
        - **Purpose:** "What is the risk of my current portfolio going forward?"
        - **Output:** Single VaR estimate based on actual historical losses
        
        **Why Historical VaR:**
        - No assumptions about return distributions (normal, etc.)
        - Captures actual market crashes and extreme events
        - Handles portfolio complexity and correlations naturally
        - Widely accepted by regulators (Basel III compliant)
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
        value=1,
        step=1,
        help="The time horizon for VaR calculation in days"
    )
    
    rolling_period = st.sidebar.slider(
        "Rolling Period",
        min_value=50,
        max_value=500,
        value=252,
        step=10,
        help="Number of rolling returns used to estimate VaR. Required price days will be: rolling_period + time_horizon"
    )
    
    required_price_days = rolling_period + time_horizon
    
    st.header("📋 Portfolio Input")
    st.markdown("Please provide your current portfolio holdings for Point-in-Time VaR analysis")
    
    # Portfolio Input Methods
    input_method = st.radio(
        "Select input method:",
        ["Sample Portfolio", "Manual Entry", "Upload CSV", "Paste Text"],
        horizontal=True,
        index=0  # Default to Sample Portfolio
    )

    portfolio_df = None
    
    if input_method == "Sample Portfolio":
        st.markdown("""
        Using a sample portfolio with major tech stocks for demonstration purposes.
        You can switch to other input methods to analyze your own portfolio.
        """)
        portfolio_df = get_sample_current_portfolio()
        if portfolio_df is not None and not portfolio_df.empty:
            # Display the portfolio for non-manual entry methods
            st.markdown("### Current Portfolio")
            st.dataframe(portfolio_df, use_container_width=True)
            st.markdown("---")
        else:
            st.error("Sample portfolio data is not available. Please contact support or try another input method.")
            st.stop()
    
    elif input_method == "Manual Entry":
        st.markdown("""
        Enter your portfolio holdings manually. Symbols should match Yahoo Finance format exactly
        
        [View on Yahoo Finance →](https://finance.yahoo.com)
        """)
        
        # Create empty rows for manual entry
        num_rows = st.number_input("Number of stocks in portfolio:", min_value=1, value=3)
        
        symbols = []
        quantities = []
        
        for i in range(num_rows):
            col1, col2 = st.columns([2, 1])
            with col1:
                symbol = st.text_input(
                    f"Stock Symbol {i+1}",
                    key=f"symbol_{i}",
                    help="Enter the stock symbol as it appears on Yahoo Finance"
                )
                if symbol:
                    symbol = symbol.upper()
                    # Add a check link to Yahoo Finance
                    st.markdown(f'[Check {symbol} on Yahoo Finance](https://finance.yahoo.com/quote/{symbol})')
                symbols.append(symbol)
            with col2:
                quantity = st.number_input(
                    f"Quantity {i+1}",
                    min_value=0,
                    value=0,
                    key=f"quantity_{i}",
                    help="Number of shares/units held"
                )
                quantities.append(quantity)
        
    elif input_method == "Upload CSV":
        st.markdown("""
        Upload a CSV file with your portfolio holdings.
        The CSV should have two columns:
        - Symbol: Stock symbol (e.g., AAPL, MSFT)
        - Quantity: Number of shares
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
                
    else:  # Paste Text
        st.markdown("""
        Paste your portfolio holdings in the format:
        ```
        AAPL,100
        MSFT,50
        GOOGL,25
        ```
        One holding per line, symbol and quantity separated by comma
        """)
        
        text_data = st.text_area("Paste portfolio data here:")
        
        
    run_analysis = st.button("🚀 Run VaR Analysis", type="primary", use_container_width=True)

    if run_analysis:
        if input_method != "Sample Portfolio":
            if input_method == "Manual Entry":
                data = [(s, q) for s, q in zip(symbols, quantities) if s and s.strip() and q > 0]
                if not data:
                    # Show validation errors
                    if not any(s and s.strip() for s in symbols):
                        st.info("Please enter at least one stock symbol.")
                    elif not any(q > 0 for q in quantities):
                        st.error("Please enter a quantity greater than 0 for at least one stock.")
                    else:
                        st.error("No valid portfolio entries. Please ensure at least one stock has both a symbol and a positive quantity.")
                    st.stop()
            elif input_method == "Paste Text":
                if text_data:
                    print("flag")
                    try:
                        # Split the text into lines and parse each line
                        lines = [line.strip() for line in text_data.split('\n') if line.strip()]
                        data = [line.split(',') for line in lines]
                        # Validate data
                        if not all(len(row) == 2 for row in data):
                            st.error("Each line must contain exactly two values: Symbol and Quantity.")
                            st.stop()
                        # in each line first value should be string and second value should be numeric
                        for row in data:
                            if len(row) != 2:
                                st.error("Each line must contain exactly two values: Symbol and Quantity.")
                                st.stop()
                            if not row[0].isalpha():
                                st.error(f"Invalid symbol '{row[0]}'. Symbols should only contain letters.")
                                st.stop()
                            try:
                                quantity = float(row[1])
                                if quantity <= 0:
                                    st.error(f"Invalid quantity '{row[1]}'. Quantity must be a positive number.")
                                    st.stop()
                            except ValueError:
                                st.error(f"Invalid quantity '{row[1]}'. Quantity must be a numeric value.")
                                st.stop()
                    except Exception as e:
                        st.error(f"Error parsing input: {str(e)}")
                else:
                    st.error("Please paste your portfolio data in the text area.")
                    st.stop()
            else:  # Upload CSV
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        if set(['Symbol', 'Quantity']).issubset(df.columns):
                            portfolio_df = df[['Symbol', 'Quantity']].copy()
                            portfolio_df['Symbol'] = portfolio_df['Symbol'].str.upper()
                            portfolio_df = portfolio_df[portfolio_df['Quantity'] > 0]
                        else:
                            st.error("CSV must contain 'Symbol' and 'Quantity' columns")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                else:
                    st.error("Please upload a CSV file with your portfolio data.")
                    st.stop()
            if input_method != "Upload CSV":
                portfolio_df = pd.DataFrame(data, columns=['Symbol', 'Quantity'])
                portfolio_df['Symbol'] = portfolio_df['Symbol'].str.strip().str.upper()
                # Convert Quantity to numeric
                portfolio_df['Quantity'] = pd.to_numeric(portfolio_df['Quantity'], errors='coerce')
                portfolio_df = portfolio_df.dropna()  # Remove any rows with invalid numbers
                portfolio_df = portfolio_df[portfolio_df['Quantity'] > 0]
            # Validate symbols using VaRCalculator
            with st.spinner("Validating symbols..."):
                try:
                    temp_calculator = VaRCalculator(portfolio_data=portfolio_df, confidence_level=0.95, time_horizon=1)
                    test_data = temp_calculator.fetch_market_data(number_of_days_back=5)  # Just test with 5 days
                    
                    # Check if we got data for all symbols
                    if test_data is None or test_data.empty:
                        invalid_symbols = ", ".join(portfolio_df['Symbol'])
                        st.warning(f"Unable to fetch data for symbol(s): {invalid_symbols}. Possibly delisted on Yahoo Finance or you entered an incorrect symbol.")
                        st.stop()
                    
                    missing_symbols = set(portfolio_df['Symbol']) - set(test_data.columns)
                    if missing_symbols:
                        invalid_symbols = ", ".join(missing_symbols)
                        st.warning(f"Unable to fetch data for symbol(s): {invalid_symbols}. Possibly delisted on Yahoo Finance or you entered an incorrect symbol.")
                        st.stop()
                        
                except Exception:
                    # If there's any error, we'll show a user-friendly message without the technical details
                    st.error("Unable to validate one or more symbols.")
                    st.info("Please check that all symbols are correctly entered and match Yahoo Finance exactly.")
                    st.stop()

            # Display the portfolio first
            st.markdown("### Current Portfolio")
            st.dataframe(portfolio_df, use_container_width=True)
            st.markdown("---")

        with st.spinner("Calculating VaR... This may take a moment."):
            # Initialize VaR calculator
            var_calculator = VaRCalculator(
                portfolio_data=portfolio_df,
                confidence_level=confidence_level,
                time_horizon=time_horizon
            )

            # Fetch current prices for portfolio value calculation
            try:
                # Use today's date end for current prices
                current_price_data = var_calculator.fetch_market_data(
                    number_of_days_back=1,
                    end_date=datetime.now().date()
                )
                if current_price_data is None or current_price_data.empty:
                    st.error("Unable to fetch current prices for portfolio value.")
                    st.stop()
                current_prices = current_price_data.iloc[-1]
                portfolio_df['Value'] = portfolio_df.apply(
                    lambda row: row['Quantity'] * current_prices[row['Symbol']],
                    axis=1
                )
                total_value = portfolio_df['Value'].sum()
            except Exception as e:
                st.error(f"Error fetching current prices: {e}")
                st.stop()

            # Fetch historical price data (ending before today) for returns
            try:
                # Exclude current date
                hist_end = datetime.now().date() - timedelta(days=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hist_price_data = var_calculator.fetch_market_data(
                        number_of_days_back=required_price_days,
                        end_date=hist_end
                    )
                if hist_price_data is None or hist_price_data.empty:
                    st.error("Unable to fetch sufficient historical data for VaR returns.")
                    st.stop()
            except Exception as e:
                st.error(f"Error fetching historical data: {e}")
                st.stop()

            # Calculate daily portfolio values over historical window
            portfolio_values = pd.Series(0.0, index=hist_price_data.index)
            for symbol in portfolio_df['Symbol']:
                qty = portfolio_df.loc[portfolio_df['Symbol']==symbol, 'Quantity'].values[0]
                portfolio_values += qty * hist_price_data[symbol]

            # Rolling returns (period returns)
            returns = []
            dates = []
            if len(portfolio_values) != required_price_days:
                st.error(f"Not enough historical data. Required {required_price_days} days, but got {len(portfolio_values)}.")
                st.stop()
            # st.write(f"Rolling Period = {rolling_period}, Required Price Days = {required_price_days}")
            # st.write(f"Length of pulled history: {len(portfolio_values)} days")
            # Calculate rolling returns according to formula: Rolling Period = price_days - horizon + 1
            for i in range(rolling_period):
                v0 = portfolio_values.iloc[i]
                if i + time_horizon >= len(portfolio_values):
                    st.write(f"index {i} + time_horizon {time_horizon} exceeds available data length {len(portfolio_values)}. Stopping calculation.")
                v1 = portfolio_values.iloc[i + time_horizon]
                returns.append((v1 - v0) / v0)
                dates.append(portfolio_values.index[i])

            portfolio_returns = pd.Series(returns, index=dates).iloc[-rolling_period:]
            if len(portfolio_returns) != rolling_period:
                st.warning(f"Expected {rolling_period} returns, but got {len(portfolio_returns)}.")
                st.stop()
            
            # Display returns
            with st.expander("Rolling Returns", expanded=False):
                st.markdown(f"""
                Showing {len(portfolio_returns)} rolling returns of {time_horizon}-day horizon
                according to the formula: $Return_t = \\frac{{V_{{t+{time_horizon}}} - V_t}}{{V_t}}$
                """)
                st.dataframe(portfolio_returns.round(4).to_frame('Return'), use_container_width=True)
            
            # Calculate VaR
            var_return = portfolio_returns.quantile(1 - confidence_level)
            var_dollar = abs(var_return * total_value)
            var_percentage = abs(var_return)
            
            var_results = {
                'var_percentage': var_return,
                'var_dollar': var_dollar
            }
            
            # VaR Visualization
            st.subheader("VaR Visualization")
            fig_var = plot_var_results(portfolio_returns, var_results)
            st.pyplot(fig_var)
            
            # Risk interpretation
            st.subheader("Risk Interpretation")
            
            var_percentage = abs(var_results['var_percentage'] * 100)
            var_dollar = abs(var_results['var_dollar'])
            
            # Determine risk level
            if var_percentage > 3:
                risk_level = "High"
                risk_color = "🔴"
                risk_recommendation = "Consider diversifying your portfolio across more asset classes to reduce risk."
            elif var_percentage > 2:
                risk_level = "Medium"
                risk_color = "🟠"
                risk_recommendation = "Your portfolio has a balanced risk profile, but could benefit from some diversification."
            else:
                risk_level = "Low"
                risk_color = "🟢"
                risk_recommendation = "Your portfolio shows a conservative risk profile. Consider if this aligns with your investment goals."
            
            st.markdown(f"""
            #### {risk_color} Risk Assessment: {risk_level} Risk
            
            Based on historical market data, with {confidence_level*100:.0f}% confidence:
            
            - Your portfolio of **\${total_value:,.2f}** has a {time_horizon}-day Value at Risk of **\${var_dollar:,.2f}** ({var_percentage:.2f}%)
            - This means there is a {(1-confidence_level)*100:.0f}% chance of losing more than ${var_dollar:,.2f} over the next {time_horizon} day{'s' if time_horizon > 1 else ''}
            
            **Risk Mitigation Recommendations:** {risk_recommendation}
            """)
            
            # Download results
            st.subheader("Download Results")
            results_df = pd.DataFrame({
                'Metric': ['Portfolio Value', 'VaR (%)', 'VaR ($)', 'Confidence Level', 'Time Horizon'],
                'Value': [
                    f"${total_value:,.2f}",
                    f"{var_percentage:.2f}%",
                    f"${var_dollar:,.2f}",
                    f"{confidence_level*100:.0f}%",
                    f"{time_horizon} day(s)"
                ]
            })
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download VaR Results as CSV",
                data=csv,
                file_name=f"var_results_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
