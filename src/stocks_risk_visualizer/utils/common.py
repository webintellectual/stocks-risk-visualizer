"""
Common utilities and classes for stocks risk visualizer
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import io
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Custom CSS for better styling
def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .risk-high { border-left-color: #ff4444 !important; }
        .risk-medium { border-left-color: #ffaa00 !important; }
        .risk-low { border-left-color: #00aa00 !important; }
    </style>
    """, unsafe_allow_html=True)

class VaRCalculator:
    """Value at Risk Calculator for portfolio analysis"""
    
    def __init__(self, portfolio_data, confidence_level=0.95, time_horizon=1):
        self.portfolio_data = portfolio_data
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.returns_data = None
        self.portfolio_value = None
        
    def fetch_market_data(self, number_of_days_back=252, end_date=None, suppress_warnings=True):
        """Fetch historical market data for portfolio stocks using rolling window"""
        symbols = self.portfolio_data['Symbol'].tolist()
        
        # Add progress bar only if not suppressing warnings (for main calculations)
        if not suppress_warnings:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            data = {}
            for i, symbol in enumerate(symbols):
                if not suppress_warnings:
                    status_text.text(f'Fetching data for {symbol}...')
                    progress_bar.progress((i + 1) / len(symbols))
                
                ticker = yf.Ticker(symbol)
                
                if end_date:
                    # For backtesting - fetch data up to specific end date
                    start_date = end_date - timedelta(days=int(number_of_days_back * 1.5))  # Extra buffer for weekends
                    hist = ticker.history(start=start_date, end=end_date)
                else:
                    # For current analysis - use recent data
                    hist = ticker.history(period="5y")
                
                if not hist.empty and len(hist) >= number_of_days_back:
                    # Use the most recent number_of_days_back days
                    data[symbol] = hist['Close'].tail(number_of_days_back)
                elif not suppress_warnings:
                    st.warning(f"Insufficient data for {symbol}. Need {number_of_days_back} days, got {len(hist) if not hist.empty else 0}")
                        
            if not suppress_warnings:
                progress_bar.empty()
                status_text.empty()
            
            if data:
                return pd.DataFrame(data).dropna()
            else:
                if not suppress_warnings:
                    st.error("No valid market data could be fetched")
                return None
                
        except Exception as e:
            if not suppress_warnings:
                st.error(f"Error fetching market data: {str(e)}")
            return None
    
    def calculate_portfolio_returns(self, price_data, rolling_period=None):
        """Calculate portfolio returns using total portfolio value method following Maths.md"""
        # Calculate daily portfolio values for each day
        portfolio_values = pd.Series(0.0, index=price_data.index)
        
        for _, row in self.portfolio_data.iterrows():
            symbol = row['Symbol']
            quantity = row['Quantity']
            if symbol in price_data.columns:
                portfolio_values += quantity * price_data[symbol]
        
        # Store portfolio value (current value from last day)
        self.portfolio_value = portfolio_values.iloc[-1]
        
        # If rolling_period is not specified, calculate daily returns
        if rolling_period is None:
            # Calculate daily returns on portfolio values
            portfolio_returns = portfolio_values.pct_change().dropna()
        else:
            # Calculate rolling returns according to formula: Return_t = (V_t+horizon - V_t) / V_t
            returns = []
            dates = []
            
            for i in range(rolling_period):
                v0 = portfolio_values.iloc[i]
                v1 = portfolio_values.iloc[i + self.time_horizon]
                returns.append((v1 - v0) / v0)
                dates.append(portfolio_values.index[i])
            
            portfolio_returns = pd.Series(returns, index=dates)
        
        return portfolio_returns
    
    def calculate_var(self, portfolio_returns, method='historical', is_period_returns=False):
        """Calculate Value at Risk using historical method
        
        Args:
            portfolio_returns: Series of portfolio returns
            method: Method to use (currently only 'historical' supported)
            is_period_returns: If True, returns are already period returns (no sqrt scaling needed)
        """
        
        # Historical VaR
        var_return = portfolio_returns.quantile(1 - self.confidence_level)
        var_dollar = abs(var_return * self.portfolio_value)
        var_percentage = abs(var_return)
        # var_percentile = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
    
        
        return {
            'var_percentage': var_return,
            'var_dollar': var_dollar,
            'portfolio_value': self.portfolio_value,
            'method': method,
            'confidence_level': self.confidence_level  # Include confidence level in results
        }
    
    def calculate_conditional_var(self, portfolio_returns, is_period_returns=False):
        """Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            portfolio_returns: Series of portfolio returns
            is_period_returns: If True, returns are already period returns (no sqrt scaling needed)
        """
        var_threshold = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        cvar_percentage = portfolio_returns[portfolio_returns <= var_threshold].mean()
        
        if is_period_returns:
            # Returns are already period returns, no scaling needed
            cvar_dollar = cvar_percentage * self.portfolio_value
        else:
            # Returns are daily returns, scale by sqrt(time_horizon)
            cvar_dollar = cvar_percentage * self.portfolio_value * np.sqrt(self.time_horizon)
        
        return {
            'cvar_percentage': cvar_percentage,
            'cvar_dollar': cvar_dollar
        }

class VaRBacktester:
    """VaR Backtesting for model validation"""
    
    def __init__(self, portfolio_data, confidence_level=0.95, time_horizon=1):
        self.portfolio_data = portfolio_data
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        
    def run_backtest(self, start_date, end_date, number_of_days_back=252):
        """Run VaR backtesting over a specified period using historical VaR - OPTIMIZED VERSION"""
        
        # Generate backtest dates (business days) - ensure enough buffer from current date
        # Adjust end_date if it's too close to today to ensure data availability
        today = datetime.now().date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
        if isinstance(start_date, datetime):
            start_date = start_date.date()
            
        # Ensure end_date is at least 5 business days before today
        max_end_date = today - timedelta(days=7)
        if end_date > max_end_date:
            end_date = max_end_date
            st.warning(f"End date adjusted to {end_date} to ensure data availability")
        
        backtest_dates = pd.bdate_range(start=start_date, end=end_date, freq='W')  # Weekly backtesting
        
        # Apply frequency filter based on selection
        if hasattr(self, '_frequency_setting'):
            freq_map = {
                "Weekly (Faster)": 1,      # Every week
                "Bi-weekly (Balanced)": 2, # Every 2 weeks  
                "Monthly (Fastest)": 4     # Every 4 weeks (approximately monthly)
            }
            step = freq_map.get(self._frequency_setting, 1)
            backtest_dates = backtest_dates[::step]
        
        # OPTIMIZATION 1: Fetch all market data once at the beginning
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Fetching comprehensive market data for backtesting...")
        progress_bar.progress(0.1)
        
        # Fetch a large dataset that covers the entire backtest period + rolling window
        # Use a much larger buffer to ensure we have enough data
        data_start_date = start_date - timedelta(days=int(number_of_days_back * 2.5))  # Increased buffer
        data_end_date = end_date + timedelta(days=30)  # Larger buffer
        
        # Get all market data at once
        symbols = self.portfolio_data['Symbol'].tolist()
        all_market_data = {}
        
        try:
            for i, symbol in enumerate(symbols):
                status_text.text(f'Fetching historical data for {symbol}...')
                progress_bar.progress(0.1 + (i + 1) / len(symbols) * 0.4)  # Progress from 10% to 50%
                
                ticker = yf.Ticker(symbol)
                
                # Get all data for the entire period
                hist = ticker.history(start=data_start_date, end=data_end_date)
                
                if not hist.empty and len(hist) >= number_of_days_back:
                    all_market_data[symbol] = hist['Close']
                else:
                    st.warning(f"Insufficient historical data for {symbol}. This may affect backtest results.")
            
            # Combine all data into a single dataframe
            if not all_market_data:
                st.error("No valid market data could be fetched for backtesting")
                progress_bar.empty()
                status_text.empty()
                return None
                
            all_price_data = pd.DataFrame(all_market_data)
            
            # Run the actual backtesting
            results = []
            var_breaches = 0
            
            status_text.text("Running VaR backtests...")
            
            for i, date in enumerate(backtest_dates):
                progress_bar.progress(0.5 + (i + 1) / len(backtest_dates) * 0.5)  # Progress from 50% to 100%
                
                # Get price data up to this date
                date_str = date.strftime('%Y-%m-%d')
                price_data_to_date = all_price_data[:date_str].tail(number_of_days_back)
                
                # Skip if we don't have enough data
                if len(price_data_to_date) < number_of_days_back:
                    continue
                
                # Calculate portfolio returns
                returns = price_data_to_date.pct_change().dropna()
                
                # Calculate portfolio weights
                weights = (self.portfolio_data['Quantity'] * self.portfolio_data['Current_Price']) 
                weights = weights / weights.sum()
                
                # Calculate portfolio returns
                portfolio_returns = (returns * weights.values).sum(axis=1)
                
                # Calculate VaR
                var_percentile = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
                
                # Get next day return if available (for breach checking)
                next_day = pd.bdate_range(start=date, periods=2, freq='B')[1]
                next_day_str = next_day.strftime('%Y-%m-%d')
                
                # Try to get next day data
                next_day_actual = None
                next_day_breach = False
                
                # Look for the next available data point after the VaR date
                if next_day_str in all_price_data.index:
                    next_price = all_price_data.loc[next_day_str]
                    prev_price = all_price_data.iloc[all_price_data.index.get_indexer([date_str], method='pad')[0]]
                    
                    # Calculate actual return
                    next_day_returns = (next_price - prev_price) / prev_price
                    next_day_actual = (next_day_returns * weights.values).sum()
                    
                    # Check if VaR was breached
                    next_day_breach = next_day_actual < var_percentile
                    if next_day_breach:
                        var_breaches += 1
                
                # Store results
                results.append({
                    'Date': date,
                    'VaR (%)': var_percentile * 100,
                    'Next Day Return (%)': None if next_day_actual is None else next_day_actual * 100,
                    'VaR Breach': next_day_breach
                })
            
            progress_bar.empty()
            status_text.empty()
            
            if not results:
                st.error("No valid backtest results could be generated")
                return None
                
            results_df = pd.DataFrame(results)
            
            # Calculate breach rate
            breach_rate = var_breaches / len(results_df) * 100
            expected_breach_rate = (1 - self.confidence_level) * 100
            
            return {
                'results': results_df,
                'breach_rate': breach_rate,
                'expected_breach_rate': expected_breach_rate,
                'var_breaches': var_breaches,
                'total_days': len(results_df)
            }
            
        except Exception as e:
            st.error(f"Error during backtesting: {str(e)}")
            progress_bar.empty()
            status_text.empty()
            return None
            
    def set_backtest_frequency(self, frequency):
        """Set the frequency of backtesting points"""
        self._frequency_setting = frequency

# Utility functions
def process_portfolio_data(portfolio_df):
    """Process and validate portfolio data"""
    try:
        # Check if there's a Date column and if it contains valid dates
        is_historical = False
        if 'Date' in portfolio_df.columns:
            try:
                # Convert to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(portfolio_df['Date']):
                    portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
                is_historical = True
            except:
                st.error("Date column must contain valid dates (YYYY-MM-DD)")
                return None
        
        # Define required columns based on portfolio type
        if is_historical:
            # For historical portfolios, we only need Date, Symbol, and Quantity
            required_columns = ['Date', 'Symbol', 'Quantity']
            
            # If Current_Price is not provided, add a placeholder value (will be replaced with actual prices)
            if 'Current_Price' not in portfolio_df.columns:
                portfolio_df['Current_Price'] = 1.0  # Placeholder value
        else:
            # For current portfolios, we only need Symbol and Quantity
            # Current_Price will be fetched automatically if not provided
            required_columns = ['Symbol', 'Quantity']
        
        # Check required columns
        for col in required_columns:
            if col not in portfolio_df.columns:
                st.error(f"Missing required column: {col}")
                return None
                
        # Check for numerical values
        for col in ['Quantity']:
            try:
                portfolio_df[col] = pd.to_numeric(portfolio_df[col])
            except:
                st.error(f"Column {col} must contain numerical values")
                return None
        
        # For current portfolios, fetch current prices if not provided
        if not is_historical:
            if 'Current_Price' not in portfolio_df.columns:
                st.info("Fetching current prices for your portfolio...")
                current_prices = fetch_current_prices(portfolio_df['Symbol'].tolist())
                
                if not current_prices:
                    st.error("Failed to fetch current prices for the portfolio symbols.")
                    return None
                
                # Create Current_Price column from fetched prices
                portfolio_df['Current_Price'] = portfolio_df['Symbol'].map(current_prices)
                
                # Check if any prices are missing
                missing_prices = portfolio_df[portfolio_df['Current_Price'].isna()]['Symbol'].tolist()
                if missing_prices:
                    st.error(f"Could not fetch prices for the following symbols: {', '.join(missing_prices)}")
                    return None
            else:
                # If Current_Price is provided, validate it
                try:
                    portfolio_df['Current_Price'] = pd.to_numeric(portfolio_df['Current_Price'])
                except:
                    st.error("Current_Price column must contain numerical values")
                    return None
                
                # Check for valid prices
                if (portfolio_df['Current_Price'] <= 0).any():
                    st.error("All prices must be greater than zero")
                    return None
                
        # Check for valid quantities
        if (portfolio_df['Quantity'] <= 0).any():
            st.error("All quantities must be greater than zero")
            return None
        
        # Calculate Value column for display purposes
        portfolio_df['Value'] = portfolio_df['Quantity'] * portfolio_df['Current_Price']
                
        return portfolio_df, is_historical
        
    except Exception as e:
        st.error(f"Error processing portfolio data: {str(e)}")
        return None

def load_and_process_csv(uploaded_file):
    """Load and process a CSV file containing portfolio data"""
    try:
        # Read CSV
        portfolio_df = pd.read_csv(uploaded_file)
        
        # Process the portfolio data
        result = process_portfolio_data(portfolio_df)
        if result is None:
            return None
            
        portfolio_df, is_historical = result
        return portfolio_df, is_historical
        
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def load_and_process_text(text_data):
    """Load and process text data containing portfolio information"""
    try:
        # Convert the text to a pandas dataframe
        text_data = io.StringIO(text_data)
        portfolio_df = pd.read_csv(text_data, sep=r'\s*,\s*|\s+\|\s+|\s*;\s*|\s+', engine='python')
        
        # Process the portfolio data
        result = process_portfolio_data(portfolio_df)
        if result is None:
            return None
            
        portfolio_df, is_historical = result
        return portfolio_df, is_historical
        
    except Exception as e:
        st.error(f"Error parsing text data: {str(e)}")
        st.error("Please ensure your data is formatted correctly with headers and proper separators")
        return None

def get_sample_current_portfolio():
    """Generate a sample current portfolio for demonstration"""
    return pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'],
        'Quantity': [10, 5, 3, 2, 8]
    })

def get_sample_historical_portfolio():
    """Generate a sample historical portfolio for backtesting demonstration"""
    # Generate some dates for the past 6 months
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Create a sample portfolio with dates
    data = []
    for date in dates:
        data.extend([
            {'Date': date, 'Symbol': 'AAPL', 'Quantity': 10},
            {'Date': date, 'Symbol': 'MSFT', 'Quantity': 5},
            {'Date': date, 'Symbol': 'AMZN', 'Quantity': 3},
            {'Date': date, 'Symbol': 'GOOGL', 'Quantity': 2},
            {'Date': date, 'Symbol': 'META', 'Quantity': 8}
        ])
    
    return pd.DataFrame(data)

def render_var_metrics(var_results, cvar_results):
    """Render VaR and CVaR metrics in a visually appealing way"""
    
    # Format VaR and CVaR values
    var_pct = var_results['var_percentage'] * 100
    var_dollar = var_results['var_dollar']
    portfolio_value = var_results['portfolio_value']
    cvar_pct = cvar_results['cvar_percentage'] * 100
    cvar_dollar = cvar_results['cvar_dollar']
    
    # Determine risk level
    def get_risk_class(var_pct):
        if var_pct > 3:
            return "risk-high"
        elif var_pct > 2:
            return "risk-medium"
        else:
            return "risk-low"
    
    risk_class = get_risk_class(abs(var_pct))
    
    # Create metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card {risk_class}">
            <h3>Portfolio Value</h3>
            <h2>${portfolio_value:,.2f}</h2>
            <p>Total investment value</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card {risk_class}">
            <h3>Historical VaR (95%)</h3>
            <h2>{abs(var_pct):.2f}% / ${abs(var_dollar):,.2f}</h2>
            <p>Maximum expected loss in a day</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card {risk_class}">
            <h3>Conditional VaR (ES)</h3>
            <h2>{abs(cvar_pct):.2f}% / ${abs(cvar_dollar):,.2f}</h2>
            <p>Average loss in worst scenarios</p>
        </div>
        """, unsafe_allow_html=True)
        
def plot_var_results(returns_data, var_results):
    """Plot VaR results with distribution and visualization"""
    
    portfolio_returns = returns_data
    var_threshold = var_results['var_percentage']
    confidence_level = var_results.get('confidence_level', 0.95)  # Get confidence level from results or default to 0.95
    
    # Create a figure with two subplots - one for returns, one for distribution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 2]})
    
    # Plot 1: Historical returns with VaR line
    ax1.plot(portfolio_returns.index, portfolio_returns * 100, label='Daily Returns', linewidth=1, color='navy')
    ax1.axhline(y=var_threshold * 100, color='red', linestyle='--', 
                label=f'VaR ({confidence_level*100:.0f}%): {var_threshold * 100:.2f}%')
    
    # Highlight VaR breaches
    breaches = portfolio_returns[portfolio_returns < var_threshold]
    if not breaches.empty:
        ax1.scatter(breaches.index, breaches * 100, color='red', s=30, label=f'VaR Breaches: {len(breaches)}')
    
    ax1.set_title(f'Historical Portfolio Returns with VaR ({confidence_level*100:.0f}%)', fontsize=14)
    ax1.set_ylabel('Returns (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Calculate breach percentage
    breach_pct = len(breaches) / len(portfolio_returns) * 100
    expected_breach_pct = (1 - confidence_level) * 100  # Dynamic expected breach percentage
    
    # Add text box with breach statistics
    textbox = f"VaR Breaches: {len(breaches)}/{len(portfolio_returns)} ({breach_pct:.2f}%)\n"
    textbox += f"Expected Breaches: {expected_breach_pct:.2f}%\n"
    
    # Add accuracy measure
    accuracy = 100 - abs(breach_pct - expected_breach_pct)
    textbox += f"Model Accuracy: {accuracy:.2f}%"
    
    # Place the text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.05, textbox, transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    # Plot 2: Return distribution with VaR
    sns.histplot(portfolio_returns * 100, kde=True, ax=ax2, color='navy', stat='density', alpha=0.6)
    
    # Add VaR line
    ax2.axvline(x=var_threshold * 100, color='red', linestyle='--', 
                label=f'VaR ({confidence_level*100:.0f}%): {var_threshold * 100:.2f}%')
    
    # Shade the VaR region
    xmin, xmax = ax2.get_xlim()
    x_fill = np.linspace(xmin, var_threshold * 100, 100)
    y_fill = ax2.get_ylim()[1]
    ax2.fill_between(x_fill, 0, y_fill, alpha=0.2, color='red')
    
    ax2.set_title(f'Return Distribution with VaR ({confidence_level*100:.0f}%)', fontsize=14)
    ax2.set_xlabel('Returns (%)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_backtest_results(backtest_results):
    """Plot VaR backtesting results"""
    
    results_df = backtest_results['results']
    breach_rate = backtest_results['breach_rate']
    expected_breach_rate = backtest_results['expected_breach_rate']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: VaR vs Actual Returns
    ax1.plot(results_df['Date'], results_df['VaR (%)'], 
             label='Historical VaR (95%)', color='navy', linewidth=2)
    
    # Plot actual returns where available
    valid_returns = results_df.dropna(subset=['Next Day Return (%)'])
    ax1.scatter(valid_returns['Date'], valid_returns['Next Day Return (%)'], 
                label='Actual Next-Day Returns', color='gray', alpha=0.6, s=30)
    
    # Highlight VaR breaches
    breaches = valid_returns[valid_returns['VaR Breach']]
    if not breaches.empty:
        ax1.scatter(breaches['Date'], breaches['Next Day Return (%)'],
                   color='red', s=50, label='VaR Breaches')
    
    ax1.set_title('Historical VaR Backtesting Results', fontsize=14)
    ax1.set_ylabel('Returns / VaR (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add text with breach statistics
    textbox = (f"VaR Breaches: {backtest_results['var_breaches']}/{backtest_results['total_days']} "
               f"({breach_rate:.2f}%)\n"
               f"Expected Breaches: {expected_breach_rate:.2f}%\n")
    
    # Add accuracy measure
    accuracy = 100 - abs(breach_rate - expected_breach_rate)
    textbox += f"Model Accuracy: {accuracy:.2f}%"
    
    # Place text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.05, textbox, transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    # Plot 2: Breach timeline
    dates = results_df['Date']
    breach_timeline = np.zeros(len(dates))
    
    # Set 1 for breach days
    breach_indices = results_df[results_df['VaR Breach']].index
    breach_timeline[breach_indices] = 1
    
    # Plot as stem plot
    markerline, stemlines, baseline = ax2.stem(
        dates, breach_timeline, linefmt='r-', markerfmt='ro', basefmt='k-')
    plt.setp(stemlines, linewidth=0.5)
    plt.setp(markerline, markersize=5)
    
    # Customize the plot
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No Breach', 'Breach'])
    ax2.set_title('VaR Breach Timeline', fontsize=14)
    ax2.grid(True, axis='x', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def fetch_current_prices(symbols):
    """Fetch current prices for a list of symbols using yfinance"""
    if not symbols:
        return {}
    
    try:
        # Create a progress display
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Fetching current prices for {len(symbols)} symbols...")
        
        # Fetch data for all symbols
        current_prices = {}
        for i, symbol in enumerate(symbols):
            try:
                # Update progress
                progress_bar.progress((i + 1) / len(symbols))
                status_text.text(f"Fetching price for {symbol}...")
                
                # Get current price
                ticker = yf.Ticker(symbol)
                ticker_info = ticker.info
                
                # Different price fields can be used depending on what's available
                price = None
                for field in ['currentPrice', 'regularMarketPrice', 'previousClose', 'lastPrice']:
                    if field in ticker_info and ticker_info[field] is not None:
                        price = ticker_info[field]
                        break
                
                if price is not None:
                    current_prices[symbol] = price
                else:
                    # As a fallback, try to get the last closing price from history
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        current_prices[symbol] = float(hist['Close'].iloc[-1])
                    else:
                        st.warning(f"Could not fetch current price for {symbol}")
            except Exception as e:
                st.warning(f"Error fetching price for {symbol}: {str(e)}")
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return current_prices
    except Exception as e:
        st.error(f"Error fetching current prices: {str(e)}")
        return {}
