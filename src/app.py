import streamlit as st
from stocks_risk_visualizer.utils import apply_custom_css

st.set_page_config(
    page_title="Historical VaR Portfolio Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Historical VaR Portfolio Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("**Professional Historical VaR Analysis Portal for Investment Portfolios**")

    
    st.markdown("## 📊 Available Analysis Types")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 Point-in-Time VaR
        
        **What it does:**
        - Analyzes current portfolio's risk using recent market data
        - Shows your portfolio's maximum potential loss at a specific confidence level
        - Provides risk visualization and detailed metrics
        
        **When to use it:**
        - Understand your current portfolio's risk
        - Make asset allocation decisions
        - Compare against risk benchmarks
        
        [Go to Point-in-Time VaR Analysis →](Point_in_Time_VaR)
        """)
    
    with col2:
        st.markdown("""
        ### 📆 Time Series VaR
        
        **What it does:**
        - Tracks how your portfolio's risk evolves over time
        - Shows risk trend analysis and historical patterns
        - Identifies periods of elevated or reduced risk
        - Stress-tests against past market scenarios
        
        **When to use it:**
        - Understand how your portfolio risk changes over time
        - See if recent changes made your portfolio safer
        - Compare risk levels across different market conditions
        - Track the impact of portfolio adjustments on risk
        
        [Go to Time Series VaR →](2_Time_Series_VaR)
        """)

if __name__ == "__main__":
    main()
