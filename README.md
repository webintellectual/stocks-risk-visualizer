# 📈 Historical VaR Portfolio Analyzer

**Professional Historical VaR Analysis Portal for Investment Portfolios**

---

## Overview

This project is a Streamlit-based web application for analyzing the Value at Risk (VaR) of investment portfolios using historical data. It provides two main types of risk analysis:

- **Point-in-Time VaR**: Analyze your portfolio’s risk as of today using recent market data.
- **Time Series VaR**: Track how your portfolio’s risk evolves over time.

---

## Features

- **Interactive Web UI** (Streamlit)
- **Point-in-Time VaR**:  
  - Calculates current portfolio risk at a chosen confidence level  
  - Visualizes maximum potential loss  
  - Useful for asset allocation and benchmarking

- **Time Series VaR**:  
  - Shows risk trends and historical patterns  
  - Identifies periods of elevated or reduced risk  
  - Stress-tests against past market scenarios

- **Data Sources**:  
  - Fetches historical prices using Yahoo Finance (`yfinance`)
  - Supports CSV uploads for custom portfolios

- **Visualizations**:  
  - Modern, interactive charts (Plotly, Matplotlib, Seaborn)
  - Clear risk metrics and summaries

---

## Screenshots

> Pending to put

---

## Getting Started

### Prerequisites

- Python 3.13+
- [Poetry](https://python-poetry.org/) for dependency management

### Installation

```bash
git clone https://github.com/yourusername/stocks-risk-visualizer.git
cd stocks-risk-visualizer
poetry install
```

### Running the App

```bash
poetry run streamlit run src/app.py
```

---

## Usage

1. **Home Page**: Choose between Point-in-Time VaR and Time Series VaR analysis.
2. **Point-in-Time VaR**:  
   - Upload your portfolio or use the sample  
   - Set confidence level and window  
   - View risk metrics and visualizations

3. **Time Series VaR**:  
   - Upload time series portfolio data or use the sample  
   - Analyze risk trends over time

## Author

Akshay  
[GitHub Profile](https://github.com/webintellectual)

[LinkedIn Profile](https://www.linkedin.com/in/aiiitv/)

---
