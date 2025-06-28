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

<img width="1440" alt="image" src="https://github.com/user-attachments/assets/927e7530-cfcf-498d-a8aa-6203976d04be" />

<img width="1440" alt="image" src="https://github.com/user-attachments/assets/610de9c5-049a-465d-a680-63ab0f82e6df" />

<img width="1437" alt="image" src="https://github.com/user-attachments/assets/e442e66d-aef3-4e95-8c44-a725dd3010ae" />

<img width="1440" alt="image" src="https://github.com/user-attachments/assets/d4cc25e2-e6b8-4f44-9992-aeca64474527" />



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
