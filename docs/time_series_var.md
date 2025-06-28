# Time Series VaR


| Use Case                                | Why It’s Useful                                                               |
| --------------------------------------- | ----------------------------------------------------------------------------- |
| **Risk trend analysis**                 | See how portfolio risk has changed over time                                  |
| **Portfolio evolution understanding**   | Tracks how position shifts affect future risk                                 |
| **Stress-testing historical scenarios** | See how the portfolio would’ve behaved around past crises (e.g., COVID crash) |
| **Model validation / backtesting**      | Compare predicted VaR vs actual losses                                        |


Example

"On 15 June 2025, the 1-week VaR was –2.5%"

It means:
➤ Based on market data up to 15 June,
➤ Your portfolio was at risk of losing 2.5% over the next 7 days

This VaR is tied to a specific date.

So this process helps in understanding how the risk has evolved and what trend it followed. May be 2 months ago users portfolio was at risk and now on current date it is not.


## 📉 Understand How Risk Changes Over Time

Instead of just asking:

> “How risky is my portfolio **today**?”

You're asking:

> “How has my portfolio’s **risk evolved** over the past weeks or months?”


## 🎯 Why This Is Powerful

| Insight You Get                         | What It Tells You                                        |
| --------------------------------------- | -------------------------------------------------------- |
| ✅ Rising VaR trend                      | Your portfolio became more volatile or riskier over time |
| ✅ Falling VaR trend                     | Your portfolio got safer or less exposed                 |
| ⚠️ VaR spikes during certain dates      | Those were likely **stress events** (e.g., market crash) |
| 🛡️ Low VaR today but high 2 months ago | Your portfolio is **better positioned now** than before  |


### 📌 Real Use Case:

> Imagine a user was heavily invested in tech stocks 2 months ago during a market drop. The VaR plot would show **elevated risk levels back then**.
> But today, after diversifying, the rolling VaR has dropped — showing reduced risk.

That’s **valuable insight** for the user, a fund manager, or a risk team.

