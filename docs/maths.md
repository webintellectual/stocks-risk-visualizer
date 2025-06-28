# Value at Risk (VaR):
A statistical measure that estimates how much a portfolio might lose, with a given probability, over a certain time horizon.
Example: “There is a 95% chance that the portfolio will not lose more than ₹1 crore in one day.”

## Time Horizon & Return
Time horizon is the length of time over which you measure each return. If you are interested in daily returns then time horizon is 1. if you are interested in weekly returns then time horizon is 7. If time horizon is 7 than you calculate return percentage by using the t-th day and (t+7)-th. Percentage return on (t+7)-th day relative to t-th day.
$$
\text{Return}_t^{(7)} = \frac{P_{t+7} - P_t}{P_t}
$$

In denominator, that prices comes with which respect to your calculating. So, the denominator is the starting price, i.e., the price at the beginning of the time horizon — at day $t$.

## Rolling Return
Return computed over the time horizon, sliding through the dataset
E.g. Let's say you have data of 10 days, then for time horizon 7 it will be like this:

Return of D0 to D7, then slide to return of D1 to D8 and then return of D2 to D9. Total 3 rolling returns are possible over this dataset of 10 days.

## Rolling Period
Number of such rolling returns used to estimate VaR.
If you use last 100 returns, rolling period = 100.

---
<br>

Required price days can be calculated if rolling period is provides and vice-versa. <br>
- Required price days = horizon + period
- Rolling Period = price days - horizon

To calculate VaR we can take either rolling period or time horizon as input to do VaR calculation.

# Step by Step VaR Calculation

## Setup:

| Input Parameter      | Value                     |
| -------------------- | ------------------------- |
| **Time horizon**     | 7 days                    |
| **Rolling period**   | 5                         |
| **Current date**     | 27 June                   |
| **Stocks**           | A and B                   |
| **Current holdings** | A: 10 shares, B: 5 shares |
| **Confidence level** | 95%                       |

So, required price days = 5 (rolling period) + 7 (time horizon) = 12

We’ll:
* Use fake historical prices for A and B. In real use case we download real history of real stocks of previous 11 days
* Compute rolling 7-day returns (5 of them)
* Estimate Historical VaR using those returns

**Do we include the current date (27 June) in the historical dataset?** <br>
No, typically we do not include the current date (27 June) in historical return calculations.

**Why?**

Because:

* You’re asking:
  > “If I hold **today’s portfolio (27 June)** over the next 7 days, what kind of loss might I face?”
* To answer that, you look at:
  > “How did this exact portfolio perform over similar **past 7-day periods**, ending **before today**?”

## Step 1: Create Historical Price Data

Let’s create **price data for 11 days** (from June 9 to June 26). In real world scenario we download it of fetch from finance API

| Date   | Price A | Price B |
| ------ | ------- | ------- |
| Jun 9  | ₹100    | ₹100    |
| Jun 10 | ₹100    | ₹200    |
| Jun 11 | ₹101    | ₹198    |
| Jun 12 | ₹99     | ₹199    |
| Jun 13 | ₹98     | ₹197    |
| Jun 14 | ₹96     | ₹196    |
| Jun 17 | ₹97     | ₹195    |
| Jun 18 | ₹98     | ₹198    |
| Jun 19 | ₹99     | ₹200    |
| Jun 20 | ₹100    | ₹202    |
| Jun 24 | ₹102    | ₹205    |
| Jun 26 | ₹103    | ₹206    |

## Step 2: Calculate Portfolio Value for Each Day

Use:

$$
\text{Portfolio Value}_t = 10 \times \text{Price}_A + 5 \times \text{Price}_B
$$

| Date   | Price A | Price B | Portfolio Value       |
| ------ | ------- | ------- | --------------------- |
| Jun 9  | 100     | 100     | ₹1000 + ₹500 = ₹1500  |
| Jun 10 | 100     | 200     | ₹1000 + ₹1000 = ₹2000 |
| Jun 11 | 101     | 198     | ₹1010 + ₹990 = ₹2000  |
| Jun 12 | 99      | 199     | ₹990 + ₹995 = ₹1985   |
| Jun 13 | 98      | 197     | ₹980 + ₹985 = ₹1965   |
| Jun 14 | 96      | 196     | ₹960 + ₹980 = ₹1940   |
| Jun 17 | 97      | 195     | ₹970 + ₹975 = ₹1945   |
| Jun 18 | 98      | 198     | ₹980 + ₹990 = ₹1970   |
| Jun 19 | 99      | 200     | ₹990 + ₹1000 = ₹1990  |
| Jun 20 | 100     | 202     | ₹1000 + ₹1010 = ₹2010 |
| Jun 24 | 102     | 205     | ₹1020 + ₹1025 = ₹2045 |
| Jun 26 | 103     | 206     | ₹1030 + ₹1030 = ₹2060 |

## Step 3: Compute Rolling 7-Day Returns

Use:

$$
\text{Return}_t = \frac{V_{t+7} - V_t}{V_t}
$$

| From → To       | Return Calculation   | Return     |
| --------------- | -------------------- | ---------- |
| Jun 9 → Jun 17 | (1945 - 2000) / 2000 | **–2.75%** |
| Jun 10 → Jun 17 | (1945 - 2000) / 2000 | **–2.75%** |
| Jun 11 → Jun 18 | (1970 - 2000) / 2000 | **–1.50%** |
| Jun 12 → Jun 19 | (1990 - 1985) / 1985 | **+0.25%** |
| Jun 13 → Jun 20 | (2010 - 1965) / 1965 | **+2.29%** |
| Jun 14 → Jun 24 | (2045 - 1940) / 1940 | **+5.41%** |

## Step 4: Sort the Returns

Sorted returns (ascending):

1. –2.75%
2. –1.50%
3. +0.25%
4. +2.29%
5. +5.41%

## Step 5: Compute Historical VaR

For **95% confidence**, you look for the **5th percentile**, which is roughly the **worst 1 out of 20 returns**.
But we only have 5 returns — so conservatively:
* Take the **worst** return as approximation for 95% VaR
  ➤ **VaR ≈ –2.75%**

5th percentile means that 5% of the data values are less than or equal to this value.

Which means only 5% of the bad returns (since sorted from worst to best) are equal to or less than the 5th percentile return. This gives us 95% confidence. lower the percentile, higher is the confidence level.

### Expected VaR Breaches
For a 95% confidence level VaR:
- We expect that losses will exceed VaR 5% of the time (1 - 0.95 = 0.05 or 5%)
- This means that in a sample of 100 days, we should expect about 5 breaches
- A "breach" occurs when the actual loss is worse than (more negative than) our VaR estimate

### Model Accuracy
Expected Breaches:

For 95% confidence level, we expect 5% of returns to be worse than VaR
So expected_breach_pct = 5%
Actual Breaches:

Count how many returns actually breached VaR
breach_pct = (number of breaches / total returns) × 100
Accuracy Calculation:

Take the absolute difference between actual and expected breach percentages
Subtract this difference from 100%
accuracy = 100 - |actual% - expected%|
Example:

If you have 200 returns and 95% confidence VaR:
Expected breaches = 5% = 10 breaches
If you actually get 11 breaches (5.5%):
accuracy = 100 - |5.5% - 5%| = 100 - 0.5 = 99.5%
If you get 20 breaches (10%):
accuracy = 100 - |10% - 5%| = 100 - 5 = 95%

A perfect model would have exactly 5% breaches (for 95% confidence). The further you deviate from this expected percentage:

Too many breaches (>5%) suggests your VaR is too optimistic
Too few breaches (<5%) suggests your VaR is too conservative
The accuracy percentage tells you how close you are to the ideal breach rate

## Step 6: Apply to Current Portfolio Value

Latest portfolio value = ₹2060 (on June 26)
So:

$$
\text{VaR}_{7d}^{95\%} = 2.75\% \times 2060 = \boxed{₹56.65}
$$

---

## Final Interpretation

> **"With 95% confidence, you will not lose more than ₹56.65 over the next 7 days."**
> Based on past 5 overlapping 7-day returns from June 10 to June 26.