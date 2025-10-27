What this tool does

This script pulls historical trades for settled Kalshi markets, reconstructs the daily price of the losing side for each market, 
aligns those daily prices by Days Till Event (DTE), and computes one averaged price per DTE across all markets. 
It then saves a tidy CSV and a dark-themed PNG plot of the averaged decay curve.

Use it when you want a clean, “theta-like” decay profile of prediction/event contracts aggregated over many markets.


kalshi_decay_avg_with_filter.py: allows the user to add a minumum days filter in which there must be at least x days of data/trades. Useful to eliminate shorter term markets where the market is created only a few days before event, and thus is irrelevant.
In command prompt put: python kalshi_decay_avg.py --min-coverage 59 --output-prefix ".\analysis_results\kalshi_avg_min59" if you want 59 days, change when necessary 
