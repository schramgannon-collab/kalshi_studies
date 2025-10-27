What this tool does

This script pulls historical trades for settled Kalshi markets, reconstructs the daily price of the losing side for each market, 
aligns those daily prices by Days Till Event (DTE), and computes one averaged price per DTE across all markets. 
It then saves a tidy CSV and a dark-themed PNG plot of the averaged decay curve.

Use it when you want a clean, “theta-like” decay profile of prediction/event contracts aggregated over many markets.
