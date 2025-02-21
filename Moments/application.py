# Code that generates the application of the Black-Scholes model. #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Fri 21 Feb 2025 @ 19:11:19 +0100
# Modified: Fri 21 Feb 2025 @ 20:00:34 +0100

# Packages
import numpy as np
import pandas as pd
from scipy.stats import norm

# Default Black-Scholes Model
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Check option type
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type.")

# Adjusted Black-Scholes Model
def adjusted_black_scholes(S, K, T, r, sigma, skewness, kurtosis, option_type="call"):
    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Standard normal PDF and CDF
    phi_d1 = norm.pdf(d1)
    Phi_d1 = norm.cdf(d1)

    # Gram-Charlier adjustments
    skew_adj = (skewness / 6) * (d1**3 - 3*d1)
    kurt_adj = ((kurtosis - 3) / 24) * (d1**4 - 6*d1**2 + 3)

    # Adjusted CDF
    adjusted_Phi_d1 = Phi_d1 - phi_d1 * (skew_adj + kurt_adj)
    adjusted_Phi_d2 = norm.cdf(d2)  # No direct correction applied
    
    # Check option type
    if option_type == "call":
        return S * adjusted_Phi_d1 - K * np.exp(-r * T) * adjusted_Phi_d2
    elif option_type == "put":
        return K * np.exp(-r * T) * (1 - adjusted_Phi_d2) - S * (1 - adjusted_Phi_d1)
    else:
        raise ValueError("Invalid option type.")

# Main script
def main():
    S = 137.29          # Closing price of NVIDIA stock
    r = 0.05            # Risk-free rate
    sigma = 0.35        # Estimated annualized volatility
    T = 2               # Maturity in years
    skewness = -1.5     # Estimated skewness from log returns
    kurtosis = 6        # Estimated excess kurtosis

    # Strike prices of the call options at 45, 50 and 60 euros per contract
    strike_prices = [45, 50, 60]
    
    # Call the functions and compute practical results
    results = []
    for K in strike_prices:
        bs_price = black_scholes(S, K, T, r, sigma, option_type="call")
        adjusted_price = adjusted_black_scholes(S, K, T, r, sigma, skewness,
                                                kurtosis, option_type="call")
        difference = ((adjusted_price - bs_price) / bs_price) * 100
        results.append([K, T, bs_price, adjusted_price, difference])

    option_prices = pd.DataFrame(results, columns=["Strike Price",
                                                   "Time to Maturity",
                                                   "Black-Scholes",
                                                   "MoM Black-Scholes",
                                                   "% Difference"])
    print(option_prices)

if __name__ == "__main__":
    main()
