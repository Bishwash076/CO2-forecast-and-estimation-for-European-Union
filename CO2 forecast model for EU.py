import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Historical data
years = np.array([1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                  2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                  2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                  2020, 2021, 2022, 2023])
emissions = np.array([3881.03, 3815.65, 3689.01, 3619.73, 3601.17, 3647.68, 3732.57, 3666.44, 3656.89, 3601.49,
                      3612.87, 3669.73, 3670.06, 3749.10, 3756.56, 3748.43, 3765.69, 3715.68, 3639.80, 3344.39,
                      3438.29, 3340.99, 3272.26, 3189.76, 3051.99, 3108.18, 3107.87, 3131.70, 3062.56, 2918.21,
                      2639.37, 2814.00, 2747.25, 2492.88])

# Reshape data for sklearn
X = years.reshape(-1, 1)
y = emissions

# Create polynomial regression model (degree 3 for smooth trend capturing policy impacts)
degree = 3
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(X, y)

# Forecast up to 2050
future_years = np.arange(1990, 2051).reshape(-1, 1)
predictions = polyreg.predict(future_years)

# Apply policy impact adjustments (simplified EU ETS and 20-20-20 targets effect)
# Assume continued linear reduction post-2020 due to tightened EU ETS and net-zero goals
post_2020_years = future_years[future_years > 2020].reshape(-1, 1)
post_2020_pred = polyreg.predict(post_2020_years)
# Apply a reduction factor (e.g., 2% annual reduction post-2020 to reflect policy tightening)
reduction_factor = 0.98 ** (post_2020_years - 2020)
post_2020_pred_adjusted = post_2020_pred * reduction_factor.flatten()
predictions[future_years.flatten() > 2020] = post_2020_pred_adjusted

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(years, emissions, 'bo-', label='Historical CO2 Emissions')
plt.plot(future_years, predictions, 'r--', label='Forecasted Emissions (with Policy Impact)')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (MtCO2e)')
plt.title('EU CO2 Emissions Forecast to 2050 with EU ETS and 20-20-20 Policy Impact')
plt.grid(True)
plt.legend()

# Set x-axis ticks at 5-year intervals
plt.xticks(np.arange(1990, 2051, 5))
plt.tight_layout()

# Save and show plot
plt.savefig('emission_forecast.png')
plt.show()

# Print forecasted values for 5-year intervals
forecast_years = np.arange(2025, 2051, 5)
forecast_values = polyreg.predict(forecast_years.reshape(-1, 1))
for year, value in zip(forecast_years, forecast_values):
    print(f"Forecasted CO2 Emissions for {year}: {value:.2f} MtCO2e")
