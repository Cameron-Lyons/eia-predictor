import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

solar = pd.read_csv("..\data\final_solar.csv")
solar["LST_DATE"] = pd.to_datetime(solar["LST_DATE"])

wind = pd.read_csv("..\data\final_wind.csv")
wind["DATE"] = pd.to_datetime(wind["DATE"])
wind = wind[wind["generation"] >= 0]

avg_solar = solar.groupby("LST_DATE").mean()
avg_solar["30day_rolling_avg"] = avg_solar.generation.rolling(30).mean()

avg_wind = wind.groupby("DATE").mean()
avg_wind["30day_rolling_avg"] = avg_wind.generation.rolling(30).mean()

sns.lineplot(x='DATE',
             y='generation',
             data=avg_wind,
             label='Wind Generation')

sns.lineplot(x='DATE',
             y='30day_rolling_avg',
             data=avg_wind,
             label='30 Day Rolling Avg')
plt.show()

sns.lineplot(x='DATE',
             y='generation',
             data=avg_wind,
             label='Wind Generation')

sns.lineplot(x='DATE',
             y='30day_rolling_avg',
             data=avg_wind,
             label='30 Day Rolling Avg')
plt.show()