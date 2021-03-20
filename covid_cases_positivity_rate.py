import pandas as pd

df = pd.read_csv("owid-covid-data.csv")
df = df[df.iso_code == "PHL"]

df = df[["date", "total_cases", "new_cases", "new_cases_smoothed", "total_deaths", 
         "new_deaths", "reproduction_rate", "new_tests", "new_tests_smoothed"]]
df["date"] = pd.to_datetime(df.date)
df.set_index("date", inplace=True)

df["positive_rate"] = (df.new_cases / df.new_tests)*100

cumulative_pr = sum(df.new_cases.dropna()) / sum(df.new_tests.dropna())


import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

fig, ax = plt.subplots(figsize=(12, 7))

df.positive_rate.dropna().plot(ax=plt.gca(), c="tab:blue")
ax.tick_params(axis="both", labelsize=12) 
ax.yaxis.set_major_formatter(FormatStrFormatter("%d%%"))
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Percentage", fontsize=12)
ax.set_title("Daily Positivity Rate (April 2020 - Present)", fontsize=15)

ax.text(.96, .92, "Cumulative Positivity Rate: %.2f%%" % (cumulative_pr*100), 
        verticalalignment="top", horizontalalignment="right", fontsize="13",
        bbox={"facecolor": "white", "alpha": 1, "pad": 7},
        transform=ax.transAxes)

fname = "positivity_rate_" + str(df.index[-1])[:-9]
plt.savefig(fname + ".png", dpi=100)