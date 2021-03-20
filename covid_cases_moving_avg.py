import numpy as np
import pandas as pd

df = pd.read_csv("owid-covid-data.csv")
df = df[df.iso_code == "PHL"]

df = df[["date", "total_cases", "new_cases", "new_cases_smoothed", "total_deaths", 
         "new_deaths", "reproduction_rate", "new_tests", "new_tests_smoothed"]]
df["date"] = pd.to_datetime(df.date)
df.set_index("date", inplace=True)

df = df[df.index >= "2021-01-01"]


# getting the moving averages

sr_movavg = []
interval = 7

i = 0
while i < len(df.index) - interval+1:
    group = df.new_cases[i : i+interval]
    
    group_avg = sum(group) / interval
    sr_movavg.append(group_avg)
    i += 1
    
    
front_nan = np.empty(6)
front_nan[:] = np.NaN

df["moving_avg"] = np.append(front_nan, np.array(sr_movavg))


import matplotlib.pyplot as plt

df[["new_cases", "moving_avg"]].plot()
plt.axvline(x=pd.to_datetime("2021-02-01"), label="Recent Quarantine Measure", 
            c="k", alpha=0.5, linestyle="dotted")


import pmdarima as pm

auto_model = pm.auto_arima(df.moving_avg.dropna(), start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=1,
                         start_P=0, seasonal=False,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

auto_model.summary()


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(df.moving_avg.dropna(), order=(0, 2, 1)) # from auto_model 
model_fit = model.fit(disp=-1)

# forecast, stderr, conf = model_fit.forecast(steps=16)


fig, ax = plt.subplots(2, 1, figsize=(12, 10))

df[["new_cases", "moving_avg"]].plot(ax=ax[0], color=["tab:blue", "tab:orange"])
ax[0].set_xlim(pd.to_datetime("2021-01-10"),pd.to_datetime("2021-04-01"))
ax[0].tick_params(axis="both", labelsize=12) 
ax[0].legend(loc="upper left", fontsize=12)
ax[0].axvline(x=pd.to_datetime("2021-02-01"), label="Recent Quarantine Measure", 
            c="k", alpha=0.5, linestyle="dotted")
ax[0].set_xlabel("Date", fontsize=12)
ax[0].set_ylabel("No. of new cases", fontsize=12)
ax[0].legend(loc="upper left", fontsize=12)
ax[0].set_title("New Cases vs. Moving Average", fontsize=15)

model_fit.plot_predict(2, "2021-04-01", ax=plt.gca())
ax[1].tick_params(axis="both", labelsize=12) 
ax[1].set_xlabel("Date", fontsize=12)
ax[1].set_ylabel("No. of new cases", fontsize=12)
ax[1].legend(loc="upper left", fontsize=12)
ax[1].set_title("Moving Average Forecast (Until April 1, 2021)", fontsize=15)
fig.tight_layout()

fname = "moving_avg_" + str(df.index[-1])[:-9]
plt.savefig(fname + ".png", dpi=100)

# fc starts at 2021-01-09 because earlier data are NaN
fc = model_fit.predict()
np.sqrt(np.mean((fc - df.moving_avg.dropna())**2))  # rmse

