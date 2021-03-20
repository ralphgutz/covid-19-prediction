import numpy as np
import pandas as pd

df = pd.read_csv("owid-covid-data.csv")
df = df[df.iso_code == "PHL"]

df = df[["date", "total_cases", "new_cases", "new_cases_smoothed", "total_deaths", 
         "new_deaths", "reproduction_rate", "new_tests", "new_tests_smoothed"]]
df["date"] = pd.to_datetime(df.date)
df.set_index("date", inplace=True)


import pmdarima as pm

auto_model = pm.auto_arima(df.total_cases, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=1,
                         start_P=0, seasonal=False,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

auto_model.summary()


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(df.total_cases, order=(1, 2, 3))  # from auto_model
model_fit = model.fit(disp=-1)

fc = model_fit.predict()

np.mean((fc - df.total_cases[1:])**2)**.5  # rmse


import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

mpl.rcParams["axes.prop_cycle"] = cycler(color=["tab:blue", "tab:orange"])
mpl.rcParams.update({'figure.autolayout': True})


fig, ax = plt.subplots(figsize=(12, 7))

ax.tick_params(axis="both", labelsize=12) 
model_fit.plot_predict("2021-01-01", "2021-04-01", ax=plt.gca())

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("No. of cases", fontsize=12)
ax.legend(loc="upper left", fontsize=12)
ax.set_title("ARIMA(1,2,3) Forecast of Total Cases", fontsize=15)

fname = "arima_totalcase_" + str(df.index[-1])[:-9]
plt.savefig(fname + ".png", dpi=100)


# for metrics or use df.new_cases as test (actual) values

train = df.total_cases[:400]
test = df.total_cases[400:]

model = ARIMA(train, order=(1, 2, 3))  
model_fit = model.fit(disp=-1)

fc, se, conf = model_fit.forecast(15, alpha=0.05)

np.mean((fc - test.values)**2)**.5  # rmse
np.mean(np.abs(fc - test.values)/np.abs(test.values))  # mape
