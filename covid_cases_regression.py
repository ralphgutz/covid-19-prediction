# overfitting & splitting the values are discarded
# however, splitting can also be used for metrics

import numpy as np
import pandas as pd

df = pd.read_csv("owid-covid-data.csv")

df = df[df.iso_code == "PHL"]
# df = df[df.date >= "2021-01-01"]

df = df[["date", "total_cases", "new_cases", "total_deaths", 
         "new_deaths", "reproduction_rate"]]
df["date"] = pd.to_datetime(df.date)
df.set_index("date", inplace=True)
# df_2021 = df[df.index > "2020-11-01"] # sdfsdgsdg

X = np.arange(1, len(df.index)+1).reshape(-1, 1)
y = np.array(df.total_cases).reshape(-1, 1)

x_axis = np.arange(0, len(X), 1)
total_cases = np.array(df.total_cases)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
from sklearn.metrics import r2_score, mean_squared_error

rmse = []
degrees = 5  # start degree

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(11, 9))

for i, ax in enumerate(axes.flatten()):
    
    if i == 0:
        model_linear = LinearRegression().fit(X, y)
        response_linear = model_linear.intercept_ + x_axis * model_linear.coef_
        
        ax.plot(df.index, total_cases, label="Actual Cases", c="tab:orange")
        ax.plot(df.index, response_linear[0], label="Prediction", linestyle="dashed", c="tab:blue")
        
        ax.set_title("Linear Regression", fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.set_xlabel("Date")
        ax.set_ylabel("No. of cases")
        ax.text(0.98, 0.02, "R^2 = %.4f" % float(str(r2_score(y, response_linear[0], ))[:-11]), 
                verticalalignment="bottom", horizontalalignment="right", 
                transform=ax.transAxes)
        ax.legend()
        
        rmse.append(np.sqrt(mean_squared_error(y, response_linear[0])))
        continue
    
    
    poly_obj = PolynomialFeatures(degree = degrees)
    X_poly = poly_obj.fit_transform(X)
    
    model_poly = LinearRegression().fit(X_poly, y)
    response_poly = model_poly.predict(X_poly)
    
    ax.plot(df.index, total_cases, label="Actual Cases", c="tab:orange")
    ax.plot(df.index, response_poly, label="Prediction", linestyle="dashed", c="tab:blue")
    
    ax.set_title("%dth Degree Polynomial Reg." % degrees, fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_xlabel("Date")
    ax.set_ylabel("No. of cases")
    ax.text(0.98, 0.02, "R^2 = %.4f" % float(str(r2_score(y, response_poly))[:-11]),
        verticalalignment="bottom", horizontalalignment="right",
        transform=ax.transAxes)
    ax.legend()
    
    rmse.append(np.sqrt(mean_squared_error(y, response_poly)))
    degrees += 1


fig.autofmt_xdate()
fig.tight_layout()   

fname = "regression_" + str(df.index[-1])[:-9]
plt.savefig(fname + ".png", dpi=100)

print(rmse)