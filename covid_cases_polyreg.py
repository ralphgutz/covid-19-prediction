import numpy as np
import pandas as pd

df = pd.read_csv("owid-covid-data.csv")

df = df[df.iso_code == "PHL"]

df = df[["date", "total_cases", "new_cases", "total_deaths", 
         "new_deaths", "reproduction_rate"]]
df["date"] = pd.to_datetime(df.date)
df.set_index("date", inplace=True)

df = df[df.index >= "2021-01-01"]

X = np.arange(1, len(df.index)+1).reshape(-1, 1)
y = np.array(df.new_cases).reshape(-1, 1)

x_axis = np.arange(0, 90, 1)
total_cases = np.array(df.new_cases)


from sklearn.preprocessing import PolynomialFeatures

poly_obj = PolynomialFeatures(degree = 4)
X_poly = poly_obj.fit_transform(X)


from sklearn.linear_model import LinearRegression

model_poly = LinearRegression().fit(X_poly, y)

response = model_poly.intercept_ + model_poly.coef_[0][1] * x_axis + model_poly.coef_[0][2] * x_axis**2 + model_poly.coef_[0][3] * x_axis**3 + model_poly.coef_[0][4] * x_axis**4


new_cases = np.array(df.new_cases)

new_cases = np.append(new_cases, np.zeros(12))  # adjust np.zeros(n)
new_cases[new_cases == 0] = np.nan

datelist = pd.date_range("2021-01-01", periods=90)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(datelist, response, color="tab:blue", label="Prediction")
ax.scatter(datelist, new_cases, color="tab:orange", label="New Daily Cases")
ax.tick_params(axis="both", labelsize=12) 
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("No. of cases", fontsize=12)
ax.legend(loc="upper left", fontsize=12)
ax.set_title("New Cases Prediction using Polynomial Regression (deg=4)", fontsize=15)
ax.text(0.02, 0.85, "Error: ±%.2f Cases" % np.mean((response[:78] - new_cases[:78])**2)**.5,  # adjust to get predicted values (not forecast)
        verticalalignment="top", horizontalalignment="left",
        transform=ax.transAxes, fontsize=12)

fname = "poly_reg_" + str(df.index[-1])[:-9]
plt.savefig(fname + ".png", dpi=100)


from sklearn.metrics import r2_score

new_cases[~np.isnan(new_cases)]  # catch all nan values

r2_score(new_cases[:78], response[:78])  # adjust to get predicted values (not forecast)
np.mean((response[:78] - new_cases[:78])**2)**.5  # adjust to get predicted values (not forecast)
