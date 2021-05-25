import pandas as pd
import numpy as np
from scipy.optimize import leastsq
import pylab as plt
from sklearn.metrics import r2_score
from json import dump

wind = pd.read_csv("../data/final_wind.csv")

wind["DATE"] = pd.to_datetime(wind["DATE"])
wind["DATE"] = (wind["DATE"] - wind["DATE"].min()).dt.days

avg_wind = wind.groupby("DATE").mean()
data = avg_wind.dropna(subset=["generation"])
t = data.index
data = data["generation"]

guess_mean = np.mean(data)-1000
guess_phase = -300
guess_freq = 1/50
guess_amp = 2000
guess_increase = 2.5

data_first_guess = guess_amp*np.sin(guess_freq*t+guess_phase) + guess_mean + t*guess_increase

optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] + t*x[4] - data
est_amp, est_freq, est_phase, est_mean, est_increase = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean, guess_increase])[0]

model_parameters = {"est_amp": est_amp,
                    "est_freq": est_freq,
                    "est_phase": est_phase,
                    "est_mean": est_mean,
                    "est_increase": est_increase}
dump(model_parameters, "../models/wind_sine.json")

data_fit = est_amp*np.sin(est_freq*t+est_phase) + est_mean + t*est_increase


score = r2_score(data, data_fit)
print("The r2 score of the model is {:.3f}".format(score))

plt.plot(t, data, '.', label="actual")
plt.plot(t, data_fit, label="predicted")
plt.legend()
plt.title("Sine Curve Fit to Wind Power Generation")
plt.show()
