import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import pandas as pd


column_names = ['JD', 'mag', 'error', 'good/bad', 'camera#', 'band', 'camera name']

ID = pd.read_table("13.5-14/index36.csv", sep=r'\,|\t', engine='python')

target = 420907467914 #asas-sn id

path = '13.5-14/lc36_cal/' + str(target) + '.dat'

df = pd.read_table(path, sep="\s+", names=column_names)
df.drop(df[df['good/bad'] < 1].index, inplace = True)

time = df['JD']
phot = df['mag']

#LS periodogram

frequency, power = LombScargle(time, phot).autopower()

#find best period
best_period = 1 / frequency[np.argmax(power)]

print(best_period)

#phased light curve
phase = (time / best_period) % 1

print(phase)

#plot
plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
#plt.plot(1/ frequency, power)
#plt.xscale('log')
#plt.xlabel('Period (days)')
#plt.ylabel('Power')
plt.subplot(1, 2, 1)
plt.scatter(phase, phot, s=5)
plt.xlabel('Phase')
plt.ylabel('Magnitude')
plt.title(f'Period: {best_period:.2f} days')

plt.subplot(1, 2, 2)
plt.errorbar(time-2450000, phot, yerr=df['error'], fmt='.', color='k', ecolor='r')
plt.xlabel('Time')
plt.ylabel('Flux')
plt.title(target)
#plt.colorbar(label='Phase')

plt.tight_layout()
plt.show()



#test polyfit

#degree = 10

#coeffs = np.polyfit(time, phot, degree)
#poly_function = np.poly1d(coeffs)

#fitted data
#fitted_mag = poly_function(time)

#plot
#plt.scatter(time, phot, s=5, label='Data')
#plt.plot(time, fitted_mag, color='r', label='Fit')
#plt.xlabel('Time (JD)')
#plt.ylabel('Magnitude')
#plt.gca().invert_yaxis()
#plt.legend()
#plt.show()
