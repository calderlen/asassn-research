import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

fsize = 10
tsize = 18

tdir = 'in'

major = 5.0
minor = 3.0

style = 'default'

plt.style.use(style)
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor

plt.rcParams['text.usetex'] = False

Wcolumn_names = ['JD', 'w1', 'ew1', 'w2', 'ew2', 'ignore', 'alsoignore']
column_names=["JD","mag",'error', 'good/bad', 'camera#', 'band', 'camera name']

t1 = 17180423517
t2 = 446676962403
t3 = 103080549681

m1 = '2MJ23522673$+$6317235'
m2 = '2MJ01051276$-$7229060'
m3 = '2MJ03471262$+$5711325'

g1 = "Blue Luminous"
g2 = "No Group"
g3 = "Blue Luminous"

p1 = '14.5_15/lc18_cal/' + str(t1) + '.dat'
pw1 = 'WISE/data/' + str(t1) + '.dat'
p2 = '14.5_15/lc10_cal/' + str(t2) + '.dat'
pw2 = 'WISE/data/' + str(t2) + '.dat'
p3 = '14.5_15/lc39_cal/' + str(t3) + '.dat'
pw3 = 'WISE/data/' + str(t3) + '.dat'

df1 = pd.read_csv(p1, sep="\s+", names=column_names)
df1.drop(df1[df1['good/bad']<1].index, inplace=True)
dfw1 = pd.read_csv(pw1, sep="\s+", names=Wcolumn_names)
df2 = pd.read_csv(p2, sep="\s+", names=column_names)
df2.drop(df2[df2['good/bad']<1].index, inplace=True)
dfw2 = pd.read_csv(pw2, sep="\s+", names=Wcolumn_names)
df3 = pd.read_csv(p3, sep="\s+", names=column_names)
df3.drop(df3[df3['good/bad']<1].index, inplace=True)
dfw3 = pd.read_csv(pw3, sep="\s+", names=Wcolumn_names)

def jd_to_year(jd):
    jd_epoch = 2449718.5
    year_epoch = 1995
    days_in_year = 365.25
    return year_epoch + (jd - jd_epoch) / days_in_year

def year_to_jd(year):
    jd_epoch = 2449718.5
    year_epoch = 1995
    days_in_year = 365.25
    return (year-year_epoch)*days_in_year + jd_epoch-2450000

fig = plt.figure(figsize=(10,8))

ax1 = plt.subplot(3,1,1)

dfw1['date_group'] = dfw1['JD'].astype(str).str[:5]
dfw1 = dfw1.groupby('date_group').median()
dfw1 = dfw1.drop(columns=['date_group'], errors='ignore')

dfw1= dfw1.reset_index(drop=True)

UL = dfw1['ew1'] > 5
UL2 = dfw1['ew2'] > 5

if np.abs(np.mean(dfw1['w1']) - np.mean(df1['mag'])) > 2 or np.abs(np.mean(dfw1['w2']) - np.mean(df1['mag'])) > 2:
    w1_mag = dfw1[~UL]['w1'] + np.abs(np.mean(dfw1['w1']) - np.mean(df1['mag'])) - 2 + 1.5
    w2_mag = dfw1[~UL2]['w2'] + np.abs(np.mean(dfw1['w1']) - np.mean(df1['mag'])) - 2 + 1.5
    w1_L = 'W1 + ' + str(round(np.abs(np.mean(dfw1['w1']) - np.mean(df1['mag'])) - 2,2))
    w2_L ='W2 + ' + str(round(np.abs(np.mean(dfw1['w1']) - np.mean(df1['mag'])) - 2,2))
else:
    w1_mag = dfw1[~UL]['w1'] + 0.9
    w2_mag = dfw1[~UL2]['w2'] + 0.9
    w1_L = 'W1'
    w2_L ='W2'

ax1.errorbar(df1['JD']-2450000,df1['mag'],yerr=df1['error'], fmt='o',markersize=2, color='g', ecolor='r',label='g')
ax1.errorbar(dfw1[~UL]['JD']+2400000-2450000, w1_mag, yerr=dfw1[~UL]['ew1'], xerr=None, fmt='o', color='orangered', ecolor='lightcoral', markersize=2, label='W1 + Offset')
ax1.errorbar(dfw1[~UL2]['JD']+2400000-2450000, w2_mag, yerr=dfw1[~UL2]['ew2'], xerr=None, fmt='o', color='orange', ecolor='wheat', markersize=2, label='W2 + Offset')

#ax1.set_xlim(min(dfw1.JD-2450000-300), max(dfw1.JD)-2450000+200)
ax1.set_xlim(6300,10500)
#ax1.set_ylim(min(dfw1.w1)-0.05, max(df1.mag)+0.05)
ax1.set_ylim(12.81,15.11)
ax1.tick_params(axis='x', direction='in', top=False, bottom=True, labelbottom=False, pad=-15)
#ax1.tick_params(axis='y', direction='out', right=True, pad=-25)
ax1.text(0.5, 0.95, str(m1), transform=ax1.transAxes, fontsize=12, ha='center', va='top') 
ax1.text(0.01, 0.95, str(g1), transform=ax1.transAxes, fontsize=10, ha='left', color='b', va='top')

ax1.invert_yaxis()
#ax1.legend(bbox_to_anchor=(0.0,0), loc='lower left', fontsize=9)
secax1 = ax1.secondary_xaxis('top')
years = np.arange(2008,2025)
year_ticks = [year_to_jd(year) for year in years]
secax1.xaxis.set_ticks(year_ticks)
secax1.xaxis.set_tick_params(direction='in')
#secax1.set_xlabel('Year')
secax1.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{jd_to_year(x + 2450000):.0f}"))

ax2 = plt.subplot(3,1,2)

dfw2['date_group'] = dfw2['JD'].astype(str).str[:5]
dfw2 = dfw2.groupby('date_group').median()
dfw2 = dfw2.drop(columns=['date_group'], errors='ignore')

dfw2= dfw2.reset_index(drop=True)


UL = dfw2['ew1'] > 5
UL2 = dfw2['ew2'] > 5

if np.abs(np.mean(dfw2['w1']) - np.mean(df2['mag'])) > 2 or np.abs(np.mean(dfw2['w2']) - np.mean(df2['mag'])) > 2:
    w1_mag = dfw2[~UL]['w1'] + np.abs(np.mean(dfw2['w1']) - np.mean(df2['mag'])) - 2 + 1
    w2_mag = dfw2[~UL2]['w2'] + np.abs(np.mean(dfw2['w1']) - np.mean(df2['mag'])) - 2 + 1
    w1_L = 'W1 + ' + str(round(np.abs(np.mean(dfw2['w1']) - np.mean(df2['mag'])) - 2,2))
    w2_L ='W2 + ' + str(round(np.abs(np.mean(dfw2['w1']) - np.mean(df2['mag'])) - 2,2))

else:
    w1_mag = dfw2[~UL]['w1'] + 0
    w2_mag = dfw2[~UL2]['w2'] + 0
    w1_L = 'W1'
    w2_L ='W2'

ax2.errorbar(df2.JD-2450000,df2.mag,yerr=df2.error, fmt='o',markersize=2, color='g', ecolor='r', label='g')
ax2.errorbar(dfw2[~UL]['JD']+2400000-2450000, w1_mag, yerr=dfw2[~UL]['ew1'], xerr=None, fmt='o', color='orangered', ecolor='lightcoral', markersize=2, label=w1_L)
ax2.errorbar(dfw2[~UL2]['JD']+2400000-2450000, w2_mag, yerr=dfw2[~UL2]['ew2'], xerr=None, fmt='o', color='orange', ecolor='wheat', markersize=2, label=w2_L)

ax2.set_xlim(6300,10500)
#ax2.set_ylim(min(dfw2.w1)-0.15, max(df2.mag)+0.10)
ax2.set_ylim(13.2,16.3)
ax2.tick_params(axis='x', direction='in', top=False, bottom=True, labelbottom=False, pad=-15)
#ax2.tick_params(axis='y', direction='in', right=True, pad=-30)
ax2.text(0.5, 0.95, str(m2), transform=ax2.transAxes, fontsize=12, ha='center', va='top')
ax2.text(0.01, 0.95, str(g2), transform=ax2.transAxes, fontsize=10, ha='left', va='top')
ax2.invert_yaxis()
#ax2.legend(fontsize=9)
secax2 = ax2.secondary_xaxis('top')
years = np.arange(2014,2024)
year_ticks = [year_to_jd(year) for year in years]
secax2.xaxis.set_ticks(year_ticks)
secax2.xaxis.set_tick_params(direction='in', labeltop=False)

ax3 = plt.subplot(3,1,3)

dfw3['date_group'] = dfw3['JD'].astype(str).str[:5]
dfw3 = dfw3.groupby('date_group').median()
dfw3 = dfw3.drop(columns=['date_group'], errors='ignore')

dfw3= dfw3.reset_index(drop=True)
UL = dfw3['ew1'] > 5
UL2 = dfw3['ew2'] > 5

if np.abs(np.mean(dfw3['w1']) - np.mean(df3['mag'])) > 2 or np.abs(np.mean(dfw3['w2']) - np.mean(df3['mag'])) > 2:
    w1_mag = dfw3[~UL]['w1'] + np.abs(np.mean(dfw3['w1']) - np.mean(df3['mag'])) - 2 + 1.5
    w2_mag = dfw3[~UL2]['w2'] + np.abs(np.mean(dfw3['w1']) - np.mean(df3['mag'])) - 2 + 1.5
    w1_L = 'W1 + ' + str(round(np.abs(np.mean(dfw3['w1']) - np.mean(df3['mag'])) - 2,2))
    w2_L ='W2 + ' + str(round(np.abs(np.mean(dfw3['w1']) - np.mean(df3['mag'])) - 2,2))
else:
    w1_mag = dfw3[~UL]['w1'] 
    w2_mag = dfw3[~UL2]['w2'] 
    w1_L = 'W1'
    w2_L ='W2'

ax3.errorbar(df3.JD-2450000,df3.mag,yerr=df3.error, fmt='o',markersize=2, color='g', ecolor='r', label='g')
ax3.errorbar(dfw3[~UL]['JD']+2400000-2450000, w1_mag, yerr=dfw3[~UL]['ew1'], xerr=None, fmt='o', color='orangered', ecolor='lightcoral', markersize=2, label='W1 + Offset')
ax3.errorbar(dfw3[~UL2]['JD']+2400000-2450000, w2_mag, yerr=dfw3[~UL2]['ew2'], xerr=None, fmt='o', color='orange', ecolor='wheat', markersize=2, label='W2 + Offset')

ax3.set_xlim(6300,10500)
#ax3.set_ylim(min(dfw3.w1)-0.15, max(df3.mag)+0.2)
ax3.set_ylim(13.01, 15.24)
#ax3.tick_params(axis='x', direction='in', top=False, bottom=True, pad=-15)
#ax3.tick_params(axis='y', direction='in', right=True, pad=-30)
ax3.text(0.5, 0.95, str(m3), transform=ax3.transAxes, fontsize=12, ha='center', va='top')
ax3.text(0.01, 0.95, str(g3), transform=ax3.transAxes, fontsize=10, color='b',ha='left', va='top')
ax3.invert_yaxis()
ax3.legend(bbox_to_anchor=(0.0,0), loc='lower left', fontsize=9)
#ax3.legend(fontsize=9)
secax3 = ax3.secondary_xaxis('top')
years = np.arange(2014,2024)
year_ticks = [year_to_jd(year) for year in years]
secax3.xaxis.set_ticks(year_ticks)
secax3.xaxis.set_tick_params(direction='in', labeltop=False)

plt.subplots_adjust(hspace=0, wspace=0)
fig.text(0.5, 0.04, 'JD - 2450000', ha='center', fontsize=14) 
fig.text(0.05, 0.5, 'g magnitude', va='center', rotation='vertical', fontsize=14)


plt.savefig('wiselcs.pdf', dpi=1000)
plt.show()
