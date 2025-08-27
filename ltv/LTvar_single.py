#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from astropy.stats import mad_std
from scipy import stats
from astropy.table import Table
import time

startTime = time.time()


column_names=["JD","mag",'error', 'good/bad', 'camera#', 'band', 'camera name'] #1=good, 0 =bad #1=V, 0=g

#LTvar = Table.read('LTvar.csv')

#_Target = np.array(LTvar['ASAS-SN ID']).tolist()
#Median = np.array(LTvar['Median']).tolist()
#Median_err = np.array(LTvar['Median_err']).tolist()
#Dispersion = np.array(LTvar['Dispersion']).tolist()
#Slope = np.array(LTvar['Slope']).tolist()
#Slope_err = np.array(LTvar['Slope_err']).tolist(di)

bad = False
badcam = 'bh'

ID = pd.read_table("/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/index13.csv", sep=r'\,|\t', engine='python')

target = 377957417857
path = '/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc13_cal/' + str(target) + '.dat'
#         ra = np.where(ID['#asas_sn_id'] == Target) lccal1
ra = np.where(ID['asas_sn_id'] == target) #lccal2

        
df = pd.read_table(path, sep="\s+", names=column_names)
df.drop(df[df['good/bad'] < 1].index, inplace = True)
df.head()

if target == 17181160895:
    df.drop(df[df['good/bad'] < 1].index, inplace = True)
    df.drop(df[df['JD'] < 2.458e+06].index, inplace = True)

if target == 532576065473:
    df.drop(df[df['mag'] > 14.1].index, inplace = True)


V = df[df['band'] == 1] 
g = df[df['band'] == 0]

plt.errorbar(V['JD'], V['mag'], yerr=V.error, xerr=None, fmt='o', color='darkturquoise', ecolor='r', markersize=2)
plt.errorbar(g['JD'], g['mag'], yerr=g.error, xerr=None, fmt='o', color='forestgreen', ecolor='r', markersize=2)

plt.title(target)
plt.ylim(max(df.mag)+0.25, min(df.mag)-0.25)
#plt.savefig('QS.png')
plt.show()

cams = df['camera name'].unique()
colors = plt.cm.get_cmap('tab10', len(cams))

for i, cam in enumerate(cams):
    subset = df[df['camera name'] == cam]

    plt.errorbar(subset['JD'], subset['mag'], yerr=subset['error'], fmt='o', markersize=2, label=cam, color=colors(i))

plt.legend()
plt.ylim(max(df.mag)+0.25, min(df.mag)-0.25)
plt.show()

if bad == True:

    V = V[V['camera name'] != badcam]
    g = g[g['camera name'] != badcam]

    plt.errorbar(V['JD'], V['mag'], yerr=V.error, xerr=None, fmt='o', color='darkturquoise', ecolor='r', markersize=2)
    plt.errorbar(g['JD'], g['mag'], yerr=g.error, xerr=None, fmt='o', color='forestgreen', ecolor='r', markersize=2)

    plt.title(target)
    plt.ylim(max(df.mag)+0.25, min(df.mag)-0.25)
    #plt.savefig('QS.png')
    plt.show()

RA = ID['ra_deg'].iloc[ra]
        
        #median of overall lc
        
lcMED = np.median(df.mag)
lcMED_err = mad_std(df.mag)
        
        #dispersion of overall lc (range)

lcDISP = np.ptp(df.mag)

#pstarr mag
p_mag = ID['pstarrs_g_mag'].iloc[ra]
print(p_mag)
p_mag = np.array(p_mag)
p_mag = float(p_mag)

        #seasonal medians
        
indexes = np.arange(0,30,1)
dspring = 2460023.5 


Mid = []

for n in indexes:
    date1 = dspring + 365.25*(RA-12.0)/24.0 #target overhead at midnight at date1
    date2 = date1 +  365.25/2.0 +365.25 #same RA as the sun at date2
    mid = date2 - n*365.25 #seasonal gaps (where sun is blocking target)
    Mid.append(mid)
            
mid = np.array(Mid)



mid = [x for x in mid if x < max(df.JD) and x > min(df.JD)]

#print(mid)
            
if len(mid) > 8:
    lc1 = [x for x in df['JD'] if x < mid[-1]]
    lc2 = [x for x in df['JD'] if x > mid[-1] and x < mid[-2]]
    lc3 = [x for x in df['JD'] if x > mid[-2] and x < mid[-3]]
    lc4 = [x for x in df['JD'] if x > mid[-3] and x < mid[-4]]
    lc5 = [x for x in df['JD'] if x > mid[-4] and x < mid[-5]]
    lc6 = [x for x in df['JD'] if x > mid[-5] and x < mid[-6]]
    lc7 = [x for x in df['JD'] if x > mid[-6] and x < mid[-7]]
    lc8 = [x for x in df['JD'] if x > mid[-7] and x < mid[-8]]
    lc9 = [x for x in df['JD'] if x > mid[-8] and x < mid[-9]]
    lc10 = [x for x in df['JD'] if x > mid[-9]] 

elif len (mid) <= 2:
    lc1 = [x for x in df['JD'] if x < mid[-1]]
    lc2 = [x for x in df['JD'] if x > mid[-1] and x < mid[-2]]
    lc3 = [x for x in df['JD'] if x > mid[-2]]
    lc4 = []
    lc5 = []
    lc6 = []
    lc7 = []
    lc8 = []
    lc9 = []
    lc10 = []

elif len (mid) <= 3:
    lc1 = [x for x in df['JD'] if x < mid[-1]]
    lc2 = [x for x in df['JD'] if x > mid[-1] and x < mid[-2]]
    lc3 = [x for x in df['JD'] if x > mid[-2] and x < mid[-3]]
    lc4 = [x for x in df['JD'] if x > mid[-3]]
    lc5 = []
    lc6 = []
    lc7 = []
    lc8 = []
    lc9 = []
    lc10 = []
            
elif len (mid) <= 4:
    lc1 = [x for x in df['JD'] if x < mid[-1]]
    lc2 = [x for x in df['JD'] if x > mid[-1] and x < mid[-2]]
    lc3 = [x for x in df['JD'] if x > mid[-2] and x < mid[-3]]
    lc4 = [x for x in df['JD'] if x > mid[-3] and x < mid[-4]]
    lc5 = [x for x in df['JD'] if x > mid[-4]]
    lc6 = []
    lc7 = []
    lc8 = []
    lc9 = []
    lc10 = []
            
elif len (mid) <= 5:
    lc1 = [x for x in df['JD'] if x < mid[-1]]
    lc2 = [x for x in df['JD'] if x > mid[-1] and x < mid[-2]]
    lc3 = [x for x in df['JD'] if x > mid[-2] and x < mid[-3]]
    lc4 = [x for x in df['JD'] if x > mid[-3] and x < mid[-4]]
    lc5 = [x for x in df['JD'] if x > mid[-4] and x < mid[-5]]
    lc6 = [x for x in df['JD'] if x > mid[-5]]
    lc7 = []
    lc8 = []
    lc9 = []
    lc10 = []
            
elif len (mid) <= 6:
    lc1 = [x for x in df['JD'] if x < mid[-1]]
    lc2 = [x for x in df['JD'] if x > mid[-1] and x < mid[-2]]
    lc3 = [x for x in df['JD'] if x > mid[-2] and x < mid[-3]]
    lc4 = [x for x in df['JD'] if x > mid[-3] and x < mid[-4]]
    lc5 = [x for x in df['JD'] if x > mid[-4] and x < mid[-5]]
    lc6 = [x for x in df['JD'] if x > mid[-5] and x < mid[-6]]
    lc7 = [x for x in df['JD'] if x > mid[-6]]
    lc8 = []
    lc9 = []
    lc10 = []

elif len (mid) <= 7:
    lc1 = [x for x in df['JD'] if x < mid[-1]]
    lc2 = [x for x in df['JD'] if x > mid[-1] and x < mid[-2]]
    lc3 = [x for x in df['JD'] if x > mid[-2] and x < mid[-3]]
    lc4 = [x for x in df['JD'] if x > mid[-3] and x < mid[-4]]
    lc5 = [x for x in df['JD'] if x > mid[-4] and x < mid[-5]]
    lc6 = [x for x in df['JD'] if x > mid[-5] and x < mid[-6]]
    lc7 = [x for x in df['JD'] if x > mid[-6] and x < mid[-7]]
    lc8 = [x for x in df['JD'] if x > mid[-7]]
    lc9 = []
    lc10 = []

elif len (mid) <= 8:
    lc1 = [x for x in df['JD'] if x < mid[-1]]
    lc2 = [x for x in df['JD'] if x > mid[-1] and x < mid[-2]]
    lc3 = [x for x in df['JD'] if x > mid[-2] and x < mid[-3]]
    lc4 = [x for x in df['JD'] if x > mid[-3] and x < mid[-4]]
    lc5 = [x for x in df['JD'] if x > mid[-4] and x < mid[-5]]
    lc6 = [x for x in df['JD'] if x > mid[-5] and x < mid[-6]]
    lc7 = [x for x in df['JD'] if x > mid[-6] and x < mid[-7]]
    lc8 = [x for x in df['JD'] if x > mid[-7] and x < mid[-8]]
    lc9 = [x for x in df['JD'] if x > mid[-8]]
    lc10 = [] 
            
elif len(mid) <= 9:
    lc1 = [x for x in df['JD'] if x < mid[-1]]
    lc2 = [x for x in df['JD'] if x > mid[-1] and x < mid[-2]]
    lc3 = [x for x in df['JD'] if x > mid[-2] and x < mid[-3]]
    lc4 = [x for x in df['JD'] if x > mid[-3] and x < mid[-4]]
    lc5 = [x for x in df['JD'] if x > mid[-4] and x < mid[-5]]
    lc6 = [x for x in df['JD'] if x > mid[-5] and x < mid[-6]]
    lc7 = [x for x in df['JD'] if x > mid[-6] and x < mid[-7]]
    lc8 = [x for x in df['JD'] if x > mid[-7] and x < mid[-8]]
    lc9 = [x for x in df['JD'] if x > mid[-8] and x < mid[-9]]
    lc10 = [x for x in df['JD'] if x > mid[-9]] 

print(len(lc1),len(lc2),len(lc3),len(lc4),len(lc5),len(lc6),len(lc7),len(lc8),len(lc9),len(lc10))

if (len(lc2) == 0 and len(lc3) == 0 and len(lc4) == 0 and len(lc5) > 0):
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] > max(lc7)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    df8 = pd.read_table(path, sep="\s+", names=column_names)
    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
    df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
    df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

    df9 = pd.read_table(path, sep="\s+", names=column_names)
    df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
    df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df9.drop(df9[df9['JD'] < min(lc9)].index, inplace = True)

    df10 = pd.read_table(path, sep="\s+", names=column_names)
    df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df10.drop(df10[df10['JD'] < min(lc9)].index, inplace = True)
            
    meds = [np.median(df1.mag) ,np.median(df5.mag),
    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
    
    meds_err = [mad_std(df1.mag), mad_std(df5.mag), mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag), mad_std(df9.mag), mad_std(df10.mag)]


    indexes = np.array([0,1,2,3,4,5,6])

elif(len(lc1) > 0 and len(lc2) == 0 and len(lc3) == 0 and len(lc4) > 0 and len(lc5) > 0 and len(lc9) == 0 and len(lc10) == 0):
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
    df4.drop(df4[df4['JD'] > max(lc4)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] > max(lc7)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    df8 = pd.read_table(path, sep="\s+", names=column_names)
    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
    #df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
    df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

    meds = [np.median(df1.mag),
    np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]

    meds_err = [mad_std(df1.mag),
    mad_std(df4.mag), mad_std(df5.mag), mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag)]

    indexes = np.array([0,1,2,3,4,5])


elif (len(lc3) == 0 and len(lc4) >  0 and len(lc9) == 0 and len(lc10) == 0):
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df2 = pd.read_table(path, sep="\s+", names=column_names)
    df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
    df2.drop(df2[df2['JD'] > max(lc2)].index, inplace = True)
    df2.drop(df2[df2['JD'] < min(lc2)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
    df4.drop(df4[df4['JD'] > max(lc4)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] > max(lc7)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    df8 = pd.read_table(path, sep="\s+", names=column_names)
    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

    meds = [np.median(df1.mag) ,np.median(df2.mag),
    np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]

    meds_err = [mad_std(df1.mag), mad_std(df2.mag), mad_std(df4.mag), mad_std(df5.mag), mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag)]


    indexes = np.array([0,1,2,3,4,5,6])

elif (len(lc1) > 0 and len(lc2) == 0 and len(lc3) == 0 and len(lc4) == 0 and len(lc5) == 0 and len(lc6) > 1):
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] > max(lc7)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    df8 = pd.read_table(path, sep="\s+", names=column_names)
    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
    df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
    df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

    df9 = pd.read_table(path, sep="\s+", names=column_names)
    df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
    df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df9.drop(df9[df9['JD'] < min(lc9)].index, inplace = True)

    df10 = pd.read_table(path, sep="\s+", names=column_names)
    df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df10.drop(df10[df10['JD'] < min(lc9)].index, inplace = True)

    meds = [np.median(df1.mag),
    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]

    meds_err = [mad_std(df1.mag),
    mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag), mad_std(df9.mag), mad_std(df10.mag)]


    indexes = np.array([0,1,2,3,4,5])

elif (len(lc3) == 0 and len(lc4) == 0 and len(lc5) == 0 and len(lc6) > 0):
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df2 = pd.read_table(path, sep="\s+", names=column_names)
    df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
    df2.drop(df2[df2['JD'] > max(lc2)].index, inplace = True)
    df2.drop(df2[df2['JD'] < min(lc2)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] > max(lc7)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    df8 = pd.read_table(path, sep="\s+", names=column_names)
    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
    df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
    df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

    df9 = pd.read_table(path, sep="\s+", names=column_names)
    df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
    df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df9.drop(df9[df9['JD'] < min(lc9)].index, inplace = True)

    df10 = pd.read_table(path, sep="\s+", names=column_names)
    df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df10.drop(df10[df10['JD'] < min(lc9)].index, inplace = True)
            
    meds = [np.median(df1.mag) ,np.median(df5.mag),
    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
            
    meds_err = [mad_std(df1.mag), mad_std(df5.mag),
    mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag), mad_std(df9.mag), mad_std(df10.mag)]


    indexes = np.array([0,1,2,3,4,5,6])

elif (len(lc1) > 0 and len(lc2) == 0 and len(lc3) > 0 and len(lc8) == 0):
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df3 = pd.read_table(path, sep="\s+", names=column_names)
    df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
    df3.drop(df3[df3['JD'] > max(lc3)].index, inplace = True)
    df3.drop(df3[df3['JD'] < min(lc3)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
    df4.drop(df4[df4['JD'] > max(lc4)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df4[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df4[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df4[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    meds = [np.median(df1.mag) ,np.median(df3.mag),
    np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag)]

    meds_err = [mad_std(df1.mag), mad_std(df3.mag),
    mad_std(df4.mag), mad_std(df5.mag), mad_std(df6.mag), mad_std(df7.mag)]


    indexes = np.array([0,1,2,3,4,5])

elif (len(lc1) > 0 and len(lc2) == 0 and len(lc3) == 0 and len(lc9) == 0 and len(lc8) > 0):
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
    df4.drop(df4[df4['JD'] > max(lc4)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df4[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df4[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df4[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] > max(lc7)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    df8 = pd.read_table(path, sep="\s+", names=column_names)
    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

    meds = [np.median(df1.mag), np.median(df4.mag),np.median(df5.mag),
    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]

    meds_err = [mad_std(df1.mag), mad_std(df4.mag), mad_std(df5.mag),
    mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag)]


    indexes = np.array([0,1,2,3,4,5])

elif (len(lc2) == 0 and len(lc3) == 0 and len(lc4) > 0):
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
    df4.drop(df4[df4['JD'] > max(lc4)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] > max(lc7)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    df8 = pd.read_table(path, sep="\s+", names=column_names)
    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
    df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
    df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

    df9 = pd.read_table(path, sep="\s+", names=column_names)
    df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
    df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df9.drop(df9[df9['JD'] < min(lc9)].index, inplace = True)

    df10 = pd.read_table(path, sep="\s+", names=column_names)
    df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df10.drop(df10[df10['JD'] < min(lc9)].index, inplace = True)
            
    meds = [np.median(df1.mag),np.median(df4.mag),np.median(df5.mag),
            np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
            
    meds_err = [mad_std(df1.mag),mad_std(df4.mag), mad_std(df5.mag),
    mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag), mad_std(df9.mag), mad_std(df10.mag)]


    indexes = np.array([0,1,2,3,4,5,6,7])
            
elif (len(lc3) == 0 and len(lc4) == 0 and len(lc5) > 0):
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df2 = pd.read_table(path, sep="\s+", names=column_names)
    df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
    df2.drop(df2[df2['JD'] > max(lc2)].index, inplace = True)
    df2.drop(df2[df2['JD'] < min(lc2)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] > max(lc7)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    df8 = pd.read_table(path, sep="\s+", names=column_names)
    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
    df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
    df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

    df9 = pd.read_table(path, sep="\s+", names=column_names)
    df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
    df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df9.drop(df9[df9['JD'] < min(lc9)].index, inplace = True)

    df10 = pd.read_table(path, sep="\s+", names=column_names)
    df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df10.drop(df10[df10['JD'] < min(lc9)].index, inplace = True)
            
    meds = [np.median(df1.mag), np.median(df2.mag),np.median(df5.mag),
    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
            
    meds_err = [mad_std(df1.mag), mad_std(df2.mag), mad_std(df5.mag),
    mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag), mad_std(df9.mag), mad_std(df10.mag)]


    indexes = np.array([0,1,2,3,4,5,6,7])  
            
elif (len(lc3) == 0 and len(lc4) > 0):
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df2 = pd.read_table(path, sep="\s+", names=column_names)
    df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
    df2.drop(df2[df2['JD'] > max(lc2)].index, inplace = True)
    df2.drop(df2[df2['JD'] < min(lc2)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
    df4.drop(df4[df4['JD'] > max(lc4)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] > max(lc7)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    df8 = pd.read_table(path, sep="\s+", names=column_names)
    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
    df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
    df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

    df9 = pd.read_table(path, sep="\s+", names=column_names)
    df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
    df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df9.drop(df9[df9['JD'] < min(lc9)].index, inplace = True)

    df10 = pd.read_table(path, sep="\s+", names=column_names)
    df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df10.drop(df10[df10['JD'] < min(lc9)].index, inplace = True)
            
    meds = [np.median(df1.mag), np.median(df2.mag),np.median(df4.mag),np.median(df5.mag),
    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
            
    meds_err = [mad_std(df1.mag), mad_std(df2.mag),mad_std(df4.mag), mad_std(df5.mag),
    mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag), mad_std(df9.mag), mad_std(df10.mag)]


    indexes = np.array([0,1,2,3,4,5,6,7,8])

elif (len(lc2) == 0 and len(lc3) > 0):
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df3 = pd.read_table(path, sep="\s+", names=column_names)
    df3.drop(df2[df2['good/bad'] < 1].index, inplace = True)
    df3.drop(df2[df2['JD'] > max(lc3)].index, inplace = True)
    df3.drop(df2[df2['JD'] < min(lc3)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
    df4.drop(df4[df4['JD'] > max(lc4)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] > max(lc7)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    df8 = pd.read_table(path, sep="\s+", names=column_names)
    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
    df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
    df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

    df9 = pd.read_table(path, sep="\s+", names=column_names)
    df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
    df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df9.drop(df9[df9['JD'] < min(lc9)].index, inplace = True)

    df10 = pd.read_table(path, sep="\s+", names=column_names)
    df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df10.drop(df10[df10['JD'] < min(lc9)].index, inplace = True)
            
    meds = [np.median(df1.mag), np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
            
    meds_err = [mad_std(df1.mag), mad_std(df3.mag),mad_std(df4.mag), mad_std(df5.mag),
    mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag), mad_std(df9.mag), mad_std(df10.mag)]


    indexes = np.array([0,1,2,3,4,5,6,7,8])
            
elif len(lc4) < 1:
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df2 = pd.read_table(path, sep="\s+", names=column_names)
    df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
    df2.drop(df2[df2['JD'] > max(lc2)].index, inplace = True)
    df2.drop(df2[df2['JD'] < min(lc2)].index, inplace = True)

    df3 = pd.read_table(path, sep="\s+", names=column_names)
    df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
        #     df3.drop(df3[df3['JD'] > max(lc3)].index, inplace = True)
    df3.drop(df3[df3['JD'] < min(lc3)].index, inplace = True)

    meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag)]

    meds_err = [mad_std(df1.mag), mad_std(df2.mag), mad_std(df3.mag)]

    indexes = np.array([0,1,2])

elif len(lc5) < 1:
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df2 = pd.read_table(path, sep="\s+", names=column_names)
    df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
    df2.drop(df2[df2['JD'] > max(lc2)].index, inplace = True)
    df2.drop(df2[df2['JD'] < min(lc2)].index, inplace = True)

    df3 = pd.read_table(path, sep="\s+", names=column_names)
    df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
    df3.drop(df3[df3['JD'] > max(lc3)].index, inplace = True)
    df3.drop(df3[df3['JD'] < min(lc3)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df3[df3['good/bad'] < 1].index, inplace = True)
        #     df3.drop(df3[df3['JD'] > max(lc3)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag)]

    meds_err = [mad_std(df1.mag), mad_std(df2.mag), mad_std(df3.mag),mad_std(df4.mag)]

    indexes = np.array([0,1,2,3])

elif len(lc6) < 1:
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df2 = pd.read_table(path, sep="\s+", names=column_names)
    df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
    df2.drop(df2[df2['JD'] > max(lc2)].index, inplace = True)
    df2.drop(df2[df2['JD'] < min(lc2)].index, inplace = True)

    df3 = pd.read_table(path, sep="\s+", names=column_names)
    df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
    df3.drop(df3[df3['JD'] > max(lc3)].index, inplace = True)
    df3.drop(df3[df3['JD'] < min(lc3)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
    df4.drop(df4[df4['JD'] > max(lc4)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
        #    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)


    meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag)]

    meds_err = [mad_std(df1.mag), mad_std(df2.mag), mad_std(df3.mag),mad_std(df4.mag), mad_std(df5.mag)]

    indexes = np.array([0,1,2,3,4])

elif len(lc7) < 1:
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df2 = pd.read_table(path, sep="\s+", names=column_names)
    df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
    df2.drop(df2[df2['JD'] > max(lc2)].index, inplace = True)
    df2.drop(df2[df2['JD'] < min(lc2)].index, inplace = True)

    df3 = pd.read_table(path, sep="\s+", names=column_names)
    df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
    df3.drop(df3[df3['JD'] > max(lc3)].index, inplace = True)
    df3.drop(df3[df3['JD'] < min(lc3)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
    df4.drop(df4[df4['JD'] > max(lc4)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
        #         df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                    np.median(df6.mag)]

    meds_err = [mad_std(df1.mag), mad_std(df2.mag), mad_std(df3.mag),mad_std(df4.mag), mad_std(df5.mag),
                       mad_std(df6.mag)]

    indexes = np.array([0,1,2,3,4,5])


elif len(lc8) < 1:

    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df2 = pd.read_table(path, sep="\s+", names=column_names)
    df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
    df2.drop(df2[df2['JD'] > max(lc2)].index, inplace = True)
    df2.drop(df2[df2['JD'] < min(lc2)].index, inplace = True)

    df3 = pd.read_table(path, sep="\s+", names=column_names)
    df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
    df3.drop(df3[df3['JD'] > max(lc3)].index, inplace = True)
    df3.drop(df3[df3['JD'] < min(lc3)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
    df4.drop(df4[df4['JD'] > max(lc4)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
        #         df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                    np.median(df6.mag), np.median(df7.mag)]

    meds_err = [mad_std(df1.mag), mad_std(df2.mag), mad_std(df3.mag),mad_std(df4.mag), mad_std(df5.mag),
                       mad_std(df6.mag), mad_std(df7.mag)]

    indexes = np.array([0,1,2,3,4,5,6])



elif len(lc9) < 1:
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df2 = pd.read_table(path, sep="\s+", names=column_names)
    df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
    df2.drop(df2[df2['JD'] > max(lc2)].index, inplace = True)
    df2.drop(df2[df2['JD'] < min(lc2)].index, inplace = True)

    df3 = pd.read_table(path, sep="\s+", names=column_names)
    df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
    df3.drop(df3[df3['JD'] > max(lc3)].index, inplace = True)
    df3.drop(df3[df3['JD'] < min(lc3)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
    df4.drop(df4[df4['JD'] > max(lc4)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] > max(lc7)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    df8 = pd.read_table(path, sep="\s+", names=column_names)
    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
        #         df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

    meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]

    meds_err = [mad_std(df1.mag), mad_std(df2.mag), mad_std(df3.mag),mad_std(df4.mag), mad_std(df5.mag),
                       mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag)]

    indexes = np.array([0,1,2,3,4,5,6,7])

elif len(lc10) <= 1:
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df2 = pd.read_table(path, sep="\s+", names=column_names)
    df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
    df2.drop(df2[df2['JD'] > max(lc2)].index, inplace = True)
    df2.drop(df2[df2['JD'] < min(lc2)].index, inplace = True)

    df3 = pd.read_table(path, sep="\s+", names=column_names)
    df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
    df3.drop(df3[df3['JD'] > max(lc3)].index, inplace = True)
    df3.drop(df3[df3['JD'] < min(lc3)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
    df4.drop(df4[df4['JD'] > max(lc4)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] > max(lc7)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    df8 = pd.read_table(path, sep="\s+", names=column_names)
    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
    df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
    df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

    df9 = pd.read_table(path, sep="\s+", names=column_names)
    df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df9.drop(df9[df9['JD'] < min(lc9)].index, inplace = True)

    meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag)]

    meds_err = [mad_std(df1.mag), mad_std(df2.mag), mad_std(df3.mag),mad_std(df4.mag), mad_std(df5.mag),
                       mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag), mad_std(df9.mag)]

    indexes = np.array([0,1,2,3,4,5,6,7,8])

elif len(lc10) > 1:
    df1 = pd.read_table(path, sep="\s+", names=column_names)
    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
    df1.drop(df1[df1['JD'] > max(lc1)].index, inplace = True)

    df2 = pd.read_table(path, sep="\s+", names=column_names)
    df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
    df2.drop(df2[df2['JD'] > max(lc2)].index, inplace = True)
    df2.drop(df2[df2['JD'] < min(lc2)].index, inplace = True)

    df3 = pd.read_table(path, sep="\s+", names=column_names)
    df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
    df3.drop(df3[df3['JD'] > max(lc3)].index, inplace = True)
    df3.drop(df3[df3['JD'] < min(lc3)].index, inplace = True)

    df4 = pd.read_table(path, sep="\s+", names=column_names)
    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
    df4.drop(df4[df4['JD'] > max(lc4)].index, inplace = True)
    df4.drop(df4[df4['JD'] < min(lc4)].index, inplace = True)

    df5 = pd.read_table(path, sep="\s+", names=column_names)
    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
    df5.drop(df5[df5['JD'] < min(lc5)].index, inplace = True)

    df6 = pd.read_table(path, sep="\s+", names=column_names)
    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
    df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
    df6.drop(df6[df6['JD'] < min(lc6)].index, inplace = True)

    df7 = pd.read_table(path, sep="\s+", names=column_names)
    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
    df7.drop(df7[df7['JD'] > max(lc7)].index, inplace = True)
    df7.drop(df7[df7['JD'] < min(lc7)].index, inplace = True)

    df8 = pd.read_table(path, sep="\s+", names=column_names)
    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
    df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
    df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

    df9 = pd.read_table(path, sep="\s+", names=column_names)
    df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
    df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df9.drop(df9[df9['JD'] < min(lc9)].index, inplace = True)

    df10 = pd.read_table(path, sep="\s+", names=column_names)
    df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
    df10.drop(df10[df10['JD'] < min(lc9)].index, inplace = True)

    meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]

    meds_err = [mad_std(df1.mag), mad_std(df2.mag), mad_std(df3.mag),mad_std(df4.mag), mad_std(df5.mag),
                       mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag), mad_std(df9.mag), mad_std(df10.mag)]

    indexes = np.array([0,1,2,3,4,5,6,7,8,9])


#slope
slope, intercept, r_value, p_value, std_err = stats.linregress(indexes,meds)

N = len(meds)

start = meds[0]
end = meds[-1]

            #polyfit

degree = 2

coeffs = np.polyfit(indexes,meds,degree)
poly_function = np.poly1d(coeffs)

            #fitted data
fitted_mag = poly_function(indexes)
quadratic_slope = coeffs[-3]

slope2 = coeffs[-2]            #vertex and max slope

vertex = coeffs[-2]/(2*coeffs[-3])

quad = -coeffs[-3]*(vertex**2) + coeffs[-2]*(vertex) + coeffs[-1]

diff1 = np.abs(start - quad)
diff2 = np.abs(end - quad)

if diff1 > diff2:
    max_ps = diff1/N

elif diff2 > diff1:
    max_ps = diff2/N

if slope < 0:
    max_ps = -max_ps

c1 = coeffs[-1]
c2 = coeffs[-2]


#IDX = np.where(finals['ASAS-SN ID'] == Targets)[0][0]
#finals['max implied slope'] = max_ps

        #seasonal medians
#_Target.append(Target)
#Median.append(lcMED)
#Median_err.append(lcMED_err)
#Dispersion.append(lcDISP)
#Slope.append(slope)
#Slope_err.append(slope_err)

print(target)
print(meds[0],meds[-1])
#tbl_LTvar = Table([_Target, Median, Median_err, Dispersion, Slope, Slope_err],
 #   names=('ASAS-SN ID', 'Median', 'Median_err', 'Dispersion','Slope', 'Slope_err'),
#    meta={'name': 'LTvar'})

#tbl_LTvar.write('LTvar.csv', overwrite=True)
        
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
print(len(lc1),len(lc2),len(lc3),len(lc4),len(lc5),len(lc6),len(lc7),len(lc8),len(lc9),len(lc10))
print(slope, max_ps, slope2)
