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
import os
import warnings
import time
import csv
from tqdm import tqdm
from time import sleep

warnings.filterwarnings("ignore")

df1 = pd.read_csv('bscuts.csv')
Targets = df1['ASAS-SN ID'].tolist()

column_names=["JD","mag",'error', 'good/bad', 'camera#', 'band', 'camera name'] #1=good, 0 =bad #1=V, 0=g
last = 79
directories = [str(i) for i in range(80)]

LTvar = Table.read('14.5_15/meds/LTvarMEDS.csv')

_Target = np.array(LTvar['ASAS-SN ID']).tolist()
Mag = np.array(LTvar['Pstarrs gmag']).tolist()
Median = np.array(LTvar['Median']).tolist()
Median_err = np.array(LTvar['Median_err']).tolist()
med1 = np.array(LTvar['Med1']).tolist()
med2 = np.array(LTvar['Med2']).tolist()
med3 = np.array(LTvar['Med3']).tolist()
med4 = np.array(LTvar['Med4']).tolist()
med5 = np.array(LTvar['Med5']).tolist()
med6 = np.array(LTvar['Med6']).tolist()
med7 = np.array(LTvar['Med7']).tolist()
med8 = np.array(LTvar['Med8']).tolist()
med9 = np.array(LTvar['Med9']).tolist()
med10 = np.array(LTvar['Med10']).tolist()
med11 = np.array(LTvar['Med11']).tolist()
med12 = np.array(LTvar['Med12']).tolist()

for x in directories:
    startTime = time.time()
    print('Starting ' + x + ' directory')

    ID = pd.read_table('14.5_15/index' + x + '.csv', sep=r'\,|\t', engine='python')
    directory = '14.5_15/lc' + x + '_cal/'

    files = [f for f in os.listdir(directory) if f.endswith('.dat')]

    #for filename in os.listdir(directory):

    for filename in tqdm(files, desc="Progress for "+ x + "/79"):
            
        path = directory + filename
        target = [filename.split('.')[0]]
        Target = [int(i) for i in target]
        Target = Target[0]
            #print(x,Target)

        if Target in Targets:
               
    #         ra = np.where(ID['#asas_sn_id'] == Target) lccal1
            ra = np.where(ID['asas_sn_id'] == Target) #lccal2

            df = pd.read_table(path, sep="\s+", names=column_names)
            df.drop(df[df['good/bad'] < 1].index, inplace = True)

            if Target == 17181160895:
                df.drop(df[df['good/bad'] < 1].index, inplace = True)
                df.drop(df[df['JD'] < 2.458e+06].index, inplace = True)
            
            
            RA = ID['ra_deg'].iloc[ra]
           #pstarr mag
            p_mag = ID['pstarrs_g_mag'].iloc[ra]
            p_mag = np.array(p_mag)
            p_mag = float(p_mag)

            
            #median of overall lc
            
            lcMED = np.median(df.mag)
            lcMED_err = mad_std(df.mag)
            
            #dispersion of overall lc (range)

            lcDISP = np.ptp(df.mag)
            
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

            mid_l = len(mid)

            lens = []

            lens.append(mid_l)
            

            if len(mid) > 11:
                lc1 = [x for x in df['JD'] if x < mid[-1]]
                lc2 = [x for x in df['JD'] if x > mid[-1] and x < mid[-2]]
                lc3 = [x for x in df['JD'] if x > mid[-2] and x < mid[-3]]
                lc4 = [x for x in df['JD'] if x > mid[-3] and x < mid[-4]]
                lc5 = [x for x in df['JD'] if x > mid[-4] and x < mid[-5]]
                lc6 = [x for x in df['JD'] if x > mid[-5] and x < mid[-6]]
                lc7 = [x for x in df['JD'] if x > mid[-6] and x < mid[-7]]
                lc8 = [x for x in df['JD'] if x > mid[-7] and x < mid[-8]]
                lc9 = [x for x in df['JD'] if x > mid[-8] and x < mid[-9]]
                lc10 = [x for x in df['JD'] if x > mid[-9] and x < mid[-10]]
                lc11 = [x for x in df['JD'] if x > mid[-10] and x < mid[-11]]
                lc12 = [x for x in df['JD'] if x > mid[-11]]
            
            elif len(mid) <= 2:
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
                lc11 = []
                lc12 = []
        
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
                lc11 = []
                lc12 = []
                
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
                lc11 = []
                lc12 = []
                
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
                lc11 = []
                lc12 = []
                
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
                lc11 = []
                lc12 = []

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
                lc11 = []
                lc12 = []

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
                lc11 = []
                lc12 = []
                
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
                lc11 = []
                lc12 = []

            elif len(mid) <= 10:
                lc1 = [x for x in df['JD'] if x < mid[-1]]
                lc2 = [x for x in df['JD'] if x > mid[-1] and x < mid[-2]]
                lc3 = [x for x in df['JD'] if x > mid[-2] and x < mid[-3]]
                lc4 = [x for x in df['JD'] if x > mid[-3] and x < mid[-4]]
                lc5 = [x for x in df['JD'] if x > mid[-4] and x < mid[-5]]
                lc6 = [x for x in df['JD'] if x > mid[-5] and x < mid[-6]]
                lc7 = [x for x in df['JD'] if x > mid[-6] and x < mid[-7]]
                lc8 = [x for x in df['JD'] if x > mid[-7] and x < mid[-8]]
                lc9 = [x for x in df['JD'] if x > mid[-8] and x < mid[-9]]
                lc10 = [x for x in df['JD'] if x > mid[-9] and x < mid[-10]]
                lc11 = [x for x in df['JD'] if x > mid[-10]]
                lc12 = []

           
               # print(len(lc1),len(lc2),len(lc3),len(lc4),len(lc5),len(lc6),len(lc7),len(lc8),len(lc9),len(lc10))            
           
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
                
                indexes = np.array([1,5,6,7,8,9,10])

                Med1 = meds[0]
                Med2 = np.nan
                Med3 = np.nan
                Med4 = np.nan
                Med5 = meds[1]
                Med6 = meds[2]
                Med7 = meds[3]
                Med8 = meds[4]
                Med9 = meds[5]
                Med10 = meds[6]
                Med11 = np.nan
                Med12 = np.nan
                
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
                #   df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
                df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

                meds = [np.median(df1.mag),
                np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]

                meds_err = [mad_std(df1.mag),
                mad_std(df4.mag), mad_std(df5.mag), mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag)]

                indexes = np.array([1,4,5,6,7,8])

                Med1 = meds[0]
                Med2 = np.nan
                Med3 = np.nan
                Med4 = meds[1]
                Med5 = meds[2]
                Med6 = meds[3]
                Med7 = meds[4]
                Med8 = meds[5]
                Med9 = np.nan
                Med10 = np.nan
                Med11 = np.nan
                Med12 = np.nan


            elif (len(lc1) >  0 and len(lc2) == 0 and len(lc3) == 0 and len(lc9) == 0 and len(lc8) > 0):
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
            #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
                df8.drop(df8[df8['JD'] < min(lc8)].index, inplace = True)

                meds = [np.median(df1.mag),
                np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]

                indexes = np.array([1,4,5,6,7,8])

                Med1 = meds[0]
                Med2 = np.nan
                Med3 = np.nan
                Med4 = meds[1]
                Med5 = meds[2]
                Med6 = meds[3]
                Med7 = meds[4]
                Med8 = meds[5]
                Med9 = np.nan
                Med10 = np.nan
                Med11 = np.nan
                Med12 = np.nan


            elif (len(lc1)>0 and len(lc2) == 0 and len(lc3) == 0 and len(lc4) == 0 and len(lc5) == 0 and len(lc6) > 0):
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
                df10.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                #df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
                df10.drop(df10[df10['JD'] < min(lc9)].index, inplace = True)

                meds = [np.median(df1.mag),
                np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]

                indexes = np.array([1,6,7,8,9,10])



            elif (len(lc2) == 0 and len(lc9) == 0 and len(lc10) == 0 and len(lc8) > 0):
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] <1].index, inplace=True)
                df1.drop(df1[df1['JD'] > max(lc1)].index, inplace=True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df1[df1['good/bad'] < 1].index, inplace=True)
                df3.drop(df3[df3['JD'] > max(lc3)].index, inplace=True)
                df3.drop(df3[df3['JD'] < min(lc3)].index, inplace=True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace=True)
                df4.drop(df4[df4['JD'] > max(lc4)].index, inplace=True)
                df4.drop(df4[df4['JD'] < min(lc4)].index, inplace=True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] <1 ].index, inplace=True)
                df5.drop(df5[df5['JD'] > max(lc5)].index, inplace=True)
                df5.drop(df5[df5['JD'] < min(lc5)].index, inplace=True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace=True)
                df6.drop(df6[df6['JD'] > max(lc6)].index, inplace=True)
                df6.drop(df6[df6['JD'] < min (lc6)].index, inplace=True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] <1].index, inplace=True)
                df7.drop(df7[df7['JD'] > max(lc7)].index, inplace=True)
                df7.drop(df7[df7['JD'] < min (lc7)].index, inplace=True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace=True)
                df8.drop(df8[df8['JD'] < min(lc8)].index, inplace=True)

                meds = [np.median(df1.mag), np.median(df3.mag), np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]
                indexes = np.array([1,3,4,5,6,7,8])

                Med1 = meds[0]
                Med2 = np.nan
                Med3 = meds[1]
                Med4 = meds[2]
                Med5 = meds[3]
                Med6 = meds[4]
                Med7 = meds[5]
                Med8 = meds[6]
                Med9 = np.nan
                Med10 = np.nan
                Med11 = np.nan
                Med12 = np.nan

               
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
                
                indexes = np.array([1,3,4,5,6,7])

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
            #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = Tre)
                df10.drop(df10[df10['JD'] < min(lc9)].index, inplace = True)
                
                meds = [np.median(df1.mag) ,np.median(df5.mag),
                np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
                
                indexes = np.array([1,5,6,7,8,9,10])

                Med1 = meds[0]
                Med2 = np.nan
                Med3 = np.nan
                Med4 = np.nan
                Med5 = meds[1]
                Med6 = meds[2]
                Med7 = meds[3]
                Med8 = meds[4]
                Med9 = meds[5]
                Med10 = meds[6]
                Med11 = np.nan
                Med12 = np.nan

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
                
                indexes = np.array([1,4,5,6,7,8,9,10])

                Med1 = meds[0]
                Med2 = np.nan
                Med3 = np.nan
                Med4 = meds[1]
                Med5 = meds[2]
                Med6 = meds[3]
                Med7 = meds[4]
                Med8 = meds[5]
                Med9 = meds[6]
                Med10 = meds[7]
                Med11 = np.nan
                Med12 = np.nan



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
                
                indexes = np.array([1,2,5,6,7,8,9,10])

                Med1 = meds[0]
                Med2 = meds[1]
                Med3 = np.nan
                Med4 = np.nan
                Med5 = meds[2]
                Med6 = meds[3]
                Med7 = meds[4]
                Med8 = meds[5]
                Med9 = meds[6]
                Med10 = meds[7]
                Med11 = np.nan
                Med12 = np.nan

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
                
                indexes = np.array([1,2,4,5,6,7,8,9,10])

                Med1 = meds[0]
                Med2 = meds[1]
                Med3 = np.nan
                Med4 = meds[2]
                Med5 = meds[3]
                Med6 = meds[4]
                Med7 = meds[5]
                Med8 = meds[6]
                Med9 = meds[7]
                Med10 = meds[8]
                Med11 = np.nan
                Med12 = np.nan


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
                
                indexes = np.array([1,3,4,5,6,7,8,9,10])

                Med1 = meds[0]
                Med2 = np.nan
                Med3 = meds[1]
                Med4 = meds[2]
                Med5 = meds[3]
                Med6 = meds[4]
                Med7 = meds[5]
                Med8 = meds[6]
                Med9 = meds[7]
                Med10 = meds[8]
                Med11 = np.nan
                Med12 = np.nan


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

                indexes = np.array([1,2,3])

                Med1 = meds[0]
                Med2 = meds[1]
                Med3 = meds[2]
                Med4 = np.nan
                Med5 = np.nan
                Med6 = np.nan
                Med7 = np.nan
                Med8 = np.nan
                Med9 = np.nan
                Med10 = np.nan
                Med11 = np.nan
                Med12 = np.nan


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
                
                indexes = np.array([1,2,3,4])

                Med1 = meds[0]
                Med2 = meds[1]
                Med3 = meds[2]
                Med4 = meds[3]
                Med5 = np.nan
                Med6 = np.nan
                Med7 = np.nan
                Med8 = np.nan
                Med9 = np.nan
                Med10 = np.nan
                Med11 = np.nan
                Med12 = np.nan


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

                indexes = np.array([1,2,3,4,5])

                Med1 = meds[0]
                Med2 = meds[1]
                Med3 = meds[2]
                Med4 = meds[3]
                Med5 = meds[4]
                Med6 = np.nan
                Med7 = np.nan
                Med8 = np.nan
                Med9 = np.nan
                Med10 = np.nan
                Med11 = np.nan
                Med12 = np.nan
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

                indexes = np.array([1,2,3,4,5,6])

                Med1 = meds[0]
                Med2 = meds[1]
                Med3 = meds[2]
                Med4 = meds[3]
                Med5 = meds[4]
                Med6 = meds[5]
                Med7 = np.nan
                Med8 = np.nan
                Med9 = np.nan
                Med10 = np.nan
                Med11 = np.nan
                Med12 = np.nan

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

                indexes = np.array([1,2,3,4,5,6,7])

                Med1 = meds[0]
                Med2 = meds[1]
                Med3 = meds[2]
                Med4 = meds[3]
                Med5 = meds[4]
                Med6 = meds[5]
                Med7 = meds[6]
                Med8 = np.nan
                Med9 = np.nan
                Med10 = np.nan
                Med11 = np.nan
                Med12 = np.nan

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

                indexes = np.array([1,2,3,4,5,6,7,8])

                Med1 = meds[0]
                Med2 = meds[1]
                Med3 = meds[2]
                Med4 = meds[3]
                Med5 = meds[4]
                Med6 = meds[5]
                Med7 = meds[6]
                Med8 = meds[7]
                Med9 = np.nan
                Med10 = np.nan
                Med11 = np.nan
                Med12 = np.nan

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
            #       
                df9.drop(df9[df9['JD'] < min(lc9)].index, inplace = True)

                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                        np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag)]

                indexes = np.array([1,2,3,4,5,6,7,8,9])
                Med1 = meds[0]
                Med2 = meds[1]
                Med3 = meds[2]
                Med4 = meds[3]
                Med5 = meds[4]
                Med6 = meds[5]
                Med7 = meds[6]
                Med8 = meds[7]
                Med9 = meds[8]
                Med10 = np.nan
                Med11 = np.nan
                Med12 = np.nan

            elif len(lc11) <= 1:
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
                df10.drop(df10[df10['JD'] < min(lc10)].index, inplace = True)


                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                        np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]

                indexes = np.array([1,2,3,4,5,6,7,8,9,10])

                Med1 = meds[0]
                Med2 = meds[1]
                Med3 = meds[2]
                Med4 = meds[3]
                Med5 = meds[4]
                Med6 = meds[5]
                Med7 = meds[6]
                Med8 = meds[7]
                Med9 = meds[8]
                Med10 = meds[9]
                Med11 = np.nan
                Med12 = np.nan

            elif len(lc12) <= 1:
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
                df10.drop(df10[df10['JD'] > max(lc10)].index, inplace = True)
                df10.drop(df10[df10['JD'] < min(lc10)].index, inplace = True)

                df11 = pd.read_table(path, sep="\s+", names=column_names)
                df11.drop(df11[df11['good/bad'] < 1].index, inplace = True)
                df11.drop(df11[df11['JD'] < min(lc11)].index, inplace = True)


                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                        np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag), np.median(df11.mag)]

                indexes = np.array([1,2,3,4,5,6,7,8,9,10,11])

                Med1 = meds[0]
                Med2 = meds[1]
                Med3 = meds[2]
                Med4 = meds[3]
                Med5 = meds[4]
                Med6 = meds[5]
                Med7 = meds[6]
                Med8 = meds[7]
                Med9 = meds[8]
                Med10 = meds[9]
                Med11 = meds[10]
                Med12 = np.nan


            elif len(lc12) > 1:
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
                df10.drop(df10[df10['JD'] > max(lc10)].index, inplace = True)
                df10.drop(df10[df10['JD'] < min(lc10)].index, inplace = True)

                df11 = pd.read_table(path, sep="\s+", names=column_names)
                df11.drop(df11[df11['good/bad'] < 1].index, inplace = True)
                df11.drop(df11[df11['JD'] > max(lc11)].index, inplace = True)
                df11.drop(df11[df11['JD'] < min(lc11)].index, inplace = True)

                df12 = pd.read_table(path, sep="\s+", names=column_names)
                df12.drop(df12[df12['good/bad'] < 1].index, inplace = True)
                df12.drop(df12[df12['JD'] < min(lc12)].index, inplace = True)


                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                        np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag), np.median(df11.mag), np.median(df12.mag)]

                indexes = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

                Med1 = meds[0]
                Med2 = meds[1]
                Med3 = meds[2]
                Med4 = meds[3]
                Med5 = meds[4]
                Med6 = meds[5]
                Med7 = meds[6]
                Med8 = meds[7]
                Med9 = meds[8]
                Med10 = meds[9]
                Med11 = meds[10]
                Med12 = meds[11]

#slope add back in!
          
            _Target.append(Target)
            Mag.append(p_mag)
            Median.append(lcMED)
            Median_err.append(lcMED_err)
            med1.append(Med1)
            med2.append(Med2)
            med3.append(Med3)
            med4.append(Med4)
            med5.append(Med5)
            med6.append(Med6)
            med7.append(Med7)
            med8.append(Med8)
            med9.append(Med9)
            med10.append(Med10)
            med11.append(Med11)
            med12.append(Med12)

    columns = ['ASAS-SN ID','Pstarrs gmag', 'Median', 'Median_err','Med1','Med2','Med3','Med4','Med5','Med6','Med7','Med8','Med9','Med10','Med11','Med12']
    data_lists = [_Target, Mag, Median, Median_err, med1, med2, med3, med4, med5,
              med6, med7, med8, med9, med10, med11, med12]

    max_len = max(len(col) for col in data_lists)  # Find the max length among columns

    for i in range(len(data_lists)):
        if len(data_lists[i]) == 0:
            print(f"Filling empty column: {columns[i]}")
            data_lists[i] = [np.nan] * max_len  # Fill empty columns with np.nan
        elif len(data_lists[i]) < max_len:
            data_lists[i] += [np.nan] * (max_len - len(data_lists[i]))  # Pad shorter columns

    tbl_LTvar = Table(data_lists, names=columns)
            #tbl_LTvar = Table([_Target,Mag, Median, Median_err, med1, med2, med3, med4, med5, med6, med7, med8, med9, med10, med11, med12],
        #names=('ASAS-SN ID','Pstarrs gmag', 'Median', 'Median_err','Med1','Med2','Med3','Med4','Med5','Med6','Med7','Med8','Med9','Med10','Med11','Med12'),
        #meta={'name': 'LTvar'})



#update column names

    csv_file_path = '14.5_15/meds/' + x + '.csv'

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(tbl_LTvar)


    #print(max(lens))
    print('Ending ' + x)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' +str(executionTime))
