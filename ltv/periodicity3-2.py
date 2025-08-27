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
from astropy.timeseries import LombScargle


warnings.filterwarnings("ignore")



column_names=["JD","mag",'error', 'good/bad', 'camera#', 'band', 'camera name'] #1=good, 0 =bad #1=V, 0=g

var_tw = pd.read_csv('vars2.csv')
Targets = var_tw['ASAS-SN ID'].tolist()

ID = pd.read_csv('13-13.5/index13_13.5.csv')
#ID = pd.read_csv('13.5-14/index13.5-14.csv')
#ID3 = pd.read_csv('14-14.5/index.csv')
#ID = pd.concat([ID1,ID2,ID3])

plot=False

for x in [*range(5,31)]:

#directory = 'final_candidates_lcs'
    directory = '13-13.5/lc' + str(x) + '_cal'

    for filename in os.listdir(directory):
            if filename.endswith('.dat'):
                    path = directory + '/' + filename
                    target = [filename.split('.')[0]]
                    Target = [int(i) for i in target]
                    Target = Target[0]

                    i = Target

                    if i in Targets:
                        print(str(x) + " / " + str(30) + " - " + str(Target))
                        
                        df1 = pd.read_csv(path, sep="\s+", names=column_names)
                        df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)

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
                            df10.drop(df10[df10['JD'] < min(lc10)].index, inplace = True)
                
                            meds = [np.median(df1.mag) ,np.median(df5.mag),
                            np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
                
                            indexes = np.array([0,1,2,3,4,5,6])

                            dfs = [df1,df5,df6,df7,df8,df9,df10]

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
                            df8.drop(df8[df8['JD'] < min(lc9)].index, inplace = True)

                            meds = [np.median(df1.mag),
                            np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]

                            indexes = np.array([0,1,2,3,4,5])

                            dfs = [df1,df4,df5,df6,df7,df8]


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
                            df10.drop(df10[df10['JD'] < min(lc10)].index, inplace = True)

                            meds = [np.median(df1.mag),
                            np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]

                            indexes = np.array([0,1,2,3,4,5])

                            dfs = [df1,df6,df7,df8,df9,df10]



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
                            indexes = np.array([0,1,2,3,4,5,6])

                            dfs = [df1,df3,df4,df5,df6,df7,df8]
               
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
                
                            indexes = np.array([0,1,2,3,4,5])

                            dfs = [df1,df3,df4,df5,df6,df7]

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
                            df10.drop(df10[df10['JD'] < min(lc10)].index, inplace = True)
                
                            meds = [np.median(df1.mag) ,np.median(df5.mag),
                            np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
                
                            indexes = np.array([0,1,2,3,4,5,6])

                            dfs = [df1,df2,df6,df7,df8,df9,df10]

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
                            df10.drop(df10[df10['JD'] < min(lc10)].index, inplace = True)
                
                            meds = [np.median(df1.mag),np.median(df4.mag),np.median(df5.mag),
                            np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
                
                            indexes = np.array([0,1,2,3,4,5,6,7])

                            dfs = [df1,df4,df5,df6,df7,df8,df9,df10]


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
                            df10.drop(df10[df10['JD'] < min(lc10)].index, inplace = True)
                
                            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df5.mag),
                            np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
                
                            indexes = np.array([0,1,2,3,4,5,6,7])

                            dfs = [df1,df2,df5,df6,df7,df8,df9,df10]

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
                            df10.drop(df10[df10['JD'] < min(lc10)].index, inplace = True)
                
                            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df4.mag),np.median(df5.mag),
                            np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
                
                            indexes = np.array([0,1,2,3,4,5,6,7,8])

                            dfs = [df1,df2,df4,df5,df6,df7,df8,df9,df10]

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
                
                            indexes = np.array([0,1,2,3,4,5,6,7,8])

                            dfs = [df1,df3,df4,df5,df6,df7,df8,df9,df10]

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

                            indexes = np.array([0,1,2])

                            dfs = [df1,df2,df3]

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
                
                            indexes = np.array([0,1,2,3])

                            dfs = [df1,df2,df3,df4]

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

                            indexes = np.array([0,1,2,3,4])

                            dfs = [df1,df2,df3,df4,df5]

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

                            indexes = np.array([0,1,2,3,4,5])

                            dfs = [df1,df2,df3,df4,df5,df6]

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

                            indexes = np.array([0,1,2,3,4,5,6])

                            dfs = [df1,df2,df3,df4,df5,df6,df7]

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

                            indexes = np.array([0,1,2,3,4,5,6,7])

                            dfs = [df1,df2,df3,df4,df5,df6,df7,df8]

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

                            indexes = np.array([0,1,2,3,4,5,6,7,8])

                            dfs = [df1,df2,df3,df4,df5,df6,df7,df8,df9]

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
                            df10.drop(df10[df10['JD'] < min(lc10)].index, inplace = True)

                            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                        np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]

                            indexes = np.array([0,1,2,3,4,5,6,7,8,9])

                            dfs=[df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]


#slope add back in!
                        if Target == 77310656806:
                            continue
                        elif Target == 51539980676:
                            continue
                        elif Target == 395137591432:
                            continue
                        elif Target == 180388896780:
                            continue
                        elif Target == 68719657585:
                            continue
                        elif Target == 51539954617:
                            continue
                        elif Target == 137440390128:
                            continue
                        elif Target == 412316992073:
                            continue
                        elif Target == 558346685740:
                            continue
                        elif Target == 386548333169:
                            continue
                        elif Target == 266288989631:
                            continue

                        time = df['JD']
                        mag = df['mag']

                        #if i in finals5['ASAS-SN ID'].tolist():
                            #degree = 2

                        #else:
                        degree = 1

                        coefficients = np.polyfit(time, mag, degree)
                
                        polynomial_fit = np.polyval(coefficients, time)
                
                        residuals = mag - polynomial_fit

                        time_l = len(time)

                    #ls_f = LombScargle(time, residuals)
                    #ls_freq, ls_power = ls_f.autopower(minimum_frequency=1/time_l, maximum_frequency=1/2)

                    #fapp = ls_f.false_alarm_level(1e-5)
                    #F_prob = LombScargle(time, residuals).false_alarm_probability(ls_power.max(), method='bootstrap')


                        periods = []

                        phases = []

                        amplitudes = []

                        obs_len = []

                        faps = []
                    

                        for d in dfs:
                            Time = d['JD']
                            Mag = d['mag']
                            mag_err = df['error']
                            Mag = np.array(Mag)

                            t_len = max(Time) - min(Time)

                            start = min(Time)
                            end = max(Time)

                            ID_s = np.where(df['JD'] == start)[0][0]
                            ID_e = np.where(df['JD'] == end)[0][0] + 1


                            if t_len > 1:
                                min_f = 1/t_len
                            else:
                                min_f = 1/400
                            max_f = 1/2
                        #print(len(residuals),len(mag_err)
                        
                            if len(Time) == len(residuals[ID_s:ID_e]):
                                ls = LombScargle(Time, residuals[ID_s:ID_e]) #, dy=mag_err[ID_s:ID_e])
                            #ls = LombScargle(Time, Mag[ID_s:ID_e]) #, dy=mag_err[ID_s:ID_e])




                            else:
                                print("Length mismatch: Adjusting residuals segment.")
                                residuals_segment = residuals[ID_s:ID_s + len(Time)]
                                ls = LombScargle(Time, residuals_segment)


                        
                        #ls = LombScargle(Time, residuals[ID_s:ID_e])

                            frequency, power = ls.autopower(minimum_frequency=min_f, maximum_frequency=max_f)

                        #frequency, power = LombScargle(Time, residuals[ID_s:ID_e]).autopower()
                        #print(frequency,power)

                            if len(frequency)<1:
                                continue
                            if len(power)<1:
                                continue
                            
                            period = 1 / frequency[np.argmax(power)]

                        #print(period, 1/max(frequency))
                        
                        #phase = (Time / period) % 1  # Normalize to the range
                            phase = (Time % period) / period

                            phase = np.array(phase)




                        #print(len(Time),len(residuals[ID_s:ID_e]))

                            if len(Time) > 1:

                                f_list = [566935844547,180388696014,317827871397,111669454522,206159793624,42950855497,463857423115,223339083353,266288549460,506806585013,446677287120,223339059138,489626455680,592706232639,326418220317,506807127390,635656240020,661425079419,498217055304, 541166617655, 489626274172,558346612804,154619676418,584115800887,661425331748,128850161020,25770668051,85899368559,163209152446]

                            #if i in f_list:
                                #flapp = 0.1
                            #else:
                                #flapp = ls.false_alarm_level(1e-5)
                            #print(power.max())
                            #test = ls.false_alarm_probability(power.max())
                            #print(test)
                            
                                ###fap = ls.false_alarm_probability(power.max(), method='bootstrap')
                            #print(fap)
                                #fap = np.sum(np.array(bootstrapped_powers) >= max_power) / n_bootstraps
                            #fig1, ax5 = plt.subplots(1, 1, figsize=(8, 6))
                            #ax5.plot(frequency,power, color='k')
                            #ax5.axhline(flapp, color='r')
                            #plt.show()


                            else:
                                y=1


                            sorted_indices = np.argsort(phase)
                            phase_sorted = phase[sorted_indices]
                            mag_sorted = Mag[sorted_indices]
                        #mag_err_sorted = mag_err[sorted_indices]

                            peak_power = np.max(power)
                            amp = peak_power/2

                            periods.append(period)
                            phases.append(phase)
                            amplitudes.append(amp)
                            obs_len.append(t_len)
                            ###faps.append(fap)

                    #print(obs_len)

                        max_period = np.median(obs_len)
                    
                        periods = [x for x in periods if not np.isnan(x)]
                        avg_p = np.median(periods)
                        ###faps = [x for x in faps if not np.isnan(x)]
                        ###FAP = np.mean(faps)
                    #print(FAP)
                    
                    #print(periods)
                        amplitude = np.mean(amplitudes[-1])

                        periodic_function = amplitude * np.sin(2*np.pi*time/avg_p)

                        num_seasons = len(dfs)

                        cmap = plt.get_cmap('tab10')
                        season_colors = [cmap(i) for i in np.linspace(0,1, num_seasons)]

                   # residual2 = residuals- periodic_function

                   # phases2 = []

                   # periods2 = []

                    #for d2 in dfs:

                       # Time2 = d2['JD']

                       # start2 = min(Time2)
                       # end2 = max(Time2)

                       # ID_s2 = np.where(df['JD'] == start2)[0][0]


                        if avg_p < max_period:
                        #or avg_p > max_period: #max_period

                    #print(avg_p_2)

                    #plots
                            if plot == True:
                                fig, (ax1,ax2) = plt.subplots(2,1, figsize=(12,8), sharey=True)

                                ax1.errorbar(time-2450000,mag,yerr=df.error, fmt='o',markersize=2, color='k', ecolor='r')
                                ax1.plot(time-2450000,polynomial_fit, color='teal', linewidth=3, label='Subtracted trend')
                                ax1.set_xlabel('JD-2450000', fontsize=15)
                                ax1.xaxis.set_tick_params(labelsize=13)
                                ax1.yaxis.set_tick_params(labelsize=13)
                                ax1.legend(fontsize=13)

                            #plt.ylabel('gmag')
                                ax1.set_title(Target, fontsize=20)
                                ax1.invert_yaxis()

                            #plt.subplot(3,1,2)
                            #plt.plot(time/1e+6,residuals, 'ok', markersize=2, label='Residuals', color='orange')
                            #plt.legend() 
                            #plt.gca().invert_yaxis()

                                for i in range(len(dfs)):
                                    phase = phases[i]
                                    df = dfs[i]

                                    mag_data= df['mag']
                                    mag_err = df['error']

                                    num_cycles = 2
                                #print(len(phase_sorted))
                                    for j in range(num_cycles):
                                    #ax2.errorbar(phase_sorted + j, mag_sorted, yerr=mag_err_sorted, fmt='o', color='teal', ecolor='r', markersize=2, label='Phase folded light curve')
                                        ax2.errorbar(phase_sorted + j, mag_sorted, yerr=0, fmt='o', color='teal', ecolor='r', markersize=2, label='Phase folded light curve')


                                        ax2.invert_yaxis()
                                ax2.set_xlabel('Phase', fontsize=15)
                                ax2.xaxis.set_tick_params(labelsize=13)
                                ax2.yaxis.set_tick_params(labelsize=13)


                            #plt.savefig('LP.pdf', dpi=300)
                                plt.tight_layout()

                                plt.show()



                        if i in Targets:

                            IDx=np.where(var_tw['ASAS-SN ID'] == Target)[0][0]
                            var_tw['Period'].iloc[IDx] = avg_p
                            #var_tw['FAP'].iloc[IDx] = FAP
                            #print(avg_p, var_tw['var_type'].iloc[IDx])
                    else:

                        if i in Targets:

                            IDx=np.where(var_tw['ASAS-SN ID'] == Target)[0][0]
                            var_tw['Period'].iloc[IDx] = np.nan
                            #var_tw['FAP'].iloc[IDx] = FAP
                            print(avg_p)
    var_tw.to_csv('vars2.csv', index=False)
                  

                

