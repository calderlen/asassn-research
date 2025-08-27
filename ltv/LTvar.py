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


column_names=["JD","mag",'error', 'good/bad', 'camera#', 'band', 'bsflag?', 'camera name'] #1=good, 0 =bad #1=V, 0=g
last = 30
directories = ['8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']

LTvar = Table.read('LTvar13-13.5.csv')

_Target = np.array(LTvar['ASAS-SN ID']).tolist()
Mag = np.array(LTvar['Pstarrs gmag']).tolist()
Median = np.array(LTvar['Median']).tolist()
Median_err = np.array(LTvar['Median_err']).tolist()
Dispersion = np.array(LTvar['Dispersion']).tolist()
Slope = np.array(LTvar['Slope']).tolist()
Quad_Slope = np.array(LTvar['Quad Slope']).tolist()
Coeff1 = np.array(LTvar['coeff1']).tolist()
Coeff2 = np.array(LTvar['coeff2']).tolist()
Diff = np.array(LTvar['max diff']).tolist()

for x in directories:
    startTime = time.time()
    print('Starting ' + x + ' directory')

    ID = pd.read_table('/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5//index' + x + '.csv', sep=r'\,|\t', engine='python')
    directory = '/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc' + x + '_cal/'

    files = [f for f in os.listdir(directory) if f.endswith('.dat')]

    #for filename in os.listdir(directory):

    for filename in tqdm(files, desc="Progress for "+ x + "/30"):
            
            path = directory + filename
            target = [filename.split('.')[0]]
            Target = [int(i) for i in target]
            Target = Target[0]
            #print(Target)
               
    #         ra = np.where(ID['#asas_sn_id'] == Target) lccal1
            ra = np.where(ID['asas_sn_id'] == Target) #lccal2

            df = pd.read_table(path, sep="\s+", names=column_names)
            df.drop(df[df['good/bad'] < 1].index, inplace = True)

            df['JD'] = df['JD'] + 2450000

            if Target == 17181160895:
                df.drop(df[df['good/bad'] < 1].index, inplace = True)
                df.drop(df[df['JD'] < 2.458e+06].index, inplace = True)

            if Target == 34360695533:
                continue

            if Target == 8590720263:
                continue
            
            
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

            mid = [float(x) for x in mid]

            mid = [x for x in mid if min(df.JD) < x < max(df.JD)]

            mid_l = len(mid)
            #print(mid_l)

            if mid_l == 1:
                continue

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

            elif len(mid) > 10:
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

            elif len(mid) <= 2:
                print(target)
                try: 
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
                except (IndexError):
                    continue
        
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

            #print(len(lc1),len(lc2),len(lc3),len(lc4),len(lc5),len(lc6),len(lc7),len(lc8),len(lc9),len(lc10),len(lc11),len(lc12))

            #breakpoint()
           
            if (len(lc2) == 0 and len(lc3) == 0 and len(lc4) == 0 and len(lc5) > 0):
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 > max(lc8)].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                df9 = pd.read_table(path, sep="\s+", names=column_names)
                df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 > max(lc9)].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 < min(lc9)].index, inplace = True)

                df10 = pd.read_table(path, sep="\s+", names=column_names)
                df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
            #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
                df10.drop(df10[df10['JD']+ 2450000 < min(lc9)].index, inplace = True)
                
                meds = [np.median(df1.mag) ,np.median(df5.mag),
                np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
                
                indexes = np.array([1,5,6,7,8,9,10])
                
            elif(len(lc1) > 0 and len(lc2) == 0 and len(lc3) == 0 and len(lc4) > 0 and len(lc5) > 0 and len(lc9) == 0 and len(lc10) == 0):
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                #   df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                meds = [np.median(df1.mag),
                np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]

                meds_err = [mad_std(df1.mag),
                mad_std(df4.mag), mad_std(df5.mag), mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag)]

                indexes = np.array([1,4,5,6,7,8])

            elif (len(lc1) >  0 and len(lc2) == 0 and len(lc3) == 0 and len(lc9) == 0 and len(lc8) > 0):
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                meds = [np.median(df1.mag),
                np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]

                indexes = np.array([1,4,5,6,7,8])


            elif (len(lc1)>0 and len(lc2) == 0 and len(lc3) == 0 and len(lc4) == 0 and len(lc5) == 0 and len(lc6) > 0):
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 > max(lc8)].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                df9 = pd.read_table(path, sep="\s+", names=column_names)
                df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 > max(lc9)].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 < min(lc9)].index, inplace = True)

                df10 = pd.read_table(path, sep="\s+", names=column_names)
                df10.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                #df8.drop(df8[df8['JD'] > max(lc8)].index, inplace = True)
                df10.drop(df10[df10['JD']+ 2450000 < min(lc9)].index, inplace = True)

                meds = [np.median(df1.mag),
                np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]

                indexes = np.array([1,6,7,8,9,10])



            elif (len(lc2) == 0 and len(lc9) == 0 and len(lc10) == 0 and len(lc8) > 0):
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] <1].index, inplace=True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace=True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df1[df1['good/bad'] < 1].index, inplace=True)
                df3.drop(df3[df3['JD']+ 2450000 > max(lc3)].index, inplace=True)
                df3.drop(df3[df3['JD']+ 2450000 < min(lc3)].index, inplace=True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace=True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace=True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace=True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] <1 ].index, inplace=True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace=True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace=True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace=True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace=True)
                df6.drop(df6[df6['JD']+ 2450000 < min (lc6)].index, inplace=True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] <1].index, inplace=True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace=True)
                df7.drop(df7[df7['JD']+ 2450000 < min (lc7)].index, inplace=True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace=True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace=True)

                meds = [np.median(df1.mag), np.median(df3.mag), np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]
                indexes = np.array([1,3,4,5,6,7,8])
               
            elif (len(lc1) > 0 and len(lc2) == 0 and len(lc3) > 0 and len(lc8) == 0):
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 > max(lc3)].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 < min(lc3)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df4[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df4[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df4[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                meds = [np.median(df1.mag) ,np.median(df3.mag),
                np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag)]
                
                indexes = np.array([1,3,4,5,6,7])

            elif (len(lc3) == 0 and len(lc4) == 0 and len(lc5) == 0 and len(lc6) > 0):
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df2 = pd.read_table(path, sep="\s+", names=column_names)
                df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 > max(lc2)].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 < min(lc2)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 > max(lc8)].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                df9 = pd.read_table(path, sep="\s+", names=column_names)
                df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 > max(lc9)].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 < min(lc9)].index, inplace = True)

                df10 = pd.read_table(path, sep="\s+", names=column_names)
                df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
            #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = Tre)
                df10.drop(df10[df10['JD']+ 2450000 < min(lc9)].index, inplace = True)
                
                meds = [np.median(df1.mag) ,np.median(df5.mag),
                np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
                
                indexes = np.array([1,5,6,7,8,9,10])

            elif (len(lc2) == 0 and len(lc3) == 0 and len(lc4) > 0):
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 > max(lc8)].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                df9 = pd.read_table(path, sep="\s+", names=column_names)
                df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 > max(lc9)].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 < min(lc9)].index, inplace = True)

                df10 = pd.read_table(path, sep="\s+", names=column_names)
                df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
            #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
                df10.drop(df10[df10['JD']+ 2450000 < min(lc9)].index, inplace = True)
                
                meds = [np.median(df1.mag),np.median(df4.mag),np.median(df5.mag),
                np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
                
                indexes = np.array([1,4,5,6,7,8,9,10])


            elif (len(lc3) == 0 and len(lc4) == 0 and len(lc5) > 0):
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df2 = pd.read_table(path, sep="\s+", names=column_names)
                df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 > max(lc2)].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 < min(lc2)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 > max(lc8)].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                df9 = pd.read_table(path, sep="\s+", names=column_names)
                df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 > max(lc9)].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 < min(lc9)].index, inplace = True)

                df10 = pd.read_table(path, sep="\s+", names=column_names)
                df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
            #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
                df10.drop(df10[df10['JD']+ 2450000 < min(lc9)].index, inplace = True)
                
                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df5.mag),
                np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
                
                indexes = np.array([1,2,5,6,7,8,9,10])

            elif (len(lc3) == 0 and len(lc4) > 0):
                try:
                    print(Target)
                    df1 = pd.read_table(path, sep="\s+", names=column_names)
                    df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                    df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                    df2 = pd.read_table(path, sep="\s+", names=column_names)
                    df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                    df2.drop(df2[df2['JD']+ 2450000 > max(lc2)].index, inplace = True)
                    df2.drop(df2[df2['JD']+ 2450000 < min(lc2)].index, inplace = True)

                    df4 = pd.read_table(path, sep="\s+", names=column_names)
                    df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                    df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                    df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                    df5 = pd.read_table(path, sep="\s+", names=column_names)
                    df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                    df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                    df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                    df6 = pd.read_table(path, sep="\s+", names=column_names)
                    df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                    df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                    df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                    df7 = pd.read_table(path, sep="\s+", names=column_names)
                    df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                    df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)
                    df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                    df8 = pd.read_table(path, sep="\s+", names=column_names)
                    df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                    df8.drop(df8[df8['JD']+ 2450000 > max(lc8)].index, inplace = True)
                    df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                    df9 = pd.read_table(path, sep="\s+", names=column_names)
                    df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
                    df9.drop(df9[df9['JD']+ 2450000 > max(lc9)].index, inplace = True)
                    df9.drop(df9[df9['JD']+ 2450000 < min(lc9)].index, inplace = True)

                    df10 = pd.read_table(path, sep="\s+", names=column_names)
                    df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
            #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
                    df10.drop(df10[df10['JD']+ 2450000 < min(lc9)].index, inplace = True)
                
                    meds = [np.median(df1.mag), np.median(df2.mag),np.median(df4.mag),np.median(df5.mag),
                    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
                
                    indexes = np.array([1,2,4,5,6,7,8,9,10])
                except (ValueError):
                    continue

            elif (len(lc2) == 0 and len(lc3) > 0):
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df3.drop(df2[df2['JD']+ 2450000 > max(lc3)].index, inplace = True)
                df3.drop(df2[df2['JD']+ 2450000 < min(lc3)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 > max(lc8)].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                df9 = pd.read_table(path, sep="\s+", names=column_names)
                df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 > max(lc9)].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 < min(lc9)].index, inplace = True)

                df10 = pd.read_table(path, sep="\s+", names=column_names)
                df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
            #         df9.drop(df9[df9['JD'] > max(lc9)].index, inplace = True)
                df10.drop(df10[df10['JD']+ 2450000 < min(lc9)].index, inplace = True)
                
                meds = [np.median(df1.mag), np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
                
                indexes = np.array([1,3,4,5,6,7,8,9,10])

            elif len(lc4) < 1:
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df2 = pd.read_table(path, sep="\s+", names=column_names)
                df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 > max(lc2)].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 < min(lc2)].index, inplace = True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
            #     df3.drop(df3[df3['JD'] > max(lc3)].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 < min(lc3)].index, inplace = True)

                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag)]

                indexes = np.array([1,2,3])

            elif len(lc5) < 1:
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df2 = pd.read_table(path, sep="\s+", names=column_names)
                df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 > max(lc2)].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 < min(lc2)].index, inplace = True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 > max(lc3)].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 < min(lc3)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df3[df3['good/bad'] < 1].index, inplace = True)
            #     df3.drop(df3[df3['JD'] > max(lc3)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag)]
                
                indexes = np.array([1,2,3,4])

            elif len(lc6) < 1:
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df2 = pd.read_table(path, sep="\s+", names=column_names)
                df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 > max(lc2)].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 < min(lc2)].index, inplace = True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 > max(lc3)].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 < min(lc3)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            #    df5.drop(df5[df5['JD'] > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag)]

                indexes = np.array([1,2,3,4,5])

            elif len(lc7) < 1:
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df2 = pd.read_table(path, sep="\s+", names=column_names)
                df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 > max(lc2)].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 < min(lc2)].index, inplace = True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 > max(lc3)].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 < min(lc3)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            #         df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                        np.median(df6.mag)]

                indexes = np.array([1,2,3,4,5,6])

            elif len(lc8) < 1:

                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df2 = pd.read_table(path, sep="\s+", names=column_names)
                df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 > max(lc2)].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 < min(lc2)].index, inplace = True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 > max(lc3)].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 < min(lc3)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            #         df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                        np.median(df6.mag), np.median(df7.mag)]

                indexes = np.array([1,2,3,4,5,6,7])

            elif len(lc9) < 1:
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df2 = pd.read_table(path, sep="\s+", names=column_names)
                df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 > max(lc2)].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 < min(lc2)].index, inplace = True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 > max(lc3)].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 < min(lc3)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)                       
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            #         df6.drop(df6[df6['JD'] > max(lc6)].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                        np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]

                indexes = np.array([1,2,3,4,5,6,7,8])

            elif len(lc10) <= 1:
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df2 = pd.read_table(path, sep="\s+", names=column_names)
                df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 > max(lc2)].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 < min(lc2)].index, inplace = True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 > max(lc3)].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 < min(lc3)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 > max(lc8)].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                df9 = pd.read_table(path, sep="\s+", names=column_names)
                df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
            #       
                df9.drop(df9[df9['JD']+ 2450000 < min(lc9)].index, inplace = True)

                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                        np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag)]

                indexes = np.array([1,2,3,4,5,6,7,8,9])

            elif len(lc11) <= 1:
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df2 = pd.read_table(path, sep="\s+", names=column_names)
                df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 > max(lc2)].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 < min(lc2)].index, inplace = True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 > max(lc3)].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 < min(lc3)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 > max(lc8)].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                df9 = pd.read_table(path, sep="\s+", names=column_names)
                df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 > max(lc9)].index, inplace = True)    
                df9.drop(df9[df9['JD']+ 2450000 < min(lc9)].index, inplace = True)

                df10 = pd.read_table(path, sep="\s+", names=column_names)
                df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
                df10.drop(df10[df10['JD']+ 2450000 < min(lc10)].index, inplace = True)


                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                        np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]

                indexes = np.array([1,2,3,4,5,6,7,8,9,10])

            elif len(lc12) <= 1:
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df2 = pd.read_table(path, sep="\s+", names=column_names)
                df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 > max(lc2)].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 < min(lc2)].index, inplace = True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 > max(lc3)].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 < min(lc3)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 > max(lc8)].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                df9 = pd.read_table(path, sep="\s+", names=column_names)
                df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 > max(lc9)].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 < min(lc9)].index, inplace = True)

                df10 = pd.read_table(path, sep="\s+", names=column_names)
                df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
                df10.drop(df10[df10['JD']+ 2450000 > max(lc10)].index, inplace = True)
                df10.drop(df10[df10['JD']+ 2450000 < min(lc10)].index, inplace = True)

                df11 = pd.read_table(path, sep="\s+", names=column_names)
                df11.drop(df11[df11['good/bad'] < 1].index, inplace = True)
                df11.drop(df11[df11['JD']+ 2450000 < min(lc11)].index, inplace = True)


                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                        np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag), np.median(df11.mag)]

                indexes = np.array([1,2,3,4,5,6,7,8,9,10,11])


            elif len(lc12) > 1:
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['JD']+ 2450000 > max(lc1)].index, inplace = True)

                df2 = pd.read_table(path, sep="\s+", names=column_names)
                df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 > max(lc2)].index, inplace = True)
                df2.drop(df2[df2['JD']+ 2450000 < min(lc2)].index, inplace = True)

                df3 = pd.read_table(path, sep="\s+", names=column_names)
                df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 > max(lc3)].index, inplace = True)
                df3.drop(df3[df3['JD']+ 2450000 < min(lc3)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['JD']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['JD']+ 2450000 < min(lc5)].index, inplace = True)
                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['JD']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 > max(lc7)].index, inplace = True)
                df7.drop(df7[df7['JD']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 > max(lc8)].index, inplace = True)
                df8.drop(df8[df8['JD']+ 2450000 < min(lc8)].index, inplace = True)

                df9 = pd.read_table(path, sep="\s+", names=column_names)
                df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 > max(lc9)].index, inplace = True)
                df9.drop(df9[df9['JD']+ 2450000 < min(lc9)].index, inplace = True)

                df10 = pd.read_table(path, sep="\s+", names=column_names)
                df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
                df10.drop(df10[df10['JD']+ 2450000 > max(lc10)].index, inplace = True)
                df10.drop(df10[df10['JD']+ 2450000 < min(lc10)].index, inplace = True)

                df11 = pd.read_table(path, sep="\s+", names=column_names)
                df11.drop(df11[df11['good/bad'] < 1].index, inplace = True)
                df11.drop(df11[df11['JD']+ 2450000 > max(lc11)].index, inplace = True)
                df11.drop(df11[df11['JD']+ 2450000 < min(lc11)].index, inplace = True)

                df12 = pd.read_table(path, sep="\s+", names=column_names)
                df12.drop(df12[df12['good/bad'] < 1].index, inplace = True)
                df12.drop(df12[df12['JD']+ 2450000 < min(lc12)].index, inplace = True)


                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                        np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag), np.median(df11.mag), np.median(df12.mag)]

                indexes = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
#slope add back in!

            #breakpoint()

            coeffs1 = np.polyfit(indexes,meds,1)
            slope = coeffs1[0]

            #print(indexes, meds, slope)
            #breakpoint()

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

            c1 = coeffs[-1]
            c2 = coeffs[-2]

            te = -c2/(2*quadratic_slope)
            me = c1-(c2**2)/(4*quadratic_slope)

            m0 = c1 + c2*indexes[0] + quadratic_slope*indexes[0]**2
            m1 = c1 + c2*indexes[-1] + quadratic_slope*indexes[-1]**2

            if te > indexes[0] and te < indexes[-1]:
                m1m0 = np.abs(m1-m0)
                m1me = np.abs(m1-me)
                m0me = np.abs(m0-me)

                mags = [m1m0,m1me,m0me]

                diff = max(mags)

            else:
                diff = np.abs(m1-m0)

            #min_i = min(poly_function(indexes))
            #max_i = max(poly_function(indexes))

            #diff = np.abs(min_i-max_i)

            
             #put all together
            _Target.append(Target)
            Median.append(lcMED)
            Mag.append(p_mag)
            Median_err.append(lcMED_err)
            Dispersion.append(lcDISP)
            Slope.append(slope)
            Quad_Slope.append(quadratic_slope)
            Coeff1.append(c1)
            Coeff2.append(c2)
            Diff.append(diff)


    tbl_LTvar = Table([_Target,Mag, Median, Median_err, Dispersion, Slope,Quad_Slope,Coeff1,Coeff2,Diff],
    names=('ASAS-SN ID','Pstarrs gmag', 'Median', 'Median_err','Dispersion','Slope','Quad Slope','coeff1','coeff2','max diff'),
    meta={'name': 'LTvar'})



#update column names

    csv_file_path = '13-13.5/new/' + x + '.csv'

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(tbl_LTvar)


    #print(max(lens))
    print('Ending ' + x)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' +str(executionTime))
