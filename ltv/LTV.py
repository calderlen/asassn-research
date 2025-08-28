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
import argparse

# Flags set by cli arguments
def parse_args():
    parser = argparse.ArgumentParser(prog='LTvar.py')

    parser.add_argument("--root", default="/data/poohbah/1/assassin/rowan.90/lcsv2/", type=str, help="Root folder containing mag-bin subdirectories")
    parser.add_argument("--mag-bin", default="13_13.5", type=str, help="Magnitude bin subdirectory, e.g. 13_13.5")
    parser.add_argument("--output", default="LTvar13_13.5.csv", type=str, help="Output CSV filename")
    parser.add_argument("--dir-start", type=int, default=0, help="First lcXX_cal index (inclusive)")
    parser.add_argument("--dir-end", type=int, default=30, help="Last lcXX_cal index (inclusive)")
    return parser.parse_args()

args = parse_args()
ROOT = args.root
MAG_BIN = args.mag_bin
OUTPUT = args.output
DIR_START = args.dir_start
DIR_END = args.dir_end

cuts = pd.read_csv('Cuts.csv')
cut_list = cuts['ASAS-SN ID'].tolist()

columns = ["jd", "mag", 'error', 'good/bad', 'camera', 'v/g?', 'saturated/unsaturated', 'camera,field']

directories = list(map(str, range(DIR_START, DIR_END + 1)))

# Light curve columns

ltv_path = f"LTvar{MAG_BIN.replace('_','-')}.csv"
ltv = Table.read(ltv_path)

asassn_id = np.array(ltv['ASAS-SN ID']).tolist()
slope = np.array(ltv['Slope']).tolist()
quad_slope = np.array(ltv['Quad Slope']).tolist()
c1 = np.array(ltv['coeff1']).tolist()
c2 = np.array(ltv['coeff2']).tolist()
min_g = np.array(ltv['min_date']).tolist()

for x in directories:
    start_time = time.time()
    print(f'Starting{MAG_BIN}' + x + ' directory')

    ID = pd.read_table(f'{ROOT}{MAG_BIN}/index' + x + '.csv', sep=r'\,|\t', engine='python')
    directory = f'{ROOT}{MAG_BIN}/lc' + x + '_cal/'

    files = [f for f in os.listdir(directory) if f.endswith('.dat')]
    
    for file in files:
        path = os.path.join(directory, file)
        target = [file.split('.')[0]]
        Target = [int(i) for i in target]
        Target = Target[0]

        if Target in cut_list:
            print(f'Skipping {Target} as in Cuts.csv')
            continue

        ra = np.where(ID['asas_sn_id'] == Target)
        
        df = pd.read_table(path, sep="\s+", names=columns)
        df = df[df['good/bad'] == 1] # ltv: good=1; dippers, may need both
        df = df[df['v/g?'] == 0]  # g-band: 0, V-band: 1

        if len(df) > 0: min_g_date = min(df['jd'])
        else: min_g_date = np.nan

        if Target == 17181160895:
            df.drop(df[df['good/bad'] < 1].index, inplace = True)
            df.drop(df[df['jd'] < 2.458e+06].index, inplace = True)
            
        RA = ID['ra_deg'].iloc[ra]

        #pstarr mag
        p_mag = float(np.array(ID['pstarrs_g_mag'].iloc[ra]))
        
        lc_median = np.median(df['mag'])
        lc_mad = mad_std(df['mag'])  # median absolute deviation, robust std dev
        lc_dispersion = np.ptp(df['mag'])  # peak to peak
        
        # computing seasonal medians
        indices = np.arange(0, DIR_END + 1) # Sydney had this as not inclusive of last number, why?
        dspring = 2460023.5

        Mid = [] 

        for n in indices:
            date1 = dspring + 365.25*(RA-12.0)/24.0 #target overhead at midnight at date1
            date2 = date1 +  365.25/2.0 +365.25 #same RA as the sun at date2
            mid = date2 - n*365.25 #seasonal gaps (where sun is blocking target)
            Mid.append(mid)

        mid = np.array(Mid)


        if len(df.JD) == 0:
            continue
        mid = [x for x in mid if x < max(df.JD) and x > min(df.JD)]

