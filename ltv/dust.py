import mwdust
#from astropy.coordinates import SkyCoord
#import astropy.units as u
import numpy as np
import csv
from tqdm import tqdm

# Initialize DustMap3D with Combined19 data
dust_map = mwdust.Combined19(filter='2MASS H')

#l = np.array([44.115299331810])

#b = np.array([18.796499250])

#dist = np.array([0.11438899])

input_filename = 'twins_ext.csv'
output_filename = input_filename

rows = []
valid_rows = []

with open(input_filename, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    for row in tqdm(reader, desc='Processing rows'):
        try:
            if len(row) < 3:
                raise ValueError("Row does not have enough columns")
            l_val, b_val, dist_val = map(float, row[-3:])
            if dist_val <= 0:
                raise ValueError('Distance must be positive')
            valid_rows.append([l_val,b_val,dist_val])
            rows.append(row)
        except ValueError:
            print(f"Skipping invalid row: {row}")

if valid_rows:
    l = np.array([float(row[0]) for row in valid_rows])
    b = np.array([float(row[1]) for row in valid_rows])
    dist = np.array([float(row[2]) for row in valid_rows])

    ebv = dust_map(l, b, dist)
else:
    ebv = []

with open(output_filename, 'w', newline='') as file:
    writer = csv.writer(file)

    if header:
        writer.writerow(header + ['ebv'])

    for original_row, ebv_value in zip(rows,ebv):
        writer.writerow(original_row + [ebv_value])


print(f"EBV values added to {output_filename}")





#print(ebv)

