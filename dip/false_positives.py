'''
Todo
    - Understand light curve structure so when you draft up these filters you have the right syntax
    - Implement filtering logic for the following false positives
        - Bright nearby star contamination
        - Bad point cluster (outliers)
        - Camera offset (systematic calibration jumps)
        - R Coronae Borealis variable (real variable but wrong class)
        - Contact binary (short-period eclipsing)
        - Semi-regular variable (pulsating giant)
        - T Tauri YSO (periodic/irregular young star variability)



    - Test the filtering logic against Brayden's light curves


'''

import numpy as np
import pandas as pd
import scipy
import math
import os

from tqdm import tqdm

from astropy.timeseries import LombScargle as ls

#from astropy.io import ascii
#from astropy.io import fits
#from astropy.coordinates import SkyCoord
#from astropy.time import Time
#from astropy import constants as const
#from astropy.timeseries import LombScargle as ls
#from astropy import units as u
#from astropy.timeseries import BoxLeastSquares

#import matplotlib.pyplot as pl

from dipper_processing import find_peak, custom_id, year_to_jd, jd_to_year, plot_zoom, plot_multiband


def read_lightcurve(asassn_id, path):

    # different processing for .dat and .csv files

    if os.path.exists(f"{path}/{asassn_id}.dat"):

        fname = os.path.join(path, f"{asassn_id}.dat")

        df_v = pd.DataFrame()
        df_g = pd.DataFrame()

        fdata = pd.read_fwf(fname, header=None)
        fdata.columns = ["JD", "Mag", "Mag_err", "Quality", "Cam_number", "Phot_filter", "Camera"]

        df_v = fdata.loc[fdata["Phot_filter"] == 1].reset_index(drop=True)
        df_g = fdata.loc[fdata["Phot_filter"] == 0].reset_index(drop=True)      

        df_v['Mag'].astype(float)
        df_v['JD'].astype(float)

        df_g['Mag'].astype(float)
        df_g['JD'].astype(float)

    elif os.path.exists(f"{path}/{asassn_id}.csv"):

        fname = os.path.join(path, f"{asassn_id}.csv")

        df = pd.read_csv(fname)

        df['Mag'] = pd.to_numeric(df['Mag'], errors='coerce')
        df = df.dropna()

        df['Mag'].astype(float)
        df['JD'] = df.HJD.astype(float)

        df_g = df.loc[df["Filter"] == 'g'].reset_index(drop=True)
        df_v = df.loc[df["Filter"] == 'V'].reset_index(drop=True)

    return df_v, df_g

def naive_dip_detection(df, prominence=0.17, distance=25, height=0.3, width=2):

	df['Mag'] = [float(i) for i in df['Mag']]
	df['JD'] = [float(i) for i in df['JD']]

    mag = df['Mag']
    jd = df['JD']

    mag_mean = sum(mag)/len(mag)
    df_mag_avg = [i - mag_mean for i in mag]
    
    peaks = scipy.signal.find_peaks(df_mag_avg, prominence=prominence, distance=distance, height=height, width=width) 

    peak = peaks[0]
	prop = peaks[1]
	
    length = len(peak)
	
    peak = [int(i) for i in peak]
	peak = pd.Series(peak)

    return peak, mag_mean, length


def filter_BNS(light_curve):
    # Implement filtering logic for bright nearby star contamination
    '''
    THIS IS ONE OF THE MAIN SOURCES OF FALSE POSITIVES AND SHOULD BE FOCUSED ON. The light curve will have a dip that is not real, but caused by a nearby bright star. This can be identified by looking for correlated dips in nearby stars, or by checking for known bright stars in the vicinity of the target star. One way to do this is to cross-match the target star with a catalog of bright stars and check for proximity. If a bright star is found within a certain radius, the dip can be flagged as a potential false positive.

    function components:
    - cross-match with catalog
        - astroquery with Gaia DR3, VizieR, TAP?
            - astroquery.gaia: Gaia DR3/DR2; primary check
            - astroquery.vizier: APASS, UCAC4, NOMAD, 2MASS, etc.; secondary check
            - astroquery.xmatch: direct access to CDS X-Match service for bulk cross-matches
            - astroquery.simbad: get star classifications/types, alternative IDs, basic mags; useful for checking if a nearby star is known to be variable
        - ASAS-SN neighbor light curves (internal cross-match?); build a neighbor query; access ASAS-SN Sky Patrol databases
    - cross-matching conditions
        - identify bright stars within angular radius of 15"-30", considering that the ASAS-SN pixel scale is 8" with PSF wings
        - choose what a "bright" star should be
    '''



    pass

def filter_BPC(light_curve):
    # Implement filtering logic for bad point clusters (outliers)
    '''
    It seems to me that the only way the BPC case differs from a dipper, e.g., JO73924-272916 (New) is that only one camera dips. For a dipper, all cameras dip (COUNTEREXAMPLE: J205245-713514 (New), the hot pink camera didn't dip here, so this isn't a hard rule? Ask Chris.)
    '''
    pass

def filter_CO(light_curve):
    # Implement filtering logic for camera offset (systematic calibration jumps)
    '''
    Clustering of the medians of the light curves about a median -- maybe calibrate the medians of each of the cameras, then calculate the standard deviation of the medians, then set a limit? Or would this also miss out on some cases, or would it also clean some fine curves? This failure mode that only occurs near the poles. A source almost always appears in only one field for a given camera, so the light curve intercalibration procedure assumes a single offset for each camera. However, very close to the poles, field rotations can allow a source to appear in two fields for the same camera, leading to problems if the fields need different offsets.
    '''
    pass

def filter_RCBv(light_curve):
    # Implement filtering logic for R Coronae Borealis variables (real variable but wrong class)
    '''
    In the example RCBv light curve, the time between dips is 650 days, and the dips are >0.3 mag. According to wikipedia, RBCvs can vary in luminosity in 2 modes: (1) irregularly unpredictable sudden fading by 1-9 mag, (2) low amplitude puslation of a few 1/10s of a magnitude. This raises the question: how do we distinguish between RBCvs and dippers: I think we can set a condition looks for these low-amplitude puslations iff the dip is >0.3 mag. Importantly, there will be increases in flux from the median, distinguishing it from dippers.
    '''
    pass

def filter_CB(light_curve):
    # Implement filtering logic for contact binaries (short-period eclipsing)
    '''
    When you phase-fold with a Lomb-Scargle periodogram, there's clearly a repeating pattern with a short <1 day period, double humped, minima nearly equal, and small color change.
    '''
    pass

def filter_SRv(light_curve):
    # Implement filtering logic for semi-regular variables (pulsating giants)
    '''
    Long periods (10s-100s of days), quasi-periodic, multi-periodic, larger ampltiude than noise, very red stars. I guess that we can check for the >0.3mag dip and then run a L-S periodogram and check for multiple sharp peaks (for multi-periodicity) and broad, smeared peaks (for quasi-periodicity). The quasi-peridodicity may require a stochastic or time-variable model, such as a Gaussian Process with a quasi-periodic kernel, wavelet analysis, or autoregressive model. Importantly, there will be increases in flux from the median, distinguishing it from dippers.
    '''
    pass

def filter_TTauriYSO(light_curve):
    # Implement filtering logic for T Tauri YSOs (periodic/irregular young star variability)
    '''
    Deep dimming events but are recurring, either quasi-periodically, or irregularly. In the example T Tauri light curve, the dip length is 2 days and is phase-folded with period of 8.1 days. 
    '''
    pass

def detect_dippers(light_curve):
    # Main function to detect dippers and apply false positive filters
    '''
    Apply naive dip detection first, then sequentially apply each false positive filter. If a light curve passes all filters, classify it as a dipper.
    
    single-dip eclipsing binaries
    single-dip long-period eclipsing binaries
    multi-dip eclipsing binaries
    multi-dip long-period eclipsing binaries
    dippers

    
    '''


    pass