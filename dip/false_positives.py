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

def filterBrightNearbyStar(light_curve):
    # Implement filtering logic for bright nearby star contamination
    '''
    
    '''
    pass

def filterBadPointCluster(light_curve):
    # Implement filtering logic for bad point clusters (outliers)
    '''
    It seems to me that the only way the BPC case differs from a dipper, e.g., JO73924-272916 (New) is that only one camera dips. For a dipper, all cameras dip (COUNTEREXAMPLE: J205245-713514 (New), the hot pink camera didn't dip here, so this isn't a hard rule? Ask Chris.)
    '''
    pass

def filterCameraOffset(light_curve):
    # Implement filtering logic for camera offset (systematic calibration jumps)
    '''
    Clustering of the medians of the light curves about a median -- maybe calibrate the medians of each of the cameras, then calculate the standard deviation of the medians, then set a limit? Or would this also miss out on some cases, or would it also clean some fine curves? This failure mode that only occurs near the poles. A source almost always appears in only one field for a given camera, so the light curve intercalibration procedure assumes a single offset for each camera. However, very close to the poles, field rotations can allow a source to appear in two fields for the same camera, leading to problems if the fields need different offsets.
    '''
    pass

def filterRCBV(light_curve):
    # Implement filtering logic for R Coronae Borealis variables (real variable but wrong class)
    '''

    '''
    pass

def filterCB(light_curve):
    # Implement filtering logic for contact binaries (short-period eclipsing)
    '''
    When you phase-fold with a Lomb-Scargle periodogram, there's clearly a repeating pattern.
    '''
    pass

def filterSRV(light_curve):
    # Implement filtering logic for semi-regular variables (pulsating giants)
    '''

    '''
    pass

def filterTTauriYSO(light_curve):
    # Implement filtering logic for T Tauri YSOs (periodic/irregular young star variability)
    '''

    '''
    pass
