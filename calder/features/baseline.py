import numpy as np
import pandas as pd
from utils import read_lightcurve

# need to get the vsx catalog
# get the asassn id of each surviving lc after crossmatchign with vsx
# load the asassn_id's into a a list
# then for all the asassn_id's, append the v and g mag lightcurves to them
# probably need to rewrite read_lightcurve to account for this

#df_v, df_g = read_lightcurve(asassn_id, path)


def rolling_time_median(t, y, days=30., min_points=10):
    t = np.asarray(t)
    y = np.asarray(y)

    y_med = np.full_like(y, np.nan, dtype=np.float64)

    for i, ti in enumerate(t):
        mask = (t >= ti - days/2) & (t <= ti + days/2)
        if np.sum(mask) >= min_points:
            y_med[i] = np.median(y[mask])
    
    # if there are not enough points in the window, fill with global median
    global_med = np.nanmedian(y[mask])

    return np.where(np.isnan(y_med), global_med, y_med)


def per_camera_baseline(df, days=30., min_points=10):
    
    pieces = []

    for camera, camera_subdf in df.groupby("camera#"):
        t = camera_subdf["JD"].to_numpy()
        y = camera_subdf["mag"].to_numpy()
        y_med = rolling_time_median(t, y, days=days, min_points=min_points)
        df.loc[camera_subdf.index, "baseline"] = y_med

        r = np.median(y - y_med)
        
        mad = 1.4826 * np.median(np.abs((r - np.median(r))))

        # calculate robust stddev including photometric error
        r_std = np.sqrt(mad**2 + np.median(camera_subdf["error"].to_numpy())**2 )
        r_std = np.clip(r_std, 1e-6, None)

        camera_subdf["z"] = r / r_std

        pieces.append(camera_subdf)

    return pd.concat(pieces, ignore_index=True)