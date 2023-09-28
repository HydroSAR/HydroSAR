import numpy as np
import xarray as xr
import pandas as pd
import random
import scipy as sci
import matplotlib
import matplotlib.pyplot as plt

import os
import logging
from pathlib import Path

from eofs.xarray import Eof
from geoglows import streamflow
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FactorAnalysis
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def unrot_eof(stack: xr.DataArray, variance_threshold: float = 0.8, n_modes: int = -1) -> xr.Dataset:
    """Function to perform rotated empirical othogonal function (eof) on a spatial timeseries

    args:
        stack (xr.DataArray): DataArray of spatial temporal values with coord order of (t,y,x)
        variance_threshold(float, optional): optional fall back value to select number of eof
            modes to use. Only used if n_modes is less than 1. default = 0.727
        n_modes (int, optional): number of eof modes to use. default = 4

    returns:
        xr.Dataset: rotated eof dataset with spatial modes, temporal modes, and mean values
            as variables

    """
    # extract out some dimension shape information
    shape3d = stack.shape
    spatial_shape = shape3d[1:]
    shape2d = (shape3d[0],np.prod(spatial_shape))

    # flatten the data from [t,y,x] to [t,...]
    da_flat = xr.DataArray(
        stack.values.reshape(shape2d),
        coords = [stack.time,np.arange(shape2d[1])],
        dims=['time','space']
    )
    #logger.debug(da_flat)
        
    ## find the temporal mean for each pixel
    center = da_flat.mean(dim='time')
    
    centered = da_flat - center
               
    # get an eof solver object
    # explicitly set center to false since data is already
    #solver = Eof(centered,center=False)
    solver = Eof(centered,center=False)

    # check if the n_modes keyword is set to a realistic value
    # if not get n_modes based on variance explained
    if n_modes < 0:
        n_modes = int((solver.varianceFraction().cumsum() < variance_threshold).sum())

    # calculate to spatial eof values
    eof_components = solver.eofs(neofs=n_modes).transpose()
    # get the indices where the eof is valid data
    non_masked_idx = np.where(np.logical_not(np.isnan(eof_components[:,0])))[0]

    # # waiting for release of sklean version >= 0.24
    # # until then have a placeholder function to do the rotation
    # fa = FactorAnalysis(n_components=n_modes, rotation="varimax")
    # rotated[non_masked_idx,:] = fa.fit_transform(eof_components[non_masked_idx,:])

    # get eof with valid data
    eof_components = np.asarray(eof_components)

    
    # project the original time series data on the rotated eofs
    projected_pcs = np.dot(centered[:,non_masked_idx], eof_components[non_masked_idx,:])

    # reshape the rotated eofs to a 3d array of [y,x,c]
    spatial = eof_components.reshape(spatial_shape+(n_modes,))
    
    
    # structure the spatial and temporal eof components in a Dataset
    eof_ds = xr.Dataset(
        {
            "spatial_modes": (["lat","lon","mode"],spatial[:,:,-1]),
            "temporal_modes":(["time","mode"],projected_pcs),
            "center": (["lat","lon"],center.values.reshape(spatial_shape))
        },
        coords = {
            "lon":(["lon"],stack.lon),
            "lat":(["lat"],stack.lat),
            "time":stack.time,
            "mode": np.arange(n_modes)+1
        }
    )

    return eof_ds
    
    
def sig_eof_mode(stack: xr.DataArray, option: int=0, monte_carlo_iter: int=1000) -> int:
    """
       Significant test upon the EOF analysis results.
       
       Option:
       1. Monte Carlo Test
       2. 
       3. 
       
    """          
    
    fontdict={
       'weight':'bold',
       'size':14    
    }
    matplotlib.rc('font',**fontdict)
    
    # extract out some dimension shape information
    shape3d = stack.shape
    spatial_shape = shape3d[1:]
    shape2d = (shape3d[0],np.prod(spatial_shape))

    # flatten the data from [t,y,x] to [t,...]
    da_flat = xr.DataArray(
        stack.values.reshape(shape2d),
        coords = [stack.time,np.arange(shape2d[1])],
        dims=['time','space']
    )
    #logger.debug(da_flat)
        
    ## find the temporal mean for each pixel
    center = da_flat.mean(dim='time')
    
    centered = da_flat - center
    centered = centered.dropna(dim='space')
    
    pix_num = centered.sizes['space']
    obs_num = centered.sizes['time']
    
    # Get eigenvalue from real data   
    solver_real = Eof(centered,center=False)
    real_lamb = np.array(solver_real.eigenvalues(neigs = obs_num))
       
    sig_mode = 0
    if option==1:
       print('Perform Monte Carlo significant test')
       
       rng = np.random.default_rng()      
       
       #Xp: space x time
       #Xp[,j][sample(nrow(Xp))]
       #Means for j-th column (with nrow of data), shuffule with size of nrow.
       
       mc_lamb = np.full((obs_num,monte_carlo_iter),np.nan)
             
       random.seed(1)
       for i in range(monte_carlo_iter):
         #print('Iteration: ',str(i+1))
         
         # ----- Randomize observation (space-wise randomization)
         # ----- with for loop -----
         #obs_temp = np.full((obs_num,pix_num), np.nan)         
         #for j in range(obs_num): # time-wise
         #  print('Randomization',str(j))
         #  obs_temp[j,:] = random.sample(list(centered[j,:].values), pix_num)         
         
         # ----- Some vectorization -----
         #idx = np.random.rand(obs_num, pix_num).argsort(axis=1)
         #obs_temp = np.take_along_axis(centered.values,idx,axis=1)
         
         # ----- Some vectorization 1 -----
         obs_temp = rng.permuted(centered.values, axis=1)
         
         obs_temp = xr.DataArray(
              obs_temp,
              coords=[centered.time,np.arange(pix_num)],
              dims=['time','space']
         )        
         
         #obs_temp = obs_temp.chunk({'time':10000000000,'space':10000000000})         
         solver_mc = Eof(obs_temp,center=False)
         #print(solver_mc.eigenvalues(neigs = obs_num))
         mc_lamb[:,i] = solver_mc.eigenvalues(neigs = obs_num)
          
       mc_lamb = np.transpose(mc_lamb)
       mean_mc_lamb = np.mean(mc_lamb,axis=0)
       std_mc_lamb = np.std(mc_lamb,axis=0)
       
       gt = np.greater(real_lamb, mean_mc_lamb).astype(int)
       gt_rev = gt[::-1]
       sig_mode = len(gt_rev) - np.argmax(gt_rev) - 1
       
       plt.title('Scree Plot',fontdict=fontdict)
       plt.plot(np.arange(obs_num)+1, real_lamb, marker='+')
       plt.errorbar(np.arange(obs_num)+1, mean_mc_lamb, std_mc_lamb)
       plt.legend(['Real Data','MonteCarlo Sim. Data'])
       plt.xlabel('Mode',fontdict=fontdict)
       plt.ylabel('Eigenvalue',fontdict=fontdict)
       plt.show()       
              
    return sig_mode
                                                           
    
def modes_rtpc2hydro(reof_stack: xr.Dataset, hydro_stack: xr.DataArray, spearman_r_threshold: float=0.6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:   
    """
       Calculate Spearman's R (monotonic correlation) between RTPC and hydrological data. This help determine which 
       modes are water-related. The threshold of Spearman's R is defined as >=0.6 by default based on the interpretation
       of 

    """    
    # get number of mode     
    mode_num = reof_stack.sizes['mode']    
    # get number of hydrological data sites
    site_num = hydro_stack.sizes['site']    
    
    mono_r = np.zeros((site_num, mode_num))
    mono_p = np.zeros((site_num, mode_num))
    linr_r = np.zeros((site_num, mode_num))
    linr_p = np.zeros((site_num, mode_num))
    
    
    for ct_mode in range(mode_num):
      # get mode of tpc
      tpc = reof_stack.temporal_modes.sel(mode=int(ct_mode+1))    
      
      for ct_site in range(site_num):
        # get hydrological data coincident with satellite images (TPC) 
        if hydro_stack[ct_site].sizes['time'] != reof_stack.sizes['time']:        
          hydro_site = match_dates(hydro_stack[ct_site], tpc)
        else:
          hydro_site = hydro_stack[ct_site]
    
        hydro_zscore = sci.stats.zscore(hydro_site)                
        indx_good_hydro = (np.abs(hydro_zscore) <= 3)
        
        #good_hydro = hydro_site
        #good_tpc = tpc
        
        good_hydro = hydro_site[indx_good_hydro]
        good_tpc = tpc[indx_good_hydro]    

        #plt.scatter(good_tpc, good_hydro)
        #plt.show()        
    
        # calculate monotonic correlation between hydrological data and tpc
        # as a reference to judge their connection
        mono_r[ct_site, ct_mode] = sci.stats.spearmanr(good_tpc, good_hydro)[0]
        mono_p[ct_site, ct_mode] = sci.stats.spearmanr(good_tpc, good_hydro)[1]
        linr_r[ct_site, ct_mode] = sci.stats.pearsonr(good_tpc, good_hydro)[0]
        linr_p[ct_site, ct_mode] = sci.stats.pearsonr(good_tpc, good_hydro)[1]
        
    indx_site_mode = ((np.abs(mono_r) >= spearman_r_threshold).astype(int)).nonzero()
    
    
    return mono_r, mono_p, linr_r, linr_p, indx_site_mode
        
        
def wrap_streamflow(lats: list, lons: list) -> Tuple[xr.DataArray, list]:
    """Function to get and wrap up histroical streamflow data from the GeoGLOWS server
    at different geographic coordinates

    args:
        lats (list): latitude values where to get streamflow data
        lons (list): longitude values where to get streamflow data

    returns:
        xr.DataArray: DataArray object of streamflow with datetime coordinates
    """
    site_num = len(lats)
    reaches = []
    for ct_site in range(site_num):
      if ct_site==0:
        q, reach_id = get_streamflow(lats[ct_site], lons[ct_site])
        q["time"] = q["time"].dt.strftime("%Y-%m-%d")
        q.expand_dims(dim='site')
      else:
        q1, reach_id = get_streamflow(lats[ct_site], lons[ct_site])
        q1["time"] = q1["time"].dt.strftime("%Y-%m-%d")        
        q = xr.concat( (q, q1),dim='site' ) 
      reaches.append(reach_id)
      
    # return the series as a xr.DataArray
    return q, reaches      


def reof(stack: xr.DataArray, variance_threshold: float = 0.727, n_modes: int = 100) -> xr.Dataset:
    """Function to perform rotated empirical othogonal function (eof) on a spatial timeseries

    args:
        stack (xr.DataArray): DataArray of spatial temporal values with coord order of (t,y,x)
        variance_threshold(float, optional): optional fall back value to select number of eof
            modes to use. Only used if n_modes is less than 1. default = 0.727
        n_modes (int, optional): number of eof modes to use. default = 4

    returns:
        xr.Dataset: rotated eof dataset with spatial modes, temporal modes, and mean values
            as variables

    """

    # extract out some dimension shape information
    shape3d = stack.shape
    spatial_shape = shape3d[1:]
    shape2d = (shape3d[0],np.prod(spatial_shape))

    # flatten the data from [t,y,x] to [t,...]
    da_flat = xr.DataArray(
        stack.values.reshape(shape2d),
        coords = [stack.time,np.arange(shape2d[1])],
        dims=['time','space']
    )
    #logger.debug(da_flat)
        
    ## find the temporal mean for each pixel
    center = da_flat.mean(dim='time')
    
    centered = da_flat - center
               
    # get an eof solver object
    # explicitly set center to false since data is already
    #solver = Eof(centered,center=False)
    solver = Eof(centered,center=False)

    # check if the n_modes keyword is set to a realistic value
    # if not get n_modes based on variance explained
    if n_modes < 0:
        n_modes = int((solver.varianceFraction().cumsum() < variance_threshold).sum())
    
    # get total and cumulated total variance fractions of eof (up to the max. retained mode)
    total_eof_var_frac = solver.varianceFraction(neigs=n_modes).cumsum()
    
    # Determine how many modes we need to represent 90% of the dataset's variance
    ind_90 = np.where(np.abs(total_eof_var_frac-0.9) == np.min(np.abs(total_eof_var_frac-0.9)))[0][0]
    
    # Plot the amount of variance depending on each mode
    plt.figure()
    plt.plot(total_eof_var_frac*100)
    plt.xlabel('amount of modes')
    plt.ylabel('dataset variance explained by the modes [%]')
    plt.axhline(y = 90, color = 'r', linestyle = '-')
    plt.axvline(x = ind_90, color = 'red')
    plt.title(f"90% of the variance is explained by the first {ind_90} modes") 

    # Set the amount of modes to that 90% cutoff
    n_modes = ind_90
    
    # Crop the total eof variance fraction
    total_eof_var_frac = total_eof_var_frac[:n_modes] 

    # calculate to spatial eof values
    eof_components = solver.eofs(neofs=100).transpose()

    # get the indices where the eof is valid data
    non_masked_idx = np.where(np.logical_not(np.isnan(eof_components[:,0])))[0]

    # create a "blank" array to set roated values to
    rotated = eof_components.copy()

    # # waiting for release of sklean version >= 0.24
    # # until then have a placeholder function to do the rotation
    fa = FactorAnalysis(n_components=100, rotation="varimax")
    rotated[non_masked_idx,:] = fa.fit_transform(eof_components[non_masked_idx,:])
    rotated = rotated.values[:,:n_modes] # We crop out the last mode that is usually filled with 0s

    # project the original time series data on the rotated eofs
    projected_pcs = np.dot(centered[:,non_masked_idx], rotated[non_masked_idx,:])
    
    # get variance of each rotated mode
    rot_var = np.var(projected_pcs, axis=0)
    
    # get variance of all rotated modes
    total_rot_var = rot_var.cumsum()
    
    # get variance fraction of each rotated mode
    rot_var_frac = ((rot_var/total_rot_var)*total_eof_var_frac)*100
    
    # reshape the rotated eofs to a 3d array of [y,x,c]
    spatial_rotated = rotated.reshape(spatial_shape+(n_modes,))

    # sort modes based on variance fraction of REOF
    indx_rot_var_frac_sort = np.expand_dims(((np.argsort(-1*rot_var_frac)).data), axis=0)        
    projected_pcs = np.take_along_axis(projected_pcs,indx_rot_var_frac_sort,axis=1)
    
    indx_rot_var_frac_sort = np.expand_dims(indx_rot_var_frac_sort, axis=0)
    spatial_rotated = np.take_along_axis(spatial_rotated,indx_rot_var_frac_sort,axis=2)

    # structure the spatial and temporal reof components in a Dataset. We squeeze out the last n_mode because it is empty
    reof_ds = xr.Dataset(
        {
            "spatial_modes": (["lat","lon","mode"],spatial_rotated),
            "temporal_modes":(["time","mode"],projected_pcs),
            "center": (["lat","lon"],center.values.reshape(spatial_shape))
        },
        coords = {
            "lon":(["lon"],stack.lon.values),
            "lat":(["lat"],stack.lat.values),
            "time":stack.time.values,
            "mode": np.arange(n_modes)
        }
    )


    return reof_ds


def _ortho_rotation(components: np.array, method: str = 'varimax', tol: float = 1e-6, max_iter: int = 100) -> np.array:
    """Return rotated components. Temp function"""
    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    var = 0

    for _ in range(max_iter):
        comp_rot = np.dot(components, rotation_matrix)
        if method == "varimax":
            tmp = comp_rot * np.transpose((comp_rot ** 2).sum(axis=0) / nrow)
        elif method == "quartimax":
            tmp = 0
        u, s, v = np.linalg.svd(
            np.dot(components.T, comp_rot ** 3 - tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new

    return np.dot(components, rotation_matrix)




def get_streamflow(lat: float, lon: float) -> Tuple[xr.DataArray, int]:
    """Function to get histroical streamflow data from the GeoGLOWS server
    based on geographic coordinates

    args:
        lat (float): latitude value where to get streamflow data
        lon (float): longitude value where to get streamflow data

    returns:
        xr.DataArray: DataArray object of streamflow with datetime coordinates
    """
    # ??? pass lat lon or do it by basin ???
    reach = streamflow.latlon_to_reach(lat,lon)
    # send request for the streamflow data
    q = streamflow.historic_simulation(reach['reach_id'])

    # rename column name to something not as verbose as 'streamflow_m^3/s'
    q.columns = ["discharge"]

    # rename index and drop the timezone value
    q.index.name = "time"
    q.index = q.index.tz_localize(None)

    # return the series as a xr.DataArray
    return q.discharge.to_xarray(), reach['reach_id']


def match_dates(original: xr.DataArray, matching: xr.DataArray) -> xr.DataArray:
    """Helper function to filter a DataArray from that match the data values of another.
    Expects that each xarray object has a dimesion named 'time'

    args:
        original (xr.DataArray): original DataArray with time dimension to select from
        matching (xr.DataArray): DataArray with time dimension to compare against

    returns:
        xr.DataArray: DataArray with values that have been temporally matched
    """

    # return the DataArray with only rows that match dates
    return original.where(original.time.isin(matching.time),drop=True)

def fits_to_files(fit_dict: dict,out_dir: str):
    """Procedure to save coeffient arrays stored in dictionary output from `find_fits()` to npy files

    args:
        fit_dict (dict): output from function `find_fits()`
        out_dir (str): directory to save coeffients to
    """

    out_dir = Path(out_dir)

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    for k,v in fit_dict.items():
        if k.endswith("coeffs"):
            components = k.split("_")
            name_stem = f"poly_{k.replace('_coeffs','')}"
            coeff_file = out_dir / f"{name_stem}.npy"
            np.save(str(coeff_file), v)

    return

def find_fits(reof_ds: xr.Dataset, q_df: xr.DataArray, stack: xr.DataArray, train_size: float = 0.7, random_state: int = 0, ):
    """Function to fit multiple polynomial curves on different temporal modes and test results

    """        
    
    X = q_df
    y = reof_ds.temporal_modes

    # ---- Randomly split data into 2 groups -----
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

    logger.debug(X_train)
    logger.debug(X_test)

    spatial_test = stack.where(stack.time.isin(y_test.time),drop=True)

    shape3d = spatial_test.shape
    spatial_shape = shape3d[1:]
    shape2d = (shape3d[0],np.prod(spatial_shape))

    spatial_test_flat = xr.DataArray(
        spatial_test.values.reshape(shape2d),
        coords = [spatial_test.time,np.arange(shape2d[1])],
        dims=['time','space']
    )

    non_masked_idx= np.where(np.logical_not(np.isnan(spatial_test_flat[0,:])))[0]

    modes = reof_ds.mode.values

    fit_dict = dict()
    dict_keys = ['fit_r2','pred_r','pred_rmse']

    for mode in modes:

        logger.debug(mode)
        y_train_mode = y_train.sel(mode=mode)
        y_test_mode = y_test.sel(mode=mode)

        for order in range(1,4):

            # apply polynomial fitting
            c = np.polyfit(X_train,y_train_mode,deg=order)
            f = np.poly1d(c)

            y_pred = f(X_test)

            synth_test = synthesize(reof_ds,X_test,f,mode=mode)                      
             

            synth_test_flat = xr.DataArray(
                synth_test.values.reshape(shape2d),
                coords = [synth_test.time,np.arange(shape2d[1])],
                dims=['time','space']
            )

            # calculate statistics
            # calculate the stats of fitting on a test subsample
            temporal_r2 = metrics.r2_score(y_test_mode, y_pred)

            temporal_r = -999 if temporal_r2 < 0 else np.sqrt(temporal_r2)

            # check the synthesis stats comapared to observed data
            space_r2 = metrics.r2_score(
                spatial_test_flat[:,non_masked_idx],
                synth_test_flat[:,non_masked_idx],
            )
            space_r= -999 if space_r2 < 0 else np.sqrt(space_r2)

            space_rmse = metrics.mean_squared_error(
                spatial_test_flat[:,non_masked_idx],
                synth_test_flat[:,non_masked_idx],
                squared=False
            )

            # pack the resulting statistics in dictionary for the loop
            #stats = [temporal_r2,space_r,space_rmse]
            stats = [temporal_r2, temporal_r, space_rmse]
            loop_dict = {f"mode{mode}_order{order}_{k}":stats[i] for i,k in enumerate(dict_keys)}
            loop_dict[f"mode{mode}_order{order}_coeffs"] = c
            logging.debug(loop_dict)
            # merge the loop dictionary with the larger one
            fit_dict = {**fit_dict,**loop_dict}

    return fit_dict


def sel_best_fit(fit_dict: dict, metric: str = "r",ranking: str = "max") -> tuple:
    """Function to extract out the best fit based on user defined metric and ranking

    args:
        fit_dict (dict): output from function `find_fits()`
        metric (str, optional): statitical metric to rank, options are 'r', 'r2' or 'rmse'. default = 'r'
        ranking (str, optional): type of ranking to perform, options are 'min' or 'max'. default = max

    returns:
        tuple: tuple of values containg the coefficient key, mode, and coefficients of best fit
    """
    def max_ranking(old,new):
        key,ranker = old
        k,v = new
        if v > ranker:
            key = k
            ranker = v
        return (key,ranker)
    
    def min_ranking(old,new):
        key,ranker = old
        k,v = new
        if v < ranker:
            key = k
            ranker = v
        return (key,ranker)

    if metric not in ["r","r2","rmse"]:
        raise ValueError("could not determine metric to rank, options are 'r', 'r2' or 'rmse'")

    if ranking == "max":
        ranker = -1
        ranking_f = max_ranking
    elif ranking == "min":
        ranker = 999
        ranking_f = min_ranking
    else:
        raise ValueError("could not determine ranking, options are 'min' or 'max'")

    ranked = ("",ranker)
    
    for k,v in fit_dict.items():
        if k.endswith(metric):
            ranked = ranking_f(ranked,(k,v))

    key_split = ranked[0].split("_")
    ranked_key = "_".join(key_split[:-2] + ["coeffs"])
    coeffs = fit_dict[ranked_key]
    mode = int(key_split[0].replace("mode",""))

    return (ranked_key,mode,coeffs)

def synthesize_indep(reof_ds: xr.Dataset, q_df: xr.DataArray, model_mode_order, model_path='.\\model_path\\'):
    """Function to synthesize data at time of interest and output as DataArray
   
    """
    mode_list=list(model_mode_order)
    for num_mode in mode_list:
        #for order in range(1,4):
             
        f = np.poly1d(np.load(model_path+'\poly'+'{num:0>2}'.format(num=str(num_mode))+'_deg'+'{num:0>2}'.format(num=model_mode_order[str(num_mode)])+'.npy'))
        #logger.debug('{model_mode_order[str(mode)]:0>2}'+'.npy')
        
        
        y_vals = xr.apply_ufunc(f, q_df)
        logger.debug(y_vals)
        
        synth = y_vals * reof_ds.spatial_modes.sel(mode=int(num_mode)) # + reof_ds.center
        
        synth = synth.astype(np.float32).drop("mode").sortby("time")

        return synth
             

def synthesize(reof_ds: xr.Dataset, q_df: xr.DataArray, polynomial: np.poly1d, mode: int = 1) -> xr.DataArray:
    """Function to synthesize data from reof data and regression coefficients.
    This will also format the result as a geospatially aware xr.DataArray

    args:
        reof_ds (xr.Dataset):
        q_df (xr.DataArray):
        polynomial (np.poly1d):
        mode (int, optional):


    returns:
        xr.DataArray: resulting synthesized data based on reof temporal regression
            and spatial modes
    """

    y_vals = xr.apply_ufunc(polynomial,q_df)
    
    logger.debug(y_vals)

    synth = y_vals * reof_ds.spatial_modes.sel(mode=mode) + reof_ds.center

    # drop the unneeded mode dimension
    # force sorting by time in case the array is not already
    synth = synth.astype(np.float32).drop("mode").sortby("time")
    

    return synth
