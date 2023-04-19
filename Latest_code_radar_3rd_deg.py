#!/usr/bin/env python
# coding: utf-8

# In[1]:


# conda update conda
# conda create -n radar -c conda-forge -c anaconda pandas scipy numpy netcdf4 matplotlib scikit-image h5py pygrib cdsapi scikit-learn
#!/usr/bin/env python 
# -*- coding: utf-8 -*-

__author__ = "Ligin Joseph and Hylke Beck"
__email__ = "hylke.beck@gloh2o.org"
__date__ = "February 1, 2021"

import os
import sys
import numpy as np
import pandas as pd
import logging
import time
import pdb
import h5py
import scipy.io
import glob
import shutil
import pickle
import random
import calendar
from netCDF4 import Dataset 
from datetime import datetime, timedelta
from subprocess import call
import matplotlib.pyplot as plt
import importlib
from PIL import Image
from collections import Counter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay
from scipy.ndimage import convolve
from matplotlib import pyplot as plt 
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from scipy.spatial import cKDTree


# In[2]:


# Download gifs
def download_gifs(rawfolder):
    command = 'wget --no-check-certificate --directory-prefix='+rawfolder+' --timeout=10 --no-clobber -r -A .gif "https://ncm.gov.sa/Ar/Weather/RegionWeather/Pages/KSA-IRIS.aspx"'
    print(command)
    call(command,shell=True)


# Move gifs to year-month folder
def move_gifs(rawfolder):
    gif_files = sorted(glob.glob(os.path.join(rawfolder,'ncm.gov.sa','Images','IRIS','*.gif')))
    for gif_file in gif_files:
        date_str = '20'+os.path.basename(gif_file)[3:11]
        date_obj = datetime.strptime(date_str, '%Y%m%d%H%M')
        if os.path.exists(os.path.join(rawfolder,date_obj.strftime('%Y%m')))==False:
            os.makedirs(os.path.join(rawfolder,date_obj.strftime('%Y%m')))
        dest = os.path.join(rawfolder,date_obj.strftime('%Y%m'),os.path.basename(gif_file))
        shutil.move(gif_file,dest)
        print('Moved '+gif_file+' to '+dest)


# In[3]:


# Function to list all the colors in the image in RGB format
def color_list(image_file):
    # Load image
    img = Image.open(image_file)
    img = img.convert('RGB')
    img_array = np.array(img)

    # Get the unique colors in RGB format
    unique_colors = np.unique(img_array.reshape(-1, 3), axis=0)

    # Convert the unique colors to a list of tuples
    unique_colors = [tuple(color) for color in unique_colors.tolist()]

    return unique_colors
    

# Function to find the weights for interpolating data into regular lat-lon grid
def interp_weights(xy, uv,d=2):
    tri = Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


# In[13]:


def one_time_run():
    # Extract unique colors from reference images
    color_zero = color_list(os.path.join('GAMEP_radar_gif_ref_img', 'color_zero.png'))
    color_no_data = color_list(os.path.join('GAMEP_radar_gif_ref_img', 'color_nodata.png'))
    
    # Compute color-to-value mapping
    color_bar = Image.open(os.path.join('GAMEP_radar_gif_ref_img', 'color_bar.png'))
    color_bar = color_bar.convert('RGB')
    color_bar = np.array(color_bar)
    color_bar = color_bar.reshape(-1, 3)
    color_bar = [tuple(color) for color in color_bar]
    num_colors = len(color_bar)
    value_range = 70 - 2
    value_step = value_range / (num_colors - 1)
    color_to_values = {}
    for i, color in enumerate(color_bar):
        value = 2 + i * value_step if i != num_colors - 1 else 70
        color_to_values[color] = value
    
    # Define the irregular pixel coordinates
    irregular_coords = np.array([[69.32428,59.83234],[181.87382,91.19064],[290.41243,28.31199],[750.01000,71.58157],[38.00649,107.72059],
                             [233.04376,152.04356],[343.48655,194.82697],[740.85370,149.73423],[838.08873,178.90474],[77.62977,205.64437],
                             [161.29241,280.71792],[659.54091,282.62210],[756.53285,230.27725],[879.45413,227.88688],[908.82722,238.15733],
                             [45.05603,304.19613],[426.29838,335.91905],[628.38518,308.77427],[852.91707,392.80155],[1034.09835,372.86837],
                             [153.63515,409.00738],[536.57911,403.49740],[689.80531,481.12337],[893.37090,431.16887],[945.55370,463.90466],
                             [118.56977,545.27823],[236.46724,528.01901],[735.03986,533.04282],[942.15047,513.43376],[1063.93735,525.79071],
                             [188.01178,695.42532],[294.84877,667.91591],[366.31652,678.93588],[195.32467,714.30512],[193.64331,792.78189],
                             [424.49548,745.74445],[475.26027,763.00367],[610.25490,734.07624],[477.20497,859.67149],[570.38854,895.00022],
                             [473.15351,933.73217],[976.50685,972.78824],[145.95763,1015.89577],[282.41079,1020.43341],[559.53063,1011.03402]])

        # Define the corresponding regular lat-lon coordinates
    regular_coords = np.array([[34.870310,32.000335],[37.00529,31.50556],[39.08596,32.50158],[47.86722,31.79566],[34.26619,31.21880],
                               [38.00081,30.50440],[40.101849,29.788529],[47.668612,30.552839],[49.517071,30.071945],[35.020692,29.609113],
                               [36.598637,28.372098],[46.121426,28.334305],[47.971041,29.240431],[50.313350,29.243498],[50.872055,29.073344],
                               [34.383203,27.980650],[41.690171,27.438312],[45.562941,27.906775],[49.798362,26.470819],[53.279217,26.799579],
                               [36.474975,26.207652],[43.768520,26.305203],[46.702362,24.959593],[50.564131,25.805060],[51.557958,25.267969],
                               [35.809770,23.836921],[38.063306,24.150358],[47.561805,24.062687],[51.543386,24.410962],[53.847529,24.181106],
                               [37.133763,21.188575],[39.149430,21.667012],[40.552663,21.480526],[37.268670,20.839278],[37.238345,19.434415],
                               [41.640053,20.294170],[42.618240,19.994558],[45.198527,20.500918],[42.657882,18.234320],[44.419408,17.616928],
                               [42.582626,16.898795],[52.172877,16.196145],[36.330346,15.386863],[38.917282,15.295983],[44.216461,15.480657]])


    # Fit a 2nd-degree polynomial to anchor points
    poly_reg = PolynomialFeatures(degree=3)
    irregular_poly = poly_reg.fit_transform(irregular_coords)
    regressor = LinearRegression().fit(irregular_poly, regular_coords)

    # Generate lat-lon grid from pixel coordinates
    xx, yy = np.meshgrid(np.arange(1067), np.arange(1031))
    pixel_coords = np.column_stack((xx.ravel(), yy.ravel()))

    # Apply the polynomial transformation to the pixel coordinates
    pixel_coords_poly = poly_reg.transform(pixel_coords)
    predicted_coords = regressor.predict(pixel_coords_poly)

    # Reshape the predicted coordinates into a lat-lon grid
    latlon_map = np.zeros((1031, 1067, 2))
    latlon_map[:, :, 0] = predicted_coords[:, 0].reshape((1031, 1067))
    latlon_map[:, :, 1] = predicted_coords[:, 1].reshape((1031, 1067))
    Y, X = latlon_map[:, :, 1], latlon_map[:, :, 0]

    # Define target lat-lon grid
    res = 0.025
    lon_target = np.asarray(np.arange(33 + res / 2, 55 + res / 2, res))
    lat_target = np.asarray(np.arange(34 - res / 2, 10 - res / 2, -res))
    Xi, Yi = np.asarray(np.meshgrid(lon_target, lat_target))

    # Prepare xy and uv coordinate arrays
    xy = np.zeros([X.shape[0] * X.shape[1], 2])
    xy[:, 0] = Y.flatten()
    xy[:, 1] = X.flatten()
    uv = np.zeros([Xi.shape[0] * Xi.shape[1], 2])
    uv[:, 0] = Yi.flatten()
    uv[:, 1] = Xi.flatten()

    # Compute interpolation weights
    vtx, wts = interp_weights(xy, uv)

    # Save results to a dictionary
    data = {
        'unique_colors_zero': color_zero,
        'unique_colors_no_data': color_no_data,
        'unique_colors_cb': color_bar,
        'color_to_values': color_to_values,
        'Xi': Xi,
        'Yi': Yi,
        'vtx': vtx,
        'wts': wts
        }
    
    # Save the results to disk
    with open(datapath, "wb") as f:
        pickle.dump(data, f)

    return data
    
    
# Define the function to load the results
def load_data():
    with open(datapath, "rb") as f:
        data = pickle.load(f)
    return data


# Define the function to call, which either runs the function once or loads the results
def get_data():
    if os.path.exists(datapath):
        print('Loading previously created data file')
        data = load_data()
    else:
        print('Executing one_time_run()')
        data = one_time_run()
    return data


# In[5]:



# Function to interpolate data into a regular lat-lon grid
def interpolate(values, vtx, wts, fill_value=np.nan):
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret
    
    
def fill_gaps(input_array, gap_mask, kernel_size):
    # Create a kernel for mean filtering
    mean_kernel = np.ones((kernel_size, kernel_size))

    # Perform mean filtering using convolve, ignoring gaps (masked values)
    filtered_array = convolve(np.where(gap_mask, 0, input_array), mean_kernel, mode='constant', cval=0.0)
    kernel_sum = convolve(np.where(gap_mask, 0, 1), mean_kernel, mode='constant', cval=0.0)

    # Normalize the filtered array by dividing by the kernel_sum to compute the mean
    filtered_array = filtered_array / kernel_sum

    # Fill the gaps in the original input_array with the filtered values
    output_array = input_array.copy()
    output_array[gap_mask] = filtered_array[gap_mask]

    return output_array


# In[6]:


def map_colors(img, palette):
    # Flatten the image into a 2D array
    h, w, _ = img.shape
    img_flat = np.reshape(img, (h*w, 3))

    # Build a KD-tree for the palette
    tree = cKDTree(palette)

    # Query the tree for the closest color to each pixel in the image
    _, indices = tree.query(img_flat)

    # Reshape the indices back into the original image shape
    indices = np.reshape(indices, (h, w))

    # Map the indices to the palette colors
    img_mapped = palette[indices]

    return img_mapped


# In[7]:


def gif_to_netcdf(input,output,ts,data):
    # Load image
    img = Image.open(input).convert('RGB')

    # Calculate the coordinates to crop the image
    width, height = img.size
    new_height = 1031
    new_width = 1067
    left = (width - new_width) // 2  # integer division
    top = 0
    right = left + new_width
    bottom = 1031

    # Crop the image
    img = img.crop((left, top, right, bottom))
    img = np.array(img)
    
    # Apply palette to image
    color_to_values = np.array(list(data['color_to_values'].keys()))
    unique_colors_no_data = data['unique_colors_no_data']
    unique_colors_zero = data['unique_colors_zero']
    black = [(0,0,0)]
    palette = stacked = np.vstack((color_to_values, unique_colors_no_data, unique_colors_zero, black))
    img_mapped = map_colors(img, palette)
    
    # Create mask arrays for each list
    mask_nodata = np.isin(img_mapped, data['unique_colors_no_data']).all(axis=2)
    mask_zero = np.isin(img_mapped, data['unique_colors_zero']).all(axis=2)
    mask_black = np.isin(img_mapped, [(0,0,0)]).all(axis=2)

    # Map colors to values
    data_array = np.zeros(img_mapped[:,:,0].shape, dtype=np.single)*np.NaN
    data_array[mask_zero] = 0
    for color, value in data['color_to_values'].items():
        mask = np.all(img_mapped==np.array(color), axis=2)
        data_array[mask] = value
    data_array[mask_nodata] = np.nan
    data_array[mask_black] = np.nan
    
    # Fill gaps where black lines or airport symbols were present
    data_array_int = fill_gaps(data_array, np.isnan(data_array), 5)
    
    # Interpolate the array into a regular lat-lon grid
    reflectivity = interpolate(data_array_int.flatten(), data['vtx'], data['wts'])
    reflectivity = reflectivity.reshape(data['Xi'].shape[0],data['Xi'].shape[1])
    
    # Compute precipitation using NWS Z-R relationship (could not find a reference, unfortunately)
    precipitation = (10**(reflectivity/10)/200)**0.625
    
    # Create netCDF file
    lat = data['Yi'][:,0]
    lon = data['Xi'][0,:]
    
    if os.path.exists(os.path.dirname(output))==False:
        os.makedirs(os.path.dirname(output))
    
    ncfile = Dataset(output, 'w', format='NETCDF4')
    ncfile.history = 'Created on %s' % datetime.utcnow().strftime('%Y-%m-%d %H:%M')

    ncfile.createDimension('lon', len(lon))
    ncfile.createDimension('lat', len(lat))
    ncfile.createDimension('time', None)

    ncfile.createVariable('lon', 'f4', ('lon',))
    ncfile.variables['lon'][:] = lon
    ncfile.variables['lon'].units = 'degrees_east'
    ncfile.variables['lon'].long_name = 'longitude'

    ncfile.createVariable('lat', 'f4', ('lat',))
    ncfile.variables['lat'][:] = lat
    ncfile.variables['lat'].units = 'degrees_north'
    ncfile.variables['lat'].long_name = 'latitude'

    ncfile.createVariable('time', 'f4', 'time')
    ncfile.variables['time'][:] = (pd.to_datetime(ts)-pd.to_datetime(datetime(1900, 1, 1))).total_seconds()/86400
    ncfile.variables['time'].units = 'days since 1900-1-1 00:00:00'
    ncfile.variables['time'].long_name = 'time'
    
    ncfile.createVariable('reflectivity', np.single, ('time', 'lat', 'lon'), zlib=True, complevel=1, chunksizes=(1,200,200,), fill_value=-9999, least_significant_digit=1)
    ncfile.createVariable('precipitation', np.single, ('time', 'lat', 'lon'), zlib=True, complevel=1, chunksizes=(1,200,200,), fill_value=-9999, least_significant_digit=1)

    ncfile.variables['precipitation'][0,:,:] = precipitation
    ncfile.variables['precipitation'].units = 'mm/h'
    ncfile.variables['reflectivity'][0,:,:] = reflectivity
    ncfile.variables['reflectivity'].units = 'dBZ'
    
    ncfile.close()
    


# In[8]:


def process_gif_file(gif_file, config, data):
    date_str = '20' + os.path.basename(gif_file)[3:13]
    date_obj = datetime.strptime(date_str, '%Y%m%d%H%M')
    outfolder = os.path.join(config['dir_converted'], 'GAMEP_radar_data_preliminary', date_obj.strftime('%Y%m'))
    outfilepath = os.path.join(outfolder, date_obj.strftime('%d%H%M') + '.nc')
    
    # Check if output already exists
    if os.path.isfile(outfilepath):
        return

    print('Converting ' + gif_file)
    t = time.time()
    
    # Pause the function for a moment to avoid conflicts
    time.sleep(random.uniform(0, 1))

    # Create output folder
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    
    # Convert gif to netCDF
    gif_to_netcdf(gif_file, outfilepath, date_obj, data)
    print('Time elapsed is ' + str(time.time() - t) + ' sec')


# In[9]:



def parallel_process_gif_files(folders, config, data, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for folder in folders:
            gif_files = sorted(glob.glob(os.path.join(folder, '*.gif')))
            
            # Submit tasks to the ThreadPoolExecutor
            futures = [executor.submit(process_gif_file, gif_file, config, data) for gif_file in gif_files]

            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Error processing file: {e}')


# In[10]:


def save_netcdf(file, varname, data, varunits, ts, least_sig_dig, lat, lon):

    if os.path.isfile(file)==False:        
        if os.path.exists(os.path.dirname(file))==False:
            os.makedirs(os.path.dirname(file))
        
        ncfile = Dataset(file, 'w', format='NETCDF4')
        ncfile.history = 'Created on %s' % datetime.utcnow().strftime('%Y-%m-%d %H:%M')

        ncfile.createDimension('lon', len(lon))
        ncfile.createDimension('lat', len(lat))
        ncfile.createDimension('time', None)

        ncfile.createVariable('lon', 'f4', ('lon',))
        ncfile.variables['lon'][:] = lon
        ncfile.variables['lon'].units = 'degrees_east'
        ncfile.variables['lon'].long_name = 'longitude'

        ncfile.createVariable('lat', 'f4', ('lat',))
        ncfile.variables['lat'][:] = lat
        ncfile.variables['lat'].units = 'degrees_north'
        ncfile.variables['lat'].long_name = 'latitude'

        ncfile.createVariable('time', 'f4', 'time')
        ncfile.variables['time'][:] = (pd.to_datetime(ts)-pd.to_datetime(datetime(1900, 1, 1))).total_seconds()/86400
        ncfile.variables['time'].units = 'days since 1900-1-1 00:00:00'
        ncfile.variables['time'].long_name = 'time'
    
    else:
        ncfile = Dataset(file, 'r+', format='NETCDF4')   
    
    if varname not in ncfile.variables.keys():
        ncfile.createVariable(varname, data.dtype, ('time', 'lat', 'lon'), zlib=True, complevel=1, chunksizes=(1,int(np.minimum(data.shape[0],200)),int(np.minimum(data.shape[1],200)),), fill_value=-9999, least_significant_digit=least_sig_dig)

    ncfile.variables[varname][0,:,:] = data
    ncfile.variables[varname].units = varunits

    ncfile.close()
    


# In[11]:


def compute_hourly_average(var_name, avg_type, var_units, dates_hourly, input_dir, output_dir, min_files, max_files, least_sig_dig):
    
    # If the data_loader call is slow (because of h5py), run this in terminal: export HDF5_USE_FILE_LOCKING='FALSE'
    
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    
    # Get dates of input files
    input_files = sorted(glob.glob(os.path.join(input_dir,'*.nc')))
    input_dates = []
    for ii in np.arange(len(input_files)):
        date_str = os.path.basename(input_dir)+os.path.basename(input_files[ii])[:6]
        input_dates.append(datetime.strptime(date_str, '%Y%m%d%H%M'))
    input_dates = pd.to_datetime(input_dates).floor('H')

    # Get information from existing netCDFs
    dset = Dataset(input_files[0])
    lat = np.array(dset.variables['lat'][:])
    lon = np.array(dset.variables['lon'][:])
    mapsize = (len(lat),len(lon))
    dset.close()
    
    for ii in np.arange(len(dates_hourly)):
    
        # Check if hourly file already exists
        filepath = os.path.join(output_dir, dates_hourly[ii].strftime('%Y%j.%H') + '.nc')
        if os.path.isfile(filepath): 
            continue
                
        # Check number of files
        ind = np.where(input_dates==dates_hourly[ii])[0]
        if len(ind)<min_files:
            continue

        # Load hourly data
        print('Creating '+filepath)
        t1 = time.time()        
        DATABIG = np.zeros((mapsize[0], mapsize[1], max_files), dtype=np.single)*np.NaN
        count = 0
        for jj in np.arange(len(ind)):
            try:
                dset = Dataset(input_files[ind[jj]])
                DATA = np.squeeze(dset.variables[var_name][:])
                dset.close()
                DATABIG[:,:,jj] = DATA
                count = count+1
            except:
                print(input_files[ind[jj]]+' seems to be corrupt, deleting')
                os.remove(input_files[ind[jj]])            
        if count<min_files:
            continue
        
        # Compute hourly average
        if avg_type=='sum':
            DATA = np.sum(DATABIG, 2)
        elif avg_type=='mean':
            DATA = np.mean(DATABIG, 2)
        elif avg_type=='max':
            DATA = np.max(DATABIG, 2)
        elif avg_type=='min':
            DATA = np.min(DATABIG, 2)
        else:
            error("avg_type must be sum, mean, max, or min")
        
        # Save average to netCDF
        save_netcdf(filepath, var_name, DATA, var_units, dates_hourly[ii], 1, lat, lon)
        print('Time elapsed is ' + str(time.time()-t1) + ' sec')


# In[12]:


def compute_3hourly_average(var_name, avg_type, var_units, dates_3hourly, input_dir, output_dir, min_files, max_files, least_sig_dig):
    
    # If the data_loader call is slow (because of h5py), run this in terminal: export HDF5_USE_FILE_LOCKING='FALSE'
    
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    
    # Get dates of input files
    input_files = sorted(glob.glob(os.path.join(input_dir,'*.nc')))
    input_dates = []
    for ii in np.arange(len(input_files)):
        date_str = os.path.basename(input_files[ii])[:10]
        input_dates.append(datetime.strptime(date_str, '%Y%j.%H'))
    input_dates = pd.to_datetime(input_dates).floor('3H')

    # Get information from existing netCDFs
    dset = Dataset(input_files[0])
    lat = np.array(dset.variables['lat'][:])
    lon = np.array(dset.variables['lon'][:])
    mapsize = (len(lat),len(lon))
    dset.close()
    
    for ii in np.arange(len(dates_3hourly)):
    
        # Check if 3-hourly file already exists
        filepath = os.path.join(output_dir, dates_3hourly[ii].strftime('%Y%j.%H') + '.nc')
        if os.path.isfile(filepath):
            continue
                
        # Check number of files
        ind = np.where(input_dates==dates_3hourly[ii])[0]
        if len(ind)<min_files:
            continue
        
        # Load hourly data
        print('Creating '+filepath)
        t1 = time.time()
        DATABIG = np.zeros((mapsize[0], mapsize[1], max_files), dtype=np.single)*np.NaN
        for jj in np.arange(len(ind)):
            dset = Dataset(input_files[ind[jj]])
            DATA = np.squeeze(dset.variables[var_name][:])
            dset.close()
            DATABIG[:,:,jj] = DATA

        # Compute 3-hourly average
        if avg_type=='sum':
            DATA = np.sum(DATABIG, 2)
        elif avg_type=='mean':
            DATA = np.mean(DATABIG, 2)
        elif avg_type=='max':
            DATA = np.max(DATABIG, 2)
        elif avg_type=='min':
            DATA = np.min(DATABIG, 2)
        else:
            error("avg_type must be sum, mean, max, or min")
        
        # Save average to netCDF
        save_netcdf(filepath, var_name, DATA, var_units, dates_3hourly[ii], 1, lat, lon)
        print('Time elapsed is ' + str(time.time()-t1) + ' sec')


def compute_daily_average(var_name, avg_type, var_units, dates_daily, input_dir, output_dir, min_files, max_files, least_sig_dig):
    
    # If the data_loader call is slow (because of h5py), run this in terminal: export HDF5_USE_FILE_LOCKING='FALSE'
    
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    
    # Get dates of input files
    input_files = sorted(glob.glob(os.path.join(input_dir,'*.nc')))
    input_dates = []
    for ii in np.arange(len(input_files)):
        date_str = os.path.basename(input_files[ii])[:10]
        input_dates.append(datetime.strptime(date_str, '%Y%j.%H'))
    input_dates = pd.to_datetime(input_dates).floor('D')

    # Get information from existing netCDFs
    dset = Dataset(input_files[0])
    lat = np.array(dset.variables['lat'][:])
    lon = np.array(dset.variables['lon'][:])
    mapsize = (len(lat),len(lon))
    dset.close()
    
    for ii in np.arange(len(dates_daily)):
    
        # Check if daily file already exists
        filepath = os.path.join(output_dir, dates_daily[ii].strftime('%Y%j') + '.nc')
        if os.path.isfile(filepath):
            continue
                
        # Check number of files
        ind = np.where(input_dates==dates_daily[ii])[0]
        if len(ind)<min_files:
            continue
        
        # Load 3-hourly data
        print('Creating '+filepath)
        t1 = time.time()
        DATABIG = np.zeros((mapsize[0], mapsize[1], max_files), dtype=np.single)*np.NaN
        for jj in np.arange(len(ind)):
            dset = Dataset(input_files[ind[jj]])
            DATA = np.squeeze(dset.variables[var_name][:])
            dset.close()
            DATABIG[:,:,jj] = DATA

        # Compute daily average
        if avg_type=='sum':
            DATA = np.sum(DATABIG, 2)
        elif avg_type=='mean':
            DATA = np.mean(DATABIG, 2)
        elif avg_type=='max':
            DATA = np.max(DATABIG, 2)
        elif avg_type=='min':
            DATA = np.min(DATABIG, 2)
        else:
            error("avg_type must be sum, mean, max, or min")
        
        # Save average to netCDF
        save_netcdf(filepath, var_name, DATA, var_units, dates_daily[ii], 1, lat, lon)
        print('Time elapsed is ' + str(time.time()-t1) + ' sec')


# In[ ]:


if __name__ == "__main__":
    script = sys.argv[0]
    settings_file = sys.argv[1]

    config = importlib.import_module(settings_file, package=None)
    config = config.config

    # Define a global variable for the path to the saved data
    datapath = os.path.join('GAMEP_radar_gif_ref_img','data.pkl')

    # Define number of parallel workers
    max_workers = 16

    t1 = time.time()

    sys.path.insert(1,config['dir_modules'].rstrip(os.sep))
    import library as lb
    lb.mkdir(os.path.join(config['dir_raw'],'GAMEP_raw_radar_gifs'))
    free_space = lb.get_free_space_mb(config['dir_raw'])
    if free_space<5000: raise ValueError('Not enough disk space, terminating to avoid file corruption')

    # Launch conversion
    #download_gifs(os.path.join(config['dir_raw'],'GAMEP_raw_radar_gifs'))
    #move_gifs(os.path.join(config['dir_raw'],'GAMEP_raw_radar_gifs'))
    data = get_data()
    #process_gif_file('/mnt/datawaha/hyex/beckhe/DATA_RAW/GAMEP_raw_radar_gifs/202303/KSA230312143503.MAX74SN.gif',config,data)
    folders = sorted(glob.glob(os.path.join(config['dir_raw'], 'GAMEP_raw_radar_gifs', '20*')))
    parallel_process_gif_files(folders, config, data, max_workers)
    
    # Delete most recent hourly file, to ensure it includes all 12 5-minute files
    input_files = sorted(glob.glob(os.path.join(config['dir_converted'],'GAMEP_radar_data_preliminary','Hourly')))
    try:
        os.remove(input_files[-2:])
    except:
        pass
    
    # Compute hourly means
    folders = sorted(glob.glob(os.path.join(config['dir_converted'], 'GAMEP_radar_data_preliminary', '20*')))
    for folder in folders:        
        date_obj = datetime.strptime(os.path.basename(folder), '%Y%m')
        number_of_days_in_month = calendar.monthrange(date_obj.year, date_obj.month)[1]
        dates_hourly = pd.date_range(start=date_obj, end=date_obj.replace(day=number_of_days_in_month).replace(hour=23), freq='H')
        input_dir = folder
        output_dir = os.path.join(config['dir_converted'],'GAMEP_radar_data_preliminary','Hourly')
        compute_hourly_average('precipitation', 'mean', 'mm/h', dates_hourly, input_dir, output_dir, 10,12,1)
        
    # Compute 3-hourly sums
    folders = sorted(glob.glob(os.path.join(config['dir_converted'], 'GAMEP_radar_data_preliminary', '20*')))
    dates_3hourly = pd.date_range(start=datetime.strptime(os.path.basename(folders[0]), '%Y%m'), end=datetime.strptime(os.path.basename(folders[-1]), '%Y%m').replace(day=31).replace(hour=21), freq='3H')
    input_dir = os.path.join(config['dir_converted'],'GAMEP_radar_data_preliminary','Hourly')
    output_dir = os.path.join(config['dir_converted'],'GAMEP_radar_data_preliminary','3hourly')
    compute_3hourly_average('precipitation', 'sum', 'mm/3h', dates_3hourly, input_dir, output_dir, 3,3,1)    

    # Compute daily sums
    folders = sorted(glob.glob(os.path.join(config['dir_converted'], 'GAMEP_radar_data_preliminary', '20*')))
    dates_daily = pd.date_range(start=datetime.strptime(os.path.basename(folders[0]), '%Y%m'), end=datetime.strptime(os.path.basename(folders[-1]), '%Y%m').replace(day=31), freq='D')
    input_dir = os.path.join(config['dir_converted'],'GAMEP_radar_data_preliminary','3hourly')
    output_dir = os.path.join(config['dir_converted'],'GAMEP_radar_data_preliminary','Daily')
    compute_3hourly_average('precipitation', 'sum', 'mm/d', dates_daily, input_dir, output_dir, 8,8,1)
    
    '''
    # Testing
    dset = Dataset('/mnt/datawaha/hyex/beckhe/DATA_PROCESSED/GAMEP_radar_data_preliminary/202303/121435.nc')
    precipitation = np.squeeze(dset.variables['precipitation'][:])
    reflectivity = np.squeeze(dset.variables['reflectivity'][:])

    plt.figure(0)
    plt.imshow(reflectivity,vmin=-5, vmax=40)

    plt.figure(3)
    plt.imshow(precipitation,vmin=-1, vmax=3)

    plt.show()

    pdb.set_trace()

    '''       

    print('\n\n\nScript done, total script run time is ' + str(time.time()-t1) + ' sec')

