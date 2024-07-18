import re
import os
import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat


def parse_module(filepath):
    """
    Converts a .mat file into a netCDF file, preserving metadata
    
    Parameters
    ----------
    filepath: str
        The path to the .mat file to load and process into a netCDF
        file
    
    Returns
    -------
    ds: xarray.Dataset
        A dataset with the wind module data contained in the .mat file
        parsed and processed into an xarray dataset object
    """
    # First, parse out the deployment number and the serial number
    # from the filepath
    basename = os.path.basename(filepath)
    serial_number = re.search(r'[0-9]{3}', basename).group()
    
    # Get the deployment number
    deployment = re.search(r'000[0-9]{2}', filepath).group()
    
    # Load the data
    mat = loadmat(filepath, squeeze_me=True, simplify_cells=True)
    
    # Get the different data pieces
    meta = mat["meta"]
    header = {key.replace("_",""): mat[key] for key in mat if key.startswith("_")}
    data = {key: mat[key] for key in mat if not key.startswith(("_", "meta"))}
    
    #==============================================================
    # Filter the data if the timestamp == 1
    good, = np.where(data["mday"] != 1)
    for key in data:
        data[key] = data[key][good]
    
    #==============================================================
    #Now parse the attributes
    ATTRS_DS = header
    ATTRS_DS.update({'serial_number': serial_number})
    ATTRS_VARS = {}
    
    # Iterate through the meta data and construct a new ATTRIBUTES Dictionary
    for key in meta:
        # Check if contains metadata related to the variables
        if key == "data":
            ATTRS_VARS.update(meta[key])
        else:
            if type(meta[key]) is dict:
                ATTRS_DS.update(meta[key])
            else:
                ATTRS_DS.update({key: meta[key]})
                
    # Fix the keys in the ATTRS_VAR
    keys = ATTRS_VARS.keys()
    for key in list(keys):
        if key == "tilt_x":
            ATTRS_VARS['tiltx'] = ATTRS_VARS.pop(key)
        elif key == "tilt_y":
            ATTRS_VARS['tilty'] = ATTRS_VARS.pop(key)
        else:
            pass
        
    # Fix the dtypes
    for key in ATTRS_DS:
        item = ATTRS_DS.get(key)
        if type(item) is bytes:
            item = item.decode('utf-8')
            ATTRS_DS[key] = item
        elif type(item) is np.ndarray:
            item = ", ".join(item)
            ATTRS_DS[key] = item
        elif type(item) is list:
            item = ' '.join(item)
        else:
            pass
        
    for var in ATTRS_VARS:
        for key in ATTRS_VARS[var]:
            item = ATTRS_VARS[var].get(key)
            if type(item) is np.ndarray:
                item = " ".join(item)
                ATTRS_VARS[var][key] = item
            else:
                pass
        
    # =================================================================
    # Construct the dataset
    # Parse the datetime
    timestamps = pd.to_datetime(data["mday"]-719529, unit='D')
    
    # Create a deployment value
    deployment = int(deployment)
    deployment = deployment*np.ones(timestamps.shape, int)

    # Iterate through the dictionary and construct a variables dict
    data_vars = {key: (["time"], data.get(key), ATTRS_VARS[key]) for key in data if key != "mday"}
    data_vars.update({"deployment": (["time"], deployment)})
    coords = {'time': timestamps}
    attrs = ATTRS_DS
    
    # Construct the dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs
    )
    
    return ds


def main():
    """Parse all the swnd .mat files in the directory to .nc files"""
    parser = argparse.ArgumentParser()
    parser.add_argument('dirname', required=True, help='name of the directory where the WND module files are saved')
    args = parser.parse_args()
    dirname = args.dirname
    
    # Get the full path name of the directory
    basepath = os.path.abspath(dirname)
    fullpath = "/".join((basepath, "**", "*WND*.mat"))
    
    # Identify all of the wind .mat files in the directory and subdirectories
    mat_files = glob.glob(fullpath, recursive=True)

    # Now iterate through each file and rework into a netCDF file, saving back as a .nc in the same directory
    for file in mat_files:
        # parse the .mat into a netCDF
        ds = parse_module(file)
        # Save the netCDF
        new_file = file.replace('.mat','.nc')
        print(new_file)
        ds.to_netcdf(new_file, format='netcdf4', engine='h5netcdf')
    

if __name__ == '__main__':
    main()