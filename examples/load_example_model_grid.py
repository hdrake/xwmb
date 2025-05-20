import urllib.request
import shutil
import os
import xarray as xr
import xgcm

def download_MOM6_example_data(file_name):
    # download the data
    url = 'https://zenodo.org/record/15420739/files/'
    destination_path = f"../data/{file_name}"
    if not os.path.exists(destination_path):
        print(f"File '{file_name}' being downloaded to {destination_path}.")
        with urllib.request.urlopen(url + file_name) as response, open(destination_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"File '{file_name}' has completed download to {destination_path}.")
    else:
        print(f"File '{file_name}' already exists at {destination_path}. Skipping download.")
    return destination_path

def load_MOM6_example_grid(file_name):
    destination_path = download_MOM6_example_data(file_name)
    ds = xr.open_dataset(destination_path, chunks=-1).fillna(0.)
    return construct_grid(ds)

def load_MOM6_coarsened_diagnostics():
    file_name = 'MOM6_global_example_sigma2_budgets_v0_0_6.nc'
    return load_MOM6_example_grid(file_name)

def construct_grid(ds):
    coords={
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'},
        'Z': {'center': 'sigma2_l', 'outer': 'sigma2_i'},
    }
    boundary = {'X':'periodic', 'Y':'extend', 'Z':'extend'}

    metrics = {('X','Y'):'areacello'}
    grid = xgcm.Grid(ds, coords=coords, metrics=metrics, boundary=boundary, autoparse_metadata=False)
    return grid