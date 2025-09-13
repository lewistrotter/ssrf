

import xarray as xr
import rioxarray

import ssrf

def main():

    ds_cloud = xr.open_dataset(
        r'E:\PMA Unmixing\data\storage\07_apply_masks\2018-02-21.nc',
        mask_and_scale=False,
        decode_coords='all'
    ).drop_vars('spatial_ref')

    ds_clear = xr.open_dataset(
        r'E:\PMA Unmixing\data\storage\07_apply_masks\2018-02-11.nc',
        mask_and_scale=False,
        decode_coords='all'
    ).drop_vars('spatial_ref')

    da_x = ds_clear.to_array()
    da_y = ds_cloud.to_array()

    nodata = -999
    n_samples = 2000000

    xgb_params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.1,
        'max_depth': 8,
        'device': 'cuda',
        'nthread': -1
    }


    da_out = ssrf.run(
        da_x=da_x,  # inputs, features, predictors
        da_y=da_y,  # target
        nodata=nodata,
        n_samples=n_samples,
        xgb_params=xgb_params
    )

    return da_out


if __name__ == '__main__':

    da_out = main()
