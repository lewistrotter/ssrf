

import dask
import xarray as xr
import rioxarray

from dask.distributed import LocalCluster, Client

def main(client = None):

    ds_cloud = xr.open_dataset(
        r'E:\PMA Unmixing\data\storage\07_apply_masks\2018-02-21.nc',
        mask_and_scale=False,
        decode_coords='all',
        chunks={}
    ).drop_vars('spatial_ref')

    ds_clear = xr.open_dataset(
        r'E:\PMA Unmixing\data\storage\07_apply_masks\2018-02-11.nc',
        mask_and_scale=False,
        decode_coords='all',
        chunks={}
    ).drop_vars('spatial_ref')

    da_x = ds_clear.to_array()#.compute()
    da_y = ds_cloud.to_array()#.compute()

    #da_x = da_x.isel(x=slice(5000, None), y=slice(0, 5000))
    #da_y = da_y.isel(x=slice(5000, None), y=slice(0, 5000))

    da_x = da_x.chunk({'variable': -1, 'y': 1024, 'x': 1024})#.persist()  # 'y': 1024, 'x': 1024
    da_y = da_y.chunk({'variable': -1, 'y': 1024, 'x': 1024})#.persist()  # 'y': 1024, 'x': 1024

    nodata = -999
    n_samples = 2000000

    xgb_params = {
        'num_boost_round': 100,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.1,
        'max_depth': 8,
        'device': 'cuda',
        'nthread': -1
    }

    da_out = ssrf.testing.run(
        da_x=da_x,  # inputs, features, predictors
        da_y=da_y,  # target
        nodata=nodata,
        n_samples=n_samples,
        xgb_params=xgb_params,
        xgb_client=client
    )

    return da_out


if __name__ == '__main__':

    client = None
    #cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    #client = Client(cluster)

    # TODO: use optimum to determine genral best params

    da_out = main(client)

