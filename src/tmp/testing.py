
import time
import warnings
import dask
import dask.array as darray
import numpy as np
import numba as nb
import xarray as xr
import rioxarray
import xgboost as xgb
import xgboost.dask as xgbd

from dask.array.overlap import overlap
from dask.array import map_blocks
from dask.array import map_overlap

from dask.diagnostics import ProgressBar
from dask.distributed import LocalCluster
from dask.distributed import Client


def _is_lazy(arr: np.ndarray) -> bool:

    if isinstance(arr, darray.Array):
        return True

    return False


@nb.njit
def _is_any_nodata(
        arr: np.ndarray,
        nodata: int | float
) -> bool:

    if arr.ndim == 1:
        for i in range(arr.shape[0]):
            if arr[i] == nodata or np.isnan(arr[i]):
                return True

    elif arr.ndim == 2:
        for yi in range(arr.shape[0]):
            for xi in range(arr.shape[1]):
                if arr[yi, xi] == nodata or np.isnan(arr[yi, xi]):
                    return True

    elif arr.ndim == 3:
        for bi in range(arr.shape[0]):
            for yi in range(arr.shape[1]):
                for xi in range(arr.shape[2]):
                    if arr[bi, yi, xi] == nodata or np.isnan(arr[bi, yi, xi]):
                        return True

    else:
        raise ValueError('Arrays with > 3 dimensions not supported.')

    return False


@nb.njit(parallel=True)
def _make_train_grid(
        arr_x: np.ndarray,
        arr_y: np.ndarray
) -> np.ndarray:

    y_size, x_size = arr_x.shape
    arr_out = np.zeros((y_size, x_size), np.bool)

    for yi in nb.prange(1, y_size - 1):
        for xi in range(1, x_size - 1):

            if not arr_y[yi, xi]:  # true == nodata was present
                arr_x_sel = arr_x[yi - 1:yi + 1 + 1, xi - 1:xi + 1 + 1]

                if not _is_any_nodata(arr_x_sel, True):
                    arr_out[yi, xi] = True

    return arr_out


def _random_sampling(
        arr: np.ndarray,
        n_samples: int,
        rand_seed: int
) -> np.ndarray:

    arr_idx = np.flatnonzero(arr.ravel())
    if arr_idx.size == 0:
        return np.zeros(arr.shape, dtype=np.bool_)

    rng = np.random.default_rng(seed=rand_seed)
    arr_idx = rng.choice(
        arr_idx,
        size=n_samples,
        replace=False
    )

    arr_out = np.zeros(arr.shape, dtype=np.bool_).ravel()
    arr_out[arr_idx] = True
    arr_out = arr_out.reshape(arr.shape)

    return arr_out


def _random_sampling_lazy(
        arr: np.ndarray,
        n_samples: int,
        rand_seed: int,
        block_info=None
) -> np.ndarray:

    chunk_size = np.prod(arr.shape)
    total_size = np.prod(block_info[None]['shape'])

    n_sub_samples = int(chunk_size / total_size * n_samples)

    arr_out = _random_sampling(
        arr,
        n_sub_samples,
        rand_seed
    )

    return arr_out


@nb.njit(parallel=True)
def _extract_train_x_set(
        arr: np.ndarray,
        arr_smp: np.ndarray
) -> np.ndarray:

    n_vars = arr.shape[0] * 9  # 9 pix per var per win

    n_samples = arr_smp[1:-1, 1:-1].sum()  # exclude edges for valid sample size
    if n_samples == 0:
        return np.empty((0, n_vars), arr.dtype)

    arr_idx = np.empty((n_samples, 2), np.int64)

    i = 0
    for yi in range(1, arr_smp.shape[0] - 1):
        for xi in range(1, arr_smp.shape[1] - 1):
            if arr_smp[yi, xi]:
                arr_idx[i, 0], arr_idx[i, 1] = yi, xi
                i += 1

    arr_out = np.empty((n_samples, n_vars), arr.dtype)

    for i in nb.prange(n_samples):
        yi, xi = arr_idx[i, 0], arr_idx[i, 1]
        arr_sel = arr[:, yi - 1:yi + 1 + 1, xi - 1:xi + 1 + 1]
        arr_out[i, :] = arr_sel.ravel()

    return arr_out


def _extract_train_x_set_lazy(
        arr: np.ndarray,
        arr_smp: np.ndarray,
        nodata: int | float
) -> np.ndarray:

    n_vars = arr.shape[0] * 9  # 9 pix per var per win

    arr_pad = overlap(
        arr,
        depth=(0, 1, 1),
        boundary=nodata,
    )

    arr_smp_pad = overlap(
        arr_smp,
        depth=(1, 1),
        boundary=False   # false == invalid pixels
    )

    # output is ragged thus delayed approach better
    X1 = darray.concatenate([
        darray.from_delayed(
            dask.delayed(_extract_train_x_set)(x, smp),
            shape=(np.nan, n_vars),
            dtype=arr.dtype
        )
        for x, smp in zip(
            arr_pad.to_delayed().ravel(),
            arr_smp_pad.to_delayed().ravel()
        )
    ])

    X1 = X1.compute()

    # # this works... based on dask blog  FIXME: output slightly different to above.
    # X2 = map_blocks(
    #     _extract_train_x_set,
    #     arr_pad,
    #     arr_smp_pad,
    #     drop_axis=(2,),
    #     new_axis=(1, ),
    #     dtype=np.int16,
    #     meta=np.empty((0, 90), dtype=np.int16)
    # )
    #
    # X2 = X2.compute()


    X = None


    return X


@nb.njit(parallel=True)
def _extract_train_y_set(
        arr: np.ndarray,
        arr_smp: np.ndarray
) -> np.ndarray:

    n_vars = arr.shape[0]

    n_samples = arr_smp[1:-1, 1:-1].sum()  # exclude edges for valid sample size
    if n_samples == 0:
        return np.empty((0, n_vars), arr.dtype)

    arr_idx = np.empty((n_samples, 2), np.int64)

    i = 0
    for yi in range(1, arr_smp.shape[0] - 1):
        for xi in range(1, arr_smp.shape[1] - 1):
            if arr_smp[yi, xi]:
                arr_idx[i, 0], arr_idx[i, 1] = yi, xi
                i += 1

    arr_out = np.empty((n_samples, n_vars), arr.dtype)

    for i in nb.prange(n_samples):
        yi, xi = arr_idx[i, 0], arr_idx[i, 1]
        arr_out[i, :] = arr[:, yi, xi]

    return arr_out


def _extract_train_y_set_lazy(
        arr: np.ndarray,
        arr_smp: np.ndarray,
        nodata: int | float
) -> np.ndarray:

    n_vars = arr.shape[0] * 9  # 9 pix per var per win

    arr_pad = overlap(
        arr,
        depth=(0, 1, 1),
        boundary=nodata,
    )

    arr_smp_pad = overlap(
        arr_smp,
        depth=(1, 1),
        boundary=False  # false == invalid pixels
    )

    # output is ragged thus delayed approach better
    y = darray.concatenate([
        darray.from_delayed(
            dask.delayed(_extract_train_y_set)(y, smp),
            shape=(np.nan, n_vars),
            dtype=arr.dtype
        )
        for y, smp in zip(
            arr_pad.to_delayed().ravel(),
            arr_smp_pad.to_delayed().ravel()
        )
    ])

    return y


def extract_train_set(
        arr_x: np.ndarray | darray.Array,
        arr_y: np.ndarray | darray.Array,
        nodata: int | float,
        n_samples: int | None,
        rand_seed: int = 0
) -> tuple:

    is_x_lazy = _is_lazy(arr_x)
    is_y_lazy = _is_lazy(arr_y)

    if not is_x_lazy and not is_y_lazy:
        arr_smp = _make_train_grid(
            np.any(arr_x == nodata, axis=0),
            np.any(arr_y == nodata, axis=0)
        )  # true where valid for train in output

        if n_samples:
            arr_smp = _random_sampling(
                arr_smp,
                n_samples,
                rand_seed
            )

        X = _extract_train_x_set(arr_x, arr_smp)
        y = _extract_train_y_set(arr_y, arr_smp)

    elif is_x_lazy and is_y_lazy:
        arr_smp = map_overlap(
            _make_train_grid,
            darray.any(arr_x == nodata, axis=0),
            darray.any(arr_y == nodata, axis=0),
            depth=(1, 1),
            boundary=True,  # true == invalid going in, always
            dtype=np.bool_
        )  # true where valid for train in output

        if n_samples:
            arr_smp = map_blocks(
                _random_sampling_lazy,
                arr_smp,
                n_samples=n_samples,
                rand_seed=rand_seed,
                dtype=np.bool_
            )

        # FIXME: do a persist here to prevent repeat computes

        # FIXME: consider extracting per-chunk counts of true (try with and without pute) for ragged output

        X = _extract_train_x_set_lazy(arr_x, arr_smp, nodata)
        y = _extract_train_y_set_lazy(arr_y, arr_smp, nodata)

    else:
        raise TypeError('Only numpy or dask arrays are supported.')

    return X, y


def _default_xgb_params():

    xgb_params = {
        'num_boost_round': 100,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.1,
        'max_depth': 8,
        'device': 'cuda',
        'nthread': -1
    }

    return xgb_params


def train_xgb_models(
        X: np.ndarray | darray.Array,
        y: np.ndarray | darray.Array,
        xgb_params: dict = None,
        xgb_client=None
) -> dict:

    if xgb_params is None:
        xgb_params = _default_xgb_params()

    xgb_nbr = xgb_params.pop('num_boost_round', None)
    if xgb_nbr is None:
        raise ValueError('XGBoost num_boost_round must be specified in xgb_params.')

    # TODO: evals
    # TODO: early_stopping_rounds

    is_x_lazy = _is_lazy(X)
    is_y_lazy = _is_lazy(y)

    if not is_x_lazy and not is_y_lazy:
        models = {}
        for i in range(y.shape[1]):
            print(f'Training y variable {i + 1}.')
            dtrain = xgb.DMatrix(X, y[:, i])
            models[i] = xgb.train(xgb_params, dtrain, xgb_nbr)

    elif is_x_lazy and is_y_lazy:
        if xgb_client is None:
            raise ValueError('Must specify xgb_client if using dask arrays.')

        if np.isnan(X.shape[0]):
            warnings.warn('Unknown X shape, computing size.')
            X = X.compute_chunk_sizes()

        if np.isnan(y.shape[0]):
            warnings.warn('Unknown y shape, computing size.')
            y = y.compute_chunk_sizes()

        models = {}
        for i in range(y.shape[1]):
            print(f'Training y variable {i + 1}.')
            dtrain = xgbd.DaskDMatrix(xgb_client, X, y[:, i])
            models[i] = xgbd.train(xgb_client, xgb_params, dtrain, xgb_nbr)

    else:
        raise TypeError('Only numpy or dask arrays are supported.')

    return models


@nb.njit(parallel=True)
def _make_predict_grid(
        arr_x: np.ndarray,
        arr_y: np.ndarray
) -> np.ndarray:

    y_size, x_size = arr_x.shape
    arr_out = np.zeros((y_size, x_size), np.bool)

    for yi in nb.prange(1, y_size - 1):
        for xi in range(1, x_size - 1):

            if arr_y[yi, xi]:  # true == nodata was present
                arr_x_sel = arr_x[yi - 1:yi + 1 + 1, xi - 1:xi + 1 + 1]

                if not _is_any_nodata(arr_x_sel, True):
                    arr_out[yi, xi] = True

    return arr_out


@nb.njit
def _extract_predict_idx_set(
        arr_smp: np.ndarray,
        arr_yi_global: np.ndarray,
        arr_xi_global: np.ndarray
) -> np.ndarray:

    n_samples = arr_smp.sum()
    if n_samples == 0:
        return np.empty((0, 2), np.int64)

    arr_i = np.empty((n_samples, 2), np.int64)

    i = 0
    for yi in range(1, arr_smp.shape[0] - 1):
        for xi in range(1, arr_smp.shape[1] - 1):
            if arr_smp[yi, xi]:
                arr_i[i, 0] = arr_yi_global[yi, xi]
                arr_i[i, 1] = arr_xi_global[yi, xi]
                i += 1

    return arr_i


@nb.njit
def _extract_predict_idx_set_lazy(
        arr_smp: np.ndarray,
        arr_yi_global: np.ndarray,
        arr_xi_global: np.ndarray
) -> np.ndarray:

    n_samples = arr_smp[1:-1, 1:-1].sum()  # exclude edges for valid sample size
    if n_samples == 0:
        return np.empty((0, 2), np.int64)

    arr_i = np.empty((n_samples, 2), np.int64)

    i = 0
    for yi in range(1, arr_smp.shape[0] - 1):
        for xi in range(1, arr_smp.shape[1] - 1):
            if arr_smp[yi, xi]:
                arr_i[i, 0] = arr_yi_global[yi, xi]
                arr_i[i, 1] = arr_xi_global[yi, xi]
                i += 1

    return arr_i


# TODO: turn all arr_idx to arr_i

@nb.njit(parallel=True)
def _extract_predict_x_set(
        arr: np.ndarray,
        arr_smp: np.ndarray
) -> np.ndarray:

    n_vars = arr.shape[0] * 9  # 9 pix per var per win

    n_samples = arr_smp[1:-1, 1:-1].sum()  # exclude edges for valid sample size
    if n_samples == 0:
        return np.empty((0, n_vars), arr.dtype)

    arr_i = np.empty((n_samples, 2), np.int64)

    i = 0
    for yi in range(1, arr_smp.shape[0] - 1):
        for xi in range(1, arr_smp.shape[1] - 1):
            if arr_smp[yi, xi]:
                arr_i[i, 0], arr_i[i, 1] = yi, xi
                i += 1

    arr_out = np.empty((n_samples, n_vars), arr.dtype)

    for i in nb.prange(n_samples):
        yi, xi = arr_i[i, 0], arr_i[i, 1]
        arr_sel = arr[:, yi - 1:yi + 1 + 1, xi - 1:xi + 1 + 1]
        arr_out[i, :] = arr_sel.ravel()

    return arr_out


def _extract_predict_x_set_lazy(
        arr: np.ndarray,
        arr_smp: np.ndarray,
        nodata: int | float
) -> np.ndarray:

    n_vars = arr.shape[0] * 9  # 9 pix per var per win

    arr_pad = overlap(
        arr,
        depth=(0, 1, 1),
        boundary=nodata,
    )

    arr_smp_pad = overlap(
        arr_smp,
        depth=(1, 1),
        boundary=False   # false == invalid pixels
    )

    # output is ragged thus delayed approach better
    X = darray.concatenate([
        darray.from_delayed(
            dask.delayed(_extract_predict_x_set)(x, smp),
            shape=(np.nan, n_vars),
            dtype=arr.dtype
        )
        for x, smp in zip(
            arr_pad.to_delayed().ravel(),
            arr_smp_pad.to_delayed().ravel()
        )
    ])

    return X



def extract_predict_set(
        arr_x: np.ndarray | darray.Array,
        arr_y: np.ndarray | darray.Array,
        nodata: int | float
) -> tuple:

    is_x_lazy = _is_lazy(arr_x)
    is_y_lazy = _is_lazy(arr_y)

    if not is_x_lazy and not is_y_lazy:
        arr_smp = _make_predict_grid(
            np.any(arr_x == nodata, axis=0),
            np.any(arr_y == nodata, axis=0)
        )

        arr_global_yi, arr_global_xi = np.indices(arr_smp.shape)

        arr_i = _extract_predict_idx_set(
            arr_smp,
            arr_global_yi,
            arr_global_xi
        )

        X = _extract_predict_x_set(
            arr_x,
            arr_smp
        )


    elif is_x_lazy and is_y_lazy:
        arr_smp = map_overlap(
            _make_predict_grid,
            darray.any(arr_x == nodata, axis=0),
            darray.any(arr_y == nodata, axis=0),
            depth=(1, 1),
            boundary=True,  # true == invalid going in, always
            dtype=np.bool_
        )

        arr_global_yi, arr_global_xi = darray.indices(
            arr_smp.shape,
            chunks=(arr_smp.shape)
        )

        arr_i = darray.concatenate([
            darray.from_delayed(
                dask.delayed(_extract_predict_idx_set_lazy)(yi, xi, smp),
                shape=(np.nan, 2),
                dtype=np.int64
            )
            for yi, xi, smp in zip(
                arr_smp.to_delayed().ravel(),
                arr_global_yi.to_delayed().ravel(),
                arr_global_xi.to_delayed().ravel()
            )
        ])

        X = _extract_predict_x_set_lazy(
            arr_x,
            arr_smp,
            nodata
        )

        # TODO: consider persist here

    else:
        raise TypeError('Only numpy or dask arrays are supported.')

    return arr_i, X




#@nb.njit
def predict(
        arr_yi_global,
        arr_xi_global,
        arr: np.ndarray,
        arr_i: np.ndarray
) -> np.ndarray:

    #for yi in nb.prange(arr_yi_global.shape[0)]



    return None


def run(
        da_x: xr.DataArray,
        da_y: xr.DataArray,
        nodata: int | float,
        n_samples: int | None,
        xgb_params: dict = None,
        xgb_client=None
) -> xr.DataArray:

    if not isinstance(da_x, xr.DataArray):
        raise TypeError('Input da_x must be of type xr.DataArray.')
    if not isinstance(da_y, xr.DataArray):
        raise TypeError('Input da_y must be of type xr.DataArray.')

    if da_x.ndim != 3:
        raise TypeError('Input da_x must be of shape (vars, y, x).')
    if da_y.ndim != 2 and da_y.ndim != 3:
        raise TypeError('Input da_y must be of shape (y, x) or (vars, y, x).')

    if da_y.ndim == 2:
        da_y = da_y.expand_dims(dim={'variable': [0]})

    if da_y.shape[1] != da_x.shape[1] or da_y.shape[2] != da_x.shape[2]:
        raise ValueError('Inputs da_y and da_x spatial extents must match.')

    # TODO: ensure dtypes match

    if nodata is None:
        raise ValueError('Input nodata value must be provided.')

    X, y = extract_train_set(
        da_x.data,
        da_y.data,
        nodata,
        n_samples
    )

    # if arr_x.size == 0 or arr_y.size == 0:
    #     raise ValueError('No training samples were returned.')

    X = X.compute_chunk_sizes()
    y = y.compute_chunk_sizes()

    models = train_xgb_models(
        X,
        y,
        xgb_params,
        xgb_client
    )

    arr_i, X = extract_predict_set(
        da_x.data,
        da_y.data,
        nodata
    )

    arr_i = arr_i.compute_chunk_sizes()
    X = X.compute_chunk_sizes()


    dtrain = xgb.DMatrix(X)

    y_pred = []
    for i, model in models.items():
        print(f'Predicting y variable {i + 1}.')
        y_pred.append(model.predict(dtrain))

    y_pred = np.column_stack(y_pred)


    arr_out = darray.full(
        da_x.data.shape,
        nodata,
        chunks=da_x.data.chunks,
        dtype=da_x.data.dtype
    )

    # yi, xi = arr_i[:, 0], arr_i[:, 1]
    # arr_out[:, yi, xi] = y_pred.T

    # predict(
    #     arr_yi_global,
    #     arr_xi_global,
    #     arr=y_pred,
    #     arr_i=arr_i,
    #     #dtype=da_x.data.dtype
    # )


    # out = map_blocks(
    #     predict,
    #     arr_yi=arr_yi_global,
    #     arr_xi=arr_xi_global,
    #     arr=y_pred,
    #     arr_i=arr_i,
    #     dtype=da_x.data.dtype
    # )

    def scatter_block(block, block_info=None):

        b_start, b_stop = block_info[0]['array-location'][0]
        y_start, y_stop = block_info[0]['array-location'][1]
        x_start, x_stop = block_info[0]['array-location'][2]

        mask = (
                (arr_i[:, 0] >= y_start) & (arr_i[:, 0] < y_stop) &
                (arr_i[:, 1] >= x_start) & (arr_i[:, 1] < x_stop)
        )

        if not np.any(mask):
            return block

        local_ij = arr_i[mask]
        local_vals = y_pred[mask]

        local_i = local_ij[:, 0] - y_start
        local_j = local_ij[:, 1] - x_start

        block[:, local_i, local_j] = local_vals.T

        return block



    scattered = arr_out.map_blocks(scatter_block, dtype=arr_out.dtype)


    da_out = xr.DataArray(
        arr_out,
        coords=da_x.coords,
        dims=da_x.dims,
        attrs=da_x.attrs
    )



    return


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

    da_x = da_x.isel(x=slice(5000, None), y=slice(0, 5000))
    da_y = da_y.isel(x=slice(5000, None), y=slice(0, 5000))

    da_x = da_x.chunk({'variable': -1, 'y': 1024, 'x': 1024})#.persist()  # 'y': 1024, 'x': 1024
    da_y = da_y.chunk({'variable': -1, 'y': 1024, 'x': 1024})#.persist()  # 'y': 1024, 'x': 1024

    nodata = -999
    n_samples = 10000 #2000000

    xgb_params = {
        'num_boost_round': 100,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.1,
        'max_depth': 8,
        #'device': 'cuda',
        'nthread': -1
    }

    da_out = run(
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
    #cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    #client = Client(cluster)

    # TODO: use optimum to determine genral best params

    s = time.time()

    da_out = main(client)

    e = time.time()
    print(e - s)
