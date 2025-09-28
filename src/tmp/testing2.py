
import time
import dask
import dask.array as darray
import numpy as np
import numba as nb
import xarray as xr
import rioxarray

from dask.array.overlap import overlap
from dask.array import map_overlap
from dask.array import map_blocks

from dask.diagnostics import ProgressBar


# region helpers

def _is_lazy(arr: np.ndarray) -> bool:

    if isinstance(arr, darray.Array):
        return True

    return False


def _extract_chunk_offsets(arr: darray.Array) -> list:

    if arr.ndim == 2:
        y_chunks = arr.chunks[0]
        x_chunks = arr.chunks[1]
    elif arr.ndim == 3:
        y_chunks = arr.chunks[1]
        x_chunks = arr.chunks[2]
    else:
        raise TypeError('Dask arrays of 2 or 3 dimensions supported only.')

    # get global offsets (start indices) of each chunk
    y_offsets = np.cumsum((0,) + y_chunks[:-1])
    x_offsets = np.cumsum((0,) + x_chunks[:-1])

    # create pairs of x, y chunk offsets
    offsets = []
    for y_offset in y_offsets:
        for x_offset in x_offsets:
            offsets.append((int(y_offset), int(x_offset)))

    return offsets


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

# endregion


# region training

@nb.njit(parallel=True)
def _make_train_grid(
        arr_x: np.ndarray,
        arr_y: np.ndarray
) -> np.ndarray:

    y_size, x_size = arr_x.shape
    arr_out = np.zeros((y_size, x_size), np.bool)

    for yi in nb.prange(1, y_size - 1):
        for xi in range(1, x_size - 1):

            if not arr_y[yi, xi]:
                arr_x_sel = arr_x[yi - 1:yi + 1 + 1, xi - 1:xi + 1 + 1]

                if not _is_any_nodata(arr_x_sel, True):
                    arr_out[yi, xi] = True

    return arr_out


def _random_sampling_np(
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


def _random_sampling_dk(
        arr: np.ndarray,
        n_samples: int,
        rand_seed: int,
        block_info: dict | None = None
) -> np.ndarray:

    chunk_size = np.prod(arr.shape)
    total_size = np.prod(block_info[None]['shape'])

    n_sub_samples = int(chunk_size / total_size * n_samples)

    arr_out = _random_sampling_np(
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

    i = 0  # safe as prange not used
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


# TODO: combine X, y func here like predict
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
            arr_smp = _random_sampling_np(
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
                _random_sampling_dk,
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


# endregion


# region prediction

@nb.njit(parallel=True)
def _make_predict_grid(
        arr_x: np.ndarray,
        arr_y: np.ndarray
) -> np.ndarray:

    y_size, x_size = arr_x.shape
    arr_out = np.zeros((y_size, x_size), np.bool)

    for yi in nb.prange(1, y_size - 1):
        for xi in range(1, x_size - 1):

            if arr_y[yi, xi]:
                arr_x_sel = arr_x[yi - 1:yi + 2, xi - 1:xi + 2]

                if not _is_any_nodata(arr_x_sel, True):
                    arr_out[yi, xi] = True

    return arr_out


@nb.njit
def _extract_predict_i_x_set(
        arr_grid: np.ndarray,
        arr_x: np.ndarray,
) -> tuple:

    v_size, y_size, x_size = arr_x.shape
    v_size *= 9  # 9 pix per var per win

    n_samples = 0
    for yi in range(1, y_size - 1):
        for xi in range(1, x_size - 1):
            if arr_grid[yi, xi]:
                n_samples += 1

    if n_samples == 0:
        return (
            np.empty((0, 2), np.int64),
            np.empty((0, v_size), arr_x.dtype)
        )

    arr_i = np.empty((n_samples, 2), np.int64)
    arr_X = np.empty((n_samples, v_size), arr_x.dtype)

    i = 0
    for yi in range(1, y_size - 1):
        for xi in range(1, x_size - 1):
            if arr_grid[yi, xi]:
                arr_i[i, 0], arr_i[i, 1] = yi, xi
                arr_X[i, :] = arr_x[:, yi - 1:yi + 2, xi - 1:xi + 2].ravel()
                i += 1

    return arr_i, arr_X


def _extract_predict_set_np(
        arr_x: np.ndarray,
        arr_y: np.ndarray,
        nodata: int | float
) -> tuple:

    # clamp nodata across vars. true == nodata
    arr_nd_x = np.any(arr_x == nodata, axis=0)
    arr_nd_y = np.any(arr_y == nodata, axis=0)

    # predict grid. y = nodata and x win all valid. true = predict
    arr_grid = _make_predict_grid(arr_nd_x, arr_nd_y)

    # extract pixel idx and x wins where grid == true
    arr_i, arr_X = _extract_predict_i_x_set(arr_grid, arr_x)

    return arr_i, arr_X


@dask.delayed
def _extract_predict_set_dk(
        arr_x: np.ndarray,
        arr_y: np.ndarray,
        nodata: int | float,
        offsets: tuple,
) -> tuple:

    arr_i, arr_X = _extract_predict_set_np(
        arr_x,
        arr_y,
        nodata
    )

    # convert per-chunk local idx to global
    arr_i[:, 0] += offsets[0]  # y
    arr_i[:, 1] += offsets[1]  # x

    return arr_i, arr_X


def _apply_extract_predict_set_dk(
        arr_x: darray.Array,
        arr_y: darray.Array,
        nodata: int | float
) -> tuple:

    arr_x_pad = overlap(
        arr_x,
        depth=(0, 1, 1),
        boundary=nodata,
    )

    arr_y_pad = overlap(
        arr_y,
        depth=(0, 1, 1),
        boundary=nodata,
    )

    n_vars = arr_x.shape[0] * 9  # 9 pix per var per win

    # extract chunk offsets for local-global conversion
    offsets = _extract_chunk_offsets(arr_x_pad)

    x_delays = arr_x_pad.to_delayed().ravel()
    y_delays = arr_y_pad.to_delayed().ravel()

    arr_i, arr_X = [], []
    for x, y, offset in zip(x_delays, y_delays, offsets):
        delay = dask.delayed(_extract_predict_set_dk)(
            x, y, nodata, offset
        )

        # TODO: determine n_rows for below

        # unpack idx output. wrap as dask. shape == np.nan for ragged
        arr_i.append(
            darray.from_delayed(
                dask.delayed(lambda d: d[0])(delay),
                shape=(np.nan, 2),
                dtype=np.int64
            )
        )

        # as above but for X output only
        arr_X.append(
            darray.from_delayed(
                dask.delayed(lambda d: d[1])(delay),
                shape=(np.nan, n_vars),
                dtype=arr_x_pad.dtype
            )
        )

    arr_i = darray.concatenate(arr_i, axis=0)
    arr_X = darray.concatenate(arr_X, axis=0)

    return arr_i, arr_X


def extract_predict_set(
        arr_x: np.ndarray | darray.Array,
        arr_y: np.ndarray | darray.Array,
        nodata: int | float
) -> tuple:

    is_x_lazy = _is_lazy(arr_x)
    is_y_lazy = _is_lazy(arr_y)

    if not is_x_lazy and not is_y_lazy:
        arr_i, arr_X = _extract_predict_set_np(
            arr_x,
            arr_y,
            nodata
        )
    elif is_x_lazy and is_y_lazy:
        arr_i, arr_X = _apply_extract_predict_set_dk(
            arr_x,
            arr_y,
            nodata
        )
    else:
        raise TypeError('Numpy or dask arrays supported only.')

    return arr_i, arr_X


@nb.njit(parallel=True)
def _fill_via_predict_np(
    arr_fill: np.ndarray,
    arr_i: np.ndarray,
    arr_y_pred: np.ndarray,
) -> np.ndarray:

    n_row = arr_i.shape[0]

    for i in nb.prange(n_row):
        yi, xi = arr_i[i, 0], arr_i[i, 1]
        arr_fill[:, yi, xi] = arr_y_pred[i]

    return arr_fill


def _fill_via_predict_dk(
        arr_fill: np.ndarray,
        arr_i: np.ndarray,
        arr_y_pred: np.ndarray,
        block_info: dict | None = None
) -> np.ndarray:

    # block_info gives global slice for this chunk
    chunk_extent = block_info[None]['array-location']
    y_min, y_max = chunk_extent[1]
    x_min, x_max = chunk_extent[2]

    # subset global indices within this chunk
    arr_mask = (
            (arr_i[:, 0] >= y_min) &
            (arr_i[:, 0] < y_max)  &
            (arr_i[:, 1] >= x_min) &
            (arr_i[:, 1] < x_max)
    )

    if not np.any(arr_mask):
        return arr_fill

    # convert global to local indices
    arr_j = np.column_stack((
        arr_i[arr_mask, 0] - y_min,
        arr_i[arr_mask, 1] - x_min
    ))

    arr_out = _fill_via_predict_np(
        arr_fill,
        arr_j,
        arr_y_pred[arr_mask]  # only y values in this chunk
    )

    return arr_fill

def _apply_fill_via_predict_dk(
        arr_fill: darray.Array,
        arr_i: darray.Array,
        arr_y_pred: darray.Array,
        nodata: int | float
) -> np.ndarray:

    # prevent multiple compute below
    arr_i, arr_y_pred = dask.persist(
        arr_i,
        arr_y_pred
    )

    arr_out = arr_fill.map_overlap(
        _fill_via_predict_dk,
        arr_i=arr_i,
        arr_y_pred=arr_y_pred,
        depth=(0, 1, 1),
        boundary=nodata,
        meta=np.array((), dtype=arr_fill.dtype)
    )

    return arr_out


def fill_via_predict(
        arr_fill: np.ndarray | darray.Array,
        arr_i: np.ndarray | darray.Array,
        arr_y_pred: np.ndarray | darray.Array,
        nodata: int | float
):

    is_i_lazy = _is_lazy(arr_i)
    is_y_lazy = _is_lazy(arr_y_pred)
    is_f_lazy = _is_lazy(arr_fill)

    if not is_i_lazy and not is_y_lazy and not is_f_lazy:
        arr_y = _fill_via_predict_np(
            arr_fill,
            arr_i,
            arr_y_pred
        )
    elif is_i_lazy and is_y_lazy and is_f_lazy:
        # TODO: check if need persist i, y_pred before this
        arr_y = _apply_fill_via_predict_dk(
            arr_fill,
            arr_i,
            arr_y_pred,
            nodata
        )
    else:
        raise TypeError('Numpy or dask arrays supported only.')

    return arr_y

# endregion



def run(
        da_x: xr.DataArray,
        da_y: xr.DataArray,
        nodata: int | float,
        n_samples: int | None,
        xgb_params: dict = None,
        xgb_client = None
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

    arr_i, arr_X = extract_predict_set(
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

    arr_y_pred = arr_X[:, 4 + 9*np.arange(10)]

    arr_fill = da_y.data  # TODO: if fill_inplace use arr_y, else .full()

    arr_out = fill_via_predict(
        arr_fill,
        arr_i,
        arr_y_pred,
        nodata
    )

    da_out = xr.DataArray(
        arr_out,
        dims=da_y.dims,
        coords=da_y.coords,
        attrs=da_y.attrs
    )

    return da_out


def main():

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

    da_x = da_x.chunk({'variable': -1})#.persist()  # 'y': 1024, 'x': 1024
    da_y = da_y.chunk({'variable': -1})#.persist()  # 'y': 1024, 'x': 1024

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

    da_out = run(
        da_x=da_x,  # inputs, features, predictors
        da_y=da_y,  # target
        nodata=nodata,
        n_samples=n_samples,
        xgb_params=xgb_params,
        #xgb_client=client
    )

    with ProgressBar():
        da_out = da_out.compute()

    return da_out


if __name__ == '__main__':

    s = time.time()

    main()

    e = time.time()
    print(e - s)