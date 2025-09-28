
import gc
import time
import itertools
import dask
import numpy as np
import numba as nb
import xarray as xr
import xgboost as xgb
import xgboost.dask as xgbd
from dask.diagnostics import ProgressBar


def _is_lazy(arr: np.ndarray) -> bool:

    if isinstance(arr, dask.array.Array):
        return True

    return False


def _get_xy_chunk_sizes(arr: dask.array.Array):

    y_chunks = arr.chunks[1]
    x_chunks = arr.chunks[2]

    n_samples = []
    for y_chunk, x_chunk in itertools.product(y_chunks, x_chunks):
        n_samples.append(y_chunk * x_chunk)

    return n_samples


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

    return False


@nb.njit
def _is_all_nodata(
        arr: np.ndarray,
        nodata: int | float
) -> bool:

    if arr.ndim == 1:
        for i in range(arr.shape[0]):
            if arr[i] != nodata and not np.isnan(arr[i]):
                return False

    elif arr.ndim == 2:
        for yi in range(arr.shape[0]):
            for xi in range(arr.shape[1]):
                if arr[yi, xi] != nodata and not np.isnan(arr[yi, xi]):
                    return False

    elif arr.ndim == 3:
        for bi in range(arr.shape[0]):
            for yi in range(arr.shape[1]):
                for xi in range(arr.shape[2]):
                    if arr[bi, yi, xi] != nodata and not np.isnan(arr[bi, yi, xi]):
                        return False

    return True


@nb.njit
def _extract_train_samples_numpy(
        arr_x: np.ndarray,
        arr_y: np.ndarray,
        nodata: int | float,
        n_samples: int | None
) -> np.ndarray:

    n_x_vars = arr_x.shape[0] * 9  # 9 pixels per window
    n_y_vars = arr_y.shape[0]

    arr_mask = np.zeros((arr_x.shape[1], arr_x.shape[2]), np.bool_)

    for yi in nb.prange(1, arr_x.shape[1] - 1):
        for xi in range(1, arr_x.shape[2] - 1):

            arr_y_sel = arr_y[:, yi, xi]
            if not _is_any_nodata(arr_y_sel, nodata):

                arr_x_sel = arr_x[:, yi - 1:yi + 1 + 1, xi - 1:xi + 1 + 1]
                if not _is_any_nodata(arr_x_sel, nodata):
                    arr_mask[yi, xi] = True

    n_valid = np.sum(arr_mask)
    arr_idx = np.empty((n_valid, 2), np.int32)
    if arr_idx.size == 0:
        arr_out = np.empty((0, n_x_vars + n_y_vars), arr_y.dtype)
        return arr_out

    # note: i parallel safe as prange not used
    i = 0
    for yi in range(arr_x.shape[1]):
        for xi in range(arr_x.shape[2]):
            if arr_mask[yi, xi]:
                arr_idx[i, 0], arr_idx[i, 1] = yi, xi
                i += 1

    # random sample valid indices
    n_idx = arr_idx.shape[0]
    if n_samples > n_idx or n_samples is None:
        n_samples = n_idx

    # fisher-yates non-replacement shuffle
    arr_rnd = np.arange(n_idx)
    for i in range(n_samples):
        j = i + np.random.randint(0, n_idx - i)
        arr_rnd[i], arr_rnd[j] = arr_rnd[j], arr_rnd[i]

    arr_idx = arr_idx[arr_rnd[:n_samples]]
    if arr_idx.size == 0:
        arr_out = np.empty((0, n_x_vars + n_y_vars), arr_y.dtype)
        return arr_out

    # extract real values
    n_idx = arr_idx.shape[0]
    arr_out = np.empty((n_idx, n_x_vars + n_y_vars), arr_y.dtype)

    for i in nb.prange(n_idx):
        ri, ci = arr_idx[i, 0], arr_idx[i, 1]

        arr_out[i, -n_y_vars:] = arr_y[:, ri, ci]  # y vars

        arr_x_sel = arr_x[:, ri - 1:ri + 1 + 1, ci - 1:ci + 1 + 1]
        arr_out[i, :n_x_vars] = arr_x_sel.ravel()  # x vars

    return arr_out


def _extract_train_samples_dask(
        arr_x: dask.array.Array,
        arr_y: dask.array.Array,
        nodata: int | float,
        n_samples: int | None
) -> tuple:

    if arr_x.chunks[1] != arr_y.chunks[1]:
        raise ValueError('Chunk sizes of y axis not equal between arr_x and arr_y.')
    if arr_x.chunks[2] != arr_y.chunks[2]:
        raise ValueError('Chunk sizes of x axis not equal between arr_x and arr_y.')

    if len(arr_x.chunks[0]) != 1 or len(arr_y.chunks[0]) != 1:
        raise ValueError('Chunk size of variables must be 1 (i.e,., chunk={variable: -1}.')

    arr_x_pad = dask.array.overlap.overlap(
        arr_x,
        depth=(0, 1, 1),
        boundary=nodata
    )

    arr_y_pad = dask.array.overlap.overlap(
        arr_y,
        depth=(0, 1, 1),
        boundary=nodata
    )

    x_delays = arr_x_pad.to_delayed().ravel()
    y_delays = arr_y_pad.to_delayed().ravel()

    # TODO: ensure band chunk size is 1 (i.e., -1)

    n_x_vars = arr_x.shape[0] * 9  # 9 pixels per window
    n_y_vars = arr_y.shape[0]

    total_size = arr_x.shape[1] * arr_x.shape[2]
    chunk_sizes = _get_xy_chunk_sizes(arr_x)

    arr_chunks = []
    for x_chunk, y_chunk, chunk_size in zip(x_delays, y_delays, chunk_sizes):

        n_sub_samples = round(n_samples * (chunk_size / total_size))

        chunk = dask.delayed(_extract_train_samples_numpy)(
            x_chunk,
            y_chunk,
            nodata=nodata,
            n_samples=n_sub_samples
        )

        arr_chunk = dask.array.from_delayed(
            chunk,
            shape=(np.nan, n_x_vars + n_y_vars),
            #shape=(n_sub_samples, n_x_vars + n_y_vars),  # TODO: use n_sub_samples instead of np.nan
            dtype=arr_x.dtype
        )

        arr_chunks.append(arr_chunk)

    arr_xy = dask.array.vstack(arr_chunks)

    return arr_xy


def extract_train_samples(
        arr_x: np.ndarray | dask.array.Array,
        arr_y: np.ndarray | dask.array.Array,
        nodata: int | float,
        n_samples: int | None
) -> tuple:

    is_x_lazy = _is_lazy(arr_x)
    is_y_lazy = _is_lazy(arr_y)

    if not is_x_lazy and not is_y_lazy:
        arr_xy = _extract_train_samples_numpy(
            arr_x,
            arr_y,
            nodata,
            n_samples
        )

    elif is_x_lazy and is_x_lazy:
        arr_xy = _extract_train_samples_dask(
            arr_x,
            arr_y,
            nodata,
            n_samples
        )

    else:
        raise TypeError('Inputs arr_x and arr_y must be numpy or dask arrays.')

    # unpack xy -> x, y. here, x are all vars up to n y vars.
    # FIXME: this ends up computing arr_xy twice... may need new logic to do the arr_x, arr_y sample separate.
    arr_x_out = arr_xy[:, :-arr_y.shape[0]]
    arr_y_out = arr_xy[:, -arr_y.shape[0]:]


    with ProgressBar():
        arr_x_out.compute()
        arr_y_out.compute()

    raise


    return arr_x_out, arr_y_out


def train_xgb_models(
        arr_x: np.ndarray | dask.array.Array,
        arr_y: np.ndarray | dask.array.Array,
        xgb_params: dict = None,
        xgb_client = None
) -> dict:

    if xgb_params is None:
        xgb_params = _default_xgb_params()

    xgb_nbr = xgb_params.pop('num_boost_round', None)
    if xgb_nbr is None:
        raise ValueError('XGBoost num_boost_round must be specified in xgb_params.')

    # TODO: evals
    # TODO: early_stopping_rounds

    is_x_lazy = _is_lazy(arr_x)
    is_y_lazy = _is_lazy(arr_y)

    if is_x_lazy and is_x_lazy and xgb is None:
        raise ValueError('Must specify xgb_client if using dask arrays.')

    models = {}
    if not is_x_lazy and not is_y_lazy:
        for i in range(arr_y.shape[1]):
            print(f'Training variable {i + 1}.')
            dtrain = xgb.DMatrix(arr_x, arr_y[:, i])
            models[i] = xgb.train(xgb_params, dtrain, xgb_nbr)

    elif is_x_lazy and is_x_lazy:
        for i in range(arr_y.shape[1]):
            print(f'Training variable {i + 1}.')
            dtrain = xgbd.DaskDMatrix(xgb_client, arr_x, arr_y[:, i])
            models[i] = xgbd.train(xgb_client, xgb_params, dtrain, xgb_nbr)

    else:
        raise TypeError('Inputs arr_x and arr_y must be numpy or dask arrays.')

    return models


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

    s = time.time()

    arr_x, arr_y = extract_train_samples(
        da_x.data,
        da_y.data,
        nodata,
        n_samples
    )

    if arr_x.size == 0 or arr_y.size == 0:
        raise ValueError('No training samples were returned.')

    models = train_xgb_models(
        arr_x,
        arr_y,
        xgb_params,
        xgb_client
    )

    e = time.time()
    print(e - s)

    raise


    # arr_i, arr_x = _extract_predict_samples_numpy(
    #     da_x.data,
    #     da_y.data,
    #     nodata
    # )

    # if arr_i.size == 0 or arr_x.size == 0:
    #     raise ValueError('Input data must have at least one prediction sample.')

    # arr_x = xgb.DMatrix(arr_x)
    # arr_y = []
    # for i, model in models.items():
    #     print(f'Predicting variable {i + 1}.')
    #     arr_y.append(model.predict(arr_x))
    #
    # arr_y = np.column_stack(arr_y)
    #
    # arr_y = _fill_via_predict_numpy(
    #     arr_i,
    #     arr_y,
    #     da_y.data
    # )
    #
    # da_out = xr.DataArray(
    #     arr_y,
    #     dims=da_y.dims,
    #     coords=da_y.coords,
    #     attrs=da_y.attrs
    # )

    #return da_out
    return
