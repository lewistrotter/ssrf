
# imports
import gc
import time

import dask
import numpy as np
import numba as nb
import xarray as xr
import xgboost as xgb


def _is_lazy(arr: np.ndarray) -> bool:

    if isinstance(arr, dask.array.Array):
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
) -> tuple:

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
        arr_y_out = np.empty((0, arr_y.shape[0]), arr_y.dtype)
        arr_x_out = np.empty((0, arr_x.shape[0] * 9), arr_x.dtype)  # 9 pix per win
        return arr_x_out, arr_y_out

    # note: i parallel safe as prange not used
    i = 0
    for yi in range(arr_x.shape[1]):
        for xi in range(arr_x.shape[2]):
            if arr_mask[yi, xi]:
                arr_idx[i, 0] = yi
                arr_idx[i, 1] = xi
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
        arr_y_out = np.empty((0, arr_y.shape[0]), arr_y.dtype)
        arr_x_out = np.empty((0, arr_x.shape[0] * 9), arr_x.dtype)  # 9 pix per win
        return arr_x_out, arr_y_out

    # extract real values
    n_idx = arr_idx.shape[0]
    arr_y_out = np.empty((n_idx, arr_y.shape[0]), arr_y.dtype)
    arr_x_out = np.empty((n_idx, arr_x.shape[0] * 9), arr_x.dtype)  # 9 pix per win

    for i in nb.prange(n_idx):
        ri, ci = arr_idx[i, 0], arr_idx[i, 1]

        arr_y_out[i, :] = arr_y[:, ri, ci]

        arr_x_sel = arr_x[:, ri - 1:ri + 1 + 1, ci - 1:ci + 1 + 1]
        arr_x_out[i, :] = arr_x_sel.ravel()

    return arr_x_out, arr_y_out


@nb.njit
def _extract_predict_samples_numpy(
        arr_x: np.ndarray,
        arr_y: np.ndarray,
        nodata: int | float
) -> tuple:

    arr_mask = np.zeros((arr_x.shape[1], arr_x.shape[2]), np.bool_)

    for yi in nb.prange(1, arr_x.shape[1] - 1):
        for xi in range(1, arr_x.shape[2] - 1):

            arr_y_sel = arr_y[:, yi, xi]
            if _is_all_nodata(arr_y_sel, nodata):

                arr_x_sel = arr_x[:, yi - 1:yi + 1 + 1, xi - 1:xi + 1 + 1]
                if not _is_any_nodata(arr_x_sel, nodata):
                    arr_mask[yi, xi] = True

    n_valid = np.sum(arr_mask)
    arr_idx = np.empty((n_valid, 2), np.int32)
    if arr_idx.size == 0:
        arr_x_out = np.empty((0, arr_x.shape[0] * 9), arr_x.dtype)  # 9 pix per win
        return arr_idx, arr_x_out

    # note: i parallel safe as prange not used
    i = 0
    for yi in range(arr_x.shape[1]):
        for xi in range(arr_x.shape[2]):
            if arr_mask[yi, xi]:
                arr_idx[i, 0] = yi
                arr_idx[i, 1] = xi
                i += 1

    if arr_idx.size == 0:
        arr_x_out = np.empty((0, arr_x.shape[0] * 9), arr_x.dtype)  # 9 pix per win
        return arr_idx, arr_x_out

    # extract real values
    n_idx = arr_idx.shape[0]
    arr_x_out = np.empty((n_idx, arr_x.shape[0] * 9), arr_x.dtype)  # 9 pix per win

    for i in nb.prange(n_idx):
        ri, ci = arr_idx[i, 0], arr_idx[i, 1]

        arr_x_sel = arr_x[:, ri - 1:ri + 1 + 1, ci - 1:ci + 1 + 1]
        arr_x_out[i, :] = arr_x_sel.ravel()

    return arr_idx, arr_x_out


@nb.njit
def _fill_via_predict_numpy(
        arr_i: np.ndarray,
        arr_y_pred: np.ndarray,
        arr_y_real: np.ndarray
) -> np.ndarray:

    arr_y_out = arr_y_real.copy()
    for i in nb.prange(arr_i.shape[0]):
        ri, ci = arr_i[i, 0], arr_i[i, 1]
        arr_y_out[:, ri, ci] = arr_y_pred[i, :]

    return arr_y_out


def run(
        da_x: xr.DataArray,
        da_y: xr.DataArray,
        nodata: int | float,
        n_samples: int | None,
        xgb_params: dict
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

    if nodata is None:
        raise ValueError('Input nodata value must be provided.')

    #is_x_lazy = _is_lazy(da_x.data)
    #is_y_lazy = _is_lazy(da_y.data)

    # if not is_x_lazy and not is_y_lazy:
    # ...

    arr_x, arr_y = _extract_train_samples_numpy(
        da_x.data,
        da_y.data,
        nodata,
        n_samples
    )

    models = {}
    for i in range(arr_y.shape[1]):
        print(f'Training variable {i + 1}.')
        dtrain = xgb.DMatrix(arr_x, arr_y[:, i])
        models[i] = xgb.train(
            params,
            dtrain,
            num_boost_round=100
        )

    del arr_y, arr_x
    gc.collect()

    arr_i, arr_x = _extract_predict_samples_numpy(
        da_x.data,
        da_y.data,
        nodata
    )

    arr_x = xgb.DMatrix(arr_x)
    arr_y = []
    for i, model in models.items():
        print(f'Predicting variable {i + 1}.')
        arr_y.append(model.predict(arr_x))

    arr_y = np.column_stack(arr_y)

    arr_y = _fill_via_predict_numpy(
        arr_i,
        arr_y,
        da_y.data
    )

    da_out = xr.DataArray(
        arr_y,
        dims=da_y.dims,
        coords=da_y.coords,
        attrs=da_y.attrs
    )

    return da_out
