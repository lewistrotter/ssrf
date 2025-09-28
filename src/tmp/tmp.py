
import itertools
import dask
from dask.array.overlap import overlap
from dask.distributed import Client

@nb.njit
def _extract_train_samples_numpy_old(
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

    arr_x_pad = overlap(
        arr_x,
        depth=(0, 1, 1),
        boundary=nodata
    )

    arr_y_pad = overlap(
        arr_y,
        depth=(0, 1, 1),
        boundary=nodata
    )

    arr_x_delay = arr_x_pad.to_delayed().ravel()
    arr_y_delay = arr_y_pad.to_delayed().ravel()

    # TODO: ensure band chunk size is 1 (i.e., -1)

    _, y_chunks, x_chunks = arr_x.chunks
    total_size = arr_x.shape[1] * arr_x.shape[2]

    n_samples_per_chunk = []
    for y_chunk, x_chunk in itertools.product(y_chunks, x_chunks):
        chunk_size = y_chunk * x_chunk
        n_sub_samples = n_samples * (chunk_size / total_size)
        n_samples_per_chunk.append(int(round(n_sub_samples)))

    x_data = []
    y_data = []
    for x, y, c in list(zip(arr_x_delay, arr_y_delay, n_samples_per_chunk)):
        block = dask.delayed(_extract_train_samples_numpy)(
            x,
            y,
            nodata=nodata,
            n_samples=c
        )

        # block returns (X_chunk, y_chunk)
        #X_delayed = dask.array.from_delayed(block[0], shape=(c, 90), dtype=np.float32)  # TODO: dynamic size
        #y_delayed = dask.array.from_delayed(block[1], shape=(c, 10), dtype=np.float32)  # TODO: dynamic size

        x_delayed = dask.array.from_delayed(
            dask.delayed(lambda b: b[0])(block),
            shape=(c, 90),  # TODO: dynamic size
            dtype=arr_x.dtype, #np.float32
        )

        y_delayed = dask.array.from_delayed(
            dask.delayed(lambda b: b[1])(block),
            shape=(c, 10),
            dtype=arr_y.dtype, #np.float32
        )

        # TODO: check if .rechunk({0: 10000}) fixes mem issues here.
        x_delayed = x_delayed.rechunk({0: 10000})
        y_delayed = y_delayed.rechunk({0: 10000})

        x_data.append(x_delayed)
        y_data.append(y_delayed)

    arr_x_out = dask.array.concatenate(x_data, axis=0)
    arr_y_out = dask.array.concatenate(y_data, axis=0)

    return arr_x_out, arr_y_out




# elif is_x_lazy and is_y_lazy:
#     models = {}
#     for i in range(arr_y.shape[1]):
#         print(f'Training variable {i + 1}.')
#
#
#         arr_x, arr_y = arr_x.persist(), arr_y.persist()
#
#         print('whoa')
#
#         dtrain = xgbd.DaskDMatrix(client, arr_x, arr_y[:, i])
#         models[i] = xgbd.train(
#             client,
#             params,
#             dtrain,
#             num_boost_round=100
#         )





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







def _benroulli_sampling(
        arr_mask: np.ndarray,
        n_samples: int | float
) -> np.ndarray:

    n_true = arr_mask.sum()
    m_int = np.iinfo(np.int32).max
    thresh = (n_samples * m_int / n_true).astype(np.int32)

    arr_rnd = np.random.randint(
        0,
        m_int,
        size=arr_mask.shape,
        dtype=np.int32
    )

    arr_out = arr_mask & (arr_rnd < thresh)

    return arr_out


def _benroulli_sampling_dask(
        arr_mask: np.ndarray,
        n_samples: int | float,
        block_info = None
) -> np.ndarray:

    chunk_size = np.prod(arr_mask.shape)
    total_size = np.prod(block_info[None]['shape'])
    n_sub_sample = int(round(chunk_size / total_size * n_samples))

    if n_sub_sample == 0:
        return arr_mask

    arr_out = _benroulli_sampling(
        arr_mask,
        n_sub_sample
    )

    return arr_out


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
    if n_samples == 0:
        return np.empty((0, n_x_vars + n_y_vars), arr_y.dtype)

    arr_idx = np.empty((n_valid, 2), np.int32)

    i = 0  # note: i parallel safe as prange not used
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
            # shape=(n_sub_samples, n_x_vars + n_y_vars),  # TODO: use n_sub_samples instead of np.nan
            dtype=arr_x.dtype
        )

        arr_chunks.append(arr_chunk)

    arr_xy = dask.array.vstack(arr_chunks)

    return arr_xy












def train_xgb_models(
        arr_x: np.ndarray | dask.array.Array,
        arr_y: np.ndarray | dask.array.Array,
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



@nb.njit
def _build_predict_grid(
        arr_x: np.ndarray,
        arr_y: np.ndarray
) -> np.ndarray:
    arr_out = np.zeros((arr_x.shape[0], arr_x.shape[1]), np.bool)

    for yi in nb.prange(1, arr_x.shape[0] - 1):
        for xi in range(1, arr_x.shape[1] - 1):

            if arr_y[yi, xi]:  # true == nodata was present
                arr_x_sel = arr_x[yi - 1:yi + 1 + 1, xi - 1:xi + 1 + 1]
                if not _is_any_nodata(arr_x_sel, True):
                    arr_out[yi, xi] = True

    return arr_out





def _random_sampling_dask(
        arr_mask: np.ndarray,
        n_samples: int | float,
        block_info=None
) -> np.ndarray:
    if not arr_mask.any():
        return arr_mask

    chunk_size = np.prod(arr_mask.shape)
    total_size = np.prod(block_info[None]['shape'])
    n_sub_sample = int(chunk_size / total_size * n_samples)

    arr_out = _random_sampling(
        arr_mask,
        n_sub_sample
    )

    return arr_out


@nb.njit
def _extract_x_train_set(
        arr_x: np.ndarray,
        arr_mask: np.ndarray
) -> np.ndarray:
    n_vars = arr_x.shape[0] * 9  # 9 pixels per window
    n_samples = arr_mask[1:-1, 1:-1].sum()  # exclude edges for valid sample size

    if n_samples == 0:
        return np.empty((0, n_vars), arr_x.dtype)

    arr_out = np.empty((n_samples, n_vars), arr_x.dtype)

    i = 0  # TODO: not parallel, can we fix?
    for yi in range(1, arr_mask.shape[0] - 1):
        for xi in range(1, arr_mask.shape[1] - 1):
            if arr_mask[yi, xi]:
                arr_x_sel = arr_x[:, yi - 1:yi + 1 + 1, xi - 1:xi + 1 + 1]
                arr_out[i, :] = arr_x_sel.ravel()
                i += 1

    return arr_out


@nb.njit
def _extract_y_train_set(
        arr_y: np.ndarray,
        arr_mask: np.ndarray
) -> np.ndarray:
    n_vars = arr_y.shape[0]
    n_samples = arr_mask[1:-1, 1:-1].sum()  # exclude edges for valid sample size

    if n_samples == 0:
        return np.empty((0, n_vars), arr_y.dtype)

    arr_out = np.empty((n_samples, n_vars), arr_y.dtype)

    i = 0  # TODO: not parallel, can we fix?
    for yi in range(1, arr_mask.shape[0] - 1):
        for xi in range(1, arr_mask.shape[1] - 1):
            if arr_mask[yi, xi]:
                arr_out[i, :] = arr_y[:, yi, xi]
                i += 1

    return arr_out