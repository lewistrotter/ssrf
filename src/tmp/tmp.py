
import itertools
import dask
from dask.array.overlap import overlap
from dask.distributed import Client

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