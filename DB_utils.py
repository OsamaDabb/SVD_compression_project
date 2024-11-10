import tiledb
import numpy as np
import os
import shutil
from config import *


def createDB(name, attribute_name, db_shape, tile_shape, sparse=False, data_type=np.float32):

    # Number of images and image dimensions
    height, width, length = db_shape
    tile_height, tile_width, tile_thickness = tile_shape

    # Define the array domain with tiling
    domain = tiledb.Domain(
        tiledb.Dim(name="height", domain=(0, height - 1), tile=tile_height, dtype=np.int32),
        tiledb.Dim(name="width", domain=(0, width - 1), tile=tile_width, dtype=np.int32),
        tiledb.Dim(name="length", domain=(0, length - 1), tile=tile_thickness, dtype=np.int32)
    )

    # Define the array attribute for storing pixel values (float32)
    attribute = tiledb.Attr(name=attribute_name, dtype=data_type)

    # Create the schema for a dense array
    schema = tiledb.ArraySchema(
        domain=domain,
        attrs=[attribute],
        sparse=sparse  # Dense array
    )

    # Create the TileDB array
    if not sparse:
        tiledb.DenseArray.create(name, schema)

    else:
        tiledb.SparseArray.create(name, schema)


def populateArrayD():

    chunksize = image_height * image_width * tile_thickness

    max_val, min_val = 0, 10_000 # just defaults
    data_avg = 0

    with tiledb.DenseArray("arrayD", mode="w") as array:

        for i in range(num_images // tile_thickness):

            offset = chunksize * i * 4

            image = np.fromfile(
                "Redsea_t2_4k_gan.dat", count=chunksize, offset=offset, dtype=np.float32
            ).reshape(tile_thickness, image_height, image_width)

            max_val = max(np.max(image), max_val)
            min_val = min(np.min(image), min_val)

            data_avg += np.mean(image) / (num_images//tile_thickness)

            array[i*tile_thickness:i*tile_thickness+tile_thickness] = image  # Write each image into the array

    return max_val, min_val, data_avg


def createConfig():

    cfg = tiledb.Ctx().config()
    cfg.update(
        {
            'py.init_buffer_bytes': 1024**2 * 50
        }
    )
    tiledb.default_ctx(cfg)


def deleteArrays(D):

    for file in os.listdir():

        if "array" in file:

            if "arrayD" in file and D is False:

                continue

            shutil.rmtree(file)


def getDirectorySize(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def getCompressionRatio():

    arrayD_size = getDirectorySize("arrayD")
    other_arrays_size = getDirectorySize("arrayUr") + getDirectorySize("arraySr") + getDirectorySize("arrayVr")
    error_array_size = getDirectorySize("arrayE")
    print(arrayD_size)
    print(other_arrays_size, error_array_size)

    print(f"Compression ratio p: {arrayD_size / (error_array_size + other_arrays_size)}")
    return arrayD_size / (error_array_size + other_arrays_size)
