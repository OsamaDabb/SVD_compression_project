num_images = 4000
image_height, image_width = 855, 1215
num_values = num_images * image_height * image_width

# Define the tile size
tile_height, tile_width = 855, 1
tile_thickness = 400
min_val = 225.5895
max_val = 310.55
data_avg = 292.37
vRange = max_val - min_val

k = 20
eps = 1e-2

SVD_size = (num_images, image_height, 1)
SVD_shape = (SVD_size[0], SVD_size[1])
num_SVD_matrices = num_images * image_height * image_width / (SVD_size[0] * SVD_size[1] * SVD_size[2])
