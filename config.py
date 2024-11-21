num_images = 4000
image_height, image_width = 855, 1215
num_values = num_images * image_height * image_width

# Define the tile size
tile_height, tile_width = 1, 1215
tile_thickness = 400

k = 40
eps = 1e-2

SVD_size = (num_images, 1, image_width)
SVD_shape = (SVD_size[0], SVD_size[2])
num_SVD_matrices = num_images * image_height * image_width / (SVD_size[0] * SVD_size[1] * SVD_size[2])
