num_images = 4000
image_height, image_width = 855, 1215
num_values = num_images * image_height * image_width

# Define the tile size
tile_height, tile_width = 855, 1
tile_thickness = 400

k = 400
eps = 1e-3

SVD_size = (num_images, image_height, 1)
SVD_shape = (SVD_size[0], SVD_size[1])
num_SVD_matrices = num_images * image_height * image_width / (SVD_size[0] * SVD_size[1] * SVD_size[2])
