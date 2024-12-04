num_images = 4000
image_height, image_width = 855, 1215
num_values = num_images * image_height * image_width

# Define the tile size
tile_height, tile_width = 855, 5
tile_thickness = 1000

k = 70
eps = 1e-2

SVD_size = (num_images, image_height, 5)
SVD_shape = (SVD_size[0], SVD_size[1] * SVD_size[2])
num_SVD_matrices = num_images * image_height * image_width / (SVD_size[0] * SVD_size[1] * SVD_size[2])
