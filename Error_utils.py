import tiledb
from config import *
from tqdm import tqdm
import numpy as np
import torch


def populateErrorArray(max_error, min_error, data_avg, do_quant):

    with tiledb.open("arrayD", "r") as arrayD, tiledb.open("arrayUr", "r") as arrayUr, \
            tiledb.open("arrayVr", "r") as arrayVr, tiledb.open("arraySr", "r") as arraySr, \
            tiledb.open("arrayE","w") as arrayE:

        T_length, Y_length, X_length = SVD_size
        SVD_count = 0

        # populating arrayE
        for t, y, x in tqdm(np.ndindex(num_images // T_length, image_height // Y_length, image_width // X_length),
                            total=num_images//T_length * image_height // Y_length * image_width // X_length):

            y_index = y * Y_length
            x_index = x * X_length
            t_index = t * T_length

            U, S, V = arrayUr[SVD_count]["U"], arraySr[SVD_count]["S"].reshape(-1, k), arrayVr[SVD_count]["V"]

            U = torch.tensor(U, dtype=torch.float, device="cuda")
            S = torch.tensor(S, dtype=torch.float, device="cuda")
            V = torch.tensor(V, dtype=torch.float, device="cuda")

            di_prime = (U * S @ V) + data_avg

            di = arrayD[
                 t_index:t_index + T_length, y_index:y_index + Y_length, x_index:x_index + X_length
                 ]["Temperatures"].reshape(SVD_shape)

            di = torch.tensor(di, dtype=torch.float, device="cuda")

            distances = di - di_prime
            problem_indices = torch.abs(distances).cpu().numpy() / vRange > eps

            rows, cols = np.nonzero(problem_indices)
            length_array = [SVD_count] * len(rows)

            if do_quant:
                normalized_errors = (distances - min_error) / (max_error - min_error)

                normalized_errors *= 255
                normalized_errors = normalized_errors.cpu().numpy().astype(np.uint8)

            else:
                normalized_errors = distances.cpu().numpy()

            arrayE[length_array, rows, cols] = normalized_errors[rows, cols]

            SVD_count += 1


def reconstructAndCheck(max_error, min_error, data_avg, do_quant):

    with tiledb.open("arrayD", "r") as arrayD, tiledb.open("arrayUr", "r") as arrayUr, \
            tiledb.open("arrayVr", "r") as arrayVr, tiledb.open("arraySr", "r") as arraySr, \
            tiledb.open("arrayE","r") as arrayE:

        T_length, Y_length, X_length = SVD_size
        SVD_count = 0

        # populating arrayE_hat
        for t, y, x in tqdm(np.ndindex(num_images // T_length, image_height // Y_length, image_width // X_length),
                            total=num_images//T_length * image_height // Y_length * image_width // X_length):

            y_index = y * Y_length
            x_index = x * X_length
            t_index = t * T_length

            U, S, V = arrayUr[SVD_count]["U"], arraySr[SVD_count]["S"].reshape(-1, k), arrayVr[SVD_count]["V"]

            di_prime = (U * S @ V) + data_avg

            di = arrayD[
                 t_index:t_index + T_length, y_index:y_index + Y_length, x_index:x_index + X_length
                 ]["Temperatures"].reshape(SVD_shape)

            error_data = arrayE[SVD_count]

            if do_quant:
                ei = error_data["E"].astype(np.float32) / 255.0 * (max_error - min_error) + min_error
            else:
                ei = error_data["E"]

            indices = error_data["width"], error_data["length"]
            di_prime_plus_ei = di_prime.copy()

            for row, col, value in zip(indices[0], indices[1], ei):
                di_prime_plus_ei[row, col] += value

            distances = di - di_prime_plus_ei
            assert np.all(distances / vRange <= eps), "Error correction failed"

            SVD_count += 1

        print("Check successful")
