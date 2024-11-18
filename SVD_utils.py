from DB_utils import *
from tqdm import tqdm
import random
import torch
from config import *


def SVD_experiment(SVD_size, SVD_shape, k, test_percent, epsilon):

    with tiledb.open("arrayD", "r") as arrayD:

        T_length, Y_length, X_length = SVD_size

        vRange = 80

        epsilon_necessary = 0
        total_vals = 0

        assert num_images % T_length == 0 and image_height % Y_length == 0 and image_width % X_length == 0

        for t, y, x in tqdm(np.ndindex(num_images // T_length, image_height // Y_length, image_width // X_length),
                            total=num_images//T_length * image_height // Y_length * image_width // X_length):

            if random.random() > test_percent:

                continue

            y_index = y * Y_length
            x_index = x * X_length
            t_index = t * T_length
            temp = arrayD[
                   t_index:t_index + T_length, y_index:y_index + Y_length, x_index:x_index + X_length
                   ]["Temperatures"].reshape(SVD_shape)

            temp_gpu = torch.tensor(temp, device="cuda", dtype=torch.float32)  # Move the data to GPU

            # Perform SVD on the GPU
            U, S, V = torch.svd(temp_gpu)

            epsilon_necessary += torch.sum(
                torch.abs(temp_gpu - U[:,0:k] * S[0:k] @ V[:, 0:k].t())
                / vRange
                > epsilon).item()

            total_vals += T_length * Y_length * X_length

            # U, S, V = U.cpu().numpy(), S.cpu().numpy(), V.cpu().numpy()

    print(epsilon_necessary / total_vals)
    print(f"Number of parameters {k * sum(SVD_shape)} down from {T_length * Y_length * X_length}")
    print(f"Ratio of {k * sum(SVD_shape) / (T_length * Y_length * X_length) + epsilon_necessary / total_vals}")


def compressData(data_avg, vRange):

    # open required arrays
    with tiledb.open("arrayD", "r") as arrayD, tiledb.open("arrayUr", "w") as arrayUr, \
            tiledb.open("arrayVr", "w") as arrayVr, tiledb.open("arraySr", "w") as arraySr:

        T_length, Y_length, X_length = SVD_size

        SVD_entry_count = 0
        max_error, min_error, percent_error = 0, 0, 0

        assert num_images % T_length == 0 and image_height % Y_length == 0 and image_width % X_length == 0

        for t, y, x in tqdm(np.ndindex(num_images // T_length, image_height // Y_length, image_width // X_length),
                            total=num_images//T_length * image_height // Y_length * image_width // X_length):

            # loading data from D
            y_index = y * Y_length
            x_index = x * X_length
            t_index = t * T_length

            temp = arrayD[
                   t_index:t_index + T_length, y_index:y_index + Y_length, x_index:x_index + X_length
                ]["Temperatures"].reshape(SVD_shape)

            temp_gpu = torch.tensor(temp, device="cuda", dtype=torch.float32)  # Move the data to GPU

            # center data around 0 for SVD quality
            temp_gpu -= data_avg

            # Perform SVD on the GPU
            U, S, V = torch.svd(temp_gpu)

            # calculate these values to verify validity of quant later
            error_margin = temp_gpu - U[:, :k] * S[:k] @ V.t()[:k]
            max_error = max(max_error, torch.max(error_margin).item())
            min_error = min(min_error, torch.min(error_margin).item())
            percent_error += torch.sum(torch.abs(error_margin) / vRange > eps).item()

            # load USV to their arrays
            U, S, V = U[:, :k].cpu().numpy().reshape(-1, k), S[:k].cpu().numpy().reshape(k, 1), V[:, :k].t().cpu().numpy().reshape(k, -1)

            arrayUr[SVD_entry_count] = U
            arraySr[SVD_entry_count] = S
            arrayVr[SVD_entry_count] = V

            SVD_entry_count += 1

    percent_error /= num_values

    return max_error, min_error, percent_error
