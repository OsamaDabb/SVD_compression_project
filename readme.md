# SVD Compression ods Redsea re-analysis data

## Installation
After downloading the project, ensure that the 

> Redsea_t2_4k_gan.dat file is in the root directory under that name.

All dependencies can be found/installed using the requirements.txt file. 


## Running
All necessary configuration parameters can be modified in config.py,
controlling expected dataset dimensions, SVD dimensionality, rank reduction size k, and 
desired error threshold eps (epsilon).

From there, the program is run by opening experiments.ipynb and running all cells sequentially, or simply selecting 

> run all

in your jupyter IDE. The results will be stored in the directory as the arrays
arrayD, arrayUr, arraySr, arrayVr, arrayE and  the reconstructed arrayD_prime which satisfies the error threshold.

## Results

achieves a best-case compression ratio of $\rho = 17.8$ for $\eps = 10^{-2}$