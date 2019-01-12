#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include "cnn_utils.h"

kernel *load_single_kernel(char *data_set, int size, int previous_filters, int current_filters);

float* load_bias(int data_set_number);

float **load_biases();

kernel **load_kernels();

kernel *load_kernel_by_number(int number);

float *load_batch_normalization_beta(int n);

float *load_batch_normalization_gamma(int n);

float *load_batch_normalization_means(int n);

float *load_batch_normalization_variances(int n);

