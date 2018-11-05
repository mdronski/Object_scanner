#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include "cnn_utils.h"

kernel *load_single_kernel(char *data_set, int size, int previous_filters, int current_filters);

double* load_bias(char *data_set);

double **load_biases();

kernel **load_kernels();

kernel *load_kernel_by_number(int number);
