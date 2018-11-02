#include "model_loader.h"
#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include "cnn_utils.h"

kernel *load_single_kernel(char *data_set, int size, int previous_filters, int current_filters);

void add_bias(kernel *K, char *data_set);

kernel **load_kernels();