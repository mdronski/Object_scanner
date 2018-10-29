#include "model_loader.h"
#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <cnn_utils.h>

#define FILE "yolotiny.h5"


int kernel_sizes[13][4] = {
        {3, 3, 3, 16}
        { 3, 3, 16, 32 }
        { 3, 3, 32, 64 }
        { 3, 3, 64, 128 }
        { 3, 3, 128, 256 }
        { 3, 3, 256, 512 }
        { 3, 3, 512, 1024 }
        { 1, 1, 1024, 256 }
        { 3, 3, 256, 512 }
        { 1, 1, 512, 255 }
        { 1, 1, 256, 128 }
        { 3, 3, 384, 256 }
        { 1, 1, 256, 255 }
};

char kernels_data_sets[13] = {
        "/model_weights/conv2d_1/conv2d_1/kernel:0",
        "/model_weights/conv2d_2/conv2d_2/kernel:0",
        "/model_weights/conv2d_3/conv2d_3/kernel:0",
        "/model_weights/conv2d_4/conv2d_4/kernel:0",
        "/model_weights/conv2d_5/conv2d_5/kernel:0",
        "/model_weights/conv2d_6/conv2d_6/kernel:0",
        "/model_weights/conv2d_7/conv2d_7/kernel:0",
        "/model_weights/conv2d_8/conv2d_8/kernel:0",
        "/model_weights/conv2d_9/conv2d_9/kernel:0",
        "/model_weights/conv2d_10/conv2d_10/kernel:0",
        "/model_weights/conv2d_11/conv2d_11/kernel:0",
        "/model_weights/conv2d_12/conv2d_12/kernel:0",
        "/model_weights/conv2d_13/conv2d_13/kernel:0"
};

char bias_data_sets[2] = {
        "/model_weights/conv2d_10/conv2d_10/bias:0",
        "/model_weights/conv2d_13/conv2d_13/bias:0"
}

kernel *load_single_kernel(char *data_set, int size, int previous_filters, int current_filters){
    hid_t file_id, dataset_id;  /* identifiers */
    herr_t status;
    kernel *K = allocate_kernel(size, previous_filters, current_filters);

    /* Open an existing file. */
    file_id = H5Fopen(FILE, H5F_ACC_RDWR, H5P_DEFAULT);

    /* Open an existing dataset. */
    dataset_id = H5Dopen2(file_id, data_set, H5P_DEFAULT);

    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     K->weights);


    /* Close the dataset. */
    status = H5Dclose(dataset_id);
    /* Close the file. */
    status = H5Fclose(file_id);

    return K;
}

void add_bias(kernel *K, char *data_set){
    double bias[255];

    hid_t file_id, dataset_id;  /* identifiers */
    herr_t status;



    /* Open an existing file. */
    file_id = H5Fopen(FILE, H5F_ACC_RDWR, H5P_DEFAULT);
    /* Open an existing dataset. */
    dataset_id = H5Dopen2(file_id, data_set, H5P_DEFAULT);

    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     bias);


    for (int f = 0; f < 255; ++f) {
        for (int l = 0; l < 256; ++l) {
            K->weights[f][l][0][0] += bias[f];
        }
    }


    /* Close the dataset. */
    status = H5Dclose(dataset_id);

    /* Close the file. */
    status = H5Fclose(file_id);

}


kernel **load_kernels() {
    kernel **kernels = malloc(13 * sizeof(kernel * ));

    for (int i = 0; i < 13; ++i) {
        kernel_sizes[i] = load_single_kernel(kernels_data_sets[i], kernel_sizes[0], kernel_sizes[2],
                                             kernel_sizes[3])
    }

    add_bias(kernels[9], bias_data_sets[0]);
    add_bias(kernels[12], bias_data_sets[1]);

    return kernels;
}