//#include "model_loader.h"
#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include "cnn_utils.h"
#define FILE_NAME "yolotiny.h5"


int kernel_sizes[13][4] = {
        { 3, 3, 3, 16 },
        { 3, 3, 16, 32 },
        { 3, 3, 32, 64 },
        { 3, 3, 64, 128 },
        { 3, 3, 128, 256 },
        { 3, 3, 256, 512 },
        { 3, 3, 512, 1024 },
        { 1, 1, 1024, 256 },
        { 3, 3, 256, 512 },
        { 1, 1, 512, 255 },
        { 1, 1, 256, 128 },
        { 3, 3, 384, 256 },
        { 1, 1, 256, 255 }
};

char *kernels_data_sets[13] = {
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

char *bias_data_sets[2] = {
        "/model_weights/conv2d_10/conv2d_10/bias:0",
        "/model_weights/conv2d_13/conv2d_13/bias:0"
};

char *batch_normalization_betas[11] = {
        "/model_weights/batch_normalization_1/batch_normalization_1/beta:0",
        "/model_weights/batch_normalization_2/batch_normalization_2/beta:0",
        "/model_weights/batch_normalization_3/batch_normalization_3/beta:0",
        "/model_weights/batch_normalization_4/batch_normalization_4/beta:0",
        "/model_weights/batch_normalization_5/batch_normalization_5/beta:0",
        "/model_weights/batch_normalization_6/batch_normalization_6/beta:0",
        "/model_weights/batch_normalization_7/batch_normalization_7/beta:0",
        "/model_weights/batch_normalization_8/batch_normalization_8/beta:0",
        "/model_weights/batch_normalization_9/batch_normalization_9/beta:0",
        "/model_weights/batch_normalization_10/batch_normalization_10/beta:0",
        "/model_weights/batch_normalization_11/batch_normalization_11/beta:0",
};

char *batch_normalization_gammas[11] = {
        "/model_weights/batch_normalization_1/batch_normalization_1/gamma:0",
        "/model_weights/batch_normalization_2/batch_normalization_2/gamma:0",
        "/model_weights/batch_normalization_3/batch_normalization_3/gamma:0",
        "/model_weights/batch_normalization_4/batch_normalization_4/gamma:0",
        "/model_weights/batch_normalization_5/batch_normalization_5/gamma:0",
        "/model_weights/batch_normalization_6/batch_normalization_6/gamma:0",
        "/model_weights/batch_normalization_7/batch_normalization_7/gamma:0",
        "/model_weights/batch_normalization_8/batch_normalization_8/gamma:0",
        "/model_weights/batch_normalization_9/batch_normalization_9/gamma:0",
        "/model_weights/batch_normalization_10/batch_normalization_10/gamma:0",
        "/model_weights/batch_normalization_11/batch_normalization_11/gamma:0",
};

char *batch_normalization_means[11] = {
        "/model_weights/batch_normalization_1/batch_normalization_1/moving_mean:0",
        "/model_weights/batch_normalization_2/batch_normalization_2/moving_mean:0",
        "/model_weights/batch_normalization_3/batch_normalization_3/moving_mean:0",
        "/model_weights/batch_normalization_4/batch_normalization_4/moving_mean:0",
        "/model_weights/batch_normalization_5/batch_normalization_5/moving_mean:0",
        "/model_weights/batch_normalization_6/batch_normalization_6/moving_mean:0",
        "/model_weights/batch_normalization_7/batch_normalization_7/moving_mean:0",
        "/model_weights/batch_normalization_8/batch_normalization_8/moving_mean:0",
        "/model_weights/batch_normalization_9/batch_normalization_9/moving_mean:0",
        "/model_weights/batch_normalization_10/batch_normalization_10/moving_mean:0",
        "/model_weights/batch_normalization_11/batch_normalization_11/moving_mean:0",
};

char *batch_normalization_variances[11] = {
        "/model_weights/batch_normalization_1/batch_normalization_1/moving_variance:0",
        "/model_weights/batch_normalization_2/batch_normalization_2/moving_variance:0",
        "/model_weights/batch_normalization_3/batch_normalization_3/moving_variance:0",
        "/model_weights/batch_normalization_4/batch_normalization_4/moving_variance:0",
        "/model_weights/batch_normalization_5/batch_normalization_5/moving_variance:0",
        "/model_weights/batch_normalization_6/batch_normalization_6/moving_variance:0",
        "/model_weights/batch_normalization_7/batch_normalization_7/moving_variance:0",
        "/model_weights/batch_normalization_8/batch_normalization_8/moving_variance:0",
        "/model_weights/batch_normalization_9/batch_normalization_9/moving_variance:0",
        "/model_weights/batch_normalization_10/batch_normalization_10/moving_variance:0",
        "/model_weights/batch_normalization_11/batch_normalization_11/moving_variance:0",
};


float *load_batch_normalization_means(int n){

    float *means = malloc(kernel_sizes[n][3] * sizeof(float) * 124);

    hid_t file_id, dataset_id;  /* identifiers */
    herr_t status;
    /* Open an existing file. */
    file_id = H5Fopen(FILE_NAME, H5F_ACC_RDWR, H5P_DEFAULT);
    /* Open an existing dataset. */
    dataset_id = H5Dopen2(file_id, batch_normalization_means[n], H5P_DEFAULT);

    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     means);


    /* Close the dataset. */
    status = H5Dclose(dataset_id);
    /* Close the file. */
    status = H5Fclose(file_id);

//    printf("Means =\n");
//    for (int i = 0; i < kernel_sizes[n][3]; ++i) {
//        printf("%lf ", means[i]);
//    }
//    printf("\n\n");

    return means;
}


float *load_batch_normalization_variances(int n){


    float *variances = malloc(kernel_sizes[n][3] * sizeof(float) * 124);

    hid_t file_id, dataset_id;  /* identifiers */
    herr_t status;
    /* Open an existing file. */
    file_id = H5Fopen(FILE_NAME, H5F_ACC_RDWR, H5P_DEFAULT);
    /* Open an existing dataset. */
    dataset_id = H5Dopen2(file_id, batch_normalization_variances[n], H5P_DEFAULT);

    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     variances);


    /* Close the dataset. */
    status = H5Dclose(dataset_id);
    /* Close the file. */
    status = H5Fclose(file_id);
////
//    printf("Variances =\n");
//    for (int i = 0; i < kernel_sizes[n][3]; ++i) {
//        printf("%lf ", variances[i]);
//    }
//    printf("\n\n");

    return variances;
}



float *load_batch_normalization_beta(int n){

    float *beta = malloc(kernel_sizes[n][3] * sizeof(float) * 124);

    hid_t file_id, dataset_id;  /* identifiers */
    herr_t status;
    /* Open an existing file. */
    file_id = H5Fopen(FILE_NAME, H5F_ACC_RDONLY, H5P_DEFAULT);
    /* Open an existing dataset. */
    dataset_id = H5Dopen2(file_id, batch_normalization_betas[n], H5P_DEFAULT);

    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     beta);



    /* Close the dataset. */
    status = H5Dclose(dataset_id);

    /* Close the file. */
    status = H5Fclose(file_id);


//    printf("Betas =\n");
//    for (int i = 0; i < kernel_sizes[n][3]; ++i) {
//        printf("%lf ", beta[i]);
//    }
//    printf("\n\n");

    return beta;
}


float *load_batch_normalization_gamma(int n){

    float *gamma = malloc(kernel_sizes[n][3] * sizeof(float) * 124);

    hid_t file_id, dataset_id;  /* identifiers */
    herr_t status;
    /* Open an existing file. */
    file_id = H5Fopen(FILE_NAME, H5F_ACC_RDWR, H5P_DEFAULT);
    /* Open an existing dataset. */
    dataset_id = H5Dopen2(file_id, batch_normalization_gammas[n], H5P_DEFAULT);

    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     gamma);

    /* Close the dataset. */
    status = H5Dclose(dataset_id);
    /* Close the file. */
    status = H5Fclose(file_id);

//    printf("Gammas =\n");
//    for (int i = 0; i < kernel_sizes[n][3]; ++i) {
//        printf("%lf ", gamma[i]);
//    }
//    printf("\n\n");

    return gamma;
}


kernel *load_single_kernel(char *data_set, int size, int n_layers, int n_filters) {

    float *array = malloc(size*size*n_layers*n_filters* sizeof(float));

    hid_t file_id, dataset_id;  /* identifiers */
    herr_t status;
    kernel *K = allocate_kernel(size, n_layers, n_filters);
    /* Open an existing file. */
    file_id = H5Fopen(FILE_NAME, H5F_ACC_RDWR, H5P_DEFAULT);
    /* Open an existing dataset. */
    dataset_id = H5Dopen2(file_id, data_set, H5P_DEFAULT);

    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     array);

    for (int h = 0; h < size; ++h) {
        for (int w = 0; w < size; ++w) {
            for (int l = 0; l < n_layers; ++l) {
                for (int f = 0; f < n_filters; ++f) {
                    K->weights[f][l][h][w] =
                            array[h*size*n_layers*n_filters
                            + w*n_layers*n_filters
                            + l*n_filters
                            + f];
                }
            }
        }
    }

    free(array);

    /* Close the dataset. */
    status = H5Dclose(dataset_id);
    /* Close the file. */
    status = H5Fclose(file_id);

    return K;
}



float* load_bias(int n) {
    float *bias = malloc(255 * sizeof(float));

    hid_t file_id, dataset_id;  /* identifiers */
    herr_t status;


    /* Open an existing file. */
    file_id = H5Fopen(FILE_NAME, H5F_ACC_RDWR, H5P_DEFAULT);
    /* Open an existing dataset. */
    dataset_id = H5Dopen2(file_id, bias_data_sets[n], H5P_DEFAULT);

    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     bias);


    /* Close the dataset. */
    status = H5Dclose(dataset_id);

    /* Close the file. */
    status = H5Fclose(file_id);

    return bias;
}

float **load_biases(){
    float **biases = malloc(2 * sizeof(float*));

    biases[0] = load_bias(0);
    biases[1] = load_bias(1);

    return biases;
}


kernel **load_kernels() {
    kernel **kernels = malloc(5 * sizeof(kernel *));

    for (int i = 0; i < 5; ++i) {
        kernels[i] = load_single_kernel(kernels_data_sets[i], kernel_sizes[i][0], kernel_sizes[i][2],
                                             kernel_sizes[i][3]);
        fprintf(stderr, "%d\n", i);
    }

    return kernels;
}

kernel *load_kernel_by_number(int number){
    return load_single_kernel(kernels_data_sets[number], kernel_sizes[number][0], kernel_sizes[number][2],
                              kernel_sizes[number][3]);
}

//int main() {
////    printf("xd\n");
////    printf("%s\n%d %d %d\n", kernels_data_sets[0], kernel_sizes[0][0], kernel_sizes[0][2], kernel_sizes[0][3]);
////    kernel * K = load_single_kernel(kernels_data_sets[0], kernel_sizes[0][0], kernel_sizes[0][2], kernel_sizes[0][3]);
////    printf("xd\n");
////
////    printf("%d %d %d\n", K->size, K->n_layers, K->n_filters);
//////    for (int i = 0; i < 64; ++i) {
//////        printf("%lf ", K->weights[0][0][0][i]);
//////    }
////    print_kernel(K);
//
//    kernel *K = load_single_kernel(kernels_data_sets[6], 3, 512, 1024);
//    print_kernel(K);
//    free_kernel(K);
//
//    return 0;
//}