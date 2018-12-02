#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include "model_loader.c"


conv_layer *load_image(){
    char buffer[256];
    size_t len = 256;
    FILE *f = fopen("out.ppm", "r");
    getline(&buffer, &len, f);
    getline(&buffer, &len, f);
    getline(&buffer, &len, f);
    int width = 640;
    int height = 480;
    size_t img_size = (size_t) (3 * width * height);
    __uint8_t *image = malloc(width*height*3 * sizeof(__uint8_t));

    fread(image, img_size, 1, f);
//    printf("%c\n", image[0]);
    conv_layer *L = allocate_conv_layer(height, width, 3);

    for (int h = 0; h < L->height; ++h) {
        for (int w = 0; w < L->width; ++w) {
            for (int l = 0; l < 3; ++l) {
//                printf("%d\n", image[h*width*3 + 3*w + l]);
                L->values[l][h][w] = (double) image[h*width*3 + 3*w + l];
//                printf("%lf\n", L->values[l][h][w]);
            }
        }
    }


    conv_layer *L2 = allocate_conv_layer(416, 416, 3);

    for (int h = 0; h < 416; ++h) {
        for (int w = 0; w < 416; ++w) {
            for (int l = 0; l < 3; ++l) {
                L2->values[l][h][w] = L->values[l][h][w];
            }
        }
    }

    free_conv_layer(L);

    return L2;
}

conv_layer *load_resized_image(){
    char buffer[256];
    size_t len = 256;
    FILE *f = fopen("out_resized.ppm", "r");
    getline(&buffer, &len, f);
    getline(&buffer, &len, f);
    getline(&buffer, &len, f);
    int width = 416;
    int height = 416;
    size_t img_size = (size_t) (3 * width * height);
    __uint8_t *image = malloc(width*height*3 * sizeof(__uint8_t));

    fread(image, img_size, 1, f);
    conv_layer *L = allocate_conv_layer(height, width, 3);

    for (int h = 0; h < L->height; ++h) {
        for (int w = 0; w < L->width; ++w) {
            for (int l = 0; l < 3; ++l) {
                L->values[l][h][w] = (double) image[h*width*3 + 3*w + l];
            }
        }
    }


    return L;
}

void save_image(conv_layer *L){
    int width = 416;
    int height = 416;
    FILE *f = fopen("load_function_test.ppm", "wc");
    fprintf(f, "P6\n%d %d\n255\n", width, height);


    for (int h = 0; h < L->height; ++h) {
        for (int w = 0; w < L->width; ++w) {
            for (int l = 0; l < 3; ++l) {
                __uint8_t  c = (__uint8_t) L->values[l][h][w];
                if (h == 0 && w == 0){
                    printf("%d %lf\n", c, L->values[l][h][w]);
                }
                fprintf(f, "%c", c);
            }
        }
    }


}

conv_layer *batch_norm_wrapper(conv_layer *L, int n){
    return batch_normalization(L, load_batch_normalization_means(n), load_batch_normalization_variances(n),
                               load_batch_normalization_gamma(n), load_batch_normalization_beta(n));
}

conv_layer *conv_block_wrapper_with_pool(conv_layer *L, int n, int pool_size, int pool_stride){
    printf("Block nr %d\n", n);
    kernel *K = load_kernel_by_number(n);
    print_kernel(K);
    conv_layer *L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    conv_layer *L3 = batch_norm_wrapper(L2, n);
    conv_layer *L4 = leaky_ReLu(L3);
    conv_layer *L5 = max_pool(L4, pool_size, pool_stride);
    printf("After pooling: ");
    print_conv_layer(L5);
    printf("\n");
    return L5;
}

conv_layer *conv_block_wrapper_no_pool(conv_layer *L, int n){
    printf("Block nr %d\n", n);
    kernel *K = load_kernel_by_number(n);
    print_kernel(K);
    conv_layer *L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    conv_layer *L3 = batch_norm_wrapper(L2, n);
    conv_layer *L4 = leaky_ReLu(L3);
    printf("\n");

    return L4;
}

void run_model(){

    kernel *K;
    conv_layer *L;
    conv_layer *L1;
    conv_layer *L2;
    conv_layer *L3;
    conv_layer *L4;
    conv_layer *L5;
    L = load_resized_image();

////  1
//    K = load_kernel_by_number(0);
////    print_kernel(K);
//    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
////
////    printf("\n%d %d\n", L2->height, L2->n_layers);
////    printf("\n%lf\n", L2->values[0][0][0]);
////    printf("\n%lf\n", L2->values[0][0][1]);
////    printf("\n%lf\n", L2->values[0][0][2]);
////
//
////    print_conv_layer(L2);
////    free_conv_layer(L);
//
//    L3 = batch_norm_wrapper(L2, 0);
////    printf("\n%d %d\n", L3->height, L3->n_layers);
////    printf("\n%lf\n", L3->values[0][0][0]);
////    printf("\n%lf\n", L3->values[0][0][1]);
////    printf("\n%lf\n", L3->values[0][0][2]);
//
//    L4 = leaky_ReLu(L3);
//
////    printf("\n%d %d\n", L4->height, L4->n_layers);
////    printf("\n%lf\n", L4->values[0][0][0]);
////    printf("\n%lf\n", L4->values[0][0][1]);
////    printf("\n%lf\n", L4->values[0][0][2]);
//
//    L5 = max_pool(L4, 2, 2);
//    printf("After pooling: ");
//
////    printf("\n%d %d\n", L5->height, L5->n_layers);
////    printf("\n%lf\n", L5->values[0][0][0]);
////    printf("\n%lf\n", L5->values[0][0][1]);
////    printf("\n%lf\n", L5->values[0][0][2]);
////    print_conv_layer_one_l(L4);

    L1 = conv_block_wrapper_with_pool(L, 0, 2, 2);
    L2 = conv_block_wrapper_with_pool(L1, 1, 2, 2);

    L1 = conv_block_wrapper_with_pool(L2, 2, 2, 2);
    L2 = conv_block_wrapper_with_pool(L1, 3, 2, 2);

    L1 = conv_block_wrapper_with_pool(L2, 4, 2, 2);
    L2 = conv_block_wrapper_with_pool(L1, 5, 2, 1);

//    printf("\n%dx%d %d\n", L2->height, L2->width, L2->n_layers);
//    printf("\n%lf", L2->values[0][0][0]);
//    printf("\n%lf", L2->values[0][0][1]);
//    printf("\n%lf\n\n", L2->values[0][0][2]);


    L1 = conv_block_wrapper_no_pool(L2, 6);
    L2 = conv_block_wrapper_no_pool(L1, 7);

    L1 = conv_block_wrapper_no_pool(L2, 8);
    L2 = conv_block_wrapper_no_pool(L1, 9);


//    printf("\n%dx%d %d\n", L1->height, L->width, L1->n_layers);
//    printf("\n%lf\n", L1->values[0][0][0]);
//    printf("\n%lf\n", L1->values[0][0][1]);
//    printf("\n%lf\n", L1->values[0][0][2]);

//    printf("\n%dx%d %d\n", L2->height, L2->width, L2->n_layers);
//    printf("\n%lf\n", L2->values[0][0][0]);
//    printf("\n%lf\n", L2->values[0][0][1]);
//    printf("\n%lf\n", L2->values[0][0][2]);

    printf("\n%dx%d %d\n", L2->height, L2->width, L2->n_layers);
    printf("\n%lf ", L2->values[0][0][0]);
    printf("\n%lf ", L2->values[1][0][0]);
    printf("\n%lf ", L2->values[2][0][0]);
    printf("\n%lf ", L2->values[3][0][0]);
    printf("\n%lf ", L2->values[4][0][0]);
    printf("\n%lf\n", L2->values[5][0][0]);



    return;

//    L = L4;
//    free_kernel(K);
//    free_conv_layer(L2);
//    free_conv_layer(L3);
////    free_conv_layer(L4);
//
//
////  2
//    K = load_kernel_by_number(1);
////    print_kernel(K);
//    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
//    print_conv_layer_one_l(L2);
//    free_conv_layer(L);
//    L3 = leaky_ReLu(L2);
//    L4 = max_pool(L3, 2, 2);
//    printf("After pooling: ");
//    print_conv_layer_one_l(L4);
//    L = L4;
//    free_kernel(K);
//    free_conv_layer(L2);
//    free_conv_layer(L3);
////    free_conv_layer(L4);
//
////  3
//    K = load_kernel_by_number(2);
////    print_kernel(K);
//
//    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
//    print_conv_layer_one_l(L2);
//    free_conv_layer(L);
//    L3 = leaky_ReLu(L2);
//    L4 = max_pool(L3, 2, 2);
//    printf("After pooling: ");
//    print_conv_layer_one_l(L4);
//    L = L4;
//    free_kernel(K);
//    free_conv_layer(L2);
//    free_conv_layer(L3);
////    free_conv_layer(L4);
//
//
////  4
//    K = load_kernel_by_number(3);
////    print_kernel(K);
//
//    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
//    print_conv_layer_one_l(L2);
//    free_conv_layer(L);
//    L3 = leaky_ReLu(L2);
//    L4 = max_pool(L3, 2, 2);
//    printf("After pooling: ");
//    print_conv_layer_one_l(L4);
//    L = L4;
//    free_kernel(K);
//    free_conv_layer(L2);
//    free_conv_layer(L3);
////    free_conv_layer(L4);
//
////  5
//    K = load_kernel_by_number(4);
//    print_kernel(K);
//
//    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
//    print_conv_layer_one_l(L2);
//    free_conv_layer(L);
//    L3 = leaky_ReLu(L2);
//    L4 = max_pool(L3, 2, 2);
//    printf("After pooling: ");
//    print_conv_layer_one_l(L4);
//    L = L4;
//    free_kernel(K);
//    free_conv_layer(L2);
//    free_conv_layer(L3);
////    free_conv_layer(L4);
//
////  6
//    K = load_kernel_by_number(5);
//    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
//    print_conv_layer_one_l(L2);
//    free_conv_layer(L);
//    L3 = leaky_ReLu(L2);
//    L4 = max_pool(L3, 2, 1);
//    printf("After pooling: ");
//    print_conv_layer_one_l(L4);
//    L = L4;
//    free_kernel(K);
//    free_conv_layer(L2);
//    free_conv_layer(L3);
////    free_conv_layer(L4);
//
////  7
//    K = load_kernel_by_number(6);
//    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
//    print_conv_layer_one_l(L2);
//    free_conv_layer(L);
//    L3 = leaky_ReLu(L2);
//    L = L3;
//    free_kernel(K);
//    free_conv_layer(L2);
////    free_conv_layer(L3);
//
//
////  8
//    K = load_kernel_by_number(7);
//    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
//    print_conv_layer_one_l(L2);
//    free_conv_layer(L);
//    L3 = leaky_ReLu(L2);
//    L = L3;
//    free_kernel(K);
//    free_conv_layer(L2);
////    free_conv_layer(L3);
//
//
////  9
//    K = load_kernel_by_number(8);
//    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
//    print_conv_layer_one_l(L2);
//    free_conv_layer(L);
//    L3 = leaky_ReLu(L2);
//    L = L3;
//    free_kernel(K);
//    free_conv_layer(L2);
////    free_conv_layer(L3);
//
////  10
//    K = load_kernel_by_number(9);
//    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
//    print_conv_layer_one_l(L2);
//    free_conv_layer(L);
//    double *bias = load_bias(0);
//    L3 = add_bias(L2, bias);
//    L = L3;
//    free_kernel(K);
//    free_conv_layer(L2);
////    free_conv_layer(L3);
//
//    double ***anchors = load_anchors(L, 3);
//
//    print_conv_layer(L);
//    free_conv_layer(L);
//
//    for (int j = 0; j < 85; ++j) {
//        printf("%lf\t%lf\t%lf\n", anchors[0][0][j], anchors[0][1][j], anchors[0][2][j]);
////    }




}



int main(){



    run_model();

//    conv_layer *L = test_conv_layer(10, 10, 3);
//    conv_layer *L2 = max_pool(L, 2, 1);
//    print_conv_layer(L2);
//    kernel *K = test_kernel(1, 1024, 256);
//    conv_layer *L2 = conv3D(L, K, 1, NO_PADDING);
//
//    print_conv_layer(L);
//    print_kernel(K);
//    print_conv_layer(L2);
//        conv_layer *L2 = max_pool(L, 2, 2);

//    print_conv_layer(L);
//    print_conv_layer(L2);




    return 0;
}