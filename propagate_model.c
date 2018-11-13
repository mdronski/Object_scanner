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

    return L;
}


void run_model(){

//    Pooling is wrong! check
    kernel *K;
    conv_layer *L;
    conv_layer *L2;
    conv_layer *L3;
    conv_layer *L4;
    printf("Input layer: ");
    printf("\n");
//    L = test_conv_layer(416, 416, 3);
    L = load_image();
    print_conv_layer(L);

//  1
    K = load_kernel_by_number(0);
    print_kernel(K);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L4 = max_pool(L3, 2, 2);
    printf("After pooling: ");
    print_conv_layer(L4);
    L = L4;
    free_kernel(K);
    free_conv_layer(L2);
    free_conv_layer(L3);
//    free_conv_layer(L4);

//  2
    K = load_kernel_by_number(1);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L4 = max_pool(L3, 2, 2);
    printf("After pooling: ");
    print_conv_layer(L4);
    L = L4;
    free_kernel(K);
    free_conv_layer(L2);
    free_conv_layer(L3);
//    free_conv_layer(L4);

//  3
    K = load_kernel_by_number(2);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L4 = max_pool(L3, 2, 2);
    printf("After pooling: ");
    print_conv_layer(L4);
    L = L4;
    free_kernel(K);
    free_conv_layer(L2);
    free_conv_layer(L3);
//    free_conv_layer(L4);


//  4
    K = load_kernel_by_number(3);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L4 = max_pool(L3, 2, 2);
    printf("After pooling: ");
    print_conv_layer(L4);
    L = L4;
    free_kernel(K);
    free_conv_layer(L2);
    free_conv_layer(L3);
//    free_conv_layer(L4);

//  5
    K = load_kernel_by_number(4);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L4 = max_pool(L3, 2, 2);
    printf("After pooling: ");
    print_conv_layer(L4);
    L = L4;
    free_kernel(K);
    free_conv_layer(L2);
    free_conv_layer(L3);
//    free_conv_layer(L4);

//  6
    K = load_kernel_by_number(5);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L4 = max_pool(L3, 2, 1);
    printf("After pooling: ");
    print_conv_layer(L4);
    L = L4;
    free_kernel(K);
    free_conv_layer(L2);
    free_conv_layer(L3);
//    free_conv_layer(L4);

//  7
    K = load_kernel_by_number(6);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L = L3;
    free_kernel(K);
    free_conv_layer(L2);
//    free_conv_layer(L3);


//  8
    K = load_kernel_by_number(7);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L = L3;
    free_kernel(K);
    free_conv_layer(L2);
//    free_conv_layer(L3);


//  9
    K = load_kernel_by_number(8);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L = L3;
    free_kernel(K);
    free_conv_layer(L2);
//    free_conv_layer(L3);

//  10
    K = load_kernel_by_number(9);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    double *bias = load_bias(0);
    L3 = add_bias(L2, bias);
    L = L3;
    free_kernel(K);
    free_conv_layer(L2);
//    free_conv_layer(L3);

    double ***anchors = load_anchors(L, 3);

    print_conv_layer(L);
    free_conv_layer(L);

    for (int j = 0; j < 85; ++j) {
        printf("%lf\t%lf\t%lf\n", anchors[0][0][j], anchors[0][1][j], anchors[0][2][j]);
    }




}



int main(){



    run_model();

//    conv_layer *L = test_conv_layer(10, 10, 3);
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