#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include "yolo_utils.h"


conv_layer *load_image() {
    char buffer[256];
    size_t len = 256;
    FILE *f = fopen("out.ppm", "r");
    getline(&buffer, &len, f);
    getline(&buffer, &len, f);
    getline(&buffer, &len, f);
    int width = 640;
    int height = 480;
    size_t img_size = (size_t) (3 * width * height);
    __uint8_t *image = malloc(width * height * 3 * sizeof(__uint8_t));

    fread(image, img_size, 1, f);
//    printf("%c\n", image[0]);
    conv_layer *L = allocate_conv_layer(height, width, 3);

    for (int h = 0; h < L->height; ++h) {
        for (int w = 0; w < L->width; ++w) {
            for (int l = 0; l < 3; ++l) {
//                printf("%d\n", image[h*width*3 + 3*w + l]);
                L->values[l][h][w] = (double) image[h * width * 3 + 3 * w + l];
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

conv_layer *load_resized_image() {
    char *buffer = malloc(256 * sizeof(char));
    size_t len = 256;
    FILE *f = fopen("correct_resized.ppm", "r");

    if (f == NULL) {
        printf("Load error\n");
    }
    getline(&buffer, &len, f);
    getline(&buffer, &len, f);
    getline(&buffer, &len, f);
    int width = 416;
    int height = 416;
    size_t img_size = (size_t) (3 * width * height);
    __uint8_t *image = malloc(width * height * 3 * sizeof(__uint8_t));

    fread(image, img_size, 1, f);
    conv_layer *L = allocate_conv_layer(height, width, 3);

    for (int h = 0; h < L->height; ++h) {
        for (int w = 0; w < L->width; ++w) {
            for (int l = 0; l < 3; ++l) {
                L->values[l][h][w] = (double) image[h * width * 3 + 3 * w + l];
            }
        }
    }


    return L;
}

void save_image(conv_layer *L) {
    int width = 416;
    int height = 416;
    FILE *f = fopen("load_function_test.ppm", "wc");
    fprintf(f, "P6\n%d %d\n255\n", width, height);


    for (int h = 0; h < L->height; ++h) {
        for (int w = 0; w < L->width; ++w) {
            for (int l = 0; l < 3; ++l) {
                __uint8_t c = (__uint8_t) L->values[l][h][w];
                if (h == 0 && w == 0) {
                    printf("%d %lf\n", c, L->values[l][h][w]);
                }
                fprintf(f, "%c", c);
            }
        }
    }


}

conv_layer *batch_norm_wrapper(conv_layer *L, int n) {
    return batch_normalization(L, load_batch_normalization_means(n), load_batch_normalization_variances(n),
                               load_batch_normalization_gamma(n), load_batch_normalization_beta(n));
}

conv_layer *conv_block_wrapper_with_pool(conv_layer *L, int n, int pool_size, int pool_stride) {
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

conv_layer *conv_block_wrapper_no_pool(conv_layer *L, int n) {
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

void run_model() {

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

//  1-4
    L1 = conv_block_wrapper_with_pool(L, 0, 2, 2);
//  5-8
    L2 = conv_block_wrapper_with_pool(L1, 1, 2, 2);

//  9-12
    L1 = conv_block_wrapper_with_pool(L2, 2, 2, 2);
//  13-16
    L2 = conv_block_wrapper_with_pool(L1, 3, 2, 2);

//  17-20
    L1 = conv_block_wrapper_with_pool(L2, 4, 2, 2);
//  21-24
    L2 = conv_block_wrapper_with_pool(L1, 5, 2, 1);

//    printf("\n%dx%d %d\n", L2->height, L2->width, L2->n_layers);
//    printf("\n%lf", L2->values[0][0][0]);
//    printf("\n%lf", L2->values[0][0][1]);
//    printf("\n%lf\n\n", L2->values[0][0][2]);

//  25-27
    L1 = conv_block_wrapper_no_pool(L2, 6);
//  28-30
    L2 = conv_block_wrapper_no_pool(L1, 7);

//    printf("\n%dx%d %d\n", L2->height, L2->width, L2->n_layers);
//    printf("\n%lf\n", L2->values[0][0][0]);
//    printf("\n%lf\n", L2->values[1][0][0]);
//    printf("\n%lf\n", L2->values[2][0][0]);
//

    L1 = conv_block_wrapper_no_pool(L2, 8);


//    printf("\n%dx%d %d\n", L1->height, L1->width, L1->n_layers);
//    printf("\n%lf\n", L1->values[0][0][0]);
//    printf("\n%lf\n", L1->values[1][0][0]);
//    printf("\n%lf\n", L1->values[2][0][0]);
//
//    person 0.87 (42, 118) (309, 351)
//    height = 267 width = 233
//    center = (175.5, 234.5)
//    0.15511817799961136


    K = load_kernel_by_number(9);
    print_kernel(K);
    L2 = conv3D_paralel(L1, K, 1, ZERO_PADDING);
    print_conv_layer(L2);

//    printf("\n%dx%d %d\n", L2->height, L2->width, L2->n_layers);
//    printf("\n%lf\n", L2->values[0][0][0]);
//    printf("\n%lf\n", L2->values[1][0][0]);
//    printf("\n%lf\n", L2->values[2][0][0]);
//


    yolo_box ****boxes = malloc(13 * sizeof(yolo_box ****));
    for (int i = 0; i < 13; ++i) {
        boxes[i] = malloc(13 * sizeof(yolo_box ***));
        for (int j = 0; j < 13; ++j) {
            boxes[i][j] = malloc(3 * sizeof(yolo_box **));
        }
    }

    for (int i = 0; i < 13; ++i) {
        for (int j = 0; j < 13; ++j) {
            for (int k = 0; k < 3; ++k) {
                double tx = L2->values[0 + k * 85][i][j];
                double ty = L2->values[1 + k * 85][i][j];
                double tw = L2->values[2 + k * 85][i][j];
                double th = L2->values[3 + k * 85][i][j];
                double prob = L2->values[4 + k * 85][i][j];
                int cell_x = i;
                int cell_y = j;
                int anchor_width = 344;
                int anchor_height = 318;
                int image_width = 416;
                int image_height = 416;
                boxes[i][j][k] = get_yolo_box(tx, ty, tw, th, prob, cell_x, cell_y, anchor_width, anchor_height,
                                              image_width, image_height);
            }
        }
    }

    softmax(boxes, L2);

    for (int i = 0; i < 13; ++i) {
        for (int j = 0; j < 13; ++j) {
            for (int k = 0; k < 3; ++k) {
                if(boxes[i][j][k]->confidence > 2 && boxes[i][j][k]->class_probability > 0.8){
                    double tx = L2->values[0 + k * 85][i][j];
                    double ty = L2->values[1 + k * 85][i][j];
                    double tw = L2->values[2 + k * 85][i][j];
                    double th = L2->values[3 + k * 85][i][j];
                    printf("%lf\n", boxes[i][j][k]->confidence);
                    printf("%d\n", boxes[i][j][k]->class);
                    printf("%d %d %d %lf\n", i, j, k, boxes[i][j][k]->class_probability);
//                    printf("%lf %lf %lf %lf\n", tx,ty,tw,th);
                    printf("(%.1lf, %.1lf), (%.1lf, %1.lf)\n", boxes[i][j][k]->left_up_x, boxes[i][j][k]->left_up_y, boxes[i][j][k]->right_bottom_x, boxes[i][j][k]->right_bottom_y );
                    printf("\n");
                }

//                if (boxes[i][j][k]->confidence > 10.0) {
//                    printf("%lf\n", boxes[i][j][k]->confidence);
//                    printf("%d\n", boxes[i][j][k]->class);
//                    printf("%lf\n", boxes[i][j][k]->class_probability);
//                    printf("\n");
//                }
            }
        }
    }



//    for (int i = 0; i < 13; ++i) {
//        for (int j = 0; j < 13; ++j) {
//            for (int k = 0; k < 3; ++k) {
//                if (boxes[i][j][k]->confidence > 10.0){
//                    print_yolo_box(boxes[i][j][k]);
//                }
////                if (i==7 && j==5 && k==2){
////                    printf("%lf, %lf, %lf, %lf, %d, %d\n", tx, ty, tw, th, cell_x, cell_y);
////                }
//            }
//        }
//    }

//
//    printf("\n%dx%d %d\n", L2->height, L2->width, L2->n_layers);
//    printf("\n%lf\n", L2->values[0][0][0]);
//    printf("\n%lf\n", L2->values[0][0][1]);
//    printf("\n%lf\n", L2->values[0][0][2]);
//    printf("\n%lf\n", L2->values[0][0][L2->width-1]);
//

//    [[32.303703   3.1947038 -8.289908  30.665462  48.937305 ]]
//    2 [[ -2.4656894 -14.658622  -11.760943   -3.6249452 136.50597  ]]


//    printf("\n%dx%d %d\n", L2->height, L2->width, L2->n_layers);
//    printf("\n%lf ", L2->values[0][0][0]);
//    printf("\n%lf ", L2->values[1][0][0]);
//    printf("\n%lf ", L2->values[2][0][0]);
//    printf("\n%lf ", L2->values[3][0][0]);
//    printf("\n%lf ", L2->values[4][0][0]);
//    printf("\n%lf\n", L2->values[5][0][0]);



}


int main() {


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