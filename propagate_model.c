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
                L->values[l][h][w] = (float) image[h * width * 3 + 3 * w + l];
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
                L->values[l][h][w] = (float) image[h * width * 3 + 3 * w + l];
            }
        }
    }


    return L;
}

void save_image(conv_layer *L) {
    int width = 416;
    int height = 416;
    FILE *f = fopen("test_bounding_box.ppm", "wc");
    fprintf(f, "P6\n%d %d\n255\n", width, height);


    for (int h = 0; h < L->height; ++h) {
        for (int w = 0; w < L->width; ++w) {
            for (int l = 0; l < 3; ++l) {
                __uint8_t c = (__uint8_t) L->values[l][h][w];
                if (h == 0 && w == 0) {
//                    printf("%d %lf\n", c, L->values[l][h][w]);
                }
                fprintf(f, "%c", c);
            }
        }
    }


}

void draw_boxes(conv_layer *L, yolo_box_node *list) {
    yolo_box_node *ptr = list;
    while (ptr != NULL) {
        if (ptr->box->confidence > 10 &&
            ptr->box->class_probability > 0.3
            ) {
            printf("%.1lf (%d %d)  ", ptr->box->confidence, (int) ptr->box->x_min, (int) ptr->box->x_max);
            printf("(%d %d) \n", (int) ptr->box->y_min, (int) ptr->box->y_max);
            int x = (int) ptr->box->x;
            int y = (int) ptr->box->y;

            L->values[0][y][x] = 0;
            L->values[0][y][x + 1] = 0;
            L->values[0][y + 1][x] = 0;
            L->values[0][y + 1][x + 1] = 0;


            L->values[1][y][x] = 0;
            L->values[1][y][x + 1] = 0;
            L->values[1][y + 1][x] = 0;
            L->values[1][y + 1][x + 1] = 0;


            L->values[2][y][x] = 0;
            L->values[2][y][x + 1] = 0;
            L->values[2][y + 1][x] = 0;
            L->values[2][y + 1][x + 1] = 0;


            for (int h = (int) ptr->box->y_min; h < ptr->box->y_max; ++h) {
                int ws = (int) ptr->box->x_min + 1;
                int wk = (int) ptr->box->x_max - 1;
                L->values[0][h][ws] = 0;
                L->values[1][h][ws] = 0;
                L->values[2][h][ws] = 0;

                L->values[0][h][wk] = 0;
                L->values[1][h][wk] = 0;
                L->values[2][h][wk] = 0;
            }

            for (int w = (int) ptr->box->x_min; w < ptr->box->x_max; ++w) {
                int hs = (int) ptr->box->y_min + 1;
                int hk = (int) ptr->box->y_max - 1;

                L->values[0][hs][w] = 0;
                L->values[1][hs][w] = 0;
                L->values[2][hs][w] = 0;

                L->values[0][hk][w] = 0;
                L->values[1][hk][w] = 0;
                L->values[2][hk][w] = 0;
            }
        }


        ptr = ptr->next;
    }
    save_image(L);

}

//void draw_boxes(conv_layer *L, yolo_box ****boxes) {
//    for (int i = 0; i < 13; ++i) {
//        for (int j = 0; j < 13; ++j) {
//            for (int k = 0; k < 3; ++k) {
//                if (boxes[i][j][k]->confidence > 5 &&
//                    boxes[i][j][k]->class_probability > 0.8){
////                    printf("xd\n");
//                    printf("(%d %d)  ", (int) boxes[i][j][k]->x_min, (int) boxes[i][j][k]->x_max);
//                    printf("(%d %d) \n", (int) boxes[i][j][k]->y_min, (int) boxes[i][j][k]->y_max);
//                        int x = (int) boxes[i][j][k] -> x;
//                        int y = (int) boxes[i][j][k] -> y;
////
//                    L->values[0][y][x] = 255;
//                    L->values[0][y][x+1] = 255;
//                    L->values[0][y+1][x] = 255;
//                    L->values[0][y+1][x+1] = 255;
//
//
//                    L->values[1][y][x] = 255;
//                    L->values[1][y][x+1] = 255;
//                    L->values[1][y+1][x] = 255;
//                    L->values[1][y+1][x+1] = 255;
//
//
//                    L->values[2][y][x] = 255;
//                    L->values[2][y][x+1] = 255;
//                    L->values[2][y+1][x] = 255;
//                    L->values[2][y+1][x+1] = 255;
//
////
////                    L->values[0][x][y] = 255;
////                    L->values[0][x][y+1] = 255;
////                    L->values[0][x+1][y] = 255;
////                    L->values[0][x+1][y+1] = 255;
////
////                    L->values[1][x][y] = 255;
////                    L->values[1][x][y+1] = 255;
////                    L->values[1][x+1][y] = 255;
////                    L->values[1][x+1][y+1] = 255;
////
////                    L->values[2][x][y] = 255;
////                    L->values[2][x][y+1] = 255;
////                    L->values[2][x+1][y] = 255;
////                    L->values[2][x+1][y+1] = 255;
//
//                    for (int h = (int) boxes[i][j][k]->y_min; h < boxes[i][j][k]->y_max; ++h) {
//                        int ws = (int) boxes[i][j][k]->x_min + 1;
//                        int wk = (int) boxes[i][j][k]->x_max - 1;
//                        L->values[0][h][ws] = 0;
//                        L->values[1][h][ws] = 0;
//                        L->values[2][h][ws] = 0;
//
//                        L->values[0][h][wk] = 0;
//                        L->values[1][h][wk] = 0;
//                        L->values[2][h][wk] = 0;
//                    }
//
//                    for (int w = (int) boxes[i][j][k]->x_min; w < boxes[i][j][k]->x_max; ++w) {
//                        int hs = (int) boxes[i][j][k]->y_min + 1;
//                        int hk = (int) boxes[i][j][k]->y_max - 1;
//
//                        L->values[0][hs][w] = 0;
//                        L->values[1][hs][w] = 0;
//                        L->values[2][hs][w] = 0;
//
//                        L->values[0][hk][w] = 0;
//                        L->values[1][hk][w] = 0;
//                        L->values[2][hk][w] = 0;
//                    }
//                }
//            }
//        }
//    }
//    save_image(L);
//}

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
    free_conv_layer(L2);
    free_conv_layer(L3);
    free_conv_layer(L4);
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
    conv_layer *leaky_relu5;
    L = load_resized_image();
    int class = 1;

//  1-4
    L1 = conv_block_wrapper_with_pool(L, 0, 2, 2);

//  5-8
    L2 = conv_block_wrapper_with_pool(L1, 1, 2, 2);

//  9-12
    L1 = conv_block_wrapper_with_pool(L2, 2, 2, 2);

//  13-16
    L2 = conv_block_wrapper_with_pool(L1, 3, 2, 2);

//  17-20
//    L5 = conv_block_wrapper_with_pool(L2, 4, 2, 2);

    printf("Block nr %d\n", 4);
    K = load_kernel_by_number(4);
    print_kernel(K);
    L1 = conv3D_paralel(L2, K, 1, ZERO_PADDING);
    print_conv_layer(L1);
    L3 = batch_norm_wrapper(L1, 4);
    leaky_relu5 = leaky_ReLu(L3);
    L5 = max_pool(leaky_relu5, 2, 2);
    printf("After pooling: ");
    print_conv_layer(L5);
    printf("\n");


//  21-24
    L2 = conv_block_wrapper_with_pool(L5, 5, 2, 1);

//  25-27
    L1 = conv_block_wrapper_no_pool(L2, 6);


//  28-30
    L4 = conv_block_wrapper_no_pool(L1, 7);
//  36, 38, 40
    L1 = conv_block_wrapper_no_pool(L4, 8);


//  42
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
    for (int h = 0; h < 13; ++h) {
        boxes[h] = malloc(13 * sizeof(yolo_box ***));
        for (int w = 0; w < 13; ++w) {
            boxes[h][w] = malloc(3 * sizeof(yolo_box **));
        }
    }

    for (int h = 0; h < 13; ++h) {
        for (int w = 0; w < 13; ++w) {
            for (int k = 0; k < 3; ++k) {
                float tx = L2->values[0 + k * 85][h][w];
                float ty = L2->values[1 + k * 85][h][w];
                float tw = L2->values[2 + k * 85][h][w];
                float th = L2->values[3 + k * 85][h][w];
                float conf = L2->values[4 + k * 85][h][w];
                int cell_y = h;
                int cell_x = w;
                int anchor_width;
                int anchor_height;
                switch (k % 3) {
                    case 0:
                        anchor_width = 82;
                        anchor_height = 81;
                        break;
                    case 1:
                        anchor_width = 169;
                        anchor_height = 135;
                        break;
                    case 2:
                        anchor_width = 344;
                        anchor_height = 319;
                        break;

                    default:
                        break;
                }

                int image_width = 416;
                int image_height = 416;
                boxes[h][w][k] = get_yolo_box(tx, ty, tw, th, conf, cell_x, cell_y, anchor_width, anchor_height,
                                              image_width, image_height, 13);
//                correct_yolo_box(boxes[h][w][k], image_width, image_height, 13);
            }
        }
    }

    softmax(boxes, L2, 13);
    printf("\n\n");

    for (int h = 0; h < 13; ++h) {
        for (int w = 0; w < 13; ++w) {
            for (int k = 0; k < 3; ++k) {
                if (
                        boxes[h][w][k]->confidence > 10.0
//                        && boxes[h][w][k]->class_probability > 0.5 && boxes[h][w][k]->class == 28
                        ) {
                    printf("%lf\n", boxes[h][w][k]->confidence);
                    printf("%d \n", boxes[h][w][k]->class);
                    printf("%d %d %d %lf\n", h, w, k, boxes[h][w][k]->class_probability);
                    printf("%.1lf, %.1lf - (%.1lf, %.1lf), (%.1lf, %1.lf)\n",boxes[h][w][k]->x, boxes[h][w][k]->y, boxes[h][w][k]->x_min, boxes[h][w][k]->x_max,
                           boxes[h][w][k]->y_min, boxes[h][w][k]->y_max);
                    printf("\n");
                }
            }
        }
    }

    yolo_box_node *l = non_max_supression(boxes, 0.3, class, 13);

//  31
    printf("Convolution %d\n", 10);
    K = load_kernel_by_number(10);
    print_kernel(K);
    L2 = conv3D_paralel(L4, K, 1, ZERO_PADDING);
    print_conv_layer(L2);

    L3 = batch_norm_wrapper(L2, 9);
    L4 = leaky_ReLu(L3);
    L1 = upscale(L4);

//  35
    L2 = concatenate(L1, leaky_relu5);
//
//    printf("\n%dx%d %d\n", L2->height, L2->width, L2->n_layers);
//    printf("\n%lf\n", L2->values[0][0][0]);
//    printf("\n%lf\n", L2->values[1][0][0]);
//    printf("\n%lf\n", L2->values[2][0][0]);
//    printf("\n%lf\n", L2->values[3][0][0]);
//    printf("\n");

    //  37
    printf("Convolution %d\n", 11);
    K = load_kernel_by_number(11);
    print_kernel(K);
    L1 = conv3D_paralel(L2, K, 1, ZERO_PADDING);
    print_conv_layer(L1);

//  39
    L3 = batch_norm_wrapper(L1, 10);

    L4 = leaky_ReLu(L3);


    printf("Convolution %d\n", 12);
    K = load_kernel_by_number(12);
    print_kernel(K);
    L1 = conv3D_paralel(L4, K, 1, ZERO_PADDING);
    print_conv_layer(L1);
    printf("\n");


    yolo_box ****boxes2 = malloc(26 * sizeof(yolo_box ****));
    for (int h = 0; h < 26; ++h) {
        boxes2[h] = malloc(26 * sizeof(yolo_box ***));
        for (int w = 0; w < 26; ++w) {
            boxes2[h][w] = malloc(3 * sizeof(yolo_box **));
        }
    }

    for (int h = 0; h < 26; ++h) {
        for (int w = 0; w < 26; ++w) {
            for (int k = 0; k < 3; ++k) {
                float tx = L1->values[0 + k * 85][h][w];
                float ty = L1->values[1 + k * 85][h][w];
                float tw = L1->values[2 + k * 85][h][w];
                float th = L1->values[3 + k * 85][h][w];
                float conf = L1->values[4 + k * 85][h][w];
                int cell_y = h;
                int cell_x = w;
                int anchor_width;
                int anchor_height;
//                anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
                switch (k % 3) {
                    case 0:
                        anchor_width = 10;
                        anchor_height = 14;
                        break;
                    case 1:
                        anchor_width = 23;
                        anchor_height = 27;
                        break;
                    case 2:
                        anchor_width = 37;
                        anchor_height = 58;
                        break;

                    default:
                        break;
                }

                int image_width = 416;
                int image_height = 416;
                boxes2[h][w][k] = get_yolo_box(tx, ty, tw, th, conf, cell_x, cell_y, anchor_width, anchor_height,
                                              image_width, image_height, 26);
//                correct_yolo_box(boxes[h][w][k], image_width, image_height, 13);
            }
        }
    }

    softmax(boxes2, L1, 26);
    printf("\n\n");

    for (int h = 0; h < 26; ++h) {
        for (int w = 0; w < 26; ++w) {
            for (int k = 0; k < 3; ++k) {
                if (boxes2[h][w][k]->confidence > 3.0 && boxes2[h][w][k]->class == class) {
                    printf("%lf\n", boxes2[h][w][k]->confidence);
                    printf("%d \n", boxes2[h][w][k]->class);
                    printf("%d %d %d %lf\n", h, w, k, boxes2[h][w][k]->class_probability);
                    printf("%.1lf, %.1lf - (%.1lf, %.1lf), (%.1lf, %1.lf)\n",boxes2[h][w][k]->x, boxes2[h][w][k]->y, boxes2[h][w][k]->x_min, boxes2[h][w][k]->x_max,
                           boxes2[h][w][k]->y_min, boxes2[h][w][k]->y_max);
                    printf("\n");
                }
            }
        }
    }



    yolo_box_node *l2 = non_max_supression(boxes2, 0.5, class, 26);
////
    yolo_box_node *ptr = l;
    while (ptr->next != NULL)
        ptr = ptr->next;
    ptr->next = l2;






    draw_boxes(load_resized_image(), l);


}

int main() {


    run_model();
//    conv_layer *L = test_conv_layer(3, 3, 3);
//    conv_layer *L2 = upscale(L);
//    print_conv_layer_one_l(L2);
//    float *f = load_batch_normalization_beta(10);
//    printf("%p\n ", f);
//
//    for (int i = 0; i < 1; ++i) {
//        printf("%f\n ", f[i]);
//    }


    return 0;
}