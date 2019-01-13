#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <tkPort.h>
#include "yolo_utils.h"

char *classes[80] = {
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
};

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

conv_layer *load_resized_image(const char *image_name) {
    char *buffer = malloc(256 * sizeof(char));
    size_t len = 256;
    FILE *f = fopen(image_name, "r");

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
                L->values[l][h][w] = (float) image[h * width * 3 + 3 * w + l] / 255;
            }
        }
    }


    return L;
}

void save_image(conv_layer *L, const char* image_name) {
    int width = 416;
    int height = 416;
    char *file_name = "predicted.ppm";
    FILE *f = fopen(file_name, "wc");
    fprintf(f, "P6\n%d %d\n255\n", width, height);


    for (int h = 0; h < L->height; ++h) {
        for (int w = 0; w < L->width; ++w) {
            for (int l = 0; l < 3; ++l) {
                unsigned char c = (unsigned char) (L->values[l][h][w] * 255);
                if (h == 0 && w == 0) {
//                    printf("%d %lf\n", c, L->values[l][h][w]);
                }
                fprintf(f, "%c", c);
            }
        }
    }
    fprintf(stderr, "saved results\n");


}

void draw_boxes(conv_layer *L, yolo_box_node *list) {
    yolo_box_node *ptr = list;

    while (ptr->next != NULL) {
        if (ptr->next->box->confidence < 0.6f || ptr->next->box->class_probability < 0.3){
            ptr = ptr->next;
            continue;
        }
//        fprintf(stderr, "drawn one box\n");

        fprintf(stderr, "%s at (%d, %d) \n", classes[ptr->next->box->class - 1], (int) ptr->next->box->x, (int) ptr->next->box->y);
//        printf("(%d %d) \n", (int) ptr->next->box->y_min, (int) ptr->next->box->y_max);
        int x = (int) ptr->next->box->x;
        int y = (int) ptr->next->box->y;
        int color = 255 - 4 * (255 / ptr->next->box->class);

        L->values[0][y][x] = color;
        L->values[0][y][x + 1] = color;
        L->values[0][y + 1][x] = color;
        L->values[0][y + 1][x + 1] = color;


        L->values[1][y][x] = color;
        L->values[1][y][x + 1] = color;
        L->values[1][y + 1][x] = color;
        L->values[1][y + 1][x + 1] = color;


        L->values[2][y][x] = color;
        L->values[2][y][x + 1] = color;
        L->values[2][y + 1][x] = color;
        L->values[2][y + 1][x + 1] = color;


        for (int h = (int) ptr->next->box->y_min; h < ptr->next->box->y_max; ++h) {
            int ws = (int) ptr->next->box->x_min + 1;
            int wk = (int) ptr->next->box->x_max - 1;
            L->values[0][h][ws] = color;
            L->values[1][h][ws] = color;
            L->values[2][h][ws] = color;

            L->values[0][h][wk] = color;
            L->values[1][h][wk] = color;
            L->values[2][h][wk] = color;
        }

        for (int w = (int) ptr->next->box->x_min; w < ptr->next->box->x_max; ++w) {
            int hs = (int) ptr->next->box->y_min + 1;
            int hk = (int) ptr->next->box->y_max - 1;

            L->values[0][hs][w] = color;
            L->values[1][hs][w] = color;
            L->values[2][hs][w] = color;

            L->values[0][hk][w] = color;
            L->values[1][hk][w] = color;
            L->values[2][hk][w] = color;
        }


        ptr = ptr->next;
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
//    if(n==5)
//        print_conv_layer_weights(L2, 8, 10, 8, 10, 16);
    conv_layer *L3 = batch_norm_wrapper(L2, n);
    conv_layer *L4 = leaky_ReLu(L3);
//    print_conv_layer_weights(L4, 200, 202, 200, 202, 16);
    conv_layer *L5 = max_pool(L4, pool_size, pool_stride);
//    print_conv_layer_weights(L5, 200, 202, 200, 202, 16);

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
//    if(n==8)
//        print_conv_layer_weights(L2, 9, 10, 9, 10, 16);
    print_conv_layer(L2);
    conv_layer *L3 = batch_norm_wrapper(L2, n);
    conv_layer *L4 = leaky_ReLu(L3);
    printf("\n");

    return L4;
}

yolo_box_node *run_model(const char *image_name) {

    kernel *K;
    conv_layer *L;
    conv_layer *L1;
    conv_layer *L2;
    conv_layer *L3;
    conv_layer *L4;
    conv_layer *L5;
    conv_layer *leaky_relu5;
    L = load_resized_image(image_name);


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


    L1 = add_bias(L2, load_bias(0));
    L2 = L1;


    float iou_threshold = 0.3f;
    float conf_threshold = 1.0f;

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
                        anchor_width = 81;
                        anchor_height = 82;
                        break;
                    case 1:
                        anchor_width = 135;
                        anchor_height = 169;
                        break;
                    case 2:
                        anchor_width = 319;
                        anchor_height = 344;
                        break;

                    default:
                        break;
                }

                int image_width = 416;
                int image_height = 416;
                boxes[h][w][k] = get_yolo_box(tx, ty, tw, th, conf, cell_x, cell_y, anchor_width, anchor_height,
                                              image_width, image_height, 13, k);
//                correct_yolo_box(boxes[h][w][k], image_width, image_height, 13);
            }
        }
    }

    softmax(boxes, L2, 13);
    printf("\n\n");

//    for (int h = 0; h < 13; ++h) {
//        for (int w = 0; w < 13; ++w) {
//            for (int k = 0; k < 3; ++k) {
//                if (
//                        boxes[h][w][k]->confidence > 1.0
////                        && boxes[h][w][k]->class_probability > 0.5 && boxes[h][w][k]->class == 28
//                        ) {
//                    printf("%lf\n", boxes[h][w][k]->confidence);
//                    printf("%d \n", boxes[h][w][k]->class);
//                    printf("%d %d %d %lf\n", h, w, k, boxes[h][w][k]->class_probability);
//                    printf("%.1lf, %.1lf - (%.1lf, %.1lf), (%.1lf, %1.lf)\n", boxes[h][w][k]->x, boxes[h][w][k]->y,
//                           boxes[h][w][k]->x_min, boxes[h][w][k]->x_max,
//                           boxes[h][w][k]->y_min, boxes[h][w][k]->y_max);
//                    printf("\n");
//                }
//            }
//        }
//    }

    yolo_box_node *l = non_max_supression(boxes, iou_threshold, 13, conf_threshold);
//    printf("\n After NMS list box size: %d\n", list_size(l));

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

//    L2 = add_bias(L1, load_bias(1));
//    L1 = L2;


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
                        anchor_width = 14;
                        anchor_height = 10;
                        break;
                    case 1:
                        anchor_width = 27;
                        anchor_height = 23;
                        break;
                    case 2:
                        anchor_width = 58;
                        anchor_height = 37;
                        break;

                    default:
                        break;
                }

                int image_width = 416;
                int image_height = 416;
                boxes2[h][w][k] = get_yolo_box(tx, ty, tw, th, conf, cell_x, cell_y, anchor_width, anchor_height,
                                               image_width, image_height, 26, 2137);
//                correct_yolo_box(boxes[h][w][k], image_width, image_height, 13);
            }
        }
    }

    softmax(boxes2, L1, 26);
//    printf("\n\n");
//
//
//    for (int h = 0; h < 26; ++h) {
//        for (int w = 0; w < 26; ++w) {
//            for (int k = 0; k < 3; ++k) {
//                if (
//                        boxes2[h][w][k]->confidence > 0.5
//                        ) {
//                    printf("%lf\n", boxes2[h][w][k]->confidence);
//                    printf("%d \n", boxes2[h][w][k]->class);
//                    printf("%d %d %d %lf\n", h, w, k, boxes2[h][w][k]->class_probability);
//                    printf("%.1lf, %.1lf - (%.1lf, %.1lf), (%.1lf, %1.lf)\n", boxes2[h][w][k]->x, boxes2[h][w][k]->y,
//                           boxes2[h][w][k]->x_min, boxes2[h][w][k]->x_max,
//                           boxes2[h][w][k]->y_min, boxes2[h][w][k]->y_max);
//                    printf("\n");
//                }
//            }
//        }
//    }


    yolo_box_node *l2 = non_max_supression(boxes2, iou_threshold, 26, conf_threshold);
//    printf("\n After NMS list box size: %d\n", list_size(l2));
    l = merge_lists(l, l2);

    final_non_max_supression(l, iou_threshold);


//    draw_boxes(load_resized_image(image_name), l);
    return l;

}

int main(int argc, char *argv[]) {

//    char *image_name = argv[1];
    char *image_name = "person_horse_dog.ppm";
    yolo_box_node *boxes = run_model(image_name);
    conv_layer *IMAGE = load_resized_image(image_name);
    draw_boxes(IMAGE, boxes);
    save_image(IMAGE, image_name);




    return 0;
}