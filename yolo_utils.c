#include <malloc.h>
#include <math.h>
#include "yolo_utils.h"

yolo_box *
get_yolo_box(double tx, double ty, double tw, double th, double prob, int cell_x, int cell_y, int anchor_width,
             int anchor_height, int image_width, int image_height, int grid_size) {

    yolo_box *box = malloc(sizeof(yolo_box));

    box->x = (1 / (1 + exp(-tx)) + (double) cell_x) * ((double) image_width / grid_size);
    box->y = (1 / (1 + exp(-ty)) + (double) cell_y) * ((double) image_height / grid_size);

//    double tmp_y = box->y;
//    box->y = box->x;
//    box->x = tmp_y;

    box->width = (anchor_width * exp(tw)) > image_width ? (double) image_width * 0.5  : (anchor_width * exp(tw)) * 0.5 ;
    box->height = (anchor_height * exp(th)) > image_height ? (double) image_height * 0.5 : (anchor_height * exp(th)) * 0.5 ;
    box->confidence = prob;

    box->x_min = (box->x - box->width/2) < 0 ? 0 : (box->x - box->width/2);
    box->y_max = (box->y + box->height/2) > image_height ? image_height : (box->y + box->height/2);

    box->x_max = (box->x + box->width/2) > image_width ? image_width : (box->x + box->width/2);
    box->y_min = (box->y - box->height/2) < 0 ? 0 : (box->y - box->height/2);

//    tmp_y = box->y_max;
//    box->y_max = box->y_min;
//    box->y_min = tmp_y;

    double jiter = 0.3;
    if (box->width < jiter*image_width|| box->height < jiter*image_height ){
        box->confidence = -1.0;
    }

//    double jiter2 = 0.9;
//    if (box->width > jiter2*image_width || box->height > jiter2*image_height){
//        box->confidence = -1.0;
//    }

//    if (box->y_max > 415.0 || box->x_min < 0.5 || box->x_max > 415.0 || box->y_min < 1.0){
//        box->confidence = -1.0;
//    }
    return box;
}

void correct_yolo_box(yolo_box *box, int image_width, int image_height, int grid_size){




}

void print_yolo_box(yolo_box *box) {
    printf("Center = (%lf, %lf), height = %lf, width = %lf confidence = %lf class = %d class_prob = %lf\n",
           box->x, box->y, box->height, box->width, box->confidence, box->class, box->class_probability);
}

void softmax(yolo_box ****boxes, conv_layer *L){
    double sum ;
    int best_class;
    for (int i = 0; i < 13; ++i) {
        for (int j = 0; j < 13; ++j) {
            for (int k = 0; k < 3; ++k) {
                sum = 0.0;
                for (int l = 5; l < 85; ++l) {
                    sum += exp(L->values[l+k*85][i][j]);
                }

                for (int l = 5; l < 85; ++l) {
                    L->values[l+k*85][i][j] = exp(L->values[l+k*85][i][j]) / sum;
                }

                best_class = k*85 + 5;
                for (int l = 5; l < 85; ++l) {
                    if (L->values[l+k*85][i][j] > L->values[best_class][i][j]){
                        best_class = l+k*85;
                    }
                }

                boxes[i][j][k]->class = (best_class % 85) - 4;
                boxes[i][j][k]->class_probability = L->values[best_class][i][j];
            }
        }
    }
}