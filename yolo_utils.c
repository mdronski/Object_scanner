#include <malloc.h>
#include <math.h>
#include "yolo_utils.h"

yolo_box *
get_yolo_box(double tx, double ty, double tw, double th, double prob, int cell_x, int cell_y, int anchor_width,
             int anchor_height, int image_width, int image_height) {
    yolo_box *box = malloc(sizeof(yolo_box));

    box->x = (1 / (1 + exp(-tx)) + (double) cell_x) * ((double) image_width / 13);
    box->y = (1 / (1 + exp(-ty)) + (double) cell_y) * ((double) image_height / 13);
    box->width = (anchor_width * exp(tw)) > image_height ? image_height : (anchor_width * exp(tw));
    box->height = (anchor_height * exp(th)) > image_width ? image_width : (anchor_height * exp(th));
    box->confidence = prob;

    if ((anchor_width * exp(tw)) > image_width || (anchor_height * exp(th)) > image_height)
        box->confidence = -1.0;

    box->left_up_x = box->x - box->width/2;
    box->left_up_y = box->y - box->height/2;

    box->right_bottom_x= box->x + box->width/2;
    box->right_bottom_y= box->y + box->height/2;

    return box;
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