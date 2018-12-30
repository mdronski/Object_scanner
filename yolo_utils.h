
#include "model_loader.h"

typedef struct yolo_box {
    double x;
    double y;
    double height;
    double width;
    double confidence;
    int class;
    double class_probability;
    double left_up_x;
    double left_up_y;
    double right_bottom_x;
    double right_bottom_y;
} yolo_box;

yolo_box *get_yolo_box(double tx, double ty, double tw, double th, double prob, int cell_x, int cell_y, int anchor_width, int anchor_height, int image_width, int image_height);

void print_yolo_box(yolo_box *box);

void scale_boxes(yolo_box ****boxes);

void softmax(yolo_box ****boxes, conv_layer *L);