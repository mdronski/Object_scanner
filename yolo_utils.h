
#include "model_loader.h"

typedef struct yolo_box {
    double x;
    double y;
    double height;
    double width;
    double confidence;
    int class;
    double class_probability;
    double x_min;
    double y_max;
    double x_max;
    double y_min;
} yolo_box;

yolo_box *get_yolo_box(double tx, double ty, double tw, double th, double prob, int cell_x, int cell_y, int anchor_width, int anchor_height, int image_width, int image_height, int grid_size);

void print_yolo_box(yolo_box *box);

void scale_boxes(yolo_box ****boxes);

void softmax(yolo_box ****boxes, conv_layer *L);

void correct_yolo_box(yolo_box *box, int image_width, int image_height, int grid_size);
