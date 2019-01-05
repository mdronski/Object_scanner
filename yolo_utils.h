
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

typedef struct yolo_box_node {
    yolo_box *box;
    struct yolo_box_node *next;
} yolo_box_node;



yolo_box *get_yolo_box(double tx, double ty, double tw, double th, double prob, int cell_x, int cell_y, int anchor_width, int anchor_height, int image_width, int image_height, int grid_size);

void print_yolo_box(yolo_box *box);

void scale_boxes(yolo_box ****boxes);

void softmax(yolo_box ****boxes, conv_layer *L);

yolo_box_node *non_max_supression(yolo_box ****boxes, double iou_threshold, int class);

double iou(yolo_box *box1, yolo_box *box2);