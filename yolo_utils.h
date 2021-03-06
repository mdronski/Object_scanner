
#include "model_loader.h"

typedef struct yolo_box {
    float x;
    float y;
    float height;
    float width;
    float confidence;
    int class;
    float class_probability;
    float x_min;
    float y_max;
    float x_max;
    float y_min;
} yolo_box;

typedef struct yolo_box_node {
    yolo_box *box;
    struct yolo_box_node *next;
} yolo_box_node;



yolo_box *get_yolo_box(float tx, float ty, float tw, float th, float conf, int cell_x, int cell_y, int anchor_width, int anchor_height, int image_width, int image_height, int grid_size, int index);

void print_yolo_box(yolo_box *box);

void scale_boxes(yolo_box ****boxes);

void softmax(yolo_box ****boxes, conv_layer *L, int grid_size);

yolo_box_node *non_max_supression(yolo_box ****boxes, float iou_threshold, int grid_size, float conf_thresh);

void final_non_max_supression(yolo_box_node *original_list, float iou_threshold);

float iou(yolo_box *box1, yolo_box *box2);

int list_size(yolo_box_node *l);

yolo_box_node *merge_lists(yolo_box_node *l1, yolo_box_node *l2);

void print_box_list(yolo_box_node *l);