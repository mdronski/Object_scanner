#include <malloc.h>
#include <math.h>
#include "yolo_utils.h"

#define MAX(X, Y)  ((X) > (Y) ? (X) : (Y))
#define MIN(X, Y)  ((X) < (Y) ? (X) : (Y))

yolo_box *
get_yolo_box(double tx, double ty, double tw, double th, double prob, int cell_x, int cell_y, int anchor_width,
             int anchor_height, int image_width, int image_height, int grid_size) {

    yolo_box *box = malloc(sizeof(yolo_box));

    box->x = (1 / (1 + exp(-tx)) + (double) cell_x) * ((double) image_width / grid_size);
    box->y = (1 / (1 + exp(-ty)) + (double) cell_y) * ((double) image_height / grid_size);

    box->width = (anchor_width * exp(tw)) > image_width ? (double) image_width * 0.5 : (anchor_width * exp(tw)) * 0.5;
    box->height = (anchor_height * exp(th)) > image_height ? (double) image_height * 0.5 : (anchor_height * exp(th)) * 0.5;
//    box->width = (anchor_width * exp(tw)) > image_width ? (double) image_width * 0.75 : (anchor_width * exp(tw)) * 0.75;
//    box->height = (anchor_height * exp(th)) > image_height ? (double) image_height * 0.75 : (anchor_height * exp(th)) * 0.75;
    box->confidence = prob;

    box->x_min = (box->x - box->width / 2) < 0 ? 0 : (box->x - box->width / 2);
    box->y_max = (box->y + box->height / 2) > image_height ? image_height : (box->y + box->height / 2);

    box->x_max = (box->x + box->width / 2) > image_width ? image_width : (box->x + box->width / 2);
    box->y_min = (box->y - box->height / 2) < 0 ? 0 : (box->y - box->height / 2);


    double jiter = 0.3;
    if (box->width < jiter * image_width || box->height < jiter * image_height) {
        box->confidence = -1.0;
    }


    return box;
}



void print_yolo_box(yolo_box *box) {
    printf("Center = (%lf, %lf), height = %lf, width = %lf confidence = %lf class = %d class_prob = %lf\n",
           box->x, box->y, box->height, box->width, box->confidence, box->class, box->class_probability);
}

void softmax(yolo_box ****boxes, conv_layer *L) {
    double sum;
    int best_class;
    for (int i = 0; i < 13; ++i) {
        for (int j = 0; j < 13; ++j) {
            for (int k = 0; k < 3; ++k) {
                sum = 0.0;
                for (int l = 5; l < 85; ++l) {
                    sum += exp(L->values[l + k * 85][i][j]);
                }

//                int flag = 0;
                for (int l = 5; l < 85; ++l) {
//                    if(L->values[l + k * 85][i][j] > 28.0)
//                        flag = 1;
                    L->values[l + k * 85][i][j] = exp(L->values[l + k * 85][i][j]) / sum;
                }

                best_class = k * 85 + 5;
                for (int l = 5; l < 85; ++l) {
                    if (L->values[l + k * 85][i][j] > L->values[best_class][i][j]) {
                        best_class = l + k * 85;
                    }
                }

                boxes[i][j][k]->class = (best_class % 85) - 4;
                boxes[i][j][k]->class_probability = L->values[best_class][i][j];

//                if (flag == 0){
//                    boxes[i][j][k]->confidence = -1.0;
//                }
            }
        }
    }
}

double iou(yolo_box *box1, yolo_box *box2) {


    double x1 = MAX(box1->x_min, box2->x_min);
    double y1 = MAX(box1->y_min, box2->y_min);

    double x2 = MIN(box1->x_max, box2->x_max);
    double y2 = MIN(box1->y_max, box2->y_max);
//    printf("(%.1lf, %.1lf), (%.1lf, %1.lf)\n", x1, x2, y1, y2);

    double intersection_area = MAX((x2 - x1), 0) * MAX((y2 - y1), 0);

    double box1_area = (box1->x_max - box1->x_min) * (box1->y_max - box1->y_min);
    double box2_area = (box2->x_max - box2->x_min) * (box2->y_max - box2->y_min);
    double union_area = box1_area + box2_area - intersection_area;

    double iou = intersection_area / union_area;

    double xd = intersection_area / box1_area;
    return xd;
//    if (iou > 0.2) {
//        printf("%lf\n", iou);
//        printf("(%.1lf, %.1lf), (%.1lf, %1.lf)\n", box1->x_min, box1->x_max, box1->y_min, box1->y_max);
//        printf("(%.1lf, %.1lf), (%.1lf, %1.lf)\n", box2->x_min, box2->x_max, box2->y_min, box2->y_max);
//        printf("(%.1lf, %.1lf), (%.1lf, %1.lf)\n\n", x1, x2, y1, y2);
//
//    }


    return iou;
}

yolo_box_node *initialise_list() {
    yolo_box_node *list = malloc(sizeof(yolo_box_node));
    list->next = NULL;
    list->box = NULL;
    return list;
}

void add_to_box_list(yolo_box *box, yolo_box_node *list) {
    yolo_box_node *ptr = list;
    yolo_box_node *node = malloc(sizeof(yolo_box_node));
    node->box = box;
    node->next = NULL;

    if (ptr->box == NULL) {
        ptr->box = box;
        return;
    }

//    if(ptr->box->confidence < box->confidence){
//        node->next = ptr;
//        list = node;
//        return;
//    }

    while (ptr->next != NULL && ptr->next->box->confidence > box->confidence) {
        ptr = ptr->next;
    }

    node->next = ptr->next;
    ptr->next = node;
}

int list_size(yolo_box_node *list) {
    yolo_box_node *ptr = list;
    int cnt = 0;

    if (ptr->box == NULL) {
        return 0;
    }

    while (ptr->next != NULL) {
        ptr = ptr->next;
        cnt += 1;
    }

    return cnt;
}

yolo_box_node *non_max_supression(yolo_box ****boxes, double iou_threshold, int class) {
    yolo_box_node *list = initialise_list();
    for (int h = 0; h < 13; ++h) {
        for (int w = 0; w < 13; ++w) {
            for (int k = 0; k < 3; ++k) {
                if (boxes[h][w][k]->class == class) {
                    add_to_box_list(boxes[h][w][k], list);
                }
            }
        }
    }

    yolo_box_node *node = list;
    yolo_box_node *ptr = list;

    while (node->next != NULL) {
        while (ptr->next != NULL) {
            if (iou(node->box, ptr->next->box) > iou_threshold) {
                ptr->next->box->confidence = -1.0;
            }
            ptr = ptr->next;
        }
        node = node->next;
        ptr = node;
    }

    list = list->next;

    return list;
}