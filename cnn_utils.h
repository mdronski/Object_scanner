#ifndef PROJEKT_CNN_UTILS_H
#define PROJEKT_CNN_UTILS_H

#endif //PROJEKT_CNN_UTILS_H

#define MAX(X, Y)  ((X) > (Y) ? (X) : (Y))


enum PADDING {
    NO_PADDING,
    ZERO_PADDING
};

enum ACTIVATION {
    lINEAR,
    RELU
};

typedef struct conv_layer {
    int height;
    int width;
    int n_layers;

    float ***values;
} conv_layer;

typedef struct kernel {
    int size;
    int n_layers;
    int n_filters;

    float ****weights;
} kernel;


void print_kernel(kernel *K);

void print_conv_layer( conv_layer *L);

void print_conv_layer_weights(conv_layer *L, int hs, int hk, int ws, int wk, int channels);

void print_conv_layer_one_l( conv_layer *L);

void print3D(float ***X, int depth, int height, int width);

void print_pred_layer_anchor(conv_layer *L);

conv_layer *allocate_conv_layer(int height, int width, int n_layers);

kernel *allocate_kernel(int size, int n_layers, int n_filters);


void free_conv_layer(conv_layer *L);

void free_kernel(kernel *K);


float max_from_2D(float **A, int height, int width, int range);

float conv_step(float ***L, float ***K, int layers, int filter_size, int h_start, int w_start);


conv_layer *test_conv_layer(int height, int width, int n_layers);

kernel *test_kernel(int size, int n_layers, int n_filters);


conv_layer *leaky_ReLu(conv_layer *L);

conv_layer *pad_0(conv_layer *L, int pad_range);


conv_layer *conv3D_paralel(conv_layer *L,  kernel *K, int stride, enum PADDING pad);

conv_layer *conv3D( conv_layer *L,  kernel *K, int stride, enum PADDING pad);

conv_layer *max_pool(conv_layer * L, int pool_size, int stride);

conv_layer *batch_normalization(conv_layer *L, float *mean, float *variance, float *gamma, float *beta);

conv_layer *add_layers(conv_layer *L1, conv_layer *L2);

conv_layer *add_bias(conv_layer *L, float* bias);

float ***load_anchors(conv_layer *L, int start);

conv_layer *upscale(conv_layer *L);

conv_layer *concatenate(conv_layer *L1, conv_layer *L2);
