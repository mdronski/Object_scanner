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

    double ***values;
} conv_layer;

typedef struct kernel {
    int size;
    int n_layers;
    int n_filters;

    double ****weights;
} kernel;


void print_kernel(kernel *K);

void print_conv_layer( conv_layer *L);

void print3D(double ***X, int depth, int height, int width);


conv_layer *allocate_conv_layer(int height, int width, int n_layers);

kernel *allocate_kernel(int size, int n_layers, int n_filters);


void free_conv_layer(conv_layer *L);

void free_kernel(kernel *K);


double max_from_2D(double **A, int height, int width, int range);

double conv_step(double ***L, double ***K, int layers, int filter_size, int h_start, int w_start);


conv_layer *test_conv_layer(int height, int width, int n_layers);

kernel *test_kernel(int size, int n_layers, int n_filters);


conv_layer *leaky_ReLu(conv_layer *L);

conv_layer *pad_0(conv_layer *L, int pad_range);


conv_layer *conv3D_paralel(conv_layer *L,  kernel *K, int stride, enum PADDING pad);

conv_layer *conv3D( conv_layer *L,  kernel *K, int stride, enum PADDING pad);

conv_layer *max_pool(conv_layer * L, int pool_size, int stride);

conv_layer *add_layers(conv_layer *L1, conv_layer *L2);

conv_layer *add_bias(conv_layer *L, double bias);
