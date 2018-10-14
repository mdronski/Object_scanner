#include <stdio.h>
#include <stdlib.h>

struct conv_layer {
    int witdh;
    int height;
    int n_layers;

    double ***values;
} conv_layer;

struct kernel {
    int size;
    int n_layers;
    int n_filters;

    double ****weights;
} kernel;

void print_kernel(struct kernel K) {
    for (int l = 0; l < K.n_layers; ++l) {
        for (int h = 0; h < K.size; ++h) {
            for (int w = 0; w < K.size; ++w) {
                printf("%lf ", K.weights[0][l][h][w]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_conv_layer(struct conv_layer L) {
    for (int l = 0; l < L.n_layers; ++l) {
        for (int h = 0; h < L.height; ++h) {
            for (int w = 0; w < L.witdh; ++w) {
                printf("%lf ", L.values[0][h][w]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

void print3D(double ***X, int depth, int height, int width) {
    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                printf("%lf ", X[d][h][w]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


double ***allocate_conv_layer(int witdh, int height, int n_layers) {
    double ***array = (double ***) malloc(n_layers * sizeof(double **));

    for (int h = 0; h < height; ++h) {
        array[h] = (double **) malloc(height * sizeof(double *));
        for (int w = 0; w < witdh; ++w) {
            array[h][w] = (double *) malloc(witdh * sizeof(double));
        }
    }
    return array;
}

double ****allocate_kernel(int witdh, int height, int n_layers, int n_filters) {
    double ****array = (double ****) malloc(n_filters * sizeof(double ***));

    for (int l = 0; l < n_layers; ++l) {
        array[l] = (double ***) malloc(n_layers * sizeof(double **));
        for (int h = 0; h < height; ++h) {
            array[l][h] = (double **) malloc(height * sizeof(double *));
            for (int w = 0; w < witdh; ++w) {
                array[l][h][w] = (double *) malloc(witdh * sizeof(double));
            }
        }
    }
    return array;
}



double conv_step(double ***L, double ***K, int layers, int filter_size, int h_start, int w_start) {
    double sum = 0.0;

    for (int l = 0; l < layers; ++l) {
        for (int h = h_start; h < h_start + filter_size; ++h) {
            for (int w = w_start; w < w_start + filter_size; ++w) {
                sum += L[l][h][w] * K[l][h - h_start][w - w_start];
            }
        }
    }
    return sum;
}

struct conv_layer conv3D(struct conv_layer L, struct kernel K) {

    if (L.n_layers != K.n_layers) {
        printf("\nWrong kernel size!\n");
        exit(EXIT_FAILURE);
    }

    struct conv_layer *NL = malloc(sizeof(struct conv_layer));
    NL->n_layers = K.n_filters;
    NL->height = NL->witdh = L.witdh - K.size + 1;

    NL->values = allocate_conv_layer(NL->witdh, NL->height,
                                     NL->n_layers);
    for (int f = 0; f < NL->n_layers; ++f) {

        for (int h = 0; h < NL->height; ++h) {

            for (int w = 0; w < NL->witdh; ++w) {

                NL->values[f][h][w] = conv_step(L.values, K.weights[f], K.n_layers, K.size, h, w);

            }
        }
    }

    return *NL;
}

struct conv_layer test_conv_layer(int size, int n_layers) {

    struct conv_layer *test_layer = malloc(sizeof(struct conv_layer));
    test_layer->witdh = test_layer->height = size;
    test_layer->n_layers = n_layers;
    test_layer->values = allocate_conv_layer(size, size, n_layers);

    for (int l = 0; l < n_layers; ++l) {
        for (int h = 0; h < size; ++h) {
            for (int w = 0; w < size; ++w) {
                test_layer->values[l][h][w] = 1.0;
            }
        }
    }
    return *test_layer;
}

struct kernel test_kernel(int size, int n_layers, int n_filters) {

    struct kernel *test_kernel = malloc(sizeof(struct kernel));
    test_kernel->n_filters = n_filters;
    test_kernel->size = size;
    test_kernel->n_layers = n_layers;
    test_kernel->weights = allocate_kernel(size, size, n_layers, n_filters);

    for (int f = 0; f < n_filters; ++f) {
        for (int l = 0; l < n_layers; ++l) {
            for (int h = 0; h < size; ++h) {
                for (int w = 0; w < size; ++w) {
                    test_kernel->weights[f][l][h][w] = 1.0;
                }
            }
        }
    }
    return *test_kernel;
}


int main() {
    struct kernel K = test_kernel(3, 3, 1);
    struct conv_layer L = test_conv_layer(10, 3);
    struct conv_layer new_layer = conv3D(L, K);

    print_conv_layer(new_layer);

    return 0;
}