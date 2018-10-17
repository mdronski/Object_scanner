#include <stdio.h>
#include <stdlib.h>

#define MAX(X, Y)  ((X) > (Y) ? (X) : (Y))

enum PADDING {
    NO_PADDING,
    ZERO_PADDING
};


typedef struct conv_layer {
    int size;
    int n_layers;

    double ***values;
} conv_layer;

typedef struct kernel {
    int size;
    int n_layers;
    int n_filters;

    double ****weights;
} kernel;

double max_from_2D(double **A, int height, int width, int range){
    double max = -999.0;
//    printf("height = %d,  width = %d,  range = %d\n", height, width, range);

    for (int h = height; h < height + range; ++h) {
        for (int w = width; w < width + range; ++w) {
            max = A[h][w] > max ? A[h][w] : max;
        }
    }
    return max;
}

void print_kernel(kernel *K) {
    for (int l = 0; l < K->n_layers; ++l) {
        for (int h = 0; h < K->size; ++h) {
            for (int w = 0; w < K->size; ++w) {
                printf("%lf ", K->weights[0][l][h][w]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_conv_layer( conv_layer *L) {
    int n_layers = 1;

    printf("Size = %d\n", L->size);
    for (int l = 0; l < n_layers; ++l) {
        for (int h = 0; h < L->size; ++h) {
            for (int w = 0; w < L->size; ++w) {
                printf("%lf ", L->values[l][h][w]);
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

conv_layer *allocate_conv_layer(int size, int n_layers) {

    conv_layer *L = malloc(sizeof(conv_layer));
    L->size=size;
    L->n_layers = n_layers;

    double ***array = (double ***) malloc(n_layers * sizeof(double **));

    for (int h = 0; h < size; ++h) {
        array[h] = (double **) malloc(size * sizeof(double *));
        for (int w = 0; w < size; ++w) {
            array[h][w] = (double *) malloc(size * sizeof(double));
        }
    }
    L->values = array;
    return L;
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

void pad_0(conv_layer *L, int pad_range){
    for (int l = 0; l < L->n_layers; ++l) {

//        Top side
        for (int h = 0; h < pad_range; ++h) {
            for (int w = 0; w < L->size; ++w) {
                L->values[l][h][w] = 0;
            }
        }

//        Right side
        for (int h = 0; h < L->size; ++h) {
            for (int w = L->size-pad_range; w < L->size; ++w) {
                L->values[l][h][w] = 0;
            }
        }

//        Bottom side
        for (int h = L->size-pad_range; h < L->size; ++h) {
            for (int w = 0; w < L->size; ++w) {
                L->values[l][h][w] = 0;
            }
        }

//        Left side
        for (int h = 0; h < L->size; ++h) {
            for (int w = 0; w < pad_range; ++w) {
                L->values[l][h][w] = 0;
            }
        }
    }
}

conv_layer *conv3D( conv_layer *L,  kernel *K, int stride, enum PADDING pad) {

    if (L->n_layers != K->n_layers) {
        printf("\nWrong kernel size!\n");
        exit(EXIT_FAILURE);
    }

    conv_layer *NL = NULL;

    switch (pad){
        case NO_PADDING:
            NL = allocate_conv_layer((((L->size - K->size)/stride) + 1), K->n_layers);
            break;
        case ZERO_PADDING:
            NL = allocate_conv_layer(L->size, K->n_layers);
            break;
    }

    for (int f = 0; f < NL->n_layers; ++f) {

        for (int h = 0; h < NL->size; ++h) {

            for (int w = 0; w < NL->size; ++w) {

                NL->values[f][h][w] = conv_step(L->values, K->weights[f], K->n_layers, K->size, h*stride, w*stride);

            }
        }
    }

    return NL;
}

conv_layer *test_conv_layer(int size, int n_layers) {

     conv_layer *test_layer = allocate_conv_layer(size, n_layers);

    for (int l = 0; l < n_layers; ++l) {
        for (int h = 0; h < size; ++h) {
            for (int w = 0; w < size; ++w) {
                test_layer->values[l][h][w] = 1.0;
            }
        }
    }
    return test_layer;
}

kernel *test_kernel(int size, int n_layers, int n_filters) {

     kernel *test_kernel = malloc(sizeof( kernel));
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
    return test_kernel;
}

conv_layer *max_pool(conv_layer * L, int pool_size){
    conv_layer *L2 = allocate_conv_layer(L->size/pool_size, L->n_layers);
    print_conv_layer(L);

    for (int l = 0; l < L2->n_layers; ++l) {
        for (int h = 0; h < L2->size; ++h) {
            for (int w = 0; w < L2->size; ++w) {
                L2->values[l][h][w] = max_from_2D(L->values[l], 2*h, 2*w, pool_size);
            }
        }
    }

    return L2;
}

conv_layer *add_layers(conv_layer *L1, conv_layer *L2){
    if (L1->n_layers != L2->n_layers || L1->size != L2->size){
        printf("Wrong parameters for add_layers function!\n");
        exit(EXIT_FAILURE);
    }
    conv_layer * L3 = allocate_conv_layer(L1->size, L1->n_layers);

    for (int l = 0; l < L3->n_layers; ++l) {
        for (int h = 0; h < L3->size; ++h) {
            for (int w = 0; w < L3->size; ++w) {
                L3->values[l][h][w] = L1->values[l][h][w] + L2->values[l][h][w];
            }
        }
    }
    return L3;
}

conv_layer *leaky_ReLu(conv_layer *L){
    conv_layer * L2 = allocate_conv_layer(L->size, L->n_layers);

    for (int l = 0; l < L->n_layers; ++l) {
        for (int h = 0; h < L->size; ++h) {
            for (int w = 0; w < L->size; ++w) {
                L2->values[l][h][w] = MAX(L->values[l][h][w], 0.01);
            }
        }
    }
    return L2;
}

int main() {
     kernel *K = test_kernel(3, 3, 1);
     conv_layer *L = test_conv_layer(10, 3);
     pad_0(L, 2);
//     conv_layer *L2 = test_conv_layer(10, 3);
//     conv_layer *new_layer = conv3D(L, K, 1);


    print_conv_layer(L);
//    print_conv_layer(new_layer);

    return 0;
}