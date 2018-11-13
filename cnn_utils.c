#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>

#include "cnn_utils.h"

//typedef struct conv_step_struct {
//    double ***L;
//    double ***K;
//    int layers;
//    int filter_size;
//    int h_start;
//    int w_start;
//    double result;
//} conv_step_struct;


typedef struct  {
    int f;
    int h;
    int stride;
    conv_layer *L;
    conv_layer *NL;
    kernel *K;

} conv_row_struct;


double max_from_2D(double **A, int height, int width, int range){
    double max = -999.0;
    for (int h = height; h < height + range; ++h) {
        for (int w = width; w < width + range; ++w) {
            max = A[h][w] > max ? A[h][w] : max;
        }
    }
    return max;
}

void print_kernel(kernel *K) {
    printf("size = %d, layers = %d, filters = %d\n", K->size, K->n_layers, K->n_filters);
//    for (int l = 0; l < K->n_layers; ++l) {
//        for (int h = 0; h < K->size; ++h) {
//            for (int w = 0; w < K->size; ++w) {
//                printf("%lf ", K->weights[0][l][h][w]);
//            }
//            printf("\n");
//        }
//        printf("\n");
//    }
}

void print_conv_layer( conv_layer *L) {
    int n_layers = 1;

    printf("Size = %d x %d, layers = %d\n", L->height, L->width, L->n_layers);
    for (int l = 0; l < n_layers; ++l) {
        for (int h = 0; h < L->height; ++h) {
            for (int w = 0; w < L->width; ++w) {
                printf("%.3lf ", L->values[l][h][w]);
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

conv_layer *allocate_conv_layer(int height, int width, int n_layers) {

    conv_layer *L = malloc(sizeof(conv_layer));
    L->height=height;
    L->width=width;
    L->n_layers = n_layers;

    double ***array = (double ***) malloc(n_layers * sizeof(double **));

    for (int l = 0; l < n_layers; ++l) {
        array[l] = (double **) malloc(height * sizeof(double *));
        for (int h = 0; h < height; ++h) {
            array[l][h] = (double *) malloc(width * sizeof(double));
        }
    }
    L->values = array;
    return L;
}

kernel *allocate_kernel(int size, int n_layers, int n_filters) {
    kernel *K = malloc(sizeof(kernel));
    K->size = size;
    K->n_layers = n_layers;
    K->n_filters = n_filters;

    double ****array = (double ****) malloc(n_filters * sizeof(double ***));

    for (int f = 0; f < n_filters; ++f) {
        array[f] = (double ***) malloc(n_layers * sizeof(double **));
        for (int l = 0; l < n_layers; ++l) {
            array[f][l] = (double **) malloc(size * sizeof(double *));
            for (int h = 0; h < size; ++h) {
//                printf("%d %d %d \n",f,l,h);
                array[f][l][h] = (double *) malloc(size * sizeof(double));

            }
        }
    }

    K->weights = array;
    return K;
}


void free_conv_layer(conv_layer *L){

    for (int l = 0; l < L->n_layers; ++l) {
        for (int h = 0; h < L->height; ++h) {
            free(L->values[l][h]);
        }
        free(L->values[l]);
    }
    free(L);
}

void free_kernel(kernel *K){

    for (int f = 0; f < K->n_filters; ++f) {
        for (int l = 0; l < K->n_layers; ++l) {
            for (int h = 0; h < K->size; ++h) {
                    free(K->weights[f][l][h]);
            }
            free(K->weights[f][l]);
        }
        free(K->weights[f]);
    }
    free(K);
}


//void *conv_step_paralel(void *args) {
//    conv_step_struct *S = (conv_step_struct *) args;
//
//    S->result = 0.0;
//
//    for (int l = 0; l < S->layers; ++l) {
//        for (int h = S->h_start; h < S->h_start + S->filter_size; ++h) {
//            for (int w = S->w_start; w < S->w_start + S->filter_size; ++w) {
//                S->result += S->L[l][h][w] * S->K[l][h - S->h_start][w - S->w_start];
//            }
//        }
//    }
//}

double conv_step(double ***L, double ***K, int layers, int filter_size, int h_start, int w_start) {
    double sum = 0.0;

    for (int l = 0; l < layers; ++l) {
        for (int h = h_start; h < h_start + filter_size; ++h) {
            for (int w = w_start; w < w_start + filter_size; ++w) {
//                printf("%d %d %d\n", l, h, w);
                sum += L[l][h][w] * K[l][h - h_start][w - w_start];
            }
        }
    }
//    fprintf(stderr, "%lf\n", sum);
    return sum;
}

conv_layer *pad_0(conv_layer *L, int pad_range){

    conv_layer *L2 = allocate_conv_layer(L->height + 2*pad_range, L->width + 2*pad_range, L->n_layers);



    for (int l = 0; l < L->n_layers; ++l) {

//        Top side
        for (int h = 0; h < pad_range; ++h) {
            for (int w = 0; w < L2->width; ++w) {
                L2->values[l][h][w] = 0;
            }
        }

//        Right side
        for (int h = 0; h < L->height; ++h) {
            for (int w = L2->width-pad_range; w < L2->width; ++w) {
                L2->values[l][h][w] = 0;
            }
        }

//        Bottom side
        for (int h = L2->height-pad_range; h < L2->height; ++h) {
            for (int w = 0; w < L2->width; ++w) {
                L2->values[l][h][w] = 0;
            }
        }

//        Left side
        for (int h = 0; h < L2->height; ++h) {
            for (int w = 0; w < pad_range; ++w) {
                L2->values[l][h][w] = 0;
            }
        }

//        Center
        for (int h = pad_range; h < L2->height-pad_range; ++h) {
            for (int w = pad_range; w < L2->width-pad_range; ++w) {
                L2->values[l][h][w] = L->values[l][h-pad_range][w-pad_range];
            }
        }
    }

    return L2;
}

conv_layer *conv3D( conv_layer *L,  kernel *K, int stride, enum PADDING pad) {

    if (L->n_layers != K->n_layers) {
        printf("\nWrong kernel size!\n");
        exit(EXIT_FAILURE);
    }

    conv_layer *NL = NULL;

    switch (pad) {

        case NO_PADDING:
            NL = allocate_conv_layer((((L->height - K->size)/stride) + 1),
                                     (((L->width - K->size)/stride) + 1), K->n_filters);
            break;
        case ZERO_PADDING:
            NL = allocate_conv_layer(L->height, L->width , K->n_filters);
            L = pad_0(L, (K->size - 1) /2) ;
            break;
    }



    for (int f = 0; f < NL->n_layers; ++f) {

        for (int h = 0; h < NL->height; ++h) {

            for (int w = 0; w < NL->width; ++w) {

                NL->values[f][h][w] = conv_step(L->values, K->weights[f], K->n_layers, K->size, h*stride, w*stride);

            }
        }
    }

    return NL;
}

void *row_convolution(void *args){
    conv_row_struct *S = (conv_row_struct *) args;

    for (int w = 0; w < S->NL->width; ++w) {

        S->NL->values[S->f][S->h][w] = conv_step(S->L->values, S->K->weights[S->f], S->K->n_layers, S->K->size, S->h*S->stride, w*S->stride);

    }
}

conv_layer *conv3D_paralel(conv_layer *L,  kernel *K, int stride, enum PADDING pad) {

    if (L->n_layers != K->n_layers) {
        printf("\nWrong kernel size!\n");
        exit(EXIT_FAILURE);
    }

    conv_layer *NL = NULL;

    switch (pad) {

        case NO_PADDING:
            NL = allocate_conv_layer((((L->height - K->size)/stride) + 1),
                                     (((L->width - K->size)/stride) + 1), K->n_filters);
            break;
        case ZERO_PADDING:
            NL = allocate_conv_layer(L->height, L->width , K->n_filters);
            L = pad_0(L, (K->size - 1) /2) ;
            break;
    }



//    printf("%d %d %d\n", NL->height, NL->width, NL->n_layers);

    pthread_t *threads = malloc(NL->height * sizeof(pthread_t));
    conv_row_struct *rows_structs = malloc(NL->height * sizeof(conv_row_struct));

    for (int f = 0; f < NL->n_layers; ++f) {

        for (int h = 0; h < NL->height; ++h) {
            rows_structs[h].f=f;
            rows_structs[h].h=h;
            rows_structs[h].K=K;
            rows_structs[h].L=L;
            rows_structs[h].NL=NL;
            rows_structs[h].stride=stride;

            pthread_create(&(threads[h]), NULL, row_convolution, (void *) &rows_structs[h]);

        }

        for (int h = 0; h < NL->height; ++h) {
            if(pthread_join(threads[h], NULL)){
                perror("JOIN ERROR");
            }
        }
//        free(threads);
//        free(rows_structs);

    }

    return NL;
}

conv_layer *test_conv_layer(int height, int width, int n_layers) {

    conv_layer *test_layer = allocate_conv_layer(height, width, n_layers);

    for (int l = 0; l < n_layers; ++l) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                test_layer->values[l][h][w] = 1.0;
            }
        }
    }
    return test_layer;
}

kernel *test_kernel(int size, int n_layers, int n_filters) {

    kernel *test_kernel = allocate_kernel(size, n_layers, n_filters);

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

conv_layer *max_pool(conv_layer * L, int pool_size, int stride){
    conv_layer *L2 = NULL;
    if (stride == 1) {
        L2 = allocate_conv_layer((L->height - pool_size + 1), (L->width - pool_size + 1), L->n_layers);
    } else {
        L2 = allocate_conv_layer(L->height/2, L->width/2, L->n_layers);
    }
//    print_conv_layer(L);

    for (int l = 0; l < L2->n_layers; ++l) {
        for (int h = 0; h < L2->height; h += stride) {
            for (int w = 0; w < L2->width; w += stride) {
                L2->values[l][h][w] = max_from_2D(L->values[l], h, w, pool_size);
            }
        }
    }

    return L2;
}

conv_layer *add_layers(conv_layer *L1, conv_layer *L2){
    if (L1->n_layers != L2->n_layers || L1->height != L2->height || L1->width != L2->width){
        printf("Wrong parameters for add_layers function!\n");
        exit(EXIT_FAILURE);
    }
    conv_layer * L3 = allocate_conv_layer(L1->height, L1->width, L1->n_layers);

    for (int l = 0; l < L3->n_layers; ++l) {
        for (int h = 0; h < L3->height; ++h) {
            for (int w = 0; w < L3->width; ++w) {
                L3->values[l][h][w] = L1->values[l][h][w] + L2->values[l][h][w];
            }
        }
    }
    return L3;
}

conv_layer *add_bias(conv_layer *L, double* bias){
    conv_layer *L2 = allocate_conv_layer(L->height, L->width, L->n_layers);

    for (int l = 0; l < L->n_layers; ++l) {
        for (int h = 0; h < L->height; ++h) {
            for (int w = 0; w < L->width; ++w) {
                L2->values[l][h][w] = L->values[l][h][w] + bias[l];
            }
        }
    }
    return L2;
}

conv_layer *leaky_ReLu(conv_layer *L){
    conv_layer * L2 = allocate_conv_layer(L->height, L->width, L->n_layers);

    for (int l = 0; l < L->n_layers; ++l) {
        for (int h = 0; h < L->height; ++h) {
            for (int w = 0; w < L->width; ++w) {
                L2->values[l][h][w] = MAX(L->values[l][h][w], 0.01);
            }
        }
    }
    return L2;
}

double ***load_anchors(conv_layer *L, int start){
    double ***anchors = malloc(L->width*L->height * sizeof(double **));

    for (int j = 0; j < L->width*L->height; ++j) {
        anchors[j] = malloc(3 * sizeof(double *));

        for (int i = 0; i < 3; ++i) {
            anchors[j][i] = malloc(85 * sizeof(double));
        }
    }

    for (int h = 0; h < L->height; ++h) {
        for (int w = 0; w < L->width; ++w) {
            for (int l = 0; l < L->n_layers; ++l) {

                for (int i = 0; i < start; ++i) {
                    for (int j = 0; j < 85; ++j) {
                        anchors[h*L->width + w][i][j] = L->values[j + i*85][h][w];
                    }

                }
            }
        }
    }

    return anchors;
}