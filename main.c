#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libnet.h>

#include "cnn_utils.h"

#define STREAM_NAME "image_stream"
FILE *gui_stream;
#define PYTHON_GUI "python_gui"


kernel *initialise_kernel(){

    kernel *kernel1 = allocate_kernel(3, 3, 2);

    for (int l = 0; l < 3; ++l) {
        kernel1->weights[0][l][0][0] = 1.0;
        kernel1->weights[0][l][0][1] = 0.0;
        kernel1->weights[0][l][0][2] = -1.0;

        kernel1->weights[0][l][1][0] = 2.0;
        kernel1->weights[0][l][1][1] = 0.0;
        kernel1->weights[0][l][1][2] = -2.0;

        kernel1->weights[0][l][2][0] = 1.0;
        kernel1->weights[0][l][2][1] = 0.0;
        kernel1->weights[0][l][2][2] = -1.0;



        kernel1->weights[1][l][0][0] = 1.0;
        kernel1->weights[1][l][0][1] = 2.0;
        kernel1->weights[1][l][0][2] = 1.0;

        kernel1->weights[1][l][1][0] = 0.0;
        kernel1->weights[1][l][1][1] = 0.0;
        kernel1->weights[1][l][1][2] = 0.0;

        kernel1->weights[1][l][2][0] = -1.0;
        kernel1->weights[1][l][2][1] = -2.0;
        kernel1->weights[1][l][2][2] = -1.0;
    }

    return kernel1;

}

static void close_stream(int sigNum, siginfo_t* info, void* vp){
    fclose(gui_stream);
    remove(PYTHON_GUI);
    exit(EXIT_SUCCESS);
}

static void initialise_stream() {
    struct sigaction sigAction;
    sigfillset(&sigAction.sa_mask);
    sigAction.sa_flags = SA_SIGINFO;
    sigAction.sa_sigaction = &close_stream;
    sigaction(SIGINT, &sigAction, NULL);

    if (mkfifo(PYTHON_GUI, 0777) == -1) {
        perror("error");
    }

    gui_stream = fopen(PYTHON_GUI, "wb");

}

void process_image(__uint8_t *image_ptr){

    conv_layer *L = allocate_conv_layer(640, 3);
    kernel *K = initialise_kernel();


//    char *buffer = malloc(100 * sizeof(char));
//    size_t size = 100;
//    getline(&buffer, &size, image_ptr);
//    getline(&buffer, &size, image_ptr);
//    getline(&buffer, &size, image_ptr);
//    fprintf(stderr, "%d ", tmp++);

    for (int h = 0; h < 640; ++h) {
        for (int w = 0; w < 640; ++w) {
            for (int l = 0; l < 3; ++l) {
                L->values[l][h][w] = (double) image_ptr[h*640*3 + 3*w + l];
            }
        }
    }



//    print_kernel(K);
//    fprintf(stderr, "%d %d\n", kernel1->n_layers, kernel1->n_filters);
//    fprintf(stderr, "%d %d\n", L->n_layers, L->size);

    conv_layer *conv_layer2 = conv3D_paralel(L, K, 1, NO_PADDING);

//    print_conv_layer(conv_layer2);

    conv_layer *conv_layer3 = allocate_conv_layer(638, 1);

//    fprintf(stderr, "%d %d\n", conv_layer2->n_layers, conv_layer2->size);
//    free(kernel1);
//    free(L);

//    fprintf(stderr, "%d %d\n", conv_layer2->n_layers, conv_layer3->n_layers);

    double max = -999.0;
    for (int h = 0; h < 638; ++h) {
        for (int w = 0; w < 638; ++w) {
//            fprintf(stderr, "%d %d\n", h, w);
//            fprintf(stderr, "%lf %lf\n", conv_layer2->values[0][h][w], conv_layer2->values[1][h][w]);
            conv_layer3->values[0][h][w] = sqrt(
                    conv_layer2->values[0][h][w] * conv_layer2->values[0][h][w]
                    + conv_layer2->values[1][h][w] * conv_layer2->values[1][h][w]
            );

            if (max < conv_layer3->values[0][h][w])
                max = conv_layer3->values[0][h][w];
        }
    }



//    print_conv_layer(conv_layer3);

    size_t img_size = 3 * 638 * 638;
    unsigned char *image_buffer = malloc(img_size + 15 +638);

//    FILE *out_file = fopen("filtered.ppm", "w");
//    fprintf(out_file, "P6\n%d %d\n255\n", 638, 638);
    image_buffer[0] = 'P';
    image_buffer[1] = '6';
    image_buffer[2] = '\n';
    image_buffer[3] = '6';
    image_buffer[4] = '3';
    image_buffer[5] = '8';
    image_buffer[6] = ' ';
    image_buffer[7] = '6';
    image_buffer[8] = '3';
    image_buffer[9] = '8';
    image_buffer[10] = '\n';
    image_buffer[11] = '2';
    image_buffer[12] = '5';
    image_buffer[13] = '5';
    image_buffer[14] = '\n';
    int i = 15;
    for (int h = 0; h < 638; ++h) {
        for (int w = 0; w < 638; ++w) {
            unsigned char c = (unsigned char) ((conv_layer3->values[0][h][w] / max) * 255.0);
//            printf("%d %d %d\n", h, w, c);
//            fwrite(&c, 1, sizeof(char), out_file);
            image_buffer[i++] = c;
            image_buffer[i++] = c;
            image_buffer[i++] = c;
//            fwrite(&c, 1, sizeof(char), out_file);
//            fwrite(&c, 1, sizeof(char), out_file);
        }
        image_buffer[i++] = 0;
        image_buffer[i++] = 0;

    }
//    fclose(out_file);
//    for (int i = 0; i < 64; ++i) {
//        fprintf(stderr, "%d ", image_buffer[i]);
//    }
//    printf("\n\n");


    FILE *out_file = fopen("filtered.ppm", "w");
    fprintf(out_file, "P6\n%d %d\n255\n", 638, 638);
    for (int h = 0; h < 638; ++h) {
        for (int w = 0; w < 638; ++w) {
            unsigned char c = (unsigned char) ((conv_layer3->values[0][h][w] / max) * 255.0);
//            printf("%d %d %d\n", h, w, c);
            fwrite(&c, 1, sizeof(char), out_file);
            fwrite(&c, 1, sizeof(char), out_file);
            fwrite(&c, 1, sizeof(char), out_file);
        }
    }

    fclose(out_file);


    fwrite(image_buffer, 1, img_size, gui_stream);
    fflush(gui_stream);
    fclose(gui_stream);
    gui_stream = fopen(PYTHON_GUI, "wb");


    free_conv_layer(L);
    free_conv_layer(conv_layer2);
    free_conv_layer(conv_layer3);
    free_kernel(K);


}


int main() {
//     FILE *fd = fopen("out.ppm", "r");
//
//     if (!fd){
//         printf("Error");
//     }
//
//    char *buffer = malloc(100 * sizeof(char));
//    size_t size = 100;
//    getline(&buffer, &size, fd);
//    getline(&buffer, &size, fd);
//     getline(&buffer, &size, fd);

//     char c;
//     conv_layer *L = allocate_conv_layer(640, 3);
//    for (int h = 0; h < 640; ++h) {
//        for (int w = 0; w < 640; ++w) {
//            for (int l = 0; l < 3; ++l) {
//                L->values[l][h][w] = (double) getc(fd);
//            }
//        }
//    }





// WORKING CODE HERE



    atexit(close_stream);

    FILE *image_stream = fopen(STREAM_NAME, "r");
    size_t img_size = 3*640*640;
    __uint8_t *image_ptr = malloc(img_size * sizeof(__uint8_t));
    initialise_stream();

    while (image_stream){
        fread(image_ptr, img_size, 1, image_stream);
//        for (int i = 0; i < 64; ++i) {
//            fprintf(stderr, "%d ", image_ptr[i]);
//        }
        fprintf(stderr, "\n\nNEXT image\n\n");

        process_image(image_ptr);

    }

    close_stream(NULL, NULL, NULL);
    free(image_ptr);
    fclose(image_stream);


// END OF WORKING CODE


//    conv_layer *L = test_conv_layer(10, 3);
//    kernel *K = test_kernel(3,3,2);
//
//    conv_layer *L2 = conv3D_paralel(L, K, 1, NO_PADDING);
//
//    print_conv_layer(L2);
//
//



//
//    conv_layer *L = allocate_conv_layer(640, 3);
//    kernel *K = initialise_kernel();
//
//
//    print_kernel(K);
////    fprintf(stderr, "%d %d\n", kernel1->n_layers, kernel1->n_filters);
////    fprintf(stderr, "%d %d\n", L->n_layers, L->size);
//
//    conv_layer *conv_layer2 = conv3D(L, K, 1, NO_PADDING);
////    fprintf(stderr, "%d %d\n", conv_layer2->n_layers, conv_layer2->size);
////    free(kernel1);
////    free(L);
//
//    conv_layer *conv_layer3 = allocate_conv_layer(638, 1);
//
////    fprintf(stderr, "%d %d\n", conv_layer2->n_layers, conv_layer3->n_layers);
//
//    double max = -999.0;
//    for (int h = 0; h < 638; ++h) {
//        for (int w = 0; w < 638; ++w) {
////            fprintf(stderr, "%d %d\n", h, w);
////            fprintf(stderr, "%lf %lf\n", conv_layer2->values[0][h][w], conv_layer2->values[1][h][w]);
//            conv_layer3->values[0][h][w] = sqrt(
//                    conv_layer2->values[0][h][w] * conv_layer2->values[0][h][w]
//                    + conv_layer2->values[1][h][w] * conv_layer2->values[1][h][w]
//            );
//
//            if (max < conv_layer3->values[0][h][w])
//                max = conv_layer3->values[0][h][w];
//        }
//    }
//
//
//    print_conv_layer(conv_layer3);
//
//    FILE *out_file = fopen("filtered.ppm", "w");
//    fprintf(out_file, "P6\n%d %d\n255\n", 638, 638);
//    for (int h = 0; h < 638; ++h) {
//        for (int w = 0; w < 638; ++w) {
//            unsigned char c = (unsigned char) ((conv_layer3->values[0][h][w] / max) * 254.0);
////            printf("%d %d %d\n", h, w, c);
//            fwrite(&c, 1, sizeof(char), out_file);
//            fwrite(&c, 1, sizeof(char), out_file);
//            fwrite(&c, 1, sizeof(char), out_file);
//        }
//    }
//    fclose(out_file);



    return 0;
}