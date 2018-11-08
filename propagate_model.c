#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include "model_loader.c"

void run_model(){
    kernel *K;
    conv_layer *L;
    conv_layer *L2;
    conv_layer *L3;
    conv_layer *L4;
    printf("Input layer: ");
    printf("\n");
    L = test_conv_layer(416, 416, 3);
    print_conv_layer(L);


    K = load_kernel_by_number(0);
    print_kernel(K);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L4 = max_pool(L3, 2, 2);
    printf("After pooling: ");
    print_conv_layer(L4);
    L = L4;
    free_kernel(K);
    free_conv_layer(L2);
    free_conv_layer(L3);
//    free_conv_layer(L4);


    K = load_kernel_by_number(1);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L4 = max_pool(L3, 2, 2);
    printf("After pooling: ");
    print_conv_layer(L4);
    L = L4;
    free_kernel(K);
    free_conv_layer(L2);
    free_conv_layer(L3);
//    free_conv_layer(L4);


    K = load_kernel_by_number(2);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L4 = max_pool(L3, 2, 2);
    printf("After pooling: ");
    print_conv_layer(L4);
    L = L4;
    free_kernel(K);
    free_conv_layer(L2);
    free_conv_layer(L3);
//    free_conv_layer(L4);


    K = load_kernel_by_number(3);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L4 = max_pool(L3, 2, 2);
    printf("After pooling: ");
    print_conv_layer(L4);
    L = L4;
    free_kernel(K);
    free_conv_layer(L2);
    free_conv_layer(L3);
//    free_conv_layer(L4);


    K = load_kernel_by_number(4);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L4 = max_pool(L3, 2, 2);
    printf("After pooling: ");
    print_conv_layer(L4);
    L = L4;
    free_kernel(K);
    free_conv_layer(L2);
    free_conv_layer(L3);
//    free_conv_layer(L4);


    K = load_kernel_by_number(5);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L4 = max_pool(L3, 2, 1);
    printf("After pooling: ");
    print_conv_layer(L4);
    L = L4;
    free_kernel(K);
    free_conv_layer(L2);
    free_conv_layer(L3);
//    free_conv_layer(L4);


    K = load_kernel_by_number(6);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L = L3;
    free_kernel(K);
    free_conv_layer(L2);
//    free_conv_layer(L3);



    K = load_kernel_by_number(7);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L = L3;
    free_kernel(K);
    free_conv_layer(L2);
//    free_conv_layer(L3);



    K = load_kernel_by_number(8);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L = L3;
    free_kernel(K);
    free_conv_layer(L2);
//    free_conv_layer(L3);


    K = load_kernel_by_number(9);
    L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
    print_conv_layer(L2);
    free_conv_layer(L);
    L3 = leaky_ReLu(L2);
    L = L3;
    free_kernel(K);
    free_conv_layer(L2);
//    free_conv_layer(L3);


    print_conv_layer(L);
    free_conv_layer(L);






//    for (int i = 0; i < 10; ++i) {
//        kernel *K = load_kernel_by_number(i);
////        print_kernel(K);
//        L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
//        print_conv_layer(L2);
//        free_conv_layer(L);
//
//        L3 = leaky_ReLu(L2);
//
//
//        if (i==7 || i==8 || i==9){
//
//        }
//        L4 = i == 5 || i == 9 ?
//             (i == 9 ? L3 : max_pool(L3, 2, 1))
//                : max_pool(L3, 2, 2);
//        printf("After pooling: ");
//        print_conv_layer(L4);
//        L = L4;
//
//        free_kernel(K);
//        free_conv_layer(L2);
//        free_conv_layer(L3);
//        printf("\n");
//    }

//    print_conv_layer(L);


}



int main(){



    run_model();

//    conv_layer *L = test_conv_layer(10, 10, 3);
//    kernel *K = test_kernel(1, 1024, 256);
//    conv_layer *L2 = conv3D(L, K, 1, NO_PADDING);
//
//    print_conv_layer(L);
//    print_kernel(K);
//    print_conv_layer(L2);
//        conv_layer *L2 = max_pool(L, 2, 2);

//    print_conv_layer(L);
//    print_conv_layer(L2);




    return 0;
}