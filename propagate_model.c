#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include "model_loader.c"

void run_model(){
    conv_layer *L = test_conv_layer(416, 416, 3);
    conv_layer *L2;
    conv_layer *L3;
    conv_layer *L4;
    print_conv_layer(L);


    for (int i = 0; i < 13; ++i) {
//        printf("1");
        kernel *K = load_kernel_by_number(i);
//        printf("2");
        L2 = conv3D_paralel(L, K, 1, ZERO_PADDING);
//        printf("3");
        print_conv_layer(L2);
//        printf("4");
        free_conv_layer(L);
//        printf("5");

        L3 = leaky_ReLu(L2);
//        printf("6");
        L4 = i == 5 ? max_pool(L3, 2, 1) : max_pool(L3, 2, 2);
        printf("After pooling: ");
        print_conv_layer(L4);
        L = L4;
//        printf("7");
//        free_kernel(K);
        free_conv_layer(L2);
        free_conv_layer(L3);
//        printf("8");
//        print_conv_layer(L);
        printf("\n");
    }

//    print_conv_layer(L);


}



int main(){



    run_model();

//    conv_layer *L = test_conv_layer(10, 10, 3);
//    kernel *K = test_kernel(1, 3, 10);
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