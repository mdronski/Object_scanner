all: compile

compile:
	gcc -O3 main.c cnn_utils.c -lm -pthread -o main
	gcc -O3 camera.c -o camera
