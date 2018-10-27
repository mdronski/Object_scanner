all: compile

compile:
	gcc main.c cnn_utils.c -lm -pthread -o main
	gcc camera.c -o camera
