#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>

static void close_all(int sigNum, siginfo_t* info, void* vp){
    kill(-getppid(), SIGTERM);
    exit(EXIT_SUCCESS);
}

static void initialise() {
    struct sigaction sigAction;
    sigfillset(&sigAction.sa_mask);
    sigAction.sa_flags = SA_SIGINFO;
    sigAction.sa_sigaction = &close_all;
    sigaction(SIGINT, &sigAction, NULL);

}


int main(){

    if (fork() == 0){
        system("./camera");
        exit(EXIT_SUCCESS);
    }
    sleep(1);

    if (fork() == 0){
        system("./main");
        exit(EXIT_SUCCESS);
    }

    sleep(1);

    if (fork() == 0){
        system("python gui.py");
        exit(EXIT_SUCCESS);
    }

    while (1){
        sleep(10);
    }

}
