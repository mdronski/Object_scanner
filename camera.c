#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <getopt.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <math.h>
#include <pthread.h>


#include <linux/videodev2.h>
#include <libv4l2.h>
#include <bits/types/siginfo_t.h>
#include <libnet.h>

#define STREAM_NAME "image_stream"

#define CLEAR(x) memset(&(x), 0, sizeof(x))
#define MAX(X, Y)  ((X) > (Y) ? (X) : (Y))
#define MIN(X, Y)  ((X) < (Y) ? (X) : (Y))
#define CLIP(X) (MIN(MAX(X, 0), 255))

struct buffer {
    void *start;
    size_t length;
};

static char *dev_name;
static int camera_fd = -1;
struct buffer *buffers;
static unsigned int n_buffers;
static int out_buf = 1;
static int force_format = 1;
static int frame_count = 1000;
int frame_counter = 0;
__u32 witdh = 640;
__u32 height = 480;

FILE *image_stream;

void v4lconvert_yuyv_to_rgb24(const unsigned char *src, unsigned char *dest,
                              int width, int height) {
    int j;
    while (--height >= 0) {
        for (j = 0; j < width; j += 2) {
            int u = src[1];
            int v = src[3];
            int u1 = (((u - 128) << 7) + (u - 128)) >> 6;
            int rg = (((u - 128) << 1) + (u - 128) +
                      ((v - 128) << 2) + ((v - 128) << 1)) >> 3;
            int v1 = (((v - 128) << 1) + (v - 128)) >> 1;

            *dest++ = CLIP(src[0] + v1);
            *dest++ = CLIP(src[0] - rg);
            *dest++ = CLIP(src[0] + u1);

            *dest++ = CLIP(src[2] + v1);
            *dest++ = CLIP(src[2] - rg);
            *dest++ = CLIP(src[2] + u1);
            src += 4;
        }
    }

}

static void errno_exit(const char *s) {
    fprintf(stderr, "%s error %d, %s\\n", s, errno, strerror(errno));
    exit(EXIT_FAILURE);
}

static int xioctl(int fh, int request, void *arg) {
    int r;

    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);

    return r;
}

static void process_image(const void *p, int size) {
    if (out_buf && frame_counter > 50 && frame_counter % 1 == 0) {
//        FILE *out_file = fopen("out.ppm", "w");
//        fprintf(out_file, "P6\n%d %d\n255\n", witdh, height);

        void *rgb_image = malloc(witdh * height * 3);
        __u8 *rgb_ptr = (__u8 *) rgb_image;

        v4lconvert_yuyv_to_rgb24(p, rgb_image, witdh, height);


//        for (int i = 0; i < 64; ++i) {
//            fprintf(stderr, "%d ", rgb_ptr[i]);
//        }

//        for (int i = 0; i < witdh * height * 3; i++) {
//            fprintf(out_file, "%c", rgb_ptr[i]);
//        }

//        fwrite(rgb_ptr, 1, witdh * height * 3, out_file);

        fwrite(rgb_ptr, 1, witdh * height * 3, image_stream);

        free(rgb_image);
//        fclose(out_file);
//        usleep(20);
    }

//    fflush(stderr);
    fprintf(stderr, ".");
//    fflush(stdout);
}

static int read_frame(void) {
    struct v4l2_buffer buf;
    unsigned int i;
    frame_counter++;

    CLEAR(buf);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(camera_fd, VIDIOC_DQBUF, &buf)) {
        switch (errno) {
            case EAGAIN:
                return 0;

            default:
                errno_exit("VIDIOC_DQBUF");
        }
    }

    assert(buf.index < n_buffers);

    process_image(buffers[buf.index].start, buf.bytesused);

    if (-1 == xioctl(camera_fd, VIDIOC_QBUF, &buf))
        errno_exit("VIDIOC_QBUF");

    return 1;
}

static void mainloop(void) {
    unsigned int count;

    count = frame_count;

    while (count-- > 0) {
//    while (1) {
        while (1) {
            fd_set fds;
            struct timeval tv;
            int r;

            FD_ZERO(&fds);
            FD_SET(camera_fd, &fds);

            /* Timeout. */
            tv.tv_sec = 2;
            tv.tv_usec = 0;

            r = select(camera_fd + 1, &fds, NULL, NULL, &tv);

            if (-1 == r) {
                if (EINTR == errno)
                    continue;
                errno_exit("select");
            }

            if (0 == r) {
                fprintf(stderr, "select timeout\\n");
                exit(EXIT_FAILURE);
            }

            if (read_frame())
                break;
            /* EAGAIN - continue select loop. */
        }
    }
}

static void stop_capturing(void) {
    enum v4l2_buf_type type;

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(camera_fd, VIDIOC_STREAMOFF, &type))
        errno_exit("VIDIOC_STREAMOFF");
}

static void start_capturing(void) {
    unsigned int i;
    enum v4l2_buf_type type;

    for (i = 0; i < n_buffers; ++i) {
        struct v4l2_buffer buf;

        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (-1 == xioctl(camera_fd, VIDIOC_QBUF, &buf))
            errno_exit("VIDIOC_QBUF");
    }
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(camera_fd, VIDIOC_STREAMON, &type))
        errno_exit("VIDIOC_STREAMON");

}

static void uninit_device(void) {
    unsigned int i;

    for (i = 0; i < n_buffers; ++i)
        if (-1 == munmap(buffers[i].start, buffers[i].length))
            errno_exit("munmap");

    free(buffers);
}

static void init_read(unsigned int buffer_size) {
    buffers = calloc(1, sizeof(*buffers));

    if (!buffers) {
        fprintf(stderr, "Out of memory\\n");
        exit(EXIT_FAILURE);
    }

    buffers[0].length = buffer_size;
    buffers[0].start = malloc(buffer_size);

    if (!buffers[0].start) {
        fprintf(stderr, "Out of memory\\n");
        exit(EXIT_FAILURE);
    }
}

static void init_mmap(void) {
    struct v4l2_requestbuffers req;

    CLEAR(req);

    req.count = 16;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(camera_fd, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            fprintf(stderr, "%s does not support "
                            "memory mappingn", dev_name);
            exit(EXIT_FAILURE);
        } else {
            errno_exit("VIDIOC_REQBUFS");
        }
    }

    if (req.count < 2) {
        fprintf(stderr, "Insufficient buffer memory on %s\\n",
                dev_name);
        exit(EXIT_FAILURE);
    }

    buffers = calloc(req.count, sizeof(*buffers));

    if (!buffers) {
        fprintf(stderr, "Out of memory\\n");
        exit(EXIT_FAILURE);
    }

    for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
        struct v4l2_buffer buf;

        CLEAR(buf);

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = n_buffers;

        if (-1 == xioctl(camera_fd, VIDIOC_QUERYBUF, &buf))
            errno_exit("VIDIOC_QUERYBUF");

        buffers[n_buffers].length = buf.length;
        buffers[n_buffers].start =
                mmap(NULL /* start anywhere */,
                     buf.length,
                     PROT_READ | PROT_WRITE /* required */,
                     MAP_SHARED /* recommended */,
                     camera_fd, buf.m.offset);

        if (MAP_FAILED == buffers[n_buffers].start)
            errno_exit("mmap");
    }
}

static void init_device(void) {
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
    unsigned int min;

    if (-1 == xioctl(camera_fd, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            fprintf(stderr, "%s is no V4L2 device\\n",
                    dev_name);
            exit(EXIT_FAILURE);
        } else {
            errno_exit("VIDIOC_QUERYCAP");
        }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "%s is no video capture device\\n",
                dev_name);
        exit(EXIT_FAILURE);
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "%s does not support streaming i/o\\n",
                dev_name);
        exit(EXIT_FAILURE);
    }

    /* Select video input, video standard and tune here. */

    CLEAR(cropcap);

    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (0 == xioctl(camera_fd, VIDIOC_CROPCAP, &cropcap)) {
        crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        crop.c = cropcap.defrect; /* reset to default */

        if (-1 == xioctl(camera_fd, VIDIOC_S_CROP, &crop)) {
            switch (errno) {
                case EINVAL:
                    fprintf(stderr, "Cropping not supported\n");
                    break;
                default:
                    break;
            }
        }
    }

    CLEAR(fmt);

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (force_format) {
        fmt.fmt.pix.width = witdh;
        fmt.fmt.pix.height = height;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

        if (-1 == xioctl(camera_fd, VIDIOC_S_FMT, &fmt))
            errno_exit("VIDIOC_S_FMT");

    } else {

        if (-1 == xioctl(camera_fd, VIDIOC_G_FMT, &fmt))
            errno_exit("VIDIOC_G_FMT");
    }


    min = fmt.fmt.pix.width * 2;
    if (fmt.fmt.pix.bytesperline < min)
        fmt.fmt.pix.bytesperline = min;
    min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
    if (fmt.fmt.pix.sizeimage < min)
        fmt.fmt.pix.sizeimage = min;


    init_mmap();

}

static void close_device() {
    if (-1 == close(camera_fd))
        errno_exit("close");

    camera_fd = -1;
}

static void open_device() {
    struct stat st;

    if (-1 == stat(dev_name, &st)) {
        fprintf(stderr, "Cannot identify '%s': %d, %s\\n",
                dev_name, errno, strerror(errno));
        exit(EXIT_FAILURE);
    }

    if (!S_ISCHR(st.st_mode)) {
        fprintf(stderr, "%s is no devicen", dev_name);
        exit(EXIT_FAILURE);
    }

    camera_fd = open(dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

    if (-1 == camera_fd) {
        fprintf(stderr, "Cannot open '%s': %d, %s\\n",
                dev_name, errno, strerror(errno));
        exit(EXIT_FAILURE);
    }
}

static void usage(FILE *fp, int argc, char **argv) {
    fprintf(fp,
            "Usage: \n"
            "-h | --help          Print this message\n"
            "-d | --device name   Video device name [%s]\n"
            "-c | --count         Number of frames to grab [%d]\n"
            "",
            dev_name, frame_count);
}

static const char short_options[] = "d:hmruofc:";

static void close_stream(int sigNum, siginfo_t* info, void* vp){
    fclose(image_stream);
    remove(STREAM_NAME);
    exit(EXIT_SUCCESS);
}

static void initialise_stream() {
    struct sigaction sigAction;
    sigfillset(&sigAction.sa_mask);
    sigAction.sa_flags = SA_SIGINFO;
    sigAction.sa_sigaction = &close_stream;
    sigaction(SIGINT, &sigAction, NULL);

    if (mkfifo(STREAM_NAME, 0777) == -1) {
        perror("error");
    }

    image_stream = fopen(STREAM_NAME, "w");

}

static const struct option
        long_options[] = {
        {"device", required_argument, NULL, 'd'},
        {"help",   no_argument,       NULL, 'h'},
        {"count",  required_argument, NULL, 'c'},
        {0, 0, 0,                           0}
};


int main(int argc, char **argv) {
    dev_name = "/dev/video0";

    for (;;) {
        int idx;
        int c;

        c = getopt_long(argc, argv,
                        short_options, long_options, &idx);

        if (-1 == c)
            break;

        switch (c) {
            case 0: /* getopt_long() flag */
                break;

            case 'd':
                dev_name = optarg;
                break;

            case 'h':
                usage(stdout, argc, argv);
                exit(EXIT_SUCCESS);

            case 'c':
                errno = 0;
                frame_count = strtol(optarg, NULL, 0);
                if (errno)
                    errno_exit(optarg);
                break;

            default:
                usage(stderr, argc, argv);
                exit(EXIT_FAILURE);
        }
    }




    atexit(close_stream);

    open_device();
    init_device();
    initialise_stream();
    start_capturing();
    mainloop();
    stop_capturing();
    uninit_device();
    close_device();
    close_stream(NULL, NULL, NULL);
    fprintf(stderr, "\n");
    return 0;
}
