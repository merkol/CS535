// utilities.h
#include <CL/cl.h>

#ifndef UTILITIES_H
#define UTILITIES_H

extern cl_context context;
extern cl_command_queue command_queue;
extern cl_program program;
extern cl_kernel kernel;
extern cl_platform_id platform;
extern cl_device_id device;

struct image_t
{
    int x;
    int y;
    int depth;	/* bytes */
    unsigned char *pixels;
};

char *load_kernel_source(const char *filename);
void setupOpenCL(const char *filename, const char *kernel_name);
void cleanupOpenCL();
int load_jpeg(char *filename, struct image_t *image);
int store_jpeg(char *filename, struct image_t *image, boolean grayscale);
void combine_and_store(struct image_t *s_x, struct image_t *s_y, struct image_t *new_image);


#endif /* UTILITIES_H */
