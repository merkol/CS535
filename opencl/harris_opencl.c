#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>
#include <jpeglib.h>
#include <time.h>
#include "utilities.h"


void harris_detection_cl(cl_command_queue command_queue, cl_kernel kernel,
                         cl_mem old_pixels_buffer, cl_mem new_pixels_buffer,
                         const int width, const int height, const int depth,
                         cl_mem x_buff, cl_mem y_buff, cl_mem A, cl_mem B, cl_mem C,
                         cl_mem det, cl_mem trace, cl_mem R, cl_event event) {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &old_pixels_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &new_pixels_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &width);
    clSetKernelArg(kernel, 3, sizeof(int), &height);
    clSetKernelArg(kernel, 4, sizeof(int), &depth);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &x_buff);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &y_buff);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), &A);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), &B);
    clSetKernelArg(kernel, 9, sizeof(cl_mem), &C);
    clSetKernelArg(kernel, 10, sizeof(cl_mem), &det);
    clSetKernelArg(kernel, 11, sizeof(cl_mem), &trace);
    clSetKernelArg(kernel, 12, sizeof(cl_mem), &R);
    size_t global_work_size[2] = { width, height };
    size_t local_work[2] = { 16, 16};
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, &global_work_size, &local_work, 0, NULL, &event);
    clWaitForEvents(1, &event);
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    printf("Execution time in milliseconds = %3.3f ms\n", (end - start) * 1.0e-6);
    clFinish(command_queue);
}

int main(int argc, char **argv){
    setupOpenCL("kernel/harris.cl", "harris");
  
	struct image_t image ,new_image;


	/* Check command line usage */
	if (argc<2) {
		fprintf(stderr,"Usage: %s image_file\n",argv[0]);
		return -1;
	}
	load_jpeg(argv[1],&image);

	new_image.x=image.x;
	new_image.y=image.y;
	new_image.depth=image.depth;
	new_image.pixels=malloc(image.x*image.y*image.depth*sizeof(char));

    const size_t size = sizeof(float) * image.x * image.y * image.depth;
    cl_mem image_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char) * image.x * image.y * image.depth, NULL, NULL);
    cl_mem new_image_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size , NULL, NULL);
    cl_mem x_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);
    cl_mem y_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);

    cl_mem A = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);
    cl_mem B = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);
    cl_mem C = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);

    cl_mem det = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);
    cl_mem trace = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);
    cl_mem R = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);


    clEnqueueWriteBuffer(command_queue, image_buffer, CL_TRUE, 0,  sizeof(char) * image.x * image.y * image.depth, image.pixels, 0, NULL, NULL);
    cl_event event;
    harris_detection_cl(command_queue, kernel, image_buffer, new_image_buffer, image.x, image.y, image.depth, x_buff, y_buff, A, B, C, det, trace, R, event);

    clEnqueueReadBuffer(command_queue, new_image_buffer, CL_TRUE, 0,  sizeof(char) * image.x * image.y * image.depth, new_image.pixels, 0, NULL, NULL);


    store_jpeg("harris.jpg", &new_image, TRUE);
    cleanupOpenCL();
    return 0;
}
