

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>
#include <jpeglib.h>
#include <CL/cl.h>
#include <time.h>
#include "utilities.h"



void generic_convolve_cl(cl_command_queue command_queue, cl_kernel kernel,
                         cl_mem old_pixels_buffer, cl_mem new_pixels_buffer,
                         const int width, const int height, const int depth,
                        cl_event event) {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &old_pixels_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &new_pixels_buffer);
	clSetKernelArg(kernel, 2, sizeof(int), &width);
	clSetKernelArg(kernel, 3, sizeof(int), &height);
	clSetKernelArg(kernel, 4, sizeof(int), &depth);
 
    size_t global_work_size[2] = { width , height};
	size_t local_work_size[2] = {32, 32};
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, &global_work_size, &local_work_size, 0, NULL, &event);
	clWaitForEvents(1, &event);
	cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    printf("Execution time in milliseconds = %3.3f ms\n", (end - start) * 1.0e-6);
    clFinish(command_queue);
}


int main(int argc, char **argv) {
	setupOpenCL("kernel/sobel.cl", "generic_convolve");
	struct image_t image, new_image;

	/* Check command line usage */
	if (argc<2) {
		fprintf(stderr,"Usage: %s image_file\n",argv[0]);
		return -1;
	}
	/* Load an image */
	load_jpeg(argv[1],&image);

	/* Allocate space for output image */
	new_image.x=image.x;
	new_image.y=image.y;
	new_image.depth=image.depth;
	new_image.pixels=malloc(image.x*image.y*image.depth*sizeof(char));


	// Create buffer for input image
	cl_mem image_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, image.x * image.y * image.depth, NULL, NULL);
    cl_mem new_image_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, image.x * image.y * image.depth, NULL, NULL);

    // Copy input image to GPU
    clEnqueueWriteBuffer(command_queue, image_buffer, CL_TRUE, 0, image.x * image.y * image.depth, image.pixels, 0, NULL, NULL);
	

	cl_event event;
    // Execute sobel operation using OpenCL
    generic_convolve_cl(command_queue, kernel, image_buffer, new_image_buffer, image.x , image.y, image.depth, event);

	
	// Enqueue reading buffers back from GPU to CPU
	clEnqueueReadBuffer(command_queue, new_image_buffer, CL_TRUE, 0, image.x * image.y * image.depth, new_image.pixels, 0, NULL, NULL);


    // Combine and store result
   	store_jpeg("sobel.jpg",&new_image, 1);

    cleanupOpenCL();


    free(new_image.pixels);
	free(image.pixels);

	return 0;
}
