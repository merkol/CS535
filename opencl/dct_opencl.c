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

void dct_transform_cl(cl_command_queue command_queue, cl_kernel kernel,
                      cl_mem old_pixels_buffer, cl_mem new_pixels_buffer,
                     cl_mem idct_output_buffer, 
                      int width, int height, cl_event event) {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &old_pixels_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &new_pixels_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &idct_output_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &width);
    clSetKernelArg(kernel, 4, sizeof(int), &height);
    size_t global_work_size[2] = { width, height };
    size_t local_work[2] = { 2,2  };
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, &global_work_size, &local_work , 0, NULL, &event);
    clWaitForEvents(1, &event);
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    printf("Execution time in milliseconds = %3.3f ms\n", (end - start) * 1.0e-6);
    clFinish(command_queue);
}

int main(int argc, char **argv)
{
    setupOpenCL("kernel/dct.cl","dct2d");

    int param = 8;
    size_t size = param * param ;


    float* test_block = (float*)malloc(size * sizeof(float));
    float* dct_output = (float*) malloc(size * sizeof(float));
    float* idct_output =  (float*)malloc(size * sizeof(float));
    for (int i = 0; i < param * param; i++) {
        test_block[i] = 10;
    }


    cl_mem test_block_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(float), NULL, NULL);
    cl_mem dct_output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(float), NULL, NULL);
    cl_mem idct_output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(float), NULL, NULL);
  
    clEnqueueWriteBuffer(command_queue, test_block_buffer, CL_TRUE, 0, size * sizeof(float), test_block, 0, NULL, NULL);

    cl_event event;
    dct_transform_cl(command_queue, kernel, test_block_buffer, dct_output_buffer, idct_output_buffer, param, param, event);

    clEnqueueReadBuffer(command_queue, dct_output_buffer, CL_TRUE, 0, size * sizeof(float), dct_output, 0, NULL, NULL);
    clEnqueueReadBuffer(command_queue, idct_output_buffer, CL_TRUE, 0, size * sizeof(float), idct_output, 0, NULL, NULL);

    // DCT output print in 8x8
    printf("DCT output: \n");
    for (int i = 0; i < param * param; i++) {
        printf("%.2f ", dct_output[i]);
        if ((i + 1) % param == 0) {
            printf("\n");
        }
    }

    // IDCT output print in 8x8
    printf("IDCT output: \n");
    for (int i = 0; i < param * param; i++) {
        printf("%.2f ", idct_output[i]);
        if ((i + 1) % param == 0) {
            printf("\n");
        }
    }
    

    return 0;
}