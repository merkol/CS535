#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>
#include <jpeglib.h>
#include <time.h>
#include "cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>	
#include <device_launch_parameters.h>



__global__ void dct2d(float* input, float* dct_output,
                int width, int height)
{
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        float pi = 3.14159265358979323846f;
        float sum = 0.0f;
        float ci, cj, dct1;
        int k, l;

        if ( x < width  && y < height) {
            if (x == 0)
                ci = 1 / sqrtf(width);
            else
                ci = sqrtf(2) / sqrtf(width);
            if (y == 0)
                cj = 1 / sqrtf(height);
            else
                cj = sqrtf(2) / sqrtf(height);
 
            sum = 0;
            for (k = 0; k < width; k++) {
                for (l = 0; l < height; l++) {
                    dct1 = input[x + (y * width)] * 
                           cos((2 * k + 1) * x * pi / (2 * width)) * 
                           cos((2 * l + 1) * y * pi / (2 * height));
                    sum += dct1;
                }
            }
            dct_output[x+ (y* width)] = ci * cj * sum; 
    }
}

__global__ void idct2d(const float* dct_output,
                float* idct_output,
                int width, int height)
{
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        float pi = 3.14159265358979323846f;
        float sum = 0.0f;
        float cu, cv;
        int u, v;
         if ( x < width  && y < height) {
            for (u = 0; u < width; u++) {
                for (v = 0; v < height; v++) {

            if (u == 0) {
                cu = 1 / sqrtf(width);
            } else {
                cu = sqrtf(2) / sqrtf(width);
              }

            if (v == 0) {
                cv = 1 / sqrtf(height);
            } else {
                cv = sqrtf(2) / sqrtf(height); 
              }   

                float idct = (dct_output[u + (v * width)] * cu * cv *      
                           cos((2 * x + 1) * u * pi / (2 * width)) * 
                           cos((2 * y + 1) * v * pi / (2 * height)));

                sum += idct;
                }               
            }
            idct_output[x + (y * width)] =  sum;    
         }
}
    





int main(int argc, char **argv)
{

    int block_size = 8;
    float* test_input = (float *)malloc(block_size * block_size * sizeof(float));
    float* test_output = (float *)malloc(block_size * block_size * sizeof(float));
    float* test_dct_output = (float *)malloc(block_size * block_size * sizeof(float));

    for (int i = 0; i < block_size * block_size; i++) {
        test_input[i] = 10;
    }

  
    float *dct_output;
    float* input;
    float* output;
	
	size_t size = block_size * block_size * sizeof(float);
    cudaMalloc((void **)&input, size);
    cudaMalloc((void **)&output, size);
    cudaMalloc((void **)&dct_output, block_size * block_size * sizeof(float));

	cudaMemcpy(input, test_input, size, cudaMemcpyHostToDevice);

    int block_dim_x = 2;
    int block_dim_y = 2;

	dim3 dimGrid(block_size, block_size);
	dim3 dimBlock(block_dim_x, block_dim_y);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dct2d<<<dimGrid, dimBlock>>>(input, dct_output, block_size, block_size);
    idct2d<<<dimGrid, dimBlock>>>(dct_output, output, block_size, block_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate:  %3.3f ms \n", time);
    
    cudaMemcpy(test_dct_output, dct_output, block_size * block_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(test_output, output, size, cudaMemcpyDeviceToHost);
    
    printf("DCT output:\n");  
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            printf("%.2f ", test_dct_output[i + (j * block_size)]);
        }
        printf("\n");

    }

    printf("\n");
    printf("IDCT\n");
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            printf("%.2f ", test_output[i + (j * block_size)]);
    
        }
        printf("\n");
    }


	return 0;
}