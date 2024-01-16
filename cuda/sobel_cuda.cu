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



__global__ void sobel(unsigned char* image, unsigned char* new_image, int width, int height, int depth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = width * depth;
    int d, k, l;
    int color;
    int sum1;
	int sum2;

  float sobel_x_filter[3][3]={
    {1.0, 0.0, -1.0},
    {2.0, 0.0, -2.0},
    {1.0, 0.0, -1.0}
  };
  
  float sobel_y_filter[3][3]={
    {1.0, 2.0, 1.0},
    {0.0, 0.0, 0.0},
    {-1.0, -2.0, -1.0}
	};

     if( x > 0 && y > 0 && x < width - 1 && y < height - 1) 
    {
        for (d = 0; d < depth; d++)
        {
            sum1 = 0;
			sum2 = 0;
            for (k = -1; k < 2; k++)
            {
                for (l = -1; l < 2; l++)
                {					
                   color = image[((y+l)*w)+(x*depth+d+k*depth)];
                   sum1 += color  *  sobel_x_filter[k + 1][l + 1];
				   sum2 += color  *  sobel_y_filter[k + 1][l + 1];   
                }
            }
            if (sum1 < 0) sum1 = 0;
            if (sum1 > 255) sum1 = 255;
			if (sum2 < 0) sum2 = 0;
			if (sum2 > 255) sum2 = 255;

            new_image[(y * w) + x * depth + d] = sum1 + sum2;
			
        }
    }
	
}


int main(int argc, char **argv)
{

    
    // empty array that contains the execution times
	struct image_t *image = (struct image_t*)malloc(sizeof(struct image_t));
	struct image_t *new_image = (struct image_t*)malloc(sizeof(struct image_t));

	unsigned char* d_old_pixels, *d_new_pixels;

	/* Check command line usage */
	if (argc<2) {
		fprintf(stderr,"Usage: %s image_file\n",argv[0]);
		return -1;
	}

	load_jpeg(argv[1], image);
	
	new_image->x=image->x;
	new_image->y=image->y;
	new_image->depth=image->depth;
    new_image->pixels = (unsigned char*)malloc(image->x * image->y * image->depth * sizeof(char));
	 
	size_t size = image->x * image->y * image->depth * sizeof(char);

    cudaMalloc((void **)&d_old_pixels, size);
    cudaMalloc((void **)&d_new_pixels, size);

	cudaMemcpy(d_old_pixels, image->pixels, size, cudaMemcpyHostToDevice);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


	dim3 dimGrid(image->x, image->y);
	dim3 dimBlock(8, 8);

    cudaEventRecord(start, 0);
    sobel<<<dimGrid, dimBlock>>>(d_old_pixels, d_new_pixels, image->x, image->y, image->depth);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Kernel elapsed time:  %3.3f ms \n", time);
 


    cudaMemcpy(new_image->pixels, d_new_pixels, size, cudaMemcpyDeviceToHost);

	store_jpeg("sobel_cuda.jpg", new_image, 1);

	cudaFree(d_old_pixels);
	cudaFree(d_new_pixels);
    



	return 0;
}