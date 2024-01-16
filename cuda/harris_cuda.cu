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




__global__ void harris(unsigned char* image, unsigned char* new_image,
                float* x_buff, float* y_buff,
                float* A, float* B, float* C,
                float* det, float* trace, float* R,
                int width, int height, int depth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = width * depth;
    int d, k, l;
    int color, color_A, color_B, color_C;
  
    float gaussian_filter[3][3] = {
    {0.0625, 0.125, 0.0625},
    {0.125, 0.25, 0.125},
    {0.0625, 0.125, 0.0625}
  };

  int sobel_x_filter[3][3]={
    {1, 0, -1},
    {2, 0, -2},
    {1, 0, -1}
  };
  
  int sobel_y_filter[3][3]={
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
	};

    int Ix , Iy, IA, IB, IC;
    if( x >= 1 && y >= 1 && x < width - 1 && y < height - 1) 
    {
        for (d = 0; d < depth; d++)
        {
            Ix = 0;
            Iy = 0;
            for (k = -1; k < 2; k++)
            {
                for (l = -1; l < 2; l++)
                {					
                   color = image[((y+l)*w)+(x*depth+d+k*depth)];
                   Ix += color  *  sobel_x_filter[k + 1][l + 1];
				   Iy += color  *  sobel_y_filter[k + 1][l + 1];   
                }
            }
            

            x_buff[(y*w)+x*depth+d] = Ix;
            y_buff[(y*w)+x*depth+d] = Iy;
        }
    

    A[(y*w)+(x*depth)] = x_buff[(y*w)+x*depth] * x_buff[(y*w)+x*depth];
    B[(y*w)+(x*depth)] = x_buff[(y*w)+x*depth] * y_buff[(y*w)+x*depth];
    C[(y*w)+(x*depth)] = y_buff[(y*w)+x*depth] * y_buff[(y*w)+x*depth];
    }

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        for (d = 0; d < depth; d++) {
            IA = 0;
            IB = 0;
            IC = 0;
            for (k = -1; k < 2; k++) {
                for (l = -1; l < 2; l++) {
                    color_A = A[((y+l)*w)+(x*depth+d+k*depth)];
                    color_B = B[((y+l)*w)+(x*depth+d+k*depth)];
                    color_C = C[((y+l)*w)+(x*depth+d+k*depth)];
                  
                    IA += color_A * gaussian_filter[k+1][l+1];
                    IB += color_B * gaussian_filter[k+1][l+1];
                    IC += color_C * gaussian_filter[k+1][l+1];
                }
            }
            
            A[(y * w) + x * depth + d] = IA;
            B[(y * w) + x * depth + d] = IB;
            C[(y * w) + x * depth + d] = IC;
        }
       
    

  float k_ = 0.04;
  det[(y*w) + x * depth] = (A[(y * w) + x * depth] * B[(y * w) + x * depth]) - (C[(y * w) + x * depth] * C[(y * w) + x * depth]);
  trace[(y*w) + x * depth] = A[(y * w) + x * depth] + B[(y * w) + x * depth];
  R[(y*w) + x * depth] = det[(y*w) + x * depth] - k_ * (trace[(y*w) + x * depth] * trace[(y*w) + x * depth]);
    


  if (R[(y*w) + x * depth] > 0.01 * INT_MAX) {
    new_image[(y * w) + x * depth] = 255;
  } else {
    new_image[(y * w) + x * depth] = 0;
  }  
    }
}


int main(int argc, char **argv)
{
	struct image_t *image = (struct image_t*)malloc(sizeof(struct image_t));
	struct image_t *new_image = (struct image_t*)malloc(sizeof(struct image_t));

    float *x_buff, *y_buff, *A, *B, *C, *det, *trace, *R;
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
    size_t size_int = image->x * image->y * image->depth * sizeof(float);

    cudaMalloc((void **)&d_old_pixels, size);
    cudaMalloc((void **)&d_new_pixels, size);

    cudaMalloc((void **)&x_buff, size_int);
    cudaMalloc((void **)&y_buff, size_int);
    cudaMalloc((void **)&A, size_int);
    cudaMalloc((void **)&B, size_int);
    cudaMalloc((void **)&C, size_int);
    cudaMalloc((void **)&det, size_int);
    cudaMalloc((void **)&trace, size_int);
    cudaMalloc((void **)&R, size_int);

	cudaMemcpy(d_old_pixels, image->pixels, size, cudaMemcpyHostToDevice);

    int block_dim_x = 16;
    int block_dim_y = 16;
	dim3 dimGrid(image->x, image->y);
	dim3 dimBlock(block_dim_x, block_dim_y);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start, 0);
    harris<<<dimGrid, dimBlock>>>(d_old_pixels, d_new_pixels, x_buff, y_buff, A, B, C, det, trace, R, image->x, image->y, image->depth);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate:  %3.3f ms \n", time);

    cudaMemcpy(new_image->pixels, d_new_pixels, size, cudaMemcpyDeviceToHost);


	store_jpeg("harris_cuda.jpg", new_image, 1);

	cudaFree(d_old_pixels);
	cudaFree(d_new_pixels);
    cudaFree(x_buff);
    cudaFree(y_buff);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(det);
    cudaFree(trace);
    cudaFree(R);

	return 0;
}