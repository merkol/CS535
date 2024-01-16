#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>
#include <jpeglib.h>
#include <time.h>
#include "utilities.h"
#include <limits.h>

void harris(unsigned char* old_pixels, 
                               unsigned char* new_pixels,
                               const int width, const int height, 
                               const int depth,
                               float* pixels_x,
                              float* pixels_y,
                              float* A, float* B, float* C,
                              float* det,  float* trace, float* R
                              ) {
int w = width * depth;
    int d, k, l;
    int color, color_A, color_B, color_C; 
    // make max global
    

     float gaussian_filter[3][3] = {
    {0.0625, 0.125, 0.0625},
    {0.125, 0.25, 0.125},
    {0.0625, 0.125, 0.0625}
  };
    // Sobel filters
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

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {

        for (d = 0; d < depth; d++) {
            Ix = 0;
            Iy = 0;
            for (k = -1; k < 2; k++) {
                for (l = -1; l < 2; l++) {
                    color = old_pixels[((y+l)*w)+(x*depth+d+k*depth)];
                    Ix += color  * sobel_x_filter[k+1][l+1];
                    Iy += color  * sobel_y_filter[k+1][l+1];
                }
            }
            
            pixels_x[(y * w) + x * depth + d] = Ix;
            pixels_y[(y * w) + x * depth + d] = Iy;
        }
       

  A[(y * w) + x * depth] = pixels_x[(y * w) + x * depth] * pixels_x[(y * w) + x * depth];
  B[(y * w) + x * depth] = pixels_y[(y * w) + x * depth] * pixels_y[(y * w) + x * depth];
  C[(y * w) + x * depth] = pixels_x[(y * w) + x * depth] * pixels_y[(y * w) + x * depth];
        }
    }

     for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
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
        
       
    

  
  // // calculate (A*B - (C*C)) - k*(A + B)*(A + B)
  float k_ = 0.04;
  det[(y*w) + x * depth] = (A[(y * w) + x * depth] * B[(y * w) + x * depth]) - (C[(y * w) + x * depth] * C[(y * w) + x * depth]);
  trace[(y*w) + x * depth] = A[(y * w) + x * depth] + B[(y * w) + x * depth];
  R[(y*w) + x * depth] = det[(y*w) + x * depth] - k_ * (trace[(y*w) + x * depth] * trace[(y*w) + x * depth]);
    }
}

 // calculate maximum pixel of R
  // for (int i = 0; i < width * height * depth; i++) {
  //   if (R[i] > max1) {
  //     max1 = R[i];
  //   }
  // }

     for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
  if (R[(y*w) + x * depth] > 0.01 * INT_MAX) {
    new_pixels[(y * w) + x * depth] = 255;
  } else {
    new_pixels[(y * w) + x * depth] = 0;
  }
    }
     }
     }
}

                            


int main(int argc, char **argv){
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

    float* pixels_x = malloc(image.x*image.y*image.depth*sizeof(float));
    float* pixels_y = malloc(image.x*image.y*image.depth*sizeof(float));
    float* A = malloc(image.x*image.y*image.depth*sizeof(float));
    float* B = malloc(image.x*image.y*image.depth*sizeof(float));
    float* C = malloc(image.x*image.y*image.depth*sizeof(float));
    float* det = malloc(image.x*image.y*image.depth*sizeof(float));
    float* trace = malloc(image.x*image.y*image.depth*sizeof(float));
    float* R = malloc(image.x*image.y*image.depth*sizeof(float));

    clock_t start = clock();
    harris(image.pixels, new_image.pixels, image.x, image.y, image.depth, pixels_x, pixels_y, A, B, C, det, trace, R);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time spent: %f\n", time_spent);
    store_jpeg("harris_serial.jpg",&new_image,1);
    
}