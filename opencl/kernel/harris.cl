__kernel void harris(__global uchar* old_pixels, 
                               __global uchar* new_pixels,
                               const int width, const int height, 
                               const int depth,
                               __global float* pixels_x,
                              __global float* pixels_y,
                              __global float* A, __global float* B, __global float* C,
                              __global float* det, __global float* trace, __global float* R
                              ) {
    int x = get_global_id(0);
    int y = get_global_id(1);
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

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
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
       
    

  
  // // calculate (A*B - (C*C)) - k*(A + B)*(A + B)
  float k_ = 0.04;
  det[(y*w) + x * depth] = (A[(y * w) + x * depth] * B[(y * w) + x * depth]) - (C[(y * w) + x * depth] * C[(y * w) + x * depth]);
  trace[(y*w) + x * depth] = A[(y * w) + x * depth] + B[(y * w) + x * depth];
  R[(y*w) + x * depth] = det[(y*w) + x * depth] - k_ * (trace[(y*w) + x * depth] * trace[(y*w) + x * depth]);


 // calculate maximum pixel of R
  // for (int i = 0; i < width * height * depth; i++) {
  //   if (R[i] > max1) {
  //     max1 = R[i];
  //   }
  // }


  if (R[(y*w) + x * depth] > 0.01 * INT_MAX) {
    new_pixels[(y * w) + x * depth] = 255;
  } else {
    new_pixels[(y * w) + x * depth] = 0;
  }
    }
}
