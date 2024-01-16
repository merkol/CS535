__kernel void generic_convolve(__global unsigned char* image, __global unsigned char* new_image, int width, int height, int depth)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
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