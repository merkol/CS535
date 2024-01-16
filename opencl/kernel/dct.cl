__kernel void dct2d(__global float* input, __global float* dct_output, __global float* idct_output,
                     int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float pi = 3.14159265358979323846f;
    float sum = 0.0f;
    float sum2 = 0.0f;
    float ci, cj, dct1;
    int k, l;
    float cu, cv;
    int u, v;


    if (x < width && y < height) {
        if (x == 0)
            ci = 1 / sqrt((float)width);
        else
            ci = sqrt(2.0f) / sqrt((float)width);
        
        if (y == 0)
            cj = 1 / sqrt((float)height);
        else
            cj = sqrt(2.0f) / sqrt((float)height);

        sum = 0;
        for (k = 0; k < width; k++) {
            for (l = 0; l < height; l++) {
                dct1 = input[k + (l * width)] *
                       cos((2.0f * k + 1) * x * pi / (2.0f * width)) *
                       cos((2.0f * l + 1) * y * pi / (2.0f * height));

                sum += dct1;
            }
        }
        dct_output[x + (y * width)] = ci * cj * sum;
    }
    

    barrier(CLK_GLOBAL_MEM_FENCE);
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( x < width  && y < height) {
            sum2 = 0;
            for (u = 0; u < width; u++) {
                for (v = 0; v < height; v++) {

            if (u == 0) {
                cu = 1 / sqrt((float)width);
            } else {
                cu = sqrt(2.0f) / sqrt((float)width);
              }

            if (v == 0) {
                cv = 1 / sqrt((float)height);
            } else {
                cv = sqrt(2.0f) / sqrt((float)height); 
              }   

                float idct = (dct_output[u + (v * width)] * cu * cv *      
                           cos((2 * x + 1) * u * pi / (2 * width)) * 
                           cos((2 * y + 1) * v * pi / (2 * height)));

                sum2 += idct;
                }               
            }
            idct_output[x + (y * width)] =  sum2;    
         }
}
