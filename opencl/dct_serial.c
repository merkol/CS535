#include <stdio.h>
#include <math.h>
#include <time.h>

#define pi 3.142857
#define m 8
#define n 8

// Function to find discrete cosine transform
void dct2d(float* matrix, float* dct) {
    int i, j, k, l;

    float ci, cj, dct1, sum;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {

            if (i == 0)
                ci = 1 / sqrt(m);
            else
                ci = sqrt(2) / sqrt(m);
            if (j == 0)
                cj = 1 / sqrt(n);
            else
                cj = sqrt(2) / sqrt(n);

            sum = 0;
            for (k = 0; k < m; k++) {
                for (l = 0; l < n; l++) {
                    dct1 = matrix[k * m + l] * 
                        cos((2 * k + 1) * i * pi / (2 * m)) * 
                        cos((2 * l + 1) * j * pi / (2 * n));
                    sum = sum + dct1;
                }
            }
            dct[i * m + j] = ci * cj * sum;
        }
    }
}

void idct2d(float* dct, float* idct) {
    int i, j, k, l;

    float ci, cj, idct1, sum;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {

            sum = 0;
            for (k = 0; k < m; k++) {
                for (l = 0; l < n; l++) {
                    if (k == 0)
                        ci = 1 / sqrt(m);
                    else
                        ci = sqrt(2) / sqrt(m);
                    if (l == 0)
                        cj = 1 / sqrt(n);
                    else
                        cj = sqrt(2) / sqrt(n);

                    idct1 = ci * cj * dct[k * m + l] * 
                        cos((2 * i + 1) * k * pi / (2 * m)) * 
                        cos((2 * j + 1) * l * pi / (2 * n));
                    sum = sum + idct1;
                }
            }
            idct[i * m + j] = sum;
        }
    }
}

// Driver code
int main() {
    // int matrix[m][n];
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         matrix[i][j] = 10;
    //     }
       
    // }

    float* matrix = (float*)malloc(sizeof(float) * m * n);
    float* dct = (float*)malloc(sizeof(float) * m * n);
    float* idct = (float*)malloc(sizeof(float) * m * n);

    for (int i = 0; i < m * n; i++) {
        matrix[i] = 10;
    }


    clock_t start = clock();
    dct2d(matrix, dct);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time spent: %f\n", time_spent);
    // Printing the DCT array
    printf("DCT output: \n");
    for (int i = 0; i < m * n; i++) {
        printf("%.2f ", dct[i]);
        if ((i + 1) % m == 0) {
            printf("\n");
        }
    }


    idct2d(dct, idct);

    // Printing the IDCT array
    printf("IDCT:\n");
    for (int i = 0; i < m * n; i++) {
        printf("%.2f ", idct[i]);
        if ((i + 1) % m == 0) {
            printf("\n");
        }
    }


    return 0;
}
