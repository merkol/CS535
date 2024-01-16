CC := nvcc cuda_utils.cu
CFLAGS := -ljpeg
all: sobel_cuda harris_cuda dct_cuda

sobel_cuda: sobel_cuda.cu
	$(CC) $(CFLAGS) -o sobel_cuda sobel_cuda.cu

harris_cuda: harris_cuda.cu
	$(CC) $(CFLAGS) -o harris_cuda harris_cuda.cu

dct_cuda: dct_cuda.cu
	$(CC) $(CFLAGS) -o dct_cuda dct_cuda.cu

clean:
	    rm -f *~ *.o sobel_cuda.jpg harris_cuda.jpg sobel_cuda harris_cuda dct_cuda
