CC = gcc utilities.c
CFLAGS = -g 
LFLAGS = -ljpeg -lm -lpthread -lOpenCL

all:   sobel_opencl harris_opencl dct_opencl harris_serial dct_serial sobel_serial

sobel_opencl:	sobel_opencl.o
		$(CC) -o sobel_opencl sobel_opencl.o $(LFLAGS)

sobel_opencl.o:	sobel_opencl.c
		$(CC) $(CFLAGS) -c sobel_opencl.c

harris_opencl: harris_opencl.o
		$(CC) -o harris_opencl harris_opencl.o $(LFLAGS)

harris_opencl.o: harris_opencl.c
		$(CC) $(CFLAGS) -c harris_opencl.c

dct_opencl: dct_opencl.o
		$(CC) -o dct_opencl dct_opencl.o $(LFLAGS)

dct_opencl.o: dct_opencl.c
		$(CC) $(CFLAGS) -c dct_opencl.c

sobel_serial: sobel_serial.o
		$(CC) -o sobel_serial sobel_serial.o $(LFLAGS)
sobel_serial.o: sobel_serial.c
		$(CC) $(CFLAGS) -c sobel_serial.c

harris_serial: harris_serial.o
		$(CC) -o harris_serial harris_serial.o $(LFLAGS)
harris_serial.o: harris_serial.c
		$(CC) $(CFLAGS) -c harris_serial.c

dct_serial: dct_serial.o
		$(CC) -o dct_serial dct_serial.o $(LFLAGS)
		
dct_serial.o: dct_serial.c
		$(CC) $(CFLAGS) -c dct_serial.c
clean:
	    rm -f *~ *.o sobel.jpg harris.jpg dct.jpg harris_serial.jpg sobel_serial.jpg sobel_seq sobel_opencl harris_opencl dct_opencl harris_serial dct_serial sobel_serial

submit:
	tar -czvf project_submit.tar.gz Makefile *.c

