#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <jpeglib.h>
#include <errno.h>
#include <math.h>
#include "utilities.h"

cl_context context;
cl_command_queue command_queue;
cl_program program;
cl_kernel kernel;
cl_platform_id platform;
cl_device_id device;


char *load_kernel_source(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Unable to open file %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    char *source_code = (char *)malloc(file_size + 1);
    if (!source_code) {
        fclose(file);
        fprintf(stderr, "Memory allocation failed while loading file\n");
        return NULL;
    }

    size_t read_size = fread(source_code, 1, file_size, file);
    if (read_size != file_size) {
        fclose(file);
        free(source_code);
        fprintf(stderr, "Error reading file\n");
        return NULL;
    }

    source_code[file_size] = '\0';
    fclose(file);

    return source_code;
}

void setupOpenCL(const char *filename, const char *kernel_name) {
    // Get the platform
	cl_platform_id platforms[3];
    cl_uint num_platforms;
    clGetPlatformIDs(3, &platforms, &num_platforms);

    // Get the device
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 2, &device, NULL);

    // Print device name
    char deviceName[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    printf("Device: %s\n", deviceName);


    // Create OpenCL context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // Create command queue
    command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    // Load OpenCL kernel code from .cl file
    char *kernel_source = load_kernel_source(filename);

    // Create program from kernel source
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);


    // Build the program and check for errors
    if (clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS) {
		printf("Error building program\n");
		exit(1);
	}

	// Create the OpenCL kernel
	kernel = clCreateKernel(program, kernel_name, NULL);
}

void cleanupOpenCL() {
    // Release OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}

int load_jpeg(char *filename, struct image_t *image) {

	FILE *fff;
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	JSAMPROW output_data;
	unsigned int scanline_len;
	int scanline_count=0;

	fff=fopen(filename,"rb");
	if (fff==NULL) {
		fprintf(stderr, "Could not load %s: %s\n",
			filename, strerror(errno));
		return -1;
	}

	/* set up jpeg error routines */
	cinfo.err = jpeg_std_error(&jerr);

	/* Initialize cinfo */
	jpeg_create_decompress(&cinfo);

	/* Set input file */
	jpeg_stdio_src(&cinfo, fff);

	/* read header */
	jpeg_read_header(&cinfo, TRUE);

	/* Start decompressor */
	jpeg_start_decompress(&cinfo);

	printf("output_width=%d, output_height=%d, output_components=%d\n",
		cinfo.output_width,
		cinfo.output_height,
		cinfo.output_components);

	image->x=cinfo.output_width;
	image->y=cinfo.output_height;
	image->depth=cinfo.output_components;

	scanline_len = cinfo.output_width * cinfo.output_components;
	image->pixels=malloc(cinfo.output_width * cinfo.output_height * cinfo.output_components);

	while (scanline_count < cinfo.output_height) {
		output_data = (image->pixels + (scanline_count * scanline_len));
		jpeg_read_scanlines(&cinfo, &output_data, 1);
		scanline_count++;
	}

	/* Finish decompressing */
	jpeg_finish_decompress(&cinfo);

	jpeg_destroy_decompress(&cinfo);

	fclose(fff);

	return 0;
}

int store_jpeg(char *filename, struct image_t *image, boolean grayscale) {

	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	int quality=90; /* % */
	int i;

	FILE *fff;

	JSAMPROW row_pointer[1];
	int row_stride;

	/* setup error handler */
	cinfo.err = jpeg_std_error(&jerr);

	/* initialize jpeg compression object */
	jpeg_create_compress(&cinfo);

	/* Open file */
	fff = fopen(filename, "wb");
	if (fff==NULL) {
		fprintf(stderr, "can't open %s: %s\n",
			filename,strerror(errno));
		return -1;
	}

	jpeg_stdio_dest(&cinfo, fff);

	/* Set compression parameters */
	cinfo.image_width = image->x;
	cinfo.image_height = image->y;
	cinfo.input_components = image->depth;
	if (grayscale)
		cinfo.in_color_space = JCS_GRAYSCALE;
	else
		cinfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, quality, TRUE);

	/* start compressing */
	jpeg_start_compress(&cinfo, TRUE);

	row_stride=image->x*image->depth;

	for(i=0;i<image->y;i++) {
		row_pointer[0] = & image->pixels[i * row_stride];
		jpeg_write_scanlines(&cinfo, row_pointer, 1);
	}

	/* finish compressing */
	jpeg_finish_compress(&cinfo);

	/* close file */
	fclose(fff);

	/* clean up */
	jpeg_destroy_compress(&cinfo);

	return 0;
}

void combine_and_store(struct image_t *s_x, struct image_t *s_y, struct image_t *new_image) {
    for (int i = 0; i < s_x->x * s_x->y * s_x->depth; i++) {
        int out = sqrt((s_x->pixels[i] * s_x->pixels[i]) + (s_y->pixels[i] * s_y->pixels[i]));
        if (out > 255) out = 255;
        if (out < 0) out = 0;
        new_image->pixels[i] = out;
    }
	if (s_x->depth == 1) {
		store_jpeg("sobel.jpg", new_image, TRUE);
	} else {
		store_jpeg("sobel.jpg", new_image, FALSE);
	}
}