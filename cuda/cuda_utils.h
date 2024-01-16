// utilities.h
#include <stdlib.h>

struct image_t
{
    int x;
    int y;
    int depth;	/* bytes */
    unsigned char *pixels;
};
int load_jpeg(char *filename, struct image_t *image);
int store_jpeg(char *filename, struct image_t *image, boolean grayscale);


