#ifndef FILEREADER
#define FILEREADER
#include"structure.h"

element *free_up(element *array,int size);
element *read(char *filename,int *nr);
#endif
