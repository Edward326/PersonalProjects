#ifndef FILESORT
#define FILESORT
#include"structure.h"

element *binaryins(element *array,int size_ofarray,int *nroftransf);
element *bubblesort(element *array,int size_ofarray,int *nroftransf);
element *bubblesort_2C(element *array,int size_ofarray);
element *shakersort(element *array,int size_ofarray,int *nroftransf);
element *quicksort(element *array,int size_ofarray);

#endif
