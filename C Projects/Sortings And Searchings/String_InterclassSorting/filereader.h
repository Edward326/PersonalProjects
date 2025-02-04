#ifndef FILEREADER
#define FILEREADER
#include"structure.h"
/*typedef struct{
  int id;char *value;
  }element;
*/
element *free_up(element *array,int size);
element *read(char *filename,int *nr);
#endif
