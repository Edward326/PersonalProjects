#ifndef FILESORTADV
#define FILESORTADV
#include"structure.h"
#include<stdlib.h>
typedef struct{
  FILE *file;
  int cindex;
}secv;
FILE *interclass3secv(element *array,int size,int *nrtransf);
#endif
