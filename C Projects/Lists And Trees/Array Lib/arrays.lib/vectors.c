#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"vectors.h"


int *fill(int *array,int nelem,int nmax){
  srand(time(NULL));
  
  if(array!=NULL)
    {perror("array already declared with some elements(static/dynamic) on memory");
      exit(-1);
    }
  if((array=malloc(sizeof(int)*nelem))==NULL){
    perror("error on allocating(possible to much elements that the stack can afford)");exit(-1);}

  for(int i=0;i<nelem;i++)
    array[i]=rand()%nmax;

  return array;
}
