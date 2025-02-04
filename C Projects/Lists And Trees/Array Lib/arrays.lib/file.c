#include<stdio.h>
#include<stdlib.h>
#include"vectors.h"

void afis(int *array,int size){
  printf("\n");
  for(int i=0;i<size;i++)
    printf("%d\t",array[i]);

  printf("\n");
}

int main(){

  int nmax=0,nelem=0,*array=NULL;
  scanf("%d%d",&nelem,&nmax);
  array=fill(array,nelem,nmax);

  afis(array,nelem);
  return 0;
}
