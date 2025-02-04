#include<stdlib.h>
#include<string.h>
#include"filesort.h"
#include<stdbool.h>
#include<stdio.h>

void quicksortimpl(element *array,int left,int right){
  int i=left,j=right;
  char sir[51],aux[51];strcpy(sir,array[(i+j)/2].value);
  do{
    while(strcmp(array[i].value,sir)<0)i++;
    while(strcmp(array[j].value,sir)>0)j--;
    if(i<=j){
      strcpy(aux,array[i].value);
      array[i].value=realloc(array[i].value,strlen(array[j].value));
      strcpy(array[i].value,array[j].value);
      array[j].value=realloc(array[j].value,strlen(aux));
      strcpy(array[j].value,aux);
      i++;j--;
      //swap elem-i elem-j
    }
  }while(i<=j);
  
  if(j>left)
    quicksortimpl(array,left,j);
  if(i<right)
    quicksortimpl(array,i,right);
}







element *binaryins(element *array,int size_ofarray,int *nr_b){
 
  int mij=0,i,j,left,right,sum=0;
  char aux[51];

   
  for(i=2;i<size_ofarray;i++){
    
    left=0;right=i-1;
    strcpy(aux,array[i].value);
    while(left<right)
      {
	mij=(left+right)/2;
	if(array[mij].value[1]>array[i].value[1])
	  right=mij-1;
	else
	  left=mij+1;
      }
    for(j=i-1;j>=left;j--)
      {
	array[j+1].value=realloc(array[j+1].value,sizeof(char)*strlen(array[j].value));
	strcpy(array[j+1].value,array[j].value);
      }
    array[left].value=realloc(array[left].value,sizeof(char)*strlen(aux));
     strcpy(array[left].value,aux);
     if(left!=i)
       sum++;
  }
  *nr_b=sum;
  return array;
}



element *bubblesort(element *array,int size_ofarray,int *nr_b){
  char sir[51];int sum=0,i,j;
  for(i=0;i<size_ofarray-1;i++){
    for(j=i+1;j<size_ofarray;j++){
      if(array[i].value[1]<array[j].value[1]){
	strcpy(sir,array[i].value);
      array[i].value=realloc(array[i].value,sizeof(char)*strlen(array[j].value));
      strcpy(array[i].value,array[j].value);
        array[j].value=realloc(array[j].value,sizeof(char)*strlen(sir));
	  strcpy(array[j].value,sir);
	  sum++;
      }
    }
  }
  *nr_b=sum;
  return array;
}

element *bubblesort_2C(element *array,int size_ofarray){
  char sir[51];int i,j;
  for(i=0;i<size_ofarray-1;i++){
    for(j=i+1;j<size_ofarray;j++){
      if(array[i].value[0]==array[j].value[0] && array[i].value[1]<array[j].value[1]){
	strcpy(sir,array[i].value);
      array[i].value=realloc(array[i].value,sizeof(char)*strlen(array[j].value));
      strcpy(array[i].value,array[j].value);
        array[j].value=realloc(array[j].value,sizeof(char)*strlen(sir));
	  strcpy(array[j].value,sir);
	 
      }
    }
  }
  return array;
}

element *shakersort(element *array,int size_ofarray,int *nr_b){
  int sus=1,jos=size_ofarray-1,ultim=size_ofarray-1,i,j,sum;
  char sir[51];
  do{

    for(i=jos;j>=sus;j--){
      if(strcmp(array[i-1].value,array[i].value)>0){
	strcpy(sir,array[i-1].value);
        array[i-1].value=realloc(array[i-1].value,strlen(array[i].value));
	strcpy(array[i-1].value,array[i].value);
	array[i].value=realloc(array[i].value,strlen(sir));
	strcpy(array[i].value,sir);
	sum++;
      }
    }
    sus=ultim+1;
    for(j=sus;j<=jos;j++){
      if(strcmp(array[j-1].value,array[j].value)>0){
	strcpy(sir,array[j-1].value);
        array[j-1].value=realloc(array[j-1].value,strlen(array[j].value));
	strcpy(array[j-1].value,array[j].value);
	array[j].value=realloc(array[j].value,strlen(sir));
	strcpy(array[j].value,sir);
	sum++;
      }
    }
    jos=ultim-1;
    
  }while(sus<jos);
  *nr_b=sum;
  return array;
}


  
element *quicksort(element *array,int size){
  quicksortimpl(array,0,size-1);
  return array;
}





