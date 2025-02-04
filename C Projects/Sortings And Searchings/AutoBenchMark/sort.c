#include <stdlib.h>
#include <string.h>
#include "sort.h"  

int equals(char *one, char *two, char *init) {
    if (strncmp(one, init, strlen(init)) == 0 && strncmp(two, init, strlen(init)) == 0)
        return 1;
    return 0;
}
int truestate(Movies *array,char *titleToFind){
int cont=0;
for(int i=0;i<array->size;i++)
if(strncmp(array->movies[i].title,titleToFind,strlen(titleToFind)))cont++;
if(cont==array->size)
return 1;
else
return 0;
}
void swapStrings(char **char1,char **char2){
  char *aux=NULL;
aux = malloc(strlen(*char1) + 1); 
         aux[strlen(*char1)]='\0';
        strcpy(aux, *char1);

*char1=realloc(*char1,strlen(*char2)+1);
(*char1)[strlen(*char2)]='\0';
strcpy(*char1,*char2);

*char2=realloc(*char2,strlen(aux)+1);
(*char2)[strlen(aux)]='\0';
strcpy(*char2,aux);
free(aux);
}


//=================================================================================================
void binaryIns(Movies *array, char *titleToFind,int caz,int *compSort,int *swapsSort) {
    int mij = 0, i, j, left, right;
    char *aux=NULL;int comp,swaps;swaps=comp=0;

    for (i = 1; i < array->size; i++) {

        left = 0;
        right = i - 1;
        aux = malloc(strlen(array->movies[i].title) + 1); 
         aux[strlen(array->movies[i].title)]='\0';
        strcpy(aux, array->movies[i].title);

        while (left <= right) {  // Changed condition from 'left < right' to 'left <= right'
            mij = (left + right) / 2;comp++;

            // Updated condition to check the substring and the remaining part of the title
            if(caz==0){
            if(strcmp(array->movies[mij].title, aux)>0)
             right = mij - 1;
            else
             left = mij + 1;
            }
            else{
               //if(truestate(array,titleToFind))return;
               
            if (equals(array->movies[mij].title, aux, titleToFind) &&
                strcmp(array->movies[mij].title + strlen(titleToFind), aux + strlen(titleToFind)) > 0)
                right = mij - 1;
            else
                left = mij + 1;
                }
        }

        for (j = i - 1; j >= left; j--) {
            // Changed memory allocation size to include null terminator
            array->movies[j + 1].title = realloc(array->movies[j + 1].title, sizeof(char) * (strlen(array->movies[j].title) + 1));
            array->movies[j + 1].title[strlen(array->movies[j].title)]='\0';
            strcpy(array->movies[j + 1].title, array->movies[j].title);swaps++;
        }

        // Changed memory allocation size to include null terminator
        array->movies[left].title = realloc(array->movies[left].title, sizeof(char) * (strlen(aux) + 1));//alocam si pentr NULLterm
         array->movies[left].title[strlen(aux)]='\0';//punem si NULLterm
        strcpy(array->movies[left].title, aux);//copiem aici continutul

        free(aux);  // Free the allocated memory for 'aux'
    }
    *compSort=comp;
    *swapsSort=swaps;
}



//=================================================================================================
void Selection(Movies *array,char *titleToFind,int caz,int *compSort,int *swapsSort){
   
   int i,j,poz; 
   //char *aux;
   int comp,sort;comp=sort=0;

for(i=0; i<array->size-1; i++)
{
poz=i;
for(j=i+1; j<array->size; j++){
    if(caz==0)
    {
if(strcmp(array->movies[poz].title,array->movies[j].title)>0) 
{
poz= j; comp++;
} 
    }
else
{
if(equals(array->movies[j].title, array->movies[poz].title, titleToFind) &&
                strcmp(array->movies[poz].title + strlen(titleToFind),array->movies[j].title + strlen(titleToFind))> 0) 
{
poz= j; comp++;
}
}
}
/*
aux = malloc(strlen(array->movies[i].title) + 1); 
         aux[strlen(array->movies[i].title)]='\0';
        strcpy(aux, array->movies[i].title);

array->movies[i].title=realloc(array->movies[i].title,strlen(array->movies[poz].title)+1);
array->movies[i].title[strlen(array->movies[poz].title)]='\0';
strcpy(array->movies[i].title,array->movies[poz].title);

array->movies[poz].title=realloc(array->movies[poz].title,strlen(aux)+1);
array->movies[poz].title[strlen(aux)]='\0';
strcpy(array->movies[poz].title,aux);
*/
swapStrings(&(array->movies[i].title),&(array->movies[poz].title));
if(strcmp(array->movies[i].title,array->movies[poz].title))sort++;

}
*swapsSort=sort;
*compSort=comp;
//free(aux);
}


//=================================================================================================
void BubbleSort(Movies *array,char *titleToFind,int caz,int *compSort,int *swapsSort){
//char *sir;
int i,j;int comp,swap;swap=comp=0;


  for(i=0;i<array->size-1;i++){
    for(j=i+1;j<array->size;j++){
        if(caz==0){

        
    if(strcmp(array->movies[j-1].title,array->movies[j].title)> 0) {

     /*sir=malloc(strlen(array->movies[j-1].title)+1);
     sir[strlen(array->movies[j-1].title)]='\0';
	strcpy(sir,array->movies[j-1].title);

      array->movies[j].title=realloc(array->movies[j].title,strlen(array->movies[j+1].title)+1);
      array->movies[j].title[strlen(array->movies[j+1].title)]='\0';
      strcpy(array->movies[j].title,array->movies[j+1].title);


        array->movies[j+1].title=realloc(array->movies[j+1].title,strlen(sir)+1);
        array->movies[j+1].title[strlen(sir)]='\0';
	  strcpy(array->movies[j+1].title,sir);*/
    swapStrings(&(array->movies[j-1].title),&(array->movies[j].title));
	  swap++;
      }

        }
        else{
      if(equals(array->movies[j-1].title, array->movies[j].title, titleToFind) &&
                strcmp(array->movies[j-1].title + strlen(titleToFind),array->movies[j].title + strlen(titleToFind))> 0) {
/*
     sir=malloc(strlen(array->movies[j].title)+1);
     sir[strlen(array->movies[j].title)]='\0';
	strcpy(sir,array->movies[j].title);

      array->movies[j].title=realloc(array->movies[j].title,strlen(array->movies[j+1].title)+1);
      array->movies[j].title[strlen(array->movies[j+1].title)]='\0';
      strcpy(array->movies[j].title,array->movies[j+1].title);


        array->movies[j+1].title=realloc(array->movies[j+1].title,strlen(sir)+1);
        array->movies[j+1].title[strlen(sir)]='\0';
	  strcpy(array->movies[j+1].title,sir);*/
     swapStrings(&(array->movies[j-1].title),&(array->movies[j].title));
	  swap++;
      }
        }
        comp++;
    }
  }
*swapsSort=swap;
*compSort=comp;
//free(sir);
}


//=================================================================================================
void ShakerSort(Movies *array,char *titleToFind,int caz,int *compSort,int *swapsSort){
int sus=1,jos=array->size-1,ultim;ultim=jos;
  //char *sir;
  int j;int comp,swap;swap=comp=0;

  do{
if(caz==0){

    for(j=jos;j>=sus;j--){    
 if(strcmp(array->movies[j-1].title,array->movies[j].title)> 0) {
  /*
	sir=malloc(strlen(array->movies[j-1].title)+1);
     sir[strlen(array->movies[j-1].title)]='\0';
	strcpy(sir,array->movies[j-1].title);

      array->movies[j-1].title=realloc(array->movies[j-1].title,strlen(array->movies[j].title)+1);
      array->movies[j-1].title[strlen(array->movies[j].title)]='\0';
      strcpy(array->movies[j-1].title,array->movies[j].title);


        array->movies[j].title=realloc(array->movies[j].title,strlen(sir)+1);
        array->movies[j].title[strlen(sir)]='\0';
	  strcpy(array->movies[j].title,sir);*/swapStrings(&(array->movies[j-1].title),&(array->movies[j].title));
	  swap++;

      }
      comp++;
      ultim=j;
    }

    sus=ultim+1;
    for(j=sus;j<=jos;j++){
      if(strcmp(array->movies[j-1].title,array->movies[j].title)> 0){
	/*sir=malloc(strlen(array->movies[j-1].title)+1);
     sir[strlen(array->movies[j-1].title)]='\0';
	strcpy(sir,array->movies[j-1].title);

      array->movies[j-1].title=realloc(array->movies[j-1].title,strlen(array->movies[j].title)+1);
      array->movies[j-1].title[strlen(array->movies[j].title)]='\0';
      strcpy(array->movies[j-1].title,array->movies[j].title);


        array->movies[j].title=realloc(array->movies[j].title,strlen(sir)+1);
        array->movies[j].title[strlen(sir)]='\0';
	  strcpy(array->movies[j].title,sir);*/
    swapStrings(&(array->movies[j-1].title),&(array->movies[j].title));
	  swap++;

      }
      comp++;
      ultim=j;
    }
    jos=ultim-1;
}
else
{

    for(j=jos;j>=sus;j--){    
 if(equals(array->movies[j-1].title, array->movies[j].title, titleToFind) &&
                strcmp(array->movies[j-1].title + strlen(titleToFind),array->movies[j].title + strlen(titleToFind))> 0) {
	/*sir=malloc(strlen(array->movies[j-1].title)+1);
     sir[strlen(array->movies[j-1].title)]='\0';
	strcpy(sir,array->movies[j-1].title);

      array->movies[j-1].title=realloc(array->movies[j-1].title,strlen(array->movies[j].title)+1);
      array->movies[j-1].title[strlen(array->movies[j].title)]='\0';
      strcpy(array->movies[j-1].title,array->movies[j].title);


        array->movies[j].title=realloc(array->movies[j].title,strlen(sir)+1);
        array->movies[j].title[strlen(sir)]='\0';
	  strcpy(array->movies[j].title,sir);*/
    swapStrings(&(array->movies[j-1].title),&(array->movies[j].title));
	  swap++;

      }
      comp++;
      ultim=j;
    }

    sus=ultim+1;
    for(j=sus;j<=jos;j++){
      if(equals(array->movies[j-1].title, array->movies[j].title, titleToFind) &&
                strcmp(array->movies[j-1].title + strlen(titleToFind),array->movies[j].title + strlen(titleToFind))> 0) {
	/*sir=malloc(strlen(array->movies[j-1].title)+1);
     sir[strlen(array->movies[j-1].title)]='\0';
	strcpy(sir,array->movies[j-1].title);

      array->movies[j-1].title=realloc(array->movies[j-1].title,strlen(array->movies[j].title)+1);
      array->movies[j-1].title[strlen(array->movies[j].title)]='\0';
      strcpy(array->movies[j-1].title,array->movies[j].title);


        array->movies[j].title=realloc(array->movies[j].title,strlen(sir)+1);
        array->movies[j].title[strlen(sir)]='\0';
	  strcpy(array->movies[j].title,sir);*/
    swapStrings(&(array->movies[j-1].title),&(array->movies[j].title));
	  swap++;

      }
      comp++;
      ultim=j;
    }
    jos=ultim-1;
}

    
  }while(sus<jos);


*swapsSort=swap;
*compSort=comp;
//free(sir);
}


//=================================================================================================
void ShellSort(Movies *array,char *titleToFind,int caz,int *compSort,int *swapsSort){
int gap=1;int i,j,swap,comp;swap=comp=0;
char *aux;
while(gap<array->size)gap=gap*3+1;

while(gap>=1){
gap/=3;
for(i=gap;i<array->size;i++){

   aux = malloc(strlen(array->movies[i].title) + 1); 
         aux[strlen(array->movies[i].title)]='\0';
        strcpy(aux, array->movies[i].title);
  
  if(caz==0){
  if(strcmp(array->movies[i-gap].title,aux)<0)comp++;
for(j=i;j>=gap && strcmp(array->movies[j-gap].title,aux)>0;j-=gap)
{
 array->movies[j].title = realloc(array->movies[j].title, (strlen(array->movies[j-gap].title) + 1));
            array->movies[j].title[strlen(array->movies[j-gap].title)]='\0';
            strcpy(array->movies[j].title, array->movies[j-gap].title);swap++;comp++;
}
 array->movies[j].title = realloc(array->movies[j].title, strlen(aux) + 1);
            array->movies[j].title[strlen(aux)]='\0';
            strcpy(array->movies[j].title, aux);
            


  }
  else
  {
if(equals(array->movies[i-gap].title, aux, titleToFind)==0 || (equals(array->movies[i-gap].title, aux, titleToFind) &&
                strcmp(array->movies[i-gap].title + strlen(titleToFind),aux + strlen(titleToFind))< 0))
                comp++;
                 
for(j=i;j>=gap && equals(array->movies[j-gap].title, aux, titleToFind) && 
         strcmp(array->movies[j-gap].title + strlen(titleToFind),aux + strlen(titleToFind))>0;j-=gap)
{
 array->movies[j].title = realloc(array->movies[j].title, (strlen(array->movies[j-gap].title) + 1));
            array->movies[j].title[strlen(array->movies[j-gap].title)]='\0';
            strcpy(array->movies[j].title, array->movies[j-gap].title);swap++;comp++;
}
 array->movies[j].title = realloc(array->movies[j].title, strlen(aux) + 1);
            array->movies[j].title[strlen(aux)]='\0';
            strcpy(array->movies[j].title, aux);
            



  }
free(aux);
}



}
}


//=========================================================================================>
void quicksortimpl(Movies *array,char *titleToFind,int caz,int left,int right,int *compSort,int *swapsSort){
  int i=left,j=right;int swap,comp;swap=comp=0;
  char *aux;
    aux = malloc(strlen(array->movies[(i+j)/2].title) + 1); 
         aux[strlen(array->movies[(i+j)/2].title)]='\0';
        strcpy(aux, array->movies[(i+j)/2].title);

  do{
    if(caz==0){
    while(strcmp(array->movies[i].title,aux)<0){i++;comp++;}
    while(strcmp(array->movies[j].title,aux)>0){j--;comp++;}

    if(i<=j){
      swapStrings(&(array->movies[i].title),&(array->movies[j].title));
      i++;j--;
      swap++;
      //swap elem-i elem-j
    }
    }
    else
    {
    while(equals(array->movies[i].title, aux, titleToFind) && 
         strcmp(array->movies[i].title + strlen(titleToFind),aux + strlen(titleToFind))<0){i++;comp++;}
    while(equals(array->movies[j].title, aux, titleToFind) && 
         strcmp(array->movies[j].title + strlen(titleToFind),aux + strlen(titleToFind))>0){j--;comp++;}

    if(i<=j){
      if(equals(array->movies[i].title, aux, titleToFind) && 
         strcmp(array->movies[i].title + strlen(titleToFind),aux + strlen(titleToFind))>0 &&
         equals(array->movies[j].title, aux, titleToFind) && 
         strcmp(array->movies[j].title + strlen(titleToFind),aux + strlen(titleToFind))<0)
      swapStrings(&(array->movies[i].title),&(array->movies[j].title));
      
      i++;j--;
      swap++;
      //swap elem-i elem-j
      
    }
    }

  }while(i<=j);
  *swapsSort=swap;
  *compSort=comp;
  free(aux);
  
  if(j>left)
    quicksortimpl(array,titleToFind,caz,left,j,*(&compSort),*(&swapsSort));
  if(i<right)
    quicksortimpl(array,titleToFind,caz,i,right,*(&compSort),*(&swapsSort));
}


void QuickSort(Movies *array,char *titleToFind,int caz,int *compSort,int *swapsSort){
quicksortimpl(array,titleToFind,caz,0,array->size-1,*(&compSort),*(&swapsSort));
}