#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include"filesortadv.h"

int viewdimension(FILE *file){
  fseek(file,0,SEEK_SET);
  int count=0;char line[55];
  
  while(fgets(line,sizeof(line),file))
    count++;
  return count;
}
void writeinit(secv *init,element *array,int size){
  FILE *file=init->file;
  for(int i=0;i<size;i++){
   
    //fprintf(file,"%d",elemsize);
    fputs(array[i].value,file);fputc('\n',file);
    // fwrite(array[i].value,elemsize*(sizeof(char)),1,file);
  }
}




void write(FILE *file,char *sir){
  //fprintf(file,"%d",size);
  fputs(sir,file);
  //fprintf(file,"%s\n",sir);
}
void read_secv(FILE *aux1,FILE *aux2,FILE *init,int k){
    fseek(init,0,SEEK_SET);
   
    int aux=k,caz=0;char value[51];
    while(feof(init)==0){
      if(caz==0){
	//fscanf(init,"%d",&size);//value=malloc(sizeof(char)*size);
	fgets(value,sizeof(value),init);//puts(value);
	write(aux1,value);
      aux--;
      if(aux==0){aux=k;caz=1;}
      }
      else
	{
	  //fscanf(init,"%d",&size);//value=malloc(sizeof(char)*size);
	  fgets(value,sizeof(value),init);
      write(aux2,value);aux--;
      if(aux==0){aux=k;caz=0;}
	}
    }
}
void copy(FILE *source,FILE *dest){
  char value[51];
 while(feof(source)==0){
   //fscanf(source,"%d",&size);//value=malloc(sizeof(char)*size);
	fgets(value,sizeof(value),source);//puts(value);
	write(dest,value);
      }
}

void interclass2secv(secv *aux1init,secv *aux2init,secv* source,int k){
 FILE *aux1=aux1init->file;
  FILE *aux2=aux2init->file;
  if(feof(aux2) && feof(aux1))return;
   
  fclose(source->file);
  source->file=fopen("init.txt","w+");
  FILE *file=source->file;
  char valueA[51],valueB[51];
  int up,down;
  
  fseek(file,0,SEEK_SET);
  fseek(aux1,0,SEEK_SET);
  fseek(aux2,0,SEEK_SET);

  up=down=1;
  while(up<=k && down<=k){
    if(feof(aux1)){
      if(feof(aux2))return;
      else{
	copy(aux2,file);return;}}
    if(feof(aux2)){
      if(feof(aux1))return;
      else{
	copy(aux1,file);return;}}
    
    fgets(valueA,sizeof(valueA),aux1);puts(valueA);
    fgets(valueB,sizeof(valueB),aux2);puts(valueB);
	
        if(strcmp(valueA,valueB)==0){
	  fputs(valueA,file);up=++down;
	}
	else{
	  if(strcmp(valueA,valueB)>0){
	  fseek(aux1,-strlen(valueA),SEEK_CUR);
	  fputs(valueB,file);down++;
	  }
	  else{
	    fseek(aux2,-strlen(valueA),SEEK_CUR);
	  fputs(valueA,file);up++;
	  }
	  
	}

	if(up>k){
    while(down){
      fputs(valueB,file);down--;
    }
	}
  else{
   if(down>k){
    while(up){
      fputs(valueA,file);up--;
      }
   }}
	  up=down=1;
  }
}


FILE* interclass3secv(element *array,int size,int *nrtransf){
 secv *source,*aux1,*aux2;
source=malloc(sizeof(secv));
aux1=malloc(sizeof(secv));
aux2=malloc(sizeof(secv));
  source->file=fopen("init.txt","w+");
 
  writeinit(source,array,size);
  int sourcesize=viewdimension(source->file);

  int ktuplu=1;
   while(ktuplu<=sourcesize){
    aux1->file=fopen("aux1.txt","w+");
  aux2->file=fopen("aux2.txt","w+");
    read_secv(aux1->file,aux2->file,source->file,ktuplu);
    //interclass2secv(aux1,aux2,source,ktuplu);
    
    fclose(aux1->file);
    fclose(aux2->file);
       ktuplu=ktuplu*2;
    }
 fclose(source->file);
 
 
  
  return NULL;

}
