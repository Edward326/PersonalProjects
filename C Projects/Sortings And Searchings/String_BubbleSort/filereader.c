#include<stdlib.h>
#include<string.h>
#include<ctype.h>
#include<stdio.h>
#include"filereader.h"
#define size_line 70

element *free_up(element *array,int size){
  if(!array){
    for(int i=0;i<size;i++){
     free(array[i].value);
    }
    free(array);
    return array;
  }
  
   return NULL;
 
}
void record(element *object,int index,char *string){
  char *id,value[51];int sizeid=0,stringlen=0;
  for(int i=0;i<strlen(string);i++)
    {
      if(isspace(string[i])==0){
        if(isdigit(string[i])){
	  sizeid++;
	  //int nr=nr*10+atoi(string[i]);
	  }
	  
	else{
	  value[stringlen]=string[i];stringlen++;
	  //object[index].value=(char*)realloc(object[index].value,sizeof(char)*(++stringlen));
	  //object[index].value[stringlen-1]=string[i];
	  //object[index].value=(char*)realloc(object[index].value,sizeof(char)*(++stringlen));
	  //object[index].value[stringlen]='\0';
	}
	  
      }
    }
  id=malloc(sizeof(char)*(sizeid+1));strncpy(id,string,sizeid);
  object[index].id=atoi(id);
  object[index].value=malloc(sizeof(char)*(stringlen+1));
  strncpy(object[index].value,value,stringlen);
  object[index].value[stringlen]='\0';
  free(id);
}

element *read(char *filename,int *arraysize){
  FILE *file;
    if((file=fopen(filename,"r"))==NULL){
    return NULL;}
    element *array=NULL;
    char line[size_line];
    int index=0;
  
    fgets(line,size_line,file);
    
    while(fgets(line,size_line,file)){
       if((array=realloc(array,sizeof(element)*(index+1)))==NULL){
	 return NULL;}
      record(array,index,line);
      index++;
      *arraysize=index;
    }
    fclose(file);
    return array;
}
