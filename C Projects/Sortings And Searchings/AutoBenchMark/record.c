#include"record.h"
#include<stdlib.h>
#include<string.h>
#include<ctype.h>
#include<stdio.h>
#define lineSize 110 

void segment(Movies *element,int index,char *line){
int caz=1,caz3=0;
int ccharindex=0;
int year=0;
int length=0;


for(int i=0;i<strlen(line)&& caz!=6;i++)
{
if(caz==1)
{
    if(isdigit(line[i]))
    year=year*10+(line[i]-'0');
    else if(isspace(line[i+1])==0) caz=2;
}
else
if(caz==2)
{
    if(isdigit(line[i]))
    length=length*10+(line[i]-'0');
    else if(isspace(line[i+1])==0) caz=3;
}
else
if(caz==3)
{
    if(isspace(line[i]) && isspace(line[i+1])==0 && caz3==0)
    caz3=1;
    else{
    if(caz3 && i!=strlen(line)-1)
    {
        element->movies[index].title=realloc(element->movies[index].title,sizeof(char)*(ccharindex+1));
        element->movies[index].title[ccharindex]=line[i];ccharindex++;
    }
    else {
   element->movies[index].title=realloc(element->movies[index].title,sizeof(char)*(ccharindex+2));
   element->movies[index].title[ccharindex]=line[i];
   element->movies[index].title[ccharindex+1]='\0';
    }
    }
}

}

element->movies[index].year=year;
element->movies[index].length=length;
}








//----------------------------------------
void record(Movies **array,char *filename){
FILE *file=fopen(filename,"r");

*array=malloc(sizeof(Movies));
(*array)->movies=NULL;

char line[lineSize];
int csize=0;


fgets(line,lineSize,file);
while(fgets(line,lineSize,file)){
if(csize==0){
(*array)->movies=realloc((*array)->movies,sizeof(Movie)*(csize+1));
(*array)->movies->title=NULL;
//(*array)->movies->type=NULL;
}
else
{
(*array)->movies=realloc((*array)->movies,sizeof(Movie)*(csize+1));
(*array)->movies[csize].title=NULL;
//(*array)->movies[csize].type=NULL;
}

segment( *array,csize,line);
csize++;
}
fclose(file);

(*array)->size=csize;


}
void free_up(Movies **array){
for(int i=0;i<(*array)->size;i++){
    free((*array)->movies[i].title);
    //free((*array)->movies[i]);
}
free((*array)->movies);
//(*array)->movies=NULL;
free((*array));
*array=NULL;
}
void make_copy(Movies **new_array,Movies *init_array){
    if(*new_array!=NULL){free_up(*(&new_array));}

*new_array=malloc(sizeof(Movies));
(*new_array)->movies=malloc(sizeof(Movie)*init_array->size);
(*new_array)->size=init_array->size;

int i,j;
//fara sa se faca aliasing
for(i=0;i<init_array->size;i++){
(*new_array)->movies[i].year=init_array->movies[i].year;
(*new_array)->movies[i].length=init_array->movies[i].length;
(*new_array)->movies[i].title=malloc(strlen((init_array)->movies[i].title)+1);


for(j=0;j<strlen((init_array)->movies[i].title);j++)
(*new_array)->movies[i].title[j]=init_array->movies[i].title[j];

(*new_array)->movies[i].title[strlen(init_array->movies[i].title)]='\0';
}

}