//program(cu fucntii cu nr variabil de parametrii) care are o functie(fill_arrays) ce ia ca parametru vectorii in care sa puna ordonat datele gasite/
//sirul in care se gaseste nr de parametrii si tipul lor si / mai apoi parametrii pe care i asociaza tipului de date primit
// spre exemplu  --   fill_arrays("ssif",string,string,int,float);

#include<stdio.h>
#include<stdarg.h>
#include<stdlib.h>
#include<string.h>

void print_funct(int *array_i,int size_i,float *array_f,int size_f,char **sir,int size_c){
    printf("\n\n");
    
  if(size_i){
    printf("integers that were found:\n");
    for(int i=0;i<size_i;i++)
      printf("%d  ",array_i[i]);
      
  }
  printf("\n\n");
  
  if(size_f){
    printf("floats that were found:\n");
    for(int i=0;i<size_f;i++)
      printf("%f  ",array_f[i]);

  }
   printf("\n\n");
   
  if(size_c){
    printf("text that was found:\n");
    for(int i=0;i<size_c;i++)
      {
	for(int j=0;j<strlen(sir[i]);j++)
	  printf("%c",sir[i][j]);
	
	printf("  ");
      }

  }
  printf("\n\n");
  
}

int occur(const char *sir,char c){
  int cont=0;
  for(int i=0;i<strlen(sir);i++){
    if(sir[i]==c)
      cont++;
  }
  return cont;
}

void fill_arrays(int *integers,char** sir,float* floats,const char *verif,...){
  va_list(va);
  va_start(va,verif);                   //asiguram ca pointerul va porni cu adresa primului element variabil(dupa *verif)
  
  int nelem=0,n_int=0,n_char=0,n_float=0;
  int n2_int=0,n2_char=0,n2_float=0;
  char *cuvant;
  nelem=strlen(verif);                      //aici vom contoriza cati de int-uri flot-uri si string-uri gasim in verif pentru a creea vectorii necesari
      n_int=occur(verif,'i');
      n_float=occur(verif,'f');
      n_char=occur(verif,'s');

      integers=(int*)(malloc(sizeof(int)*n_int));
      floats=(float*)(malloc(sizeof(float)*n_float));
      sir=(char**)(malloc(sizeof(char*)*n_char));
//alocam deja ficare vector de tipuri de date in fucntie de cate date au fost citite din verif pentru data resp

      
      //parcurgem inca odata sirul verif ,de data acesasta mergand carct cu carct si verificam de ce tip este,dupa citim variabila asociata tipului,si
      //il inscriem in vectorul care retine acel tip de data primita
      for(int i=0;i<nelem;i++){
	if(verif[i]=='i')
	  {
	    integers[n2_int]=va_arg(va,int);
	    n2_int++;
	  }
	if(verif[i]=='f')
	  {
	    floats[n2_float]=va_arg(va,double);
	    n2_float++;
	  }
	if(verif[i]=='s')
	  {
	    cuvant=va_arg(va,char*);
	    sir[n2_char]=(char*)(malloc(sizeof(char)*strlen(cuvant)));
	    strcpy(sir[n2_char],cuvant);
	    n2_char++;
	  }

      }

      print_funct(integers,n_int,floats,n_float,sir,n2_char);//afisam vectorii de date
}

int main(){
  ///declaram vectorii de sortare a datelor
  int *integers=NULL;
  char **siruri=NULL;
  float *floats=NULL;

      
  fill_arrays(integers,siruri,floats,"ssifi","ana","are",4,5.3,8);

  return 0;
}
  
