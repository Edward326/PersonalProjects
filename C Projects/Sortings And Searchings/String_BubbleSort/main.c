#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include"filereader.h"

#include"filesearch.h"
#include"filesort.h"
#include"filesortadv.h"

void printarray(element *array,int length){
  if(array==NULL){
    printf("array gol\n");exit(-1);}
  for(int i=0;i<length;i++){
    printf("id:%d value:",array[i].id);
    puts(array[i].value);printf("\n");
  }
}
void writefile(char *sir,element *array,int size){
FILE *file=fopen(sir,"w+");
for(int i=0;i<size;i++){
fprintf(file,"%d",array[i].id);
 fputs(array[i].value,file);fputc('\n',file);
}
fclose(file);
}

int main(int argv,char **argc){
  if(argv!=2){return 0;}


  //sapt3--inregistrare db si cautare prin met lineara si binara

  /*
  element *array=NULL;
  int nrofvalues=0;
  array=read(argc[1],&nrofvalues);
  //se returneza un nou array(tablou) tip TDA care e complet gol la inceput,iar dupa citirea din fisier a datelor ,fiecare index al tabloului va retine valoarea specifica din fisier pentru campul lui
  
  if(array==NULL){
    perror("fis nu exista/eroare la alocarea dinamica");
    exit(-1);}

  
  printarray(array,nrofvalues);
  



  int t1=linearsearch(array,nrofvalues,sir);
  int t2=binsearch(array,nrofvalues,sir);
  
  if(t1)
    printf("\nALGORITM CAUTARE LINIARA: %d(nr de comparatii)",t1);
  if(t2)
  printf("\nALGORITM CAUTARE BINARA: %d(nr de comparatii)\n",t2);
  */
  


  
  //sapt4--inreg pt 3 db dintr-un fisier neordonat alfabetic si sorttarea lor prin 3 metode
  /*
  printf("\n\n\n\n\n");
 int nrtransf,nrofvalues;
  nrtransf=nrofvalues=0;


  element *arrayfirst=NULL,*new_array=NULL;
  arrayfirst=read(argc[1],&nrofvalues);
  new_array=binaryins(arrayfirst,nrofvalues,&nrtransf);
  // printarray(new_array,nrofvalues);
  printf("\nALG INSERTIE_BINARA\nNR DE INTERSCHIMBARI %d\n\n",nrtransf);
  arrayfirst=free_up(arrayfirst,nrofvalues);
  new_array=free_up(new_array,nrofvalues);
  
  
  nrofvalues=nrtransf=0;
  element *array2=NULL,*new_array2=NULL;
  array2=read(argc[1],&nrofvalues);
  new_array2=bubblesort(array2,nrofvalues,&nrtransf);
  //printarray(new_array2,nrofvalues);
  printf("\nALG Bubblesort\nNR DE INTERSCHIMBARI %d\n\n",nrtransf);
  array2=free_up(array2,nrofvalues);
  new_array2=free_up(new_array2,nrofvalues);
 
  nrofvalues=nrtransf=0;
  element *array3=NULL,*new_array3=NULL;
  array3=read(argc[1],&nrofvalues);
  new_array3=shakersort(array3,nrofvalues,&nrtransf);
  //printarray(new_array3,nrofvalues);
  printf("\nALG Shakersort\nNR DE INTERSCHIMBARI %d\n",nrtransf);
  array3=free_up(array3,nrofvalues);
  new_array3=free_up(new_array3,nrofvalues);
  */

  //sapt5-->sortare prin quicksort cresc pt char[0] si desc pt char[1]
 
  int nrofvalues=0;
  element *array4=NULL,*new_array4=NULL,*new_array4_2C=NULL;
  array4=read(argc[1],&nrofvalues);
  new_array4=quicksort(array4,nrofvalues);
  writefile("quicksort.txt",new_array4,nrofvalues);
  new_array4_2C=bubblesort_2C(new_array4,nrofvalues);
 writefile("bubblesort_2C.txt",new_array4_2C,nrofvalues);
 array4=free_up(array4,nrofvalues);
  new_array4=free_up(new_array4,nrofvalues);
   new_array4_2C=free_up(new_array4_2C,nrofvalues);
   printarray(new_array4_2C,nrofvalues);
   
  return 0;
}
