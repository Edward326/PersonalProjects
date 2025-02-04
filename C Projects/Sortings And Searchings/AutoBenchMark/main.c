#include<stdio.h>
#include<stdlib.h>
#include"record.h"
#include"search.h"

//program care e menit sa caute un element dintr un fisier de elemente
//un header pentru inreg datelor in bd
//un header pentru metoda de cautare care va cintine si o metoda de sortare



int main(int argv,char **argc){
    if(argv!=2){printf("incorrect nr of arg\n");exit(-1);}
    FILE *file;
    if((file=fopen(argc[1],"r"))==NULL){perror("the file doesn't exists");exit(-1);}
     fclose(file);

    //inreg db
    Movies *array=NULL;
    record(&array,argc[1]);
    
    /*
    for(int i=0;i<array->size;i++)
    puts(array->movies[i].title);
   */
      binSearch(array);




     free_up(&array);
    


    return 0;
}
