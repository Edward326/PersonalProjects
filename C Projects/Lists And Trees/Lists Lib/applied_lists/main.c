#include<stdio.h>
#include<stdlib.h>
#include"lists.h"
#include"listsop.h"
#define nrofarg 2//schumba l in fct de nr de argumenete pe care il dai in linia de comanda(linux: ./ex "nume".txt  -->nrofarg=2)

//Lucrarea2


//Lucrare2->faza_3
//creearea in nod unei structuri care sa ti retina si nodul urm intr o lista sortata pe langa cel care iti retine nodul in nesortata(lista initiala)
int main(int argv,char** argc){
if(argv!=nrofarg){perror("\nincorrect nr of arg");exit(-1);}


  //verificam daca fisierul exista
  FILE *file;
  if((file=fopen(argc[nrofarg-1],"r+"))==NULL){perror("fisierul nu exista");exit(-1);}
  int number;simple_list *lista=NULL;


  //citim din fisier numerele si le adauigam in lista care are noduri pt care nodul urm sortat si cel nesortat va fii acelasi AICI bineintles daca exista fis 
  while(fscanf(file,"%d",&number)!=EOF)
  addsimple(&lista,number);
  
  fclose(file);

   //verificam daca lista e sortata,daca e sortata afisam lista nesortata si cea sortata  ,daca NU afisam nr de accese la memorie si afisam lista nesortata(cu *next initital) si cea sortata(*nextsorted)
  int n_acces;
  if(sortsimple(lista,&n_acces)){
    printf("\nlista a fost sortata cu succes\nNumber of Access(ess):%d\n\n\n",n_acces);
    printf("lista nesortata:\n");
    viewsimple(lista);printf("\nlista sortata:\n");
    viewsimplesorted(lista);
  }
  else{
    printf("\nlista e deja sortata/lista are un singur nod /lista nu are nicun nod\n");
    printf("lista nesortata:\n");
    viewsimple(lista);printf("\nlista sortata:\n");
    viewsimplesorted(lista);
  }
  
  //se introduce nr pe care se doreste a fii sters nodul lui, si se va sterge atat in lista sortata si cat cea nesortata
  printf("\n");
  int nrtofind;printf("Inserati nr pe care doriti sa l gasiti in lista:\t");scanf("%d",&nrtofind);
  findnr(lista,nrtofind);
 
 //afisam lista dupa stergere
 printf("\n");
  printf("lista nesortata dupa stergere:\n");
    viewsimple(lista);printf("\nlista sortata dupa stergere:\n");
    viewsimplesorted(lista);


  free_totallysimple(&lista);
  return 0;
}



/*
//Lucrare2->faza_2
int main(int argv,char** argc){
if(argv!=nrofarg){perror("\nincorrect nr of arg");exit(-1);}


  FILE *file;
  if((file=fopen(argc[nrofarg-1],"r+"))==NULL){perror("fisierul nu exista");exit(-1);}
  int number;simple_list *lista=NULL;

  while(fscanf(file,"%d",&number)!=EOF)addsimple(&lista,number);
  fclose(file);

  printf("\nlista initiala:\n");
  viewsimple(lista);

  int n_acces;
  if(sortsimple(lista,&n_acces)){
    printf("lista a fost sortata cu succes\nNumber of Access(ess):%d\n\nlista sortata:\n",n_acces);
    viewsimple(lista);
  }
  else
    printf("lista e deja sortata/lista are un singur nod /lista nu are nicun nod\n");
  
  printf("\n");
  int nrtofind;printf("Inserati nr pe care doriti sa l gasiti in lista:\t");scanf("%d",&nrtofind);printf("\n");
  findnr(lista,nrtofind);


  free_totallysimple(&lista);
  return 0;
}
*/


/*
//Lucrare2->faza_1
int main(){

  int nelem,number;
  simple_list *sirintreg=NULL;

  //datele sunt date de la tastatura si adaugate in lista -->se verifica daca lista e sortata(am dat o noi sortata) cu o fct, daca nu , se sorteaza lista cu alg BUBBLESORT fara a folosii alta lista ca VAR auxiliara

    printf("ADAUGARE ELEMENTE INTR O LISTA \n\n");
    printf("nr de elemente: ");scanf("%d",&nelem);printf("\n");

  for(int i=0;i<nelem;i++){
    scanf("%d",&number);
    addsimple(&sirintreg,number);
  }
  printf("\nSIRUL INITIAL:\n")
  viewsimple(sirintreg);
 
 int n_acces=0;
  if(sortsimple(sirintreg,&n_acces))
    printf("lista a fost sortata cu succes\nNumber of Access(ess):%d\n",n_acces);
  else
    printf("lista e deja sortata/lista are un singur nod /lista nu are nicun nod\n");
    
 viewsimple(sirintreg);
  
  free_totallysimple(&sirintreg);
  
  return 0;
}
*/

