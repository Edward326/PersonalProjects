#include<stdlib.h>
#include<stdio.h>
#include"listsop.h"
#include"lists.h"

int verifysorted(simple_list *a){
  int caz=0;
  if(a->last==NULL || a->first==a->last){perror("lista e goala//are un singur nod");exit(EXIT_FAILURE);}
   
  if(a->first->nr > a->first->next->nr)
    caz=1;
  if(a->first->nr < a->first->next->nr)
    caz=2;
  elemS *aux;
 
  
  if(caz==0)//primele 2 elem egale
    {    aux=a->first;
      while(aux->nr==aux->next->nr){
	if(aux->next->next==NULL)return 1;
	else
	aux=aux->next;
      }
      
      if(aux->nr > aux->next->nr)
	caz=1;
      else
	caz=2;
    }
  
  if(caz==1){
    aux=a->first;
    while(aux->next)
      {
	if(aux->nr < aux->next->nr)
	  return 0;
	else
	  aux=aux->next;
      }
  }
  else{
   aux=a->first;
    while(aux->next)
      {
	if(aux->nr > aux->next->nr)
	  return 0;
	else
	  aux=aux->next;
      }
  }
  return 1;
  
}
int sortsimple(simple_list *list,int *nr) {//bubblesort
    *nr=0;
    if (list==NULL || list->last == NULL || list->last == list->first)return 0;
    if (verifysorted(list)) return 0;

    elemS *a, *b, *prev, *temp;
    int swapped,nr_acces=0;

    do {
        swapped = 0;
        prev = NULL;
        a = list->firstsorted;
        b = a->nextsorted;
        nr_acces++;//accesam primul nod si de acolo avem acces la toate nodurile urm
        while (b) {
            if (a->nr > b->nr) {
                if (prev) {
                    temp = b->nextsorted;
                    prev->nextsorted = b;
                    b->nextsorted = a;
                    a->nextsorted = temp;
                } else {
                    temp = b->nextsorted;
                    list->firstsorted = b;
                    b->nextsorted = a;
                    a->nextsorted = temp;
                }
                swapped = 1;
            }
            nr_acces+=2;
            prev = a;
            a = a->nextsorted;
            b = (a) ? a->nextsorted : NULL;
        }
         nr_acces++;
    } while (swapped);
    *nr=nr_acces;
    return 1;
}



int nrofdigit(int nr){
  int count=0;
  if(nr==0)return 1;
  while(nr){nr=nr/10;count++;}
  return count;
}
void display(simple_list *lista,int nrspacing,int spacing,int nodvalue){
  viewsimple(lista);
  while(spacing){printf(" ");spacing--;}
  while(nrspacing){printf("\e[1;31m^");nrspacing--;}
  printf("\e[0;37m(nod->%d)\n",nodvalue);
}
void display2(simple_list *lista,int nrspacing,int spacing,int nodvalue){
  viewsimplesorted(lista);
  while(spacing){printf(" ");spacing--;}
  while(nrspacing){printf("\e[1;31m^");nrspacing--;}
  printf("\e[0;37m(nod->%d)\n",nodvalue);
}
void findnr2(simple_list *lista,int data){
if(lista==NULL || lista->last==NULL){printf("lista e neinit/(sau) lista e goala\n");return;}
int nod=1,spacing=0;
for(elemS *i=lista->firstsorted;i;i=i->nextsorted){
  if(i->nr==data){
    display2(lista,nrofdigit(data),spacing,nod);
    return;
  }
  else{nod++;spacing+=nrofdigit(i->nr)+1;}
}
printf("elementul nu s a gasit in lista\n");
return;
}



void findandelete(simple_list *a,int nr){
elemS *i,*aux=NULL;
for(i=a->firstsorted;i;i=i->nextsorted){
          if(i->nr==nr){
          if(i==a->firstsorted){
            a->firstsorted=a->firstsorted->next;
            free(i);
          }
          else{
            if(i->nextsorted==NULL){
            aux->nextsorted=NULL;
            free(i);
            }
            else
            {
              aux->nextsorted=i->nextsorted;
              free(i);
            }
          }
          return;
          }
          aux=i;
         }
         return;
}
void deletesimple(simple_list *a,int nr){
  if(a->first){

    if(a->first->nr==nr){
      if(a->first->next){
        elemS* i=a->first->next;
        findandelete(a,nr);
        a->first=i;
      deletesimple(a,nr);
      }
      else{
	      a->firstsorted=a->first=a->last=NULL;
      }
    }
    else
      {elemS *aux=a->first,*urm,*i,*aux2;
	for(i=a->first->next;i;i=urm){
    urm=i->next;
    if(i->nr==nr){
      if(i->next==NULL){
      findandelete(a,nr);
      aux->next=NULL;
      }
      else{
        aux2=i->next;
        findandelete(a,nr);
        aux->next=aux2;
      }
    }
	  else
	      aux=i;
	}
      }
  }
  else{
    printf("lista simpla e goala");
    exit(-1);
  }
}
void findnr(simple_list *lista,int data){
if(lista==NULL || lista->last==NULL){printf("lista e neinit/(sau) lista e goala\n");return;}
int nod=1,spacing=0;
for(elemS *i=lista->first;i;i=i->next){
  if(i->nr==data){
  printf("\npoz unde s a gasit el in lista nesortata:\n");
    display(lista,nrofdigit(data),spacing,nod);
    printf("\npoz unde s a gasit el in lista sortata:\n");
    findnr2(lista,data);
    deletesimple(lista,data);
    return;
  }
  else{nod++;spacing+=nrofdigit(i->nr)+1;}
}
printf("elementul nu s a gasit in lista\n");
}
