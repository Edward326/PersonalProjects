#include<stdlib.h>
#include<stdio.h>
#include"listsop.h"


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
        a = list->first;
        b = a->next;
        nr_acces++;//accesam primul nod si de acolo avem acces la toate nodurile urm
        while (b) {
            if (a->nr > b->nr) {
                if (prev) {
                    temp = b->next;
                    prev->next = b;
                    b->next = a;
                    a->next = temp;
                } else {
                    temp = b->next;
                    list->first = b;
                    b->next = a;
                    a->next = temp;
                }
                swapped = 1;
            }
            nr_acces+=2;
            prev = a;
            a = a->next;
            b = (a) ? a->next : NULL;
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
void display(int nrspacing,int spacing,int nodvalue){
  while(spacing){printf(" ");spacing--;}
  while(nrspacing){printf("\e[1;31m^");nrspacing--;}
  //printf("\e[0;37m(nod->%d)\n",nodvalue);
   printf("\e[0;37m");
}

void fillfrecv(simple_list *list,int **frecv){
int nmax=0;
for(elemS *i=list->first;i;i=i->next)
if(i->nr>nmax)nmax=i->nr;
*frecv=malloc(sizeof(int)*(nmax+1));int *frecv2=malloc(sizeof(int)*(nmax+1));
for(elemS *i=list->first;i;i=i->next)
frecv2[i->nr]++;
*frecv=frecv2;
}

void findnr(simple_list *lista,int data){
if(lista==NULL || lista->last==NULL){printf("lista e neinit/(sau) lista e goala\n");return;}

int *frecv;
fillfrecv(lista,&frecv);

int nod=1,spacing=0;
for(elemS *i=lista->first;i;i=i->next){
  if(i->nr==data){  viewsimple(lista);//display(nrofdigit(data),spacing,nod);
    int aux_frecv=frecv[i->nr];
    while(aux_frecv){
    display(nrofdigit(data),spacing,nod);
    spacing=1;aux_frecv--;
    }printf("\n");free(frecv);return;
  }
  else{nod++;spacing+=nrofdigit(i->nr)+1;}
}free(frecv);
printf("elementul nu s a gasit in lista\n");
}
