#include<stdlib.h>
#include<stdio.h>
#include"lists.h"

void initsimple(simple_list **a){
  *a=malloc(sizeof(simple_list));
  (*a)->last=(*a)->first=NULL;
  (*a)->last=(*a)->firstsorted=NULL;
}
void new_elemS(elemS **new,int nr){
  *new=malloc(sizeof(elemS));
  (*new)->nr=nr;
}


void viewsimple(simple_list *a){
  elemS *i;
  if(a->last){
  for(i=a->first;i;i=i->next)
      printf("%d ",i->nr);
  }
  else
    printf("empty list");
  printf("\n");
}
void viewsimplesorted(simple_list *a){
  elemS *i;
  if(a->last){
  for(i=a->firstsorted;i;i=i->nextsorted)
      printf("%d ",i->nr);
  }
  else
    printf("empty list");
  printf("\n");
}

void free_upsimple(simple_list *a){
   if(a->first)
    {elemS *i,*aux;
      for(i=a->first;i;i=aux){
	aux=i->next;
	free(i);
    }a->last=a->first=NULL;
    }
  else
    printf("simple list already empty");
}
void free_totallysimple(simple_list **a){
  if((*a)->first)
    free_upsimple((*a));
  //free((*a)->last);
  //free((*a)->first);
  free((*a));
}

void addsimple(simple_list **a,int number){
  elemS *new=NULL;
  new_elemS(&new,number);
  
   if(*a){
    if((*a)->last){
    (*a)->last->next=new;
    (*a)->last->nextsorted=new;
    (*a)->last=new;
    (*a)->last->nextsorted=NULL;
    (*a)->last->next=NULL;}
    else{
      (*a)->first=(*a)->last=new;
      (*a)->firstsorted=(*a)->last=new;
         (*a)->last->next=NULL;
      (*a)->last->nextsorted=NULL;
    }
  }
  else
    {
      initsimple(*(&a));
      (*a)->first=(*a)->last=new;
      (*a)->firstsorted=(*a)->last=new;
      (*a)->last->next=NULL;
      (*a)->last->nextsorted=NULL;
    }
}






void add_sortedsimple(simple_list **a,int nr){


  
  elemS *new;
  new_elemS(&new,nr);

  if((*a)){
    if((*a)->last==NULL){
      (*a)->first=(*a)->last=new;
      return;
    }
    if((*a)->first->nr>new->nr){
      new->next=(*a)->first;
      (*a)->first=new;
      return;
	}
    else{
      
      elemS *i=(*a)->first,*aux=(*a)->first;
      for(i=i->next;i;i=i->next){
	if(i->nr>new->nr){
	  new->next=i;
	  aux->next=new;return;
	}
	  aux=i;
      }
    }
      (*a)->last->next=new;
      (*a)->last=new;
  }
  else{
    initsimple(*(&a));
    (*a)->first=(*a)->last=new;
  }
}

void addreverse(simple_list *source,simple_list **dest){
if(*dest!=NULL || source==NULL || source->last==NULL)return;
initsimple(*(&dest));
elemS *aux=NULL;
for(elemS *i=source->first;i;i=i->next)
{
  new_elemS(&aux,i->nr);
  if((*dest)->last){
    aux->next=(*dest)->first;
    (*dest)->first=aux;
  }
  else
  (*dest)->last=(*dest)->first=aux;
}


}
int verifypalind(simple_list *lista){
if(lista->last==NULL || lista==NULL){perror("lista e goala,neinitializata");exit(-1);}
if(lista->last==lista->first)return 1;
simple_list *rev=NULL;
addreverse(lista,&rev);
elemS *j=rev->first;
for(elemS *i=lista->first;i;i=i->next){
  if(i->nr!=j->nr)
  return 0;
j=j->next;
}
return 1;



}
