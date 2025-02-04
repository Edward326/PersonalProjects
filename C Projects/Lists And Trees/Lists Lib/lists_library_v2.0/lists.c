#include<stdio.h>
#include<stdlib.h>
#include"lists.h"
#include<sys/ioctl.h>

void help(){struct winsize window;ioctl(0, TIOCGWINSZ , &window);
  printf("\e[1;35m\e[4;35mlists double/simple linked LIBRARY(v2.0)\e[4;35m\e[1;35m \033[0m\e[1;37m--dev_>>vesea_eduard_razvan--\e[1;37m\033[0m\n\n");
  printf("the structure for ~simple list:\ntypedef struct elemS{\nint nr;\nstruct elem *next;\n}elemS;\n\ntypedef struct{\nelemS *first,*last;\n}simple_list;\n\n");
  printf("the structure for ~double list:\ntypedef struct elemD{\nint nr;\nstruct elem *next,*back;\n}elemD;\n\ntypedef struct{\nelemD *first,*last;\n}double_list;\n\n\n--->enter in lists.h for more details\n\e[0;31mpress any char to continue...\e[0;31m\033[0m");
  char c;scanf("%c",&c);
  while(--window.ws_row)printf("\n");
  
}





//op standard duble-simple :afisare,init lista simpla/dubla,init nod lista simpla/dubla,stergere elemente din lista,stergerea nodurilor si dezalocarea listei



void viewsimple(simple_list *a){
  if(a==NULL){printf("lista e neinit");return;}
  elemS *i;
  if(a->last){
  for(i=a->first;i;i=i->next)
      printf("%d ",i->nr);
  }
  else printf("lista simpla goala");
  printf("\n");
}
void viewdouble(double_list *a){
   if(a==NULL){printf("lista e neinit");return;}
  elemD *i;
  if(a->last){
  for(i=a->first;i;i=i->next)
      printf("%d ",i->nr);
  }
   else printf("lista dubla goala");
  printf("\n");
}

void new_elemS(elemS **new,int nr){
  *new=malloc(sizeof(elemS));
  (*new)->nr=nr;
}
void new_elemD(elemD **new,int nr){
  *new=malloc(sizeof(elemD));
  (*new)->nr=nr;
}

void initsimple(simple_list **a){
  *a=malloc(sizeof(simple_list));
  (*a)->last=(*a)->first=NULL;
}
void initdouble(double_list **a){
  *a=malloc(sizeof(double_list));
  (*a)->last=(*a)->first=NULL;
}

void empty_simple(simple_list *a){
  if(a->last)
    {elemS *i,*aux;
      for(i=a->first;i;i=aux){
	aux=i->next;
	free(i);
    }a->last=a->first=NULL;
    }
  else
    printf("simple list already empty");
}

void empty_double(double_list *a){
  if(a->last)
    {elemD *i,*aux;
      for(i=a->first;i;i=aux){
	aux=i->next;
	free(i);
    }a->last=a->first=NULL;
    }
  else
    printf("double list already empty");
}
void trash_simple(simple_list **a){
  if((*a)->last)
    empty_simple((*a));
  //free((*a)->last);
  //free((*a)->first);
  free((*a));
}
void trash_double(double_list **a){
  if((*a)->last)
    empty_double((*a));
  //free((*a)->last);
  //free((*a)->first);
  free((*a));
}

//---------------------------------------------------------------------
//operatii cu lista simpla


void addsimple(simple_list **a,int number){
  elemS *new=NULL;
  new_elemS(&new,number);

  if(*a){
    if((*a)->last){
    (*a)->last->next=new;
    (*a)->last=new;}
    else
      (*a)->first=(*a)->last=new;
  }
  else
    {
      initsimple(*(&a));
      (*a)->first=(*a)->last=new;
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


void deletesimple(simple_list *a,int nr){
  if(a->last){

    if(a->first->nr==nr){
      if(a->first->next){
      elemS *Aux=a->first;
      a->first=a->first->next;
      free(Aux);
      deletesimple(a,nr);
      }
      else{
	     a->first=a->last=NULL;
      }
    }
    else
      {elemS *aux=a->first,*urm,*i;
	for(i=a->first->next;i;i=urm){
    urm=i->next;
	  if(i->nr==nr){
	    if(i!=a->last){
	    aux->next=urm;
	    free(i);
	    }
	    else
	      {
		a->last=aux;
		a->last->next=urm;
		free(i);
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

void deletesimple_sorted(simple_list *a,int nr){
   if(a->last){
    if(a->first->nr==nr){
      if(a->first->next){
      elemS *Aux=a->first;
      a->first=a->first->next;
      free(Aux);
      deletesimple_sorted(a,nr);
      }
      else{free(a->last);
      }
    }
    else
      {elemS *aux=a->first,*urm,*i,*j;int ok=0;
	for(i=a->first->next;i;i=urm){
	  if(i->nr==nr){
	   
	      j=i;
	      while(i->nr==nr){
                j=i->next;
		free(i);
		i=j;
		if(i==NULL){
		  ok=1;break;
		}
		
	      }
              if(ok==0){
		//free(j);
	      urm=i;
	    aux->next=urm;
	      }else
		{
		a->last=aux;
		a->last->next=NULL;
		urm=i;
		}
	  }
	  else{
	      aux=i;
	    urm=i->next;
	  }
	}
 
      }
   }
  else{
    printf("lista simpla e goala");
    exit(-1);
  }
}

int verifysortedsimple(simple_list *a){
  int caz=0;
  if(a==NULL || a->last==NULL || a->first==a->last){perror("lista e neinitializata/(sau) lista e goala/(sau) are un singur nod");exit(EXIT_FAILURE);}
   
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

int sortsimple(simple_list *list) {//bubblesort
    if (list==NULL || list->last == NULL || list->last == list->first) return 0;
    if (verifysortedsimple(list)) return 0;

    elemS *a, *b, *prev, *temp;
    int swapped;

    do {
        swapped = 0;
        prev = NULL;
        a = list->first;
        b = a->next;

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
            prev = a;
            a = a->next;
            b = (a) ? a->next : NULL;
        }
    } while (swapped);
    return 1;
}

void addreverse_simple(simple_list *source,simple_list **dest){
if(source==NULL || source->last==NULL){printf("lista sursa e goala/nu se face inversarea");return;}
if(*dest)
{
  if((*dest)->last)
  empty_simple(*dest);
}
else
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

int verifyreversed(simple_list *lista){
if(lista->last==NULL || lista==NULL){perror("lista e goala/(sau) neinitializata");exit(-1);}
if(lista->last==lista->first)return 1;
simple_list *rev=NULL;
addreverse_simple(lista,&rev);
elemS *j=rev->first;
for(elemS *i=lista->first;i;i=i->next){
  if(i->nr!=j->nr){trash_simple(&rev);
  return 0;
  }
j=j->next;
}
trash_simple(&rev);
return 1;
}





//---------------------------------------------------------------------
//operatii cu lista dubla

void makeconnection(elemD **a,elemD **b){
(*a)->next=(*b);
(*b)->back=(*a);
(*a)=(*b);
}

void addouble(double_list **a,int number){
  elemD *new=NULL;
  new_elemD(&new,number);

  if(*a){
    if((*a)->last){
    makeconnection(&((*a)->last),&new);
    }
    else
      (*a)->first=(*a)->last=new;
  }
  else
    {
      initdouble(*(&a));
      (*a)->first=(*a)->last=new;
    }
}


void add_sortedouble(double_list **a,int nr){

  elemD *new;
  new_elemD(&new,nr);

  if((*a)){
    if((*a)->last==NULL){
      (*a)->first=(*a)->last=new;
      return;
    }
    
    if((*a)->first->nr>new->nr){
      new->next=(*a)->first;
      (*a)->first->back=new;
      (*a)->first=new;
      return;
    }
    else{
      
      elemD *i=(*a)->first;
      for(i=i->next;i;i=i->next){
	if(i->nr>new->nr){
	  new->next=i;
    new->back=i->back;
    i->back->next=new;
    i->back=new;
    return;
	}
      }
    }
    makeconnection(&((*a)->last),&new);
  }
  else{
    initdouble(*(&a));
    (*a)->first=(*a)->last=new;
  }
}

void deletedouble(double_list *a,int nr){
  
 if(a->last){
    if(a->first->nr==nr){
      if(a->first->next){
      elemD *Aux=a->first;
      a->first=a->first->next;
      free(Aux);
      deletedouble(a,nr);
      }
      else{a->first=a->last=NULL;
      }
    }
    else
      {elemD *urm,*i;
	for(i=a->first->next;i;i=urm){
    urm=i->next;
	  if(i->nr==nr){
	    if(i!=a->last){
	      urm->back=i->back;
        i->back->next=urm;
	    }
	    else
	      {
		     a->last=i->back;
         a->last->next=urm;
	      }
	  }
	}
 
      }
  }
  else{
    printf("lista dubla e goala");
    exit(-1);
  }
}

int verifysortedouble(double_list *a){
  int caz=0;
  if(a==NULL || a->last==NULL || a->first==a->last){perror("lista e neinitializata/(sau) lista e goala/(sau) are un singur nod");exit(EXIT_FAILURE);}
   
  if(a->first->nr > a->first->next->nr)
    caz=1;
  if(a->first->nr < a->first->next->nr)
    caz=2;
  elemD *aux;
 
  
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

int sortdouble(double_list *list) {//bubblesort
    if (list==NULL || list->last == NULL || list->last == list->first) return 0;
    if (verifysortedouble(list)) return 0;

    elemD *a, *b, *prev, *temp;
    int swapped;

    do {
        swapped = 0;
        prev = NULL;
        a = list->first;
        b = a->next;

        while (b) {
            if (a->nr > b->nr) {
                if (prev) {
                    temp = b->next;
                    prev->next = b;
                    b->back=prev;
                    b->next = a;
                    a->back=b;
                    a->next = temp;
                    temp->back=a;
                } else {
                    temp = b->next;
                    list->first = b;
                    list->first->back=NULL;
                    b->next = a;
                    a->back=b;
                    a->next = temp;
                    temp->back=a;
                }
                swapped = 1;
            }
            prev = a;
            a = a->next;
            b = (a) ? a->next : NULL;
        }
    } while (swapped);
    return 1;
}

void addreverse_double(double_list *source,double_list **dest){
if(source==NULL || source->last==NULL){printf("lista sursa e goala/nu se face inversarea");return;}
if(*dest)
{
  if((*dest)->last)
  empty_double(*dest);
}
else
initdouble(*(&dest));

elemD *aux=NULL;
for(elemD *i=source->first;i;i=i->next)
{
  new_elemD(&aux,i->nr);
  if((*dest)->last){
    aux->next=(*dest)->first;
    (*dest)->first->back=aux;
    (*dest)->first=aux;
  }
  else
  (*dest)->last=(*dest)->first=aux;
}
}

int verifyreversed_double(double_list *lista){
if(lista->last==NULL || lista==NULL){perror("lista e goala/(sau) neinitializata");exit(-1);}
if(lista->last==lista->first)return 1;
double_list *rev=NULL;
addreverse_double(lista,&rev);
elemD *j=rev->first;
for(elemD *i=lista->first;i;i=i->next){
  if(i->nr!=j->nr){trash_double(&rev);
  return 0;
  }
j=j->next;
}
trash_double(&rev);
return 1;
}

