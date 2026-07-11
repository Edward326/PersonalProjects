#include<stdio.h>
#include<stdlib.h>
#include"lists.h"

/*
typedef struct elemS{
  int nr;
  struct elemS *next;
}elemS;

typedef struct{
  elemS *first,*last;
}simple_list;

typedef struct elemD{
  int nr;
  struct elemD *next,*back;
}elemD;

typedef struct{
  elemD *first,*last;
}double_list;

*/

void help(){
  printf("lists double/simple linked library(V1.0) --dev_>eduard_vesea\n\n");
  printf("the structure for ~simple list:\ntypedef struct elemS{\nint nr;\nstruct elem *next;\n}elemS;\n\ntypedef struct{\nelemS *first,*last;\n}simple_list;\n\n");
  printf("the structure for ~double list:\ntypedef struct elemD{\nint nr;\nstruct elem *next,*back;\n}elemD;\n\ntypedef struct{\nelemD *first,*last;\n}double_list;\n\n\n--->enter in lists.h for more details\n");
}
void viewsimple(simple_list *a){
  elemS *i;
  if(a->last){
  for(i=a->first;i;i=i->next)
    if(i)
      printf("%d ",i->nr);
  }
  printf("\n");
}
void viewdouble(double_list *a){
  elemD *i;
  if(a->last){
  for(i=a->first;i;i=i->next)
    if(i)
      printf("%d ",i->nr);
  }
  printf("\n");
}

elemS *new_elemS(int nr){
  elemS *new;
  new=malloc(sizeof(elemS));
  new->nr=nr;
  return new;
}
elemD *new_elemD(int nr){
  elemD *new;
  new=malloc(sizeof(elemD));
  new->nr=nr;
  return new;
}

simple_list *initsimple(){
  simple_list *a;
  a=malloc(sizeof(simple_list));
  a->last=a->first=NULL;
  return a;
}

double_list *initdouble(){
  double_list *a;
  a=malloc(sizeof(double_list));
  a->last=a->first=NULL;
  return a;
}

void free_upsimple(simple_list *a){
  if(a->last)
    {elemS *i,*aux;
      for(i=a->first;i;i=aux){
	if(i){
	aux=i->next;
	free(i);
	}
      }
      free(aux);
    }
  else
    printf("list already is empty");
}

void free_updouble(double_list *a){
  if(a->last)
    {elemD *i,*aux;
      for(i=a->first;i;i=aux){
	if(i){
	aux=i->next;
	free(i);
	}
      }
      free(aux);
    }
  else
    printf("list already is empty");
}

void addsimple(simple_list *a,int number){
  elemS *new=NULL;
  new=new_elemS(number);
  if(a->last){
    a->last->next=new;
    a->last=new;
  }else
    a->first=a->last=new;
}

void add_sortedsimple(simple_list *a,int nr){
  elemS *aux=a->first,*new;
  new=new_elemS(nr);int ok=0;
  if(a->last){
    
    if(a->first->nr>new->nr){
      new->next=a->first;
      a->first=new;
      ok=1;
    } 
    else{
      
      elemS *i=a->first;
      for(i=i->next;i;i=i->next){
	if(i->nr>new->nr){
	  new->next=i;
	  aux->next=new;ok=1;
	}
	if(i)
	  aux=i;
      }
    
    } 
    
    if(ok==0){
      a->last->next=new;
      a->last=new;
    }
  }
  else{
    a->first=a->last=new;
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
      else{free(a);
	a=initsimple();
      }
    }
    else
      {elemS *aux=a->first,*urm,*i;
	for(i=a->first->next;i;i=urm){
	  if(i->nr==nr){
	    if(i!=a->last){
	      urm=i->next;
	    aux->next=urm;
	    free(i);
	    }
	    else
	      {
		a->last=aux;
		a->last->next=NULL;
		urm=i->next;
		free(i);
	      }
	  }
	  else{
	    if(i){
	      aux=i;
	    urm=i->next;
	    }
	  }
	}
 
      }
  }
  else{
    printf("lista e goala");
    exit(-1);
  }
 
  
  //if(adev==0)
  //printf("elem %d nu s-a gasit//lista a ramas neschimbata\n\n",nr);

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
      else{free(a);
	a=initsimple();
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
	    if(i){
	      aux=i;
	    urm=i->next;
	    }
	  }
	}
 
      }
   }
  else{
    printf("lista e goala");
    exit(-1);
  }
  //if(adev==0)
  //printf("elem %d nu s-a gasit//lista a ramas neschimbata\n\n",nr);
}



//---------------------------------------------------------------------




void adddouble(double_list *a,int nr){
  elemD *b;b=new_elemD(nr);
   b->back=a->last;
  if(a->last){
    a->last->next=b;
    a->last=b;
  }
  else
    {
      a->first=b;
      a->last=b;
    }
  b->next=NULL;  
}


void adddouble_sorted(double_list *a,int nr){
  elemD *b;b=new_elemD(nr);
  int ok=0;
  if(a->last){
    ok=0;
    elemD *i,*aux;
    if(a->first->nr>nr){
      a->first->back=b;
      b->next=a->first;
      a->first=b;ok=1;
    }
    else{
    for(i=a->first;i;i=i->next)
      {

	if(i->nr>nr){
	  b->back=i->back;
	  b->next=i;
          i->back->next=b;
	  i->back=b;
	 
	  a->last=i;
	  ok=1;
	}
	if(i)
	  aux=i;
      }
    }
    
    if(ok==0){	  
 b->back=aux;
a->last->next=b;
    a->last=b;
    }
  
  }
  else
    {
      b->back=NULL;
      a->first=b;
      a->last=b;
    }
  a->last->next=NULL;


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
      else{free(a);
	a=initdouble();
      }
    }
    else
      {elemD *urm,*i;
	for(i=a->first;i;i=urm){
	  if(i->nr==nr){
	    if(i!=a->last){
	      urm=i->next;
	      urm->back=i->back;
	      i->back->next=urm;
	    free(i);
	    }
	    else
	      {
		a->last=i->back;
		a->last->next=NULL;
		urm=i->next;
		free(i);
	      }
	  }
	  
	}
 
      }
  }
  else{
    printf("lista e goala");
    exit(-1);
  }
 
  
  //if(adev==0)
  //printf("elem %d nu s-a gasit//lista a ramas neschimbata\n\n",nr);
}
