#include<stdio.h>
#include<stdlib.h>
#include"lists.h"

/*
typedef struct elemS{
  int nr;
  struct elem *next;
}elemS;

typedef struct{
  elemS *first,*last;
}simple_list;

typedef struct elemD{
  int nr;
  struct elem *next,*back;
}elemD;

typedef struct{
  elemD *first,*last;
}double_list;
*/

int main(){
  simple_list *total=NULL;int nrtot,nr;
  total=initsimple();
  scanf("%d",&nrtot);
  for(int i=0;i<nrtot;i++){scanf("%d",&nr);
    //addsimple(total,nr);
    add_sortedsimple(total,nr);
      printf("\n");
  }
  viewsimple(total);
  
   scanf("%d",&nrtot);
  for(int i=0;i<nrtot;i++){scanf("%d",&nr);
    //deletesimple(total,nr);
    deletesimple_sorted(total,nr);
    viewsimple(total);
      printf("\n");
      }
  free_upsimple(total);

  return 0;
}
