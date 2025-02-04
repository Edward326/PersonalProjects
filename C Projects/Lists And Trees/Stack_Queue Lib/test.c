#include<stdio.h>
#include<stdlib.h>

struct nodS{
  int nr;
  struct nodS *next;
};
struct nodD{
  int nr;
  struct nodD *back,*next;
};



//QUEUE -->COADA
void view_queue(struct nodS *first){
  if(first==NULL){printf("coada e goala");return;}printf("\n");
  for(struct nodS* i=first;i;i=i->next){
    printf("%d ",i->nr);
  }printf("\n");
}
void enqueue(struct nodS **first,int nr){
  if(*first==NULL){*first=malloc(sizeof(struct nodS)); (*first)->nr=nr;view_queue(*first);return;}
struct nodS *i,*nod=malloc(sizeof(struct nodS));nod->nr=nr;
for(i=*first;i->next;i=i->next);
i->next=nod;view_queue(*first);
}
void dequeue(struct nodS **first){
  if(*first){
  struct nodS *aux;
  aux=*first;
  *first=(*first)->next;free(aux);
  if(*first)view_queue(*first);
  }
  else
  printf("\ncoada e goala\n");
}
void free_queue(struct nodS **first){
if(*first){
  while(*first)
  dequeue(*(&first));
  printf("\ncoada e eliberata\n");
 }
}



//STACK-->STIVA
void view_stack(struct nodS *first){
  if(first==NULL){printf("stiva e goala");return;}printf("\n");
  for(struct nodS* i=first;i;i=i->next){
    printf("%d ",i->nr);
  }printf("\n");
}
struct nodS* push(struct nodS *nod,int nr){
if(nod==NULL){nod=malloc(sizeof(struct nodS ));nod->nr=nr;view_stack(nod);return nod;}
struct nodS *nodante=malloc(sizeof(struct nodS));nodante->nr=nr;
nodante->next=nod;view_stack(nodante);
return nodante;
}

void pop(struct nodS **first){
if(*first){
   struct nodS *aux;
  aux=*first;
  *first=(*first)->next;free(aux);
  if(*first)view_stack(*first);
}else 
printf("stiva e goala");
}
void free_stack(struct nodS **first){
if(*first){
  while(*first)
  pop(*(&first));
  printf("\nstiva e eliberata\n");
 }
}

int main(){
  int sir[]={1,2,3,4,5,7};



  //queue-->coada-FIFO   peek-->ultimul el din lista
  //la fel e implementarea si pt lista dubla
  
  struct nodS *first=NULL;
    printf("\nENQUEUE:\n");
  for(int i=0;i<sizeof(sir)/4;i++){
    printf("%d\n",sir[i]);
    enqueue(&first,sir[i]);printf("\n----------\n");
  }
    printf("\nDEQUEUE:\n");
  dequeue(&first);
  dequeue(&first);
  dequeue(&first);

  free_queue(&first);
  view_queue(first);//am golit intreaga coada ca sa outem adauga el de la stiva

//-------------------------------------
int n;scanf("%d",&n);
//-------------------------------------


  //stack-->stiva-LIFO   peek-->primul el din lista
  //pt dubla nu mai e necasar returnarea valorii ,cand adaugam el parcurgem de la utlimul la primul,cand stergem parc de la ult la primul si dupa de la primul stergem pe rand

  //struct nodS* first=NULL;
  printf("\nPUSH:\n");
  for(int i=0;i<sizeof(sir)/4;i++){
     printf("%d\n",sir[i]);
    first=push(first,sir[i]);printf("\n----------\n");
  }
    printf("\nPOP:\n");
  pop(&first);
  pop(&first);
  pop(&first);

  free_stack(&first);
  view_stack(first);
  free(first);
  


  return 0;
}
  
