#include"Mapping.h"
#include<stdio.h>
#include<stdlib.h>

void ZeroMapping(int **map,int downlimit,int rightlimit){
  for(int i=0;i<downlimit;i++)
    for(int j=0;j<rightlimit;j++)
       map[i][j]=0;
}

//X--coloane Y--linii
void MapMoves(int **map,Snake *snake,foodblock* food,int foodsize,int downlimit,int rightlimit){
    if(snake==NULL)return;
    ZeroMapping(map,downlimit,rightlimit);
    int j;

    for(j=0;j<foodsize;j++)
    map[food->Ypos][food->Xpos]=2;
    
    if(snake->head->Xpos<=rightlimit && snake->head->Xpos>=0 && snake->head->Ypos<=downlimit && snake->head->Ypos>=0)
    map[snake->head->Ypos][snake->head->Xpos]=1;

  int i=0;
  if(snake->bodylength==1){
  if(snake->body->Xpos<=rightlimit && snake->body->Xpos>=0 && snake->body->Ypos<=downlimit && snake->body->Ypos>=0)
    map[snake->body->Ypos][snake->body->Xpos]=1;
  }else{
    for(i=0;i<snake->bodylength;i++){
        if(snake->body[i].Xpos<=rightlimit && snake->body[i].Xpos>=0 && snake->body[i].Ypos<=downlimit && snake->body[i].Ypos>=0)
    map[snake->body[i].Ypos][snake->body[i].Xpos]=1;
    }
  }
}

void displayMap(int **map,char Ssymbol,char Fsymbol,int downlimit,int rightlimit){
  for(int i=0;i<downlimit;i++){
    for(int j=0;j<rightlimit;j++){
       if(map[i][j]==0)
       printf(" ");
      else
       if(map[i][j]==1)
       printf("%c",Ssymbol);
      else
       if(map[i][j]==2)
       printf("%c",Fsymbol);
    }   printf("\n");
  }
}

void MakeMatrix(int ***map,Window *window){
*map=malloc(sizeof(int*)*(window->downline));
for(int i=0;i<window->downline;i++)
(*map)[i]=malloc(sizeof(int)*(window->rightline));
ZeroMapping(*map,window->downline,window->rightline);
}
