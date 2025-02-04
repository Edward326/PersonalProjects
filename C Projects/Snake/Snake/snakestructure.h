#ifndef SNAKESTRUCTURE
#define SNAKESTRUCTURE

const static char foodsymbol='O';
const static char bodysymbol='X';

typedef struct{
int Xpos,Ypos;
}foodblock;


typedef struct{
int Xpos,Ypos;
}bodyblock;

typedef struct{
int bodylength;
bodyblock* body;
bodyblock* head;
}Snake;

#endif
