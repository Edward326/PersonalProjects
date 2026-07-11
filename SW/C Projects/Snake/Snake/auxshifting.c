#include<stdio.h>
//#include"snakestructure.h"
#include"WindowInterface.h"
//#include"MovingMechanism.h"
//#include"WindowDimension.h"
#include<sys/ioctl.h>
#include<unistd.h>



//algortim deplasare st-dr sus-jos fara conditii de copliziune cu revenire in ecran de la lungimea/2 a sarpelui
//viteza de deplasare a sarpelui este data de val shiftarii(mai mare) si timpul de aastaptare sleep intre deplasari
#define stdshift 10

int main(){struct winsize limit;ioctl(0, TIOCGWINSZ , &limit);

Window *window=NULL;
int block=16,shift=0,shifting=stdshift,defn=1,i,j,j1,j2,k,z,diff;diff=shifting;
setPlayWindowLimits(&window);


while(defn){
    for(i=0;i<limit.ws_row;i++)printf("\n");
for(j=0;j<window->downline/2;j++)printf("\n");


if(shift+block>window->rightline)
{
    if(shift<window->rightline){
        for(j1=0;j1<block-(window->rightline-shift)-block/2;j1++)printf("=");
        for(j1=j1+1;j1<=shift;j1++)printf(" ");
        for(j1=window->rightline-shift;j1>0;j1--)printf("=");
    }
    else
    {
    if(shift-window->rightline < block/2){
        for(j1=1;j1<=diff;j1++)printf("=");
        for(j1=block/2;j1<block;j1++)printf("=");

        diff+=shifting;
    } else
      {
        shift=0;
        for(j1=0;j1<block;j1++)printf("=");   
        
      }
    }

}
else{
    for(j1=0;j1<shift;j1++)printf(" "); 
    for(j1=0;j1<block;j1++)printf("=");
}


for(j2=j+1;j2<window->downline;j2++)printf("\n");
for(k=0;k<window->rightline;k++)printf("=");
for(z=0;z<limit.ws_row-window->downline;z++)printf("\n");

shift+=shifting;
sleep(1);
}
return 0;
}
