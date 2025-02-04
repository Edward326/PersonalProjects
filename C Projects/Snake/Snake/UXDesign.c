#include"UXDesign.h"
#include"Mapping.h"
#include"WindowInterface.h"
#include"snakestructure.h"
#include<sys/ioctl.h>
#include<stdlib.h>
//#include<unistd.h>
#include<termios.h>
#include<stdio.h>


typedef struct{
int sec;int min;
}clock;
void makeclock(clock **reff,int sec){
*reff=malloc(sizeof(clock));
(*reff)->min=0;
if(sec<60)
(*reff)->sec=sec;
else{
    (*reff)->min=sec/60;
    (*reff)->sec=sec-(60*((*reff)->min));
}
}


char getHiddenChar() {
    struct termios oldt, newt;
    char ch;

    // Dezactivam afisarea la stdin si confirmarea caracterului introdus cu enter
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    //citim normal caracterul
    ch = getchar();

    // revenirea terminal std
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}


void OpenScreen(){

    //to implement
}

void InGameScreen(int **map,char Ssymbol,char Fsymbol,int score,int time,int downlimit,int rightlimit){
     int i;   
     struct winsize limits;ioctl(0, TIOCGWINSZ , &limits);
    for(i=0;i<limits.ws_row;i++){printf("\n");}

displayMap(map,Ssymbol,Fsymbol,downlimit,rightlimit);


for(i=0;i<rightlimit;i++){printf("=");}printf("\n");
printf("\e[1;35mCurrent Score:\e[1;37m%d\n\n\e[4;32m",score);
printf("UP = W/UP_ARROW\nDOWN = S/DOWN_ARROW\nLEFT = A/LEFT_ARROW\nRIGHT = D/RIGHT_ARROW\033[0;37m\n");
for(i=downlimit+5;i<=limits.ws_row-3;i++){printf("\n");}

clock* reff=NULL;
makeclock(&reff,time);
printf("CURRENT PLAYTIME:%d:%d\n",reff->min,reff->sec);
for(i=0;i<rightlimit;i++){printf("_");}
free(reff);
}

void ExitScreen(int **map,int score,int time,int downlimit,int rightlimit){

FILE *file=fopen("score.dat","rb+");
int h_score=0,caz=1;if(feof(file)==0)fread(&h_score,sizeof(int),1,file);
if(h_score<score){fclose(file);file=fopen("score.dat","wb+");fwrite(&score,sizeof(int),1,file);fclose(file);caz=0;}

int i;
   struct winsize limits;ioctl(0, TIOCGWINSZ , &limits);
    for(i=0;i<limits.ws_row;i++){printf("\n");}


printf("\e[40m");
for(i=1;i<=downlimit/2-1;i++){printf("\n");}
for(i=1;i<=rightlimit/2-1;i++){printf(" ");}printf("\e[1;31mEND OF GAME!!\e[0;37m\e[40m\n");
for(i=downlimit/2+1;i<=downlimit;i++){printf("\n");}
for(i=0;i<rightlimit;i++){printf("=");}printf("\n");

if(caz){
printf("\e[40m\e[1;35mCurrent Score:\e[1;37m\e[40m%d\e[0;37m\e[40m but not BEATED THE HIGHSCORE OF \e[1;37m%d\e[0;37m\e[40m\n\n",score,h_score);
}
else printf("\e[1;35mCurrent Score:\e[1;37m\e[40m%d\e[0;37m\e[40m AND BEATED THE HIGHSCORE OF \e[1;37m%d \e[1;35m ==>NEW SCORE:%d\033[0;37m\e[40m\n\n",score,h_score,score);


for(i=downlimit+1;i<=limits.ws_row-3;i++){printf("\n");}
clock* reff=NULL;
makeclock(&reff,time);
printf("GAME LASTED PLAYTIME:%d:%d\n\e[0m",reff->min,reff->sec);
free(reff);
printf("\nHIT ANY KEY TO CONTINUE...");
char c=getHiddenChar();
 for(i=0;i<limits.ws_row;i++){printf("\n");}
}




//test unit
int main(){
    Window *window=NULL;setPlayWindowLimits(&window);
   int **map=NULL;MakeMatrix(&map,window);

   FILE *file=fopen("score.dat","wb+");
   fclose(file);

    InGameScreen(map,'X','O',50,1,window->downline,window->rightline);
    sleep(1);
    InGameScreen(map,'X','O',51,2,window->downline,window->rightline);
    sleep(1);
    ExitScreen(map,51,3,window->downline,window->rightline);
        sleep(1);
    ExitScreen(map,50,3,window->downline,window->rightline);

    return 0;
}