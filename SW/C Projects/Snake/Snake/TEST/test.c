#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>
#include<termios.h>
#include<unistd.h>

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

int main(){
  char ch='c';


   while(ch!='0'){
     ch=getHiddenChar();
   if(ch!='0')
     {
       if(ch==65)
    printf("\nUP");
         if(ch==66)
    printf("\nDOWN");
	   if(ch==68)
    printf("\nLEFT");
	     if(ch==67)
    printf("\nRIGHT");
	       if(ch=='\n')
    printf("\nENTER");
     }
   }
   
   
  return 0;}
