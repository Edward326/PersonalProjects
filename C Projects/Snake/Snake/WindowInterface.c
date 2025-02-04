#include<stdlib.h>
#include<sys/ioctl.h>
#include"WindowInterface.h"


void setPlayWindowLimits(Window **current){
  *current=malloc(sizeof(Window));
  struct winsize limits;ioctl(0, TIOCGWINSZ , &limits);
  (*current)->upline=0;
    (*current)->downline=((limits.ws_row/2)/2)+(limits.ws_row/2);//+(limits.ws_row/10);//o schimbi pt a lasa si gui in partea de jos
  (*current)->rightline=limits.ws_col;
    (*current)->leftline=0;    
}

