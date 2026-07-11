#ifndef WINDOWINTERFACE
#define WINDOWINTERFACE
typedef struct{
int upline,downline,leftline,rightline;
}Window;
void setPlayWindowLimits(Window **window);

#endif
