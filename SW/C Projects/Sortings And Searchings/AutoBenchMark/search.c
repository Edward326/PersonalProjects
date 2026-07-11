#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/ioctl.h>
#include<time.h>
#include<unistd.h>
#include<termios.h>
#include"search.h"
#include"sort.h"
#include"record.h"
#include<limits.h>
#define numberofSortMethods 6

typedef struct{
double totalRunTime;
int compSort;
int swapsSort;
int compSearch;
}Performance;



void PerformanceDisplay(Performance *perf){
//=========================> Performance HUD
double mintotalRunTime; int index1;
int mincompSort,index2;
int minswapsSort,index3;
int mincompSearch,index4;  mincompSearch=mincompSort=minswapsSort=mintotalRunTime=INT_MAX;
printf("\e[4;31mPerformance:\n\n\e[0m");
for(int k=0;k<numberofSortMethods;k++)
{
   printf("\e[1;31mAlgortihm %d-->\e[0m\nSortingMethod(Total nr of comparations:%d , Total nr of swaps:%d)\nSearchMethod(Total nr of comparations:%d)\n",k+1,perf[k].compSort,perf[k].swapsSort,perf[k].compSearch);
   printf("Total time spent on search algorithm(SUM TIME of sorting on each substring and searching on it):\e[1;35m%f\e[0m",perf[k].totalRunTime);
   printf("\n\n");
   if(mintotalRunTime>perf[k].totalRunTime)
   {
    mintotalRunTime=perf[k].totalRunTime;index1=k;
   }
   if(mincompSearch>perf[k].compSearch)
   {
    mincompSearch=perf[k].compSearch;index2=k;
   }
   if(minswapsSort>perf[k].swapsSort)
   {
    minswapsSort=perf[k].swapsSort;index3=k;
   }
   if(mincompSort>perf[k].compSort)
   {
    mincompSort=perf[k].compSort;index4=k;
   }
}
printf("\n\n");
printf("OVERALL PERFORMANCE:\n");
printf("By Min Total of Runtime: Algorithm\e[1;31m %d\e[0m\n",index1+1);
printf("By Min Total of Swaps In Sorting Function: Algorithm\e[1;31m %d\e[0m\n",index3+1);
printf("By Min Total of Compares In Sorting Function: Algorithm\e[1;31m %d\e[0m\n",index4+1);
printf("By Min Total of Compares in BinarySearch: Algorithm\e[1;31m %d\e[0m\n\n",index2+1);

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
void spacing(int lines){
    for(int i=0;i<lines;i++)
    printf("\n");
}
void print(Movies *array,int index,int size){
while(strncmp(array->movies[index].title,array->movies[index+1].title,size)==0)
{

    for(int i=0;i<strlen(array->movies[index].title);i++){
    if(i<size)
    printf("\e[1;35m%c",array->movies[index].title[i]);
else
    printf("\e[0m%c",array->movies[index].title[i]);
    }printf("\n");index++;
}

for(int i=0;i<strlen(array->movies[index].title);i++){
    if(i<size)
    printf("\e[1;35m%c",array->movies[index].title[i]);
else
    printf("\e[0m%c",array->movies[index].title[i]);
    }printf("\n");
}









int binSearchimpl(Movies *array,char *titleToFind,int size,int *nrofcomp){
//if(truestate(array,titleToFind))return -1;

int mij,st=0,dr=array->size-1,cmp,nrofcompInit=0;
while(st<=dr){
    mij=(st+dr)/2;
    cmp=strncmp(array->movies[mij].title,titleToFind,size);nrofcompInit++;
if(cmp==0){
*nrofcomp=nrofcompInit;
return mij;
}
if(cmp>0)
dr=mij-1;
else
st=mij+1;
}
*nrofcomp=nrofcompInit;
return -1;
}





//sorteaza in fucntie de prefixul primit de la tastaura se sorteaza(prin diferite metode) si mai apoi se cauta prin met binara prefixul
void binSearch(Movies *array){struct winsize window;ioctl(0, TIOCGWINSZ , &window);
printf("Toggle Enter to exit \e[1;35msearch mode \e[0mmother char may not exit clearly");char c;scanf("%c",&c);
spacing(window.ws_row);



char ch,*string=NULL;
int size;
Movies *auxArray=NULL;
Performance perf[numberofSortMethods];
clock_t time;

for(int k=0;k<numberofSortMethods;k++)//6 metode de sortare la fiecare masurat nr de comp si interschimbari la sortare si nr de comparatii la cautare si timpul total pentru ambele scurs
{

    make_copy(&auxArray,array);//se face copie la database


if(k==0){//===========================>BinaryIns
printf("Binary SEARCH \nToggle Enter to exit \e[1;35msearch mode \e[0mother char may not exit clearly");char c;scanf("%c",&c);
spacing(window.ws_row);

ch='0';size=0;
int compSort,swapsSort,compSearch;
compSort=swapsSort=compSearch=perf[k].compSort=perf[k].compSearch=perf[k].swapsSort=perf[k].totalRunTime=0;

binaryIns(auxArray,string,0,&compSort,&swapsSort);//sortare normala -->aici schimbi

perf[k].compSort+=compSort;
perf[k].swapsSort+=swapsSort;

/*testare daca sorteaza
for(int i=0;i<array->size;i++)
puts(array->movies[i].title);
scanf("%c",&c);
*/

while(ch!='\n'){
ch=getHiddenChar();

spacing(window.ws_row);
spacing(window.ws_row);


if(ch!='\n'){
time=clock();

string=realloc(string,size+2);
string[size]=ch;
string[size+1]='\0';
size++;printf("CURRENT SUBSTRING// \e[1;35m");
puts(string);printf("\e[0m");

binaryIns(auxArray,string,1,&compSort,&swapsSort);//sortare cu prefixul dat-->aici schimbi 
perf[k].compSort+=compSort;
perf[k].swapsSort+=swapsSort;

int findIndex=binSearchimpl(auxArray,string,size,&compSearch);//aici schimbi daca vrei metoda de cautare
perf[k].compSearch+=compSearch;
if(findIndex==-1){printf("key not found...");}
else
print(auxArray,findIndex,size);

perf[k].totalRunTime+=((double)(clock()-time))/CLOCKS_PER_SEC;
}

}
free(string);
string=NULL;
}
else
if(k==1){//======================================>Selection
printf("Selection SEARCH \nToggle Enter to exit \e[1;35msearch mode \e[0mother char may not exit clearly");char c;scanf("%c",&c);
spacing(window.ws_row);


ch='0';size=0;
int compSort,swapsSort,compSearch;
compSort=swapsSort=compSearch=perf[k].compSort=perf[k].compSearch=perf[k].swapsSort=perf[k].totalRunTime=0;

Selection(auxArray,string,0,&compSort,&swapsSort);//sortare normala -->aici schimbi

perf[k].compSort+=compSort;
perf[k].swapsSort+=swapsSort;

/*testare daca sorteaza
for(int i=0;i<array->size;i++)
puts(array->movies[i].title);
scanf("%c",&c);
*/

while(ch!='\n'){
ch=getHiddenChar();

spacing(window.ws_row);
spacing(window.ws_row);


if(ch!='\n'){
time=clock();

string=realloc(string,size+2);
string[size]=ch;
string[size+1]='\0';
size++;printf("CURRENT SUBSTRING// \e[1;35m");
puts(string);printf("\e[0m");

Selection(auxArray,string,1,&compSort,&swapsSort);//sortare cu prefixul dat-->aici schimbi 
perf[k].compSort+=compSort;
perf[k].swapsSort+=swapsSort;

int findIndex=binSearchimpl(auxArray,string,size,&compSearch);//aici schimbi daca vrei metoda de cautare
perf[k].compSearch+=compSearch;
if(findIndex==-1){printf("key not found...");}
else
print(auxArray,findIndex,size);

perf[k].totalRunTime+=((double)(clock()-time))/CLOCKS_PER_SEC;
}
}
free(string);
string=NULL;
}
else
if(k==2){//======================================>Bubblesort
printf("BubbleSort SEARCH \nToggle Enter to exit \e[1;35msearch mode \e[0mother char may not exit clearly");char c;scanf("%c",&c);
spacing(window.ws_row);


ch='0';size=0;
int compSort,swapsSort,compSearch;
compSort=swapsSort=compSearch=perf[k].compSort=perf[k].compSearch=perf[k].swapsSort=perf[k].totalRunTime=0;

BubbleSort(auxArray,string,0,&compSort,&swapsSort);//sortare normala -->aici schimbi

perf[k].compSort+=compSort;
perf[k].swapsSort+=swapsSort;

/*testare daca sorteaza
for(int i=0;i<array->size;i++)
puts(array->movies[i].title);
scanf("%c",&c);
*/

while(ch!='\n'){
ch=getHiddenChar();

spacing(window.ws_row);
spacing(window.ws_row);


if(ch!='\n'){
time=clock();

string=realloc(string,size+2);
string[size]=ch;
string[size+1]='\0';
size++;printf("CURRENT SUBSTRING// \e[1;35m");
puts(string);printf("\e[0m");

BubbleSort(auxArray,string,1,&compSort,&swapsSort);//sortare cu prefixul dat-->aici schimbi 
perf[k].compSort+=compSort;
perf[k].swapsSort+=swapsSort;

int findIndex=binSearchimpl(auxArray,string,size,&compSearch);//aici schimbi daca vrei metoda de cautare
perf[k].compSearch+=compSearch;
if(findIndex==-1){printf("key not found...");}
else
print(auxArray,findIndex,size);

perf[k].totalRunTime+=((double)(clock()-time))/CLOCKS_PER_SEC;
}
}
free(string);
string=NULL;
}
else
if(k==3){//======================================>ShakerSort
printf("ShakerSort SEARCH \nToggle Enter to exit \e[1;35msearch mode \e[0mother char may not exit clearly");char c;scanf("%c",&c);
spacing(window.ws_row);


ch='0';size=0;
int compSort,swapsSort,compSearch;
compSort=swapsSort=compSearch=perf[k].compSort=perf[k].compSearch=perf[k].swapsSort=perf[k].totalRunTime=0;

ShakerSort(auxArray,string,0,&compSort,&swapsSort);//sortare normala -->aici schimbi

perf[k].compSort+=compSort;
perf[k].swapsSort+=swapsSort;

/*
//testare daca sorteaza
for(int i=0;i<auxArray->size;i++)
puts(auxArray->movies[i].title);
scanf("%c",&c);
*/

while(ch!='\n'){
ch=getHiddenChar();

spacing(window.ws_row);
spacing(window.ws_row);


if(ch!='\n'){
time=clock();

string=realloc(string,size+2);
string[size]=ch;
string[size+1]='\0';
size++;printf("CURRENT SUBSTRING// \e[1;35m");
puts(string);printf("\e[0m");

ShakerSort(auxArray,string,1,&compSort,&swapsSort);//sortare cu prefixul dat-->aici schimbi 
perf[k].compSort+=compSort;
perf[k].swapsSort+=swapsSort;

int findIndex=binSearchimpl(auxArray,string,size,&compSearch);//aici schimbi daca vrei metoda de cautare
perf[k].compSearch+=compSearch;
if(findIndex==-1){printf("key not found...");}
else
print(auxArray,findIndex,size);

perf[k].totalRunTime+=((double)(clock()-time))/CLOCKS_PER_SEC;
}
}
free(string);
string=NULL;
}
else
if(k==4){//======================================>ShellSort
printf("ShellSort with KnuthKolatz PredeterminedSEQ SEARCH \nToggle Enter to exit \e[1;35msearch mode \e[0mother char may not exit clearly");char c;scanf("%c",&c);
spacing(window.ws_row);


ch='0';size=0;
int compSort,swapsSort,compSearch;
compSort=swapsSort=compSearch=perf[k].compSort=perf[k].compSearch=perf[k].swapsSort=perf[k].totalRunTime=0;

ShellSort(auxArray,string,0,&compSort,&swapsSort);//sortare normala -->aici schimbi

perf[k].compSort+=compSort;
perf[k].swapsSort+=swapsSort;

/*
//testare daca sorteaza
for(int i=0;i<auxArray->size;i++)
puts(auxArray->movies[i].title);
scanf("%c",&c);
*/

while(ch!='\n'){
ch=getHiddenChar();

spacing(window.ws_row);
spacing(window.ws_row);


if(ch!='\n'){
time=clock();

string=realloc(string,size+2);
string[size]=ch;
string[size+1]='\0';
size++;printf("CURRENT SUBSTRING// \e[1;35m");
puts(string);printf("\e[0m");

ShellSort(auxArray,string,1,&compSort,&swapsSort);//sortare cu prefixul dat-->aici schimbi 
perf[k].compSort+=compSort;
perf[k].swapsSort+=swapsSort;

int findIndex=binSearchimpl(auxArray,string,size,&compSearch);//aici schimbi daca vrei metoda de cautare
perf[k].compSearch+=compSearch;
if(findIndex==-1){printf("key not found...");}
else
print(auxArray,findIndex,size);

perf[k].totalRunTime+=((double)(clock()-time))/CLOCKS_PER_SEC;
}
}
free(string);
string=NULL;
}
else
if(k==5){//======================================>QuickSort
printf("QuickSort with MedianStart SEARCH \nToggle Enter to exit \e[1;35msearch mode \e[0mother char may not exit clearly");char c;scanf("%c",&c);
spacing(window.ws_row);


ch='0';size=0;
int compSort,swapsSort,compSearch;
compSort=swapsSort=compSearch=perf[k].compSort=perf[k].compSearch=perf[k].swapsSort=perf[k].totalRunTime=0;

QuickSort(auxArray,string,0,&compSort,&swapsSort);//sortare normala -->aici schimbi

perf[k].compSort+=compSort;
perf[k].swapsSort+=swapsSort;

/*
//testare daca sorteaza
for(int i=0;i<auxArray->size;i++)
puts(auxArray->movies[i].title);
scanf("%c",&c);
*/

while(ch!='\n'){
ch=getHiddenChar();

spacing(window.ws_row);
spacing(window.ws_row);


if(ch!='\n'){
time=clock();

string=realloc(string,size+2);
string[size]=ch;
string[size+1]='\0';
size++;printf("CURRENT SUBSTRING// \e[1;35m");
puts(string);printf("\e[0m");

QuickSort(auxArray,string,1,&compSort,&swapsSort);//sortare cu prefixul dat-->aici schimbi 
perf[k].compSort+=compSort;
perf[k].swapsSort+=swapsSort;

int findIndex=binSearchimpl(auxArray,string,size,&compSearch);//aici schimbi daca vrei metoda de cautare
perf[k].compSearch+=compSearch;
if(findIndex==-1){printf("key not found...");}
else
print(auxArray,findIndex,size);

perf[k].totalRunTime+=((double)(clock()-time))/CLOCKS_PER_SEC;
}
}
free(string);
string=NULL;
}
}
PerformanceDisplay(perf);



free_up(&auxArray);
}
