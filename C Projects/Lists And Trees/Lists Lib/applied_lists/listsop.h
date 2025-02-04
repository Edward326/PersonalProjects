#ifndef LISTSOP
#define LISTSOP
#include"structure.h"

//int verifysorted(simple_list *a); //nu trebuie sa fie vizibila
int sortsimple(simple_list *lista,int *n_access);//sorteaza o lista pe care o primeste ca parametru,fara a folosii var externe//sortare insituu-->Bubblesort
void findnr(simple_list *lista,int ntofind);
#endif
