#ifndef LISTS_H
#define LISTS_H

typedef struct elemS{
  int nr;
  struct elemS *next;
}elemS;

typedef struct{
  elemS *first,*last;
}simple_list;

typedef struct elemD{
  int nr;
  struct elemD *next,*back;
}elemD;

typedef struct{
  elemD *first,*last;
}double_list;


void help();


void viewsimple(simple_list *a);


void viewdouble(double_list *a);


elemS *new_elemS(int nr);


elemD *new_elemD(int nr);


simple_list *initsimple();


double_list *initdouble();


void free_upsimple(simple_list *a);


void free_updouble(double_list *a);


void addsimple(simple_list *a,int number);


void add_sortedsimple(simple_list *a,int nr);


void deletesimple(simple_list *a,int nr);


void deletesimple_sorted(simple_list *a,int nr);


void adddouble(double_list *a,int nr);


void adddouble_sorted(double_list *a,int nr);


void deletedouble(double_list *a,int nr);

#endif
