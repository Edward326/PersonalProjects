#ifndef STRUCTURE
#define STRUCTURE
typedef struct elemS{
  int nr;
  struct elemS *next;
  struct elemS *nextsorted;
}elemS;

typedef struct{
  elemS *first,*last;
  elemS *firstsorted;
}simple_list;
#endif
