#ifndef LIST_STRUCTURE
#define LIST_STRUCTURE
//header care poate fii (inclus)folosit in cazul in care avem mai multe libraries care folosesc structura definita mai jos

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
#endif
