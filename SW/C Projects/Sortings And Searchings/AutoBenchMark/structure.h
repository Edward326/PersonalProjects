#ifndef STRUCTURE
#define STRUCTURE

typedef struct{
int year,length;
char *title;
/*
char *type;
int awards;
*/
}Movie;

typedef struct{
Movie *movies;
int size;
}Movies;
#endif