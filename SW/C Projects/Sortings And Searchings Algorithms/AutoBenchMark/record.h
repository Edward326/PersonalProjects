#ifndef RECORD
#define RECORD
#include"structure.h"
void record(Movies **array,char *file);
void make_copy(Movies **new_array,Movies *init_array);
void free_up(Movies **array);//retine ** cand dimensiunea e deja specificata pe cand un * nu poate retine o structura care si modifica dimensiunea

#endif