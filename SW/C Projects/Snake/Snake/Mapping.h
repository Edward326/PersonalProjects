#ifndef MAPPING
#define MAPPING
#include"snakestructure.h"
#include"WindowInterface.h"

void ZeroMapping(int **map,int downlimit,int rightlimit);
void MapMoves(int **map,Snake *snake,foodblock* food,int foodsize,int downlimit,int rightlimit);
void displayMap(int **map,char Ssymbol,char Fsymbol,int downlimit,int rightlimit);
void MakeMatrix(int ***map,Window *window);

#endif