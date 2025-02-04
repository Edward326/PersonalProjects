#ifndef GAMEINTERFACE
#define GAMEINTERFACE

void StartGame();


/*
o fct principala din care se va intr directia
in aceasta fct:

->dam la intmamplare la rand() sec mancare pe care o pune la rand()pozitie in matrice 
apelam o alta fct care in int acesteia apelam o alta fct care afla pozitiile sarpelui(si deplaseaza)

daca nu e sfarsit de joc 
vedem daca sarpele nu a mancat din mancare geanraat,daca da se elimina din vectorul de mancare poztitia resp
apelam o fct care incarca pozitiile in matrice(plus pozitiile din vectorul de mancaruri daca exista) si apelam o alta fct 
care afiseaza matricea cu tot cu interfata jocului
se revine in fct principala unde se reactualizeaza scorul si se asteapta sleep(between_moves_delay);

daca da e sfarsit de joc
se iese mai apoi din fct si se duce in fct princip unde se reactualizeaza scorul
se vede daca e mai mare decat scorul din jocul anterior(salvat intr un fisier) si se stabileste HishScore-ul
afisam interfata de iesire si HighScore-ul stabilit
asteptam un input dupa care inchidem fereastra
*/


#endif