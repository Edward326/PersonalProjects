DOCUMENTATIE
Algoritm:

JOCUL VA RULA PENTRU O PARTIDA(pana fiecare jucator isi va pune pe masa toate cele 6 carti)!!
JOCUL NORMAL RULEAZA PANA CAND UN JUCATOR ARE UN ANUMIT NR DE PUNCTE MAXIMAL STABILIT LA INCEPUTUL JOCULUI(pucntele se aduna de la fiecare runda) , PUNCTE MAXIMALE:6,11,15,21
tura= aplicarea unei metode(care indeplineste o anumita functie) pe fiecare jucator in sensul anti_clockwise

Amesteacarea cartilor:
se alege un jucator random care sa imparta cartile(deoarece nu avem mai multe partide,nu avem un jucator castigator care sa inceapa cu el partida)
jucatorul de langa el va decide daca, cartile sunt reorganizate prin taiere sau bataie
in functie de modul ales se impart cartile
se face urmatorea tura licitatia(in urma careia se va decide cine va conduce licitatia)
cel care castiga licitatia va pune prima carte jos , ea ,fiind culoarea tromfului

se vor incepe mai apoi turele de joc (prima oara de la jucatorul din dreapta celui cel care a condus licitatia,ulterior se incepe de catre castigatorul turei trecute)
fiecarui jucator i se da posibiliatea de a anunta daca va face 20/40 runda viitoare(anuntul e VALID,adica se adauga anuntul VALID de 20/40 la scorul final, doar daca jucatorul 
respectiv castiga runda curenta si prima carte data jos e o carte din perechea de 20/40 anuntata)

POSIBILITATEA DE A ALEGE ANUNT DE 20/40 SE FACE DOAR CAND JOCUL NU E IN ULTIMA FAZA(RUNDA6) SI JUCATORUL E: 
1.CASTIGATORUL RUNDEI TRECUTE SI NU A FACUT ANUNT RUNDA TRECUTA  2.ORICE JUCATOR INAFARA DE CEL CASTIGATOR

se alege cartea de dat jos din deck-ul de carti al jucatorului
iar dupa ce s a parcurs o tura se vede cine va conduce cartile(castigatorul turei),si aceluia ii vor fii date cartile de pe masa in campul wonCards(al Player-ului castigator al turei)
ACESTE TURE SE VOR REPETA DE 6-ORI(sunt 24 de carti in totalitate,exista 4 jucatori,fiecare cu 6 carti,pe tura se dau 4 carti 24/4=6 ture)

totalPoints->la final se strang punctele pe fiecare jucator(calcularea punctelor depind si de de daca jucatorul e castigatorul licitatiei sau nu(doar el va fii penalizat daca nu acumuleaza punctele declarate in timpul licitatiei)) adunand punctele de pe toate cartile din campul wonCards al fiecarui jucator,ulterior se mai adauga si puncctele de 20 si 40 de pe ANUNTURILE VALIDE
noPoints->se calculeaza noPoints pe fiecare jucator(this.totalPoints/33)

se aduna noPoints pe jucatorul[0]&jucatorul[1] si jucatorul[2]&jucatorul[3], si se va returna echipa ce are punctajul cel mai mare
