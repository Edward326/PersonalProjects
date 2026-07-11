instrcutiuni de folosire


Unitate ALU(32b)

Cuprinde operatii de adunare ,inmultire,op logice ,adunare diferenta


totul se ruleza din fisierul "centralUnit.v" unde se configureaza:
{
    in modulul alu_tb se modifica nr dorite pentru operatie X,Y
    iar dupa se alege operatia dorita punand la op=5'd(nrOp)
    nrOp:
    0-N
    1-Sum
    2-Differnce 
    3-Multiply
    4-Divide 
    5-shiftXtoLeft
    6-shiftXtoRight 
    7-shiftYtoLeft 
    8-shiftYtoRight 
    9-makeAndOp
    10-makeOrOp 
    11-makeXorOp
}

ATENTIE:
la op de inmultire si impartire sunt necesare un nr de cicluri de tact(se conf la intial begin la cycles=NC si dupa se schimba #wait-ul la formula NC*2*run_cycleTime+run_cycleTime)
DACA NU SE DAU UN NR SUFICENT DE CICLURI DE TACT PT COMPUTAREA CORECTA A REZULTATULUI LA IESIRE SE VA RETINE REZ-0(PT A EVITA CORUPTIA)
~pt inmutlire:avg cc ∈ [20,27]
~pt impartire:fixed cc=165 pt ∀ ar fii X,Y ∈ [-2147483647,+2147483647]
~pt adunare/scadere ,opLogice sunt necesare: cc=1

DACA SE DAU UN NR SUFICEINT SAU CHIAR MAI MULT DECAT SUFICENT DE CICLURI DE TACT,
LA REZULTAT SE VA RETINE EXACT REZULTATUL REAL FARA SA SE CORUPA(DE LA PREA MULTE CICLURI,dotat cu senzori de oprire a ciclurilor) 



A NU SE FOLOSII NR PT OPERATII MAI MARI DECAT 2^30 
INTERVALUL DE FUNCTIONARE, X,Y ∈ [-2147483647,+2147483647]


CREDITS:
Mucioniu Andrei Constantin
Sperneac Catalin
Vesea Eduard

                                                                       !!!!!!A NU SE MAI MODIFF NIMIC!!!!!!
