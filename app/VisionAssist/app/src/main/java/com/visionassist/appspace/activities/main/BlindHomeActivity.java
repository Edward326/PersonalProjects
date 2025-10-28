package com.visionassist.appspace.activities.main;

//cand se apasa volum jos,sus caption/deetion
//de doua ori apasat volumul jos, voice to speech pentrua  cauta obiectul
//detection sau caption verifica, peromsiunile la camera, dupa checkPhoneStatus,
/*
 facut metoda prin care:
                -se separa fieacrer word din setcene de la voice to text model, si se elimina cuvintele inutile(where is, etc),
                -pt cuvintele ramase se cauta sinnomie din tabelul cu sinonime, pentru cele care se gasesc se incluisesc cu idul clasei yolo in secv finala, daca nu se gaseste nu se pune in secv finala
                -se ia secv finala si se de la yolo, iar cand gaseste un obiect din secv finala,

            *facut activitaeta pentru vazatorri, este live, cand obtine clasele, daca se gasesc obiectele din secv finala(se elimina din lista), se opreste activtatea, si afiseaza pe ecran imaginea cu ibiectele cu bboxuri,
            de aici daca mai sunt nuy mai sunt elem ramse in secv finala afiseaz doar butonul de home, dar daca mai sunt afiseaza butonul de home si next(repeta aceaasi activiate pana cand se gasesc sau parasete userul prin home)
            (in live butonul de volum daca este apasa cel de jos iese din activtate)

            *facut activitatea pentru nevazatori, este live, cand obtine clasele, daca se gasesc obiectele din secv finala(se elimina din lista), se opreste activtatea, se pune imaginea care in care s-au gasit si face speech cu obiectele gasite,
             de aici daca mai sunt nuy mai sunt elem ramse in secv finala afiseaz doar butonul de home, dar daca mai sunt afiseaza butonul de home si next(repeta aceaasi activiate pana cand se gasesc sau parasete userul prin home)
            (in live butonul de volum daca este apasa cel de jos iese din activtate)
 */
public class BlindHomeActivity {

}