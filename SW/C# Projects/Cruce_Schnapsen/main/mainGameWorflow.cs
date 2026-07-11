using System;
using CruceEduardVesea.entities;
using CruceEduardVesea.ui_interface;
using System.Collections.Generic;
using System.Threading;
using System.Linq;


namespace CruceEduardVesea.main{

    internal class Game{
    //intternal pentru a fii accesibili doar din spatiul main,readonly deoarece trebuie intiialoizati can se creeza o instanta
    //get;init,o fac cu propriteati, doearece ,nu trebuie sa setez o val din start obiectului(ex null care nu m ar lasa sa o declar)
    //init deaorece e readonly si nu poate sa mai fie setata dupa ce e setat in constructor
    private readonly List<Player> players;
    public readonly Interface interfaceScreen;
    private List<Card> cardsOnTable=new List<Card>();
    public Color tromf{get;private set;}
    private static Random random=new Random();
    private static bool cheatMode=false;
    
    public Game(Interface interfaceScreenReff,List<Player> playersReff)=> (interfaceScreen,players)=(interfaceScreenReff,playersReff);
    public Game(){
        this.players=new List<Player>();
        this.interfaceScreen=new Interface(5);
    }
    public static Game initialize(){
        Interface interfaceScreenReff=new Interface();
        List<Player> playersReff=interfaceScreenReff.showInitPlayers();
        return new Game(interfaceScreenReff,playersReff);
    }
    private List<Card> makeCards(){
        List<Card> cardsToPlay=new List<Card>();
        Color[] arrayColor={Color.rosu,Color.verde,Color.duba,Color.ghinda};
        for(int i=0;i<4;i++){
            cardsToPlay.Add(new Card(2,arrayColor[i]));cardsToPlay.Add(new Card(3,arrayColor[i]));
            cardsToPlay.Add(new Card(4,arrayColor[i]));cardsToPlay.Add(new Card(0,arrayColor[i]));
            cardsToPlay.Add(new Card(10,arrayColor[i]));cardsToPlay.Add(new Card(11,arrayColor[i]));
        }
        return cardsToPlay;
    }
    private List<Card> shuffle(List<Card> cardsToShuffle){
        if(cheatMode)return cardsToShuffle;
     //knuth shuffle(fisher yates algorithm)
     //aplicat de doua ori parcugrand sirul de la dr la st si de la st la dr
     List<Card> shuffledCards=new List<Card>();
     int[] randArray = new int[24];
     for (int i = 0; i < 24; i++)randArray[i] = i;

      for (int i = randArray.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            int temp = randArray[i];
            randArray[i] =randArray[j];
            randArray[j] = temp;
        }
         for (int i = 0; i <randArray.Length; i++)
        {
            int j = i+random.Next(randArray.Length-i);
            int temp = randArray[i];
            randArray[i] =randArray[j];
            randArray[j] = temp;
        }

     foreach(int reffRand in randArray)shuffledCards.Add(cardsToShuffle[reffRand]);
     
     return shuffledCards;
    }
    private void cutCardsShuffle(List<Card> cardsToShuffle){
        int lengthA=cardsOnTable.Count/2;
        for(int i=0;i<cardsToShuffle.Count/2;i++)cardsToShuffle.Add(cardsToShuffle[i]);
        cardsToShuffle.RemoveRange(0,lengthA);
    }
    private void giveCards(Player[] arrayPlayers,List<Card> cardsToShuffle){
    for(int i=0;i<6 && !cheatMode;i++){
     arrayPlayers[0].addCards(cardsToShuffle[i*4]);arrayPlayers[1].addCards(cardsToShuffle[i*4+1]);
     arrayPlayers[2].addCards(cardsToShuffle[i*4+2]);arrayPlayers[3].addCards(cardsToShuffle[i*4+3]);
     cardsToShuffle[i*4].possesor=arrayPlayers[0];cardsToShuffle[i*4+1].possesor=arrayPlayers[1];
     cardsToShuffle[i*4+2].possesor=arrayPlayers[2];cardsToShuffle[i*4+3].possesor=arrayPlayers[3];
    }//se dau cartile in clockwise si cel care imparte(shuffler primeste ultimuls)
     
     if(cheatMode){
       arrayPlayers[0].addCards(cardsToShuffle.GetRange(0,6));
    arrayPlayers[1].addCards(cardsToShuffle.GetRange(6,6));
    arrayPlayers[2].addCards(cardsToShuffle.GetRange(12,6));
    arrayPlayers[3].addCards(cardsToShuffle.GetRange(18,6));
    for(int i=0;i<24;i+=6){
        cardsToShuffle[i].possesor=arrayPlayers[i/6];cardsToShuffle[i+1].possesor=arrayPlayers[i/6];
        cardsToShuffle[i+2].possesor=arrayPlayers[i/6];cardsToShuffle[i+3].possesor=arrayPlayers[i/6];
        cardsToShuffle[i+4].possesor=arrayPlayers[i/6];cardsToShuffle[i+5].possesor=arrayPlayers[i/6];
    }
     }
    }
    private Player[] makeArrayPlayers(params Player[] reff){
        Player[] array=new Player[reff.Length];
        for(int i=0;i<reff.Length;i++)array[i]=reff[i];
        return array;
    }
    private Player setRoundWinner(){
        List<Card> tromfi=new List<Card>();
        foreach(Card reff in cardsOnTable)
        if(reff.color.Equals(tromf))tromfi.Add(reff);
        
        Player? winner=null;int max=0;
        if(tromfi.Count>0){//avem tromfi
            for(int i=0;i<tromfi.Count;i++)
            if(winner==null){max=tromfi[i].value;winner=tromfi[i].possesor;}
            else
            if(tromfi[i].value>max){max=tromfi[i].value;winner=tromfi[i].possesor;}
        }
        else{
            winner=cardsOnTable[0].possesor;Color firstColor=cardsOnTable[0].color;int maxValue=cardsOnTable[0].value;
            for(int i=1;i<cardsOnTable.Count;i++)
            if(cardsOnTable[i].color.Equals(firstColor) && cardsOnTable[i].value>maxValue){
                maxValue=cardsOnTable[i].value;winner=cardsOnTable[i].possesor;
            }
        }
        return (Player)winner;
    }
    public List<Player> playGame(ref int Team){
        //impartirea cartilor
        int shufflerIndex=random.Next(4);
        int chooserIndex=(shufflerIndex==3)?0:shufflerIndex+1;//counter_clockwise
        Player shuffler=players[shufflerIndex],chooser=players[chooserIndex];
        List<Card> cardstoShuffle=makeCards();//creearea cardurlilorde shuffleiuit (24 de carti in total),static
        cardstoShuffle=shuffle(cardstoShuffle);//shuffleuiirea intiiala,static
        int opt=interfaceScreen.shuffleScreen(chooser);
        if(opt==1)//taiere
        cutCardsShuffle(cardstoShuffle);//static
        int fIndex=(shufflerIndex==0)?3:shufflerIndex-1;int sIndex=(fIndex==0)?3:fIndex-1;
        int tIndex=(sIndex==0)?3:sIndex-1;//clockwise
        Player[] arrayPlayers=makeArrayPlayers(players[fIndex],players[sIndex],players[tIndex],players[shufflerIndex]);
        giveCards(arrayPlayers,cardstoShuffle);//static

        //licitatia
        fIndex=(shufflerIndex==3)?0:shufflerIndex+1;sIndex=(fIndex==3)?0:fIndex+1;
        tIndex=(sIndex==3)?0:sIndex+1;//counter_clockwise
        arrayPlayers=makeArrayPlayers(players[fIndex],players[sIndex],players[tIndex],players[shufflerIndex]);
        int maxPoints=0;Player? roundStarterF=null;//round starter va fii aici nullable deoarece nu primeste o val din start
        for(int i=0;i<arrayPlayers.Length;i++)arrayPlayers[i].licitate(interfaceScreen,ref maxPoints,ref roundStarterF);
        Player roundStarter=(Player)roundStarterF;
        roundStarter.setLicitor();
        
        //stablilirea tromfului
        int k;
        for(k=0;k<arrayPlayers.Length;k++)if(players[k].Equals(roundStarter))break;
        shufflerIndex=k;
        //Console.WriteLine(shufflerIndex);Thread.Sleep(5000);
        fIndex=(shufflerIndex==3)?0:shufflerIndex+1;sIndex=(fIndex==3)?0:fIndex+1;
        tIndex=(sIndex==3)?0:sIndex+1;//counter clockwise
        arrayPlayers=makeArrayPlayers(players[shufflerIndex],players[fIndex],players[sIndex],players[tIndex]);
        Card? firstThrownCard=null;
        tromf=roundStarter.setTromf(interfaceScreen,ref firstThrownCard);//se alege tromful
        cardsOnTable.Add((Card)firstThrownCard);
                
        //turele propriu zise;
        //prima tura(1 din 6) ,incepand din dreapta castigatorului licitatiei
        //round winner nu va fii niciodata null asa ca nu e necesar sa fie nullable(?)
        for(int i=1;i<arrayPlayers.Length;i++)cardsOnTable.Add(arrayPlayers[i].withdrawCard(interfaceScreen,cardsOnTable,tromf,1));
        Player roundWinner=this.setRoundWinner();roundWinner.addWonCards(cardsOnTable);cardsOnTable.Clear();
        interfaceScreen.showWinnerRound(roundWinner,1);

        //urmatoarele (6-1) ture in care se reogranizeza ordinea jucatorilor cand dau o carte(incepand cu castigatorul rundei trecute in sens counter_clockwise)
        for(int j=0;j<5;j++){
        for(k=0;k<arrayPlayers.Length;k++)if(players[k].Equals(roundWinner))break;
        shufflerIndex=k;
        fIndex=(shufflerIndex==3)?0:shufflerIndex+1;sIndex=(fIndex==3)?0:fIndex+1;
        tIndex=(sIndex==3)?0:sIndex+1;//counter clockwise
        
        players[fIndex].twentyAnnounce=players[sIndex].twentyAnnounce=players[tIndex].twentyAnnounce=false;//resetam anunturile jucatorilor necastigatori
        players[fIndex].fourtyAnnounce=players[sIndex].fourtyAnnounce=players[tIndex].fourtyAnnounce=false;//resetam anunturile jucatorilor necastigatori

        arrayPlayers=makeArrayPlayers(players[shufflerIndex],players[fIndex],players[sIndex],players[tIndex]);
        for(int i=0;i<arrayPlayers.Length;i++)cardsOnTable.Add(arrayPlayers[i].withdrawCard(interfaceScreen,cardsOnTable,tromf,j+2));
        roundWinner=this.setRoundWinner();roundWinner.addWonCards(cardsOnTable);cardsOnTable.Clear();
        interfaceScreen.showWinnerRound(roundWinner,j+2);
        }

        //calcularea punctelor finale
        for(int i=0;i<players.Count;i++)players[i].calculatePoints();
        int sumPointsFirstTeam=players[0].noPoints+players[1].noPoints;
        int sumPointsSecondTeam=players[2].noPoints+players[3].noPoints;
        List<Player> winnerTeam=(sumPointsFirstTeam>sumPointsSecondTeam)?players.GetRange(0,2):players.GetRange(2,2);
        if(sumPointsFirstTeam>sumPointsSecondTeam){
        Team=1;
        winnerTeam.AddRange(players.GetRange(2,2));
        }
        else{
        Team=2;
        winnerTeam.AddRange(players.GetRange(0,2));
        }

        //castigaorii sunt primii doi din lista, iar pierzatorii sunt ultimii doi din lista winnerTeam
        return winnerTeam;
    }
}
}