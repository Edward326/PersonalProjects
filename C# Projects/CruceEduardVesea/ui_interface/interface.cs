using System;
using System.Threading;
using System.Linq;
using CruceEduardVesea.entities;
using System.Collections.Generic;


namespace CruceEduardVesea.ui_interface{
    
    internal struct paramsScreen{
        public static int initTime=5000,startTime=3000;
        public static int FullScreenWidth,FullScreenHeight;
        public static bool initFullSCSetting=true;//active screen monitoring system
        public static string WelcomeText=paramsScreen.getWelcome();
        public static string SettingsText=paramsScreen.getSettings();
        
        public static string getWelcome(){return "Joc de Cruce in 4 jucatori\n\n";}
        public static string getSettings(){
              return "Reguli:\n" +
              "Jocul consta in asistarea celor 6 ture de joc\n" +
              "Mai intai se va decide aleatoriu jucatorul care va impartii cartile\n" +
              "Jucatorul din dreapta celui ales va alege: pastrarea modului de amestecare(Bataie) / re-stackurirea lor(Taiere)\n" +
              "Se impart cartile de la jucatorul din stanga celui care a impartit cartile\n" +
              "Pornind de la jucatorul din dreapta celui care a impartit cartile(cel care a ales bataia/taierea cartilor), se face licitatia\n" +
              "Jucatorul care a anuntat ca va strange cele mai multe puncte va fii ales ca castigatorul licitatiei\n\nPuncte disponibile:\n" +
              pointsValue.showPoints() + 
              "\n\nCastigatorul licitatiei anunta culoarea tromfului prin punerea pe masa a unei carti(prima carte)\n" +
              "Turele de joc vor incepe si vor consta astfel:\n" +
              "Se incepe de la castigatorul licitatiei(prima runda) si dupa cu cel care a castigat runda trecuta(urmatoarele runde)\n" +
              "Jucatorul curent trebuie sa dea o carte de acceasi culoare ca prima carte/culoarea tromfului/orice carte\n" +
              "Castigatorul turei va lua cartile de pe masa si va fii cel care are are cea mai mare carte de culoarea tromfului sau a primei carti date\n" +
              "La finalul jocului se aduna pentru fiecare jucator punctele cartilor luate de pe masa\n" +
              "Se decide echipa castigatoare prin adunarea punctelor coechipierilor\n"+
               "\x1b[1;31mJocul dispune de detectare automata a cartilor pentru anunt de 20/40 si cartile posibile pentru a le pune pe masa(cartea care e detectata de sistem e evidentiata sub cu ^^^^)\x1b[0m\n"+
              "\nCartile disponibile:\n"+
              Card.DisplayReffCards();
        }
    }

    public class Interface{
    private int screenWidth{get{return Console.WindowWidth;}}
    private int screenHeight{get{return Console.WindowHeight;}}
    
    private void spacing(string text){//afisare pe mijloc
      int maxCount=screenWidth/2-text.Length/2;
      for(int i=0;i<maxCount;i++)Console.Write(" ");
      Console.WriteLine($"\x1b[1;3;4;37m{text}\x1b[0m\n\n");
    }
    private void pulse(string text){
            //ansii C char format encoding 
            for(int i=232;i<=255;i++){
            Console.Clear();
            //Console.WriteLine($"FULLSCREEN({paramsScreen.FullScreenHeight}x{paramsScreen.FullScreenWidth})");
            Console.WriteLine($"\x1b[1;3m\x1b[38;5;{i}m{text}\x1b[0m");//233->255 (black->white)
            Thread.Sleep(50); 
            }
            Thread.Sleep(700);
    }
    public void verifyMaxScreen(){
        int rstTime=5000;
        string text="Please put your console into FULLSCREEN ,in order to continue the game...";
        string text2="Fullscreen detected, continuing the game...";
        if(paramsScreen.initFullSCSetting){
            int opt=0;
            while(Console.WindowHeight!=paramsScreen.FullScreenHeight && Console.WindowWidth!=paramsScreen.FullScreenWidth){
            pulse(text);opt=1;
            }
            if(opt==1){
            Console.Clear();
            Console.WriteLine($"\x1b[1m{text2}\x1b[0m");Thread.Sleep(rstTime);
            }
            Console.Clear();
        }
        else Console.Clear();
    }
    public static void getSizesFC(){
      Console.Clear();
      string curWinSize=$"\x1b[1;3;4mCurrent height x width: {Console.WindowHeight} x {Console.WindowWidth}\x1b[0m\n";
      Console.WriteLine(curWinSize);
      Console.WriteLine("\x1b[1mPlease specify the width and height(fullscreen) of your console, in order to start the game:\x1b[0m\n(Insert ENTER, to keep the current size)\n");
      string? height=null,width=null;
      Console.Write("Height: ");height=Console.ReadLine();
      if (!string.IsNullOrEmpty(height))paramsScreen.FullScreenHeight=Convert.ToInt32(height);
      else paramsScreen.FullScreenHeight=Console.WindowHeight;
      Console.Write("Width: ");width=Console.ReadLine();
      if (!string.IsNullOrEmpty(width))paramsScreen.FullScreenWidth=Convert.ToInt32(width);
      else paramsScreen.FullScreenWidth=Console.WindowWidth;

      Console.WriteLine("\n");
      curWinSize=$"\x1b[1;3;4mheight x width set to: {paramsScreen.FullScreenHeight} x {paramsScreen.FullScreenWidth},\n\x1b[0m";
      Console.Write(curWinSize+"\x1b[1;35m");
      if(paramsScreen.initFullSCSetting)
      Console.Write("with ");else Console.Write("without ");
      Console.Write("active screen monitoring system\x1b[0m");
      Thread.Sleep(paramsScreen.initTime);Console.Clear();
    }
    public Interface(){
        verifyMaxScreen();
        showStartCredidentials();
    }
    public Interface(int a){
    //nothing
    }
    public List<Player> showInitPlayers(){verifyMaxScreen();
       this.spacing("\x1b[1;3;4;38;5;214mStabilirea Jucatorilor\x1b[0m");
       
       List<Player> players=new List<Player>();
       for(int i=0;i<2;i++){
       Console.WriteLine($"\x1b[1;37mEchipa {i+1}\x1b[0m\n");
       Console.Write("Nume Jucator 1:\t\t");players.Add(new Player(Console.ReadLine()));
       Console.Write("Nume Jucator 2:\t\t");players.Add(new Player(Console.ReadLine()));
       Console.Clear();
       }
       Console.WriteLine($"\x1b[1;37mEchipa 1\x1b[0m");
       Console.WriteLine($"\x1b[3;37m{players[0].name}\t{players[1].name}\n\x1b[0m\n");
       Console.WriteLine($"\x1b[1;37mEchipa 2\x1b[0m");
       Console.WriteLine($"\x1b[3;37m{players[2].name}\t{players[3].name}\n\x1b[0m\n");Thread.Sleep(paramsScreen.startTime);
       Console.WriteLine("\n\x1b[3mplease press any key, to continue...\x1b[0m");Console.ReadKey();
       //Console.Clear();
       return players;
    }
    public int shuffleScreen(Player chooser){verifyMaxScreen();
       this.spacing("\x1b[1;3;4;38;5;214mImpartirea Cartilor\x1b[0m");

       Console.WriteLine($"Jucator \x1b[1;3;33m{chooser.name}\x1b[0m\n");
       Console.Write("Alegi:\nBataia cartilor(Enter) sau Taierea cartilor(orice caracter):\t\t");
       string opt=Console.ReadLine();//Console.Clear();
       return (string.IsNullOrEmpty(opt))?0:1;
    }
    public void showWinners(List<Player> winnerTeam,int TeamWinner){verifyMaxScreen();
       this.spacing("\x1b[1;3;4;38;5;214mEND OF THE GAME\x1b[0m");
       
       int fPoints=winnerTeam[0].noPoints+winnerTeam[1].noPoints;
       int sPoints=winnerTeam[2].noPoints+winnerTeam[3].noPoints;
       Console.WriteLine("\x1b[1;4;31mREZULTATE FINALE\x1b[0m\n");
       Console.WriteLine($"\x1b[1;34mCastigatorii jocului:\t\x1b[1;35mEchipa {TeamWinner}({fPoints} puncte)\x1b[1;34m");
       Console.WriteLine($"Membrii:\t\x1b[1;35m{winnerTeam[0].name}({winnerTeam[0].noPoints}={winnerTeam[0].totalPoints}puncte) , {winnerTeam[1].name}({winnerTeam[1].noPoints}={winnerTeam[1].totalPoints}puncte)\x1b[0m");
       Console.WriteLine();
       
       int reffA=(TeamWinner==1)?2:1;
       Console.WriteLine($"\x1b[1;31mPierzatorii jocului:\t\x1b[1;35mEchipa {reffA}({sPoints} puncte)\x1b[1;31m");
       Console.WriteLine($"Membrii:\t\x1b[1;35m{winnerTeam[2].name}({winnerTeam[2].noPoints}={winnerTeam[2].totalPoints}puncte) , {winnerTeam[3].name}({winnerTeam[3].noPoints}={winnerTeam[3].totalPoints}puncte)\x1b[0m");
   
       Thread.Sleep(3000);
       Console.WriteLine("\n\n\n\x1b[3mplease press any key, to exit the game...\x1b[0m");Console.ReadKey();
       //Console.Clear();
    }
    public string[] showCards(List<Card> cardsToShow){//verifyMaxScreen();
    string[] reffCards=new string[cardsToShow.Count];
    for(int i=0;i<cardsToShow.Count;i++){
    if(i==cardsToShow.Count-1)Console.WriteLine(cardsToShow[i]);
    else Console.Write(cardsToShow[i]+" | ");
    reffCards[i]=cardsToShow[i].ToString();
    }
    return reffCards;
    }
    public int licitateScreen(int maxValue,Player player){
      int end=1;int[] filteredArray=new int[5];int k=0;
      while(end==1){
       verifyMaxScreen();
       int[] pointsArray = new int[]
        {
            pointsValue.none,
            pointsValue.onepoint,
            pointsValue.twopoints,
            pointsValue.threepoints,
            pointsValue.fourpoint,
            pointsValue.fivepoints,
            pointsValue.sixpoints
        };
        filteredArray= pointsArray.Where(value => value > maxValue).ToArray();//filters through linq where method what points could the player choose basaed on the maxValue set by a player
       
       if(filteredArray.Length==0)return 0;
       
       this.spacing("\x1b[1;3;4;38;5;214mLicitatia\x1b[0m");
       Console.WriteLine($"Jucator \x1b[1;3;33m{player.name}\x1b[0m\n");
       Console.WriteLine("Cartile tale:");
       showCards(player.cards);Console.WriteLine();
       Console.WriteLine("Puncte disponibile pentru anunt:");
       foreach (var value in filteredArray)Console.Write(value+$"({(int)(value*33/1000)})\t");

       Console.Write("Ce anunt alegi(index(i+1))(ENTER pentru pass/MERGE):\t");
       string o=Console.ReadLine();
       if (string.IsNullOrEmpty(o)){Console.Clear();return 0;}
       else
       { k=Convert.ToInt32(o);
         k--;
       if(k+1<=filteredArray.Length && k>=0)
         end=0;
       else{Console.WriteLine("\n\n\x1b[1;31mInvalid input\x1b[0m");Thread.Sleep(2000);}
       }
       //Console.Clear();
       }
       return filteredArray[Convert.ToInt32(k)];//ascii code
    }
    public int choose40or20Screen(List<Card> cards,List<Card> reff1,List<Card> reff2,Player player){
      int end=1;int k=0;
      
      while(end==1){
       verifyMaxScreen();
       this.spacing("\x1b[1;3;4;38;5;214mAnunt de 20/40\x1b[0m");

       Console.WriteLine($"Jucator \x1b[1;3;33m{player.name}\x1b[0m\n");
       Console.WriteLine("Cartile tale:");
       showCards(cards);Console.WriteLine();

       if(reff1.Count>0){
       Console.WriteLine("Combinatii posibile pentru anunt de 20:");
       showCards(reff1);Console.WriteLine();}

       if(reff2.Count>0){
       Console.WriteLine("Combinatii posibile pentru anunt de 40:");
       showCards(reff2);Console.WriteLine();}

       if(reff1.Count>0 && reff2.Count>0){
       Console.Write("\n\n\nCe combinatie alegi(1->20 / 2->40)(ENTER pentru pass):\t");
       string o=Console.ReadLine();
       if (string.IsNullOrEmpty(o))return 0;
       else
       {
         k=Convert.ToInt32(o);
       if(k==1 || k==2)
         end=0;
       else{Console.WriteLine("\n\n\x1b[1;31mInvalid input\x1b[0m");Thread.Sleep(2000);}
       }
       //Console.Clear();
       }
       else
       if(reff1.Count>0){
       Console.Write("\n\n\nAlegi combinatia de 20(1->20)(ENTER pentru pass):\t");
       string o=Console.ReadLine();
       if (string.IsNullOrEmpty(o))return 0;
       else
       {
         k=Convert.ToInt32(o);
       if(k==1)
         end=0;
       else{Console.WriteLine("\n\n\x1b[1;31mInvalid input\x1b[0m");Thread.Sleep(2000);}
       }
       //Console.Clear();
       }
       else
       if(reff2.Count>0){
       Console.Write("\n\n\nAlegi combinatia de 40(2->40)(ENTER pentru pass):\t");
       string o=Console.ReadLine();
       if (string.IsNullOrEmpty(o))return 0;
       else
       {
         k=Convert.ToInt32(o);
       if(k==2)
         end=0;
       else{Console.WriteLine("\n\n\x1b[1;31mInvalid input\x1b[0m");Thread.Sleep(2000);}
       }
       //Console.Clear();
       }
       }
       return k;
    }
    private List<Card> possibleCombinationsMethod(List<Card> cards,List<Card> cardsOnTable,Color tromf){
      List<Card> returnCards=new List<Card>();//lista de carti in caz de culaorea primei catrti/tromf
      Color fColor=cardsOnTable[0].color;//first color

      //verificam daca avem culoarea primei carti
      foreach(Card reffCard in cards){
      if(reffCard.color.Equals(fColor))returnCards.Add(reffCard);
      }
      if(returnCards.Count>0)return returnCards;

      //verificam daca avem culaorea tromfului(lista are 0 elemente fiidnca nu s a gasit o culoarea primei carti)
      foreach(Card reffCard in cards){
      if(reffCard.color.Equals(tromf))returnCards.Add(reffCard);
      }
      if(returnCards.Count>0)return returnCards;
      
      //in rest daca nu s au indeplinit coditiile demai sus returnam toate cartile jucaotrului
      //jucatorul enavadn culoareap cartii si nici culoarea tromfului ,atunci jucatorul poate da oricare din cartile lui,adica returnam cards
      return cards;
    }
    public int showCardsScreen(List<Card> cards,List<Card> cardsOnTable,Color? tromf,Player player){
      int end=1;int k=0;
      while(end==1){
       verifyMaxScreen();
       string option=(tromf==null)?"Stabilirea culorii tromfului":"Alegerea unei carti de joc";
       string cat="\x1b[1;3;4;38;5;214m"+option+"\x1b[0m";
       this.spacing(cat);
       
       if(tromf!=null)
       Console.WriteLine("Culoarea tromfului:\t\x1b[1;31m"+tromf+"\x1b[0m\n\n");

       int actualPoints=player.calculatePointsCurrent();int actualnoPoints=actualPoints/pointsValue.onepoint;
       Console.WriteLine($"Jucator \x1b[1;3;33m{player.name}\x1b[1;33m({actualPoints}={actualnoPoints}p)\x1b[0m\n");
       if(tromf!=null){
       if(cardsOnTable.Count>0){
       Console.WriteLine("Cartile de pe masa:");
       showCards(cardsOnTable);Console.WriteLine();
       }else Console.WriteLine("Cartile de pe masa:\n\x1b[3;31mNu exista inca o carte pe masa\x1b[0m\n");
       }

       List<Card> possibleCombinations=(cardsOnTable.Count>0)?possibleCombinationsMethod(cards,cardsOnTable,(Color)tromf):cards;
       Console.WriteLine("Cartile tale:");
       string[] reffStringCards=showCards(cards);
       for(int i=0;i<cards.Count;i++)
       {
         if(possibleCombinations.Contains(cards[i])){
            Console.Write("\x1b[1;31m");
            for(int l=0;l<reffStringCards[i].Length;l++)Console.Write("^");
            Console.Write("\x1b[0m");
         }
         else {
         for(int l=0;l<reffStringCards[i].Length;l++)Console.Write(" ");
         }
         Console.Write("   ");
       }
       Console.WriteLine("\n");

       //mereu va exista cel putin o carte care poate fii data(in caz ca nue prima culoare/culaorea tromfului)
       Console.Write("\n\n\n");
       if(player.twentyAnnounce2)Console.WriteLine("\x1b[1;31mAtentie ai anuntat 20 de puncte in tura initiala,pentru a se contoriza anuntul pune o carte de 3 sau 4\x1b[0m\n");
       else 
       if(player.fourtyAnnounce2)Console.WriteLine("\x1b[1;31mAtentie ai anuntat 40 de puncte in tura initiala,pentru a se contoriza anuntul pune o carte de 3 sau 4 de culoarea tromfului\x1b[0m\n");
       
       Console.Write("Alege cartea pe care o vei da(indexul(i+1) din Cartile tale):\t");
       string conv=Console.ReadLine();
       if(string.IsNullOrEmpty(conv)){Console.WriteLine("\n\n\x1b[1;31mInvalid input\x1b[0m");Thread.Sleep(2000);}
       else{
       k=Convert.ToInt32(conv);k--;
       if(k+1>cards.Count || k<0){Console.WriteLine("\n\n\x1b[1;31mInvalid input\x1b[0m");Thread.Sleep(2000);}
       else
       if(possibleCombinations.Contains(cards[k]))end=0;
       else
       {Console.WriteLine("\n\n\x1b[1;31mInvalid input\x1b[0m");Thread.Sleep(2000);}
       }
       //Console.Clear();
       }
       return k;
    }
    public void showStartCredidentials(){verifyMaxScreen();
       
       Console.WriteLine(paramsScreen.WelcomeText);
       Console.WriteLine(paramsScreen.SettingsText);
       Thread.Sleep(paramsScreen.startTime);
       Console.WriteLine("\n\x1b[3mplease press any key, to start the game...\x1b[0m");Console.ReadKey();
       //Console.Clear(); 
    }
    public void showEndCredidentials(){verifyMaxScreen();
      this.spacing("\x1b[1;3;4;38;5;214mEnd Credidentials\x1b[0m");

       Console.Write("\x1b[1;37mMultumesc ca ai ales sa joci Cruce in 4(2 echipe)!\x1b[0m\n\n\n\n\n\n");
       Console.Write("\x1b[1;37mProject Developer:Eduard Vesea Razvan\nFacultate:Automatica si Calculatoare\nSgr:3C.03.2\x1b[0m\n\n\n");
       this.spacing("Proiect Concepte_Fundamentale_Limbaje_Programare"); 
       Thread.Sleep(5000);
       Console.WriteLine("\n\x1b[3mplease press any key, to exit game...\x1b[0m");Console.ReadKey();
       Console.Clear();Console.ResetColor();
    }
    public void showWinnerRound(Player player,int roundNo){verifyMaxScreen();
      this.spacing($"\x1b[1;3;4;38;5;214mRound \x1b[1;31m{roundNo}\x1b[1;3;4;38;5;214m winner\x1b[0m");
       
       if(player.wonCards.Count<4)return;//daca se apeleaza din greseala pe jucatorul nepotrivit

       List<Card> reff=player.wonCards.GetRange(player.wonCards.Count-4,4);
       Card? reffB=null;
       for(int i=0;i<reff.Count;i++)
       if(reff[i].possesor.name.Equals(player.name)){reffB=reff[i];break;}

       Card reffC=(Card)reffB;
       if(player.twentyAnnounce){player.twentyAnnounce2=true;player.twentyAnnounce=false;}//activam optiunea de a continua anuntul de 20 castigatorului rundei
        else
       if(player.fourtyAnnounce){player.fourtyAnnounce2=true;player.fourtyAnnounce=false;}//activam optiunea de a continua anuntul de 20 castigatorului rundei
       Console.WriteLine($"Castigatorul rundei {roundNo} este:\t\t\x1b[1;31m{player.name}\x1b[0m\nCu cartea castigatoare:\t\t{reffC.ToString()}");
       if(player.twentyAnnounce2)Console.WriteLine($"\x1b[1;31mAtentie {player.name.ToUpper()} jucatorul a anuntat 20 de puncte in tura trecuta\x1b[0m\n");
       else 
       if(player.fourtyAnnounce2)Console.WriteLine($"\x1b[1;31mAtentie {player.name.ToUpper()} a anuntat 40 de puncte in tura trecuta\x1b[0m\n");
       
       Thread.Sleep(2000);
       Console.WriteLine("\n\x1b[3mplease press any key, to continue...\x1b[0m");Console.ReadKey();
       if(roundNo==6)return;
       Console.Clear();
       this.spacing($"\x1b[1;3;4;38;5;214mRound \x1b[1;31m{roundNo+1}\x1b[1;3;4;38;5;214m starting...\x1b[0m");
       Thread.Sleep(5000);
    }
    }
}