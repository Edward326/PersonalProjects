using System;
using CruceEduardVesea.ui_interface;
using System.Collections.Generic;
using System.Threading;


namespace CruceEduardVesea.entities{

public struct pointsValue{
public static int none=0;
public static int onepoint=33,twopoints=66,threepoints=99;
public static int fourpoint=132,fivepoints=165,sixpoints=198;

public static string showPoints()=> $"\nNiciun punct:{pointsValue.none}\n1 punct:{pointsValue.onepoint}\t2 puncte:{pointsValue.twopoints}\t3 puncte:{pointsValue.threepoints}\n4 puncte:{pointsValue.fourpoint}\t5 puncte:{pointsValue.fivepoints}\t6 puncte:{pointsValue.sixpoints}\n";
}

public enum Color{
verde,rosu,duba,ghinda
}

public class Card{
    public Player possesor=new Player();
    public readonly int value;
    public readonly Color color;
    
    public Card(int reffValue,Color reffColor){
        this.value = reffValue;
        this.color = reffColor;
    }
    public static string DisplayReffCards()
    {
        string[] values = new string[] { "2", "3", "4","0", "10", "11"};
        string[] suits = new string[] { "ROSU", "GHINDA", "DUBA", "VERDE" };
        string[] suitEmojis = new string[] { "❤️", "⚜️", "🟨", "🌿" };

        // Iterăm prin fiecare combinație de valoare și culoare și le afișăm
        string result="\n";
        foreach(string suit in suits)
        {
            int suitIndex = Array.IndexOf(suits, suit);
            for (int i = 0; i < values.Length; i++)
            {
                string value = values[i];
                string emoji = suitEmojis[suitIndex];
                if(i<values.Length-1)
                result+=$"{emoji} ({suit}) de valoarea {value} | ";
                else result+=$"{emoji} ({suit}) de valoarea {value}\n";
            }
            result+="\n";
        }
        return result;
    }
    public override string ToString(){
    string[] suitEmojis = new string[] { "❤️", "⚜", "🟨", "🌿" };
    string[] suits = new string[] { "ROSU", "GHINDA", "DUBA", "VERDE" };
    string emojiReff;string suitName;
    switch(color) 
    {
    case Color.rosu:
    emojiReff=suitEmojis[0];suitName=suits[0];
    break;
    case Color.ghinda:
    emojiReff=suitEmojis[1];suitName=suits[1];
    break;
    case Color.duba:
    emojiReff=suitEmojis[2];suitName=suits[2];
    break;
    default:
    emojiReff=suitEmojis[3];suitName=suits[3];
    break;
    }
    return $"{emojiReff} ({suitName}) de valoarea {value}";
    }
}

public class Player{
    public readonly string name;
    public List<Card> cards=new List<Card>();
    public List<Card> wonCards=new List<Card>();
    public int announcedPoints{ get;private set;}
    public int totalPoints{ get;private set;}
    public int noPoints{get;private set;}
    public int hadtwentyAnnounce;
    public int hadfourtyAnnounce;
    public bool twentyAnnounce,twentyAnnounce2;
    public bool fourtyAnnounce,fourtyAnnounce2;
    public bool isLicitor;
    private List<Card> twentyList=new List<Card>();
    private List<Card> fourtyList=new List<Card>(); 
    
    public Player(string reffName)
        {
            this.name=reffName;
            this.announcedPoints = this.totalPoints=this.noPoints=0;
            this.twentyAnnounce=this.fourtyAnnounce=false;
            this.twentyAnnounce2=this.fourtyAnnounce2=false;
            this.hadfourtyAnnounce=this.hadtwentyAnnounce=0;
            this.isLicitor=false;
        }
    public Player(){this.name = "default";
            this.announcedPoints = this.totalPoints=this.noPoints=0;
            this.twentyAnnounce=this.fourtyAnnounce=false;
            this.twentyAnnounce2=this.fourtyAnnounce2=false;
            this.hadfourtyAnnounce=this.hadtwentyAnnounce=0;
            this.isLicitor=false;}
    public override bool Equals(object? obj){
         if (obj == null || GetType() != obj.GetType())
            return false;
        Player other = (Player)obj;
        return name.Equals(other.name);
    }
    public void setLicitor(){isLicitor=true;}
    public void licitate(Interface interfaceScreen,ref int maxValue,ref Player? maxPlayer){
    announcedPoints=interfaceScreen.licitateScreen(maxValue,this);//afisam intai cate puncte poate sa faca jucatoarul
    if(maxPlayer==null){maxPlayer=this;maxValue=announcedPoints;}
    else
    if(announcedPoints>maxValue){
    maxValue=announcedPoints;maxPlayer=this;
    }
    }
    public void licitateForAnnounce(Interface interfaceScreen,Color tromf){
        //afisam dupa ce s a facut licictatia(se merge inca odata peste fiecare player) pt fieacre jucator daca poate alege anunt de 40 sau anunt de 20
    List<Card> reff1=this.possibleTwenty();//la Count poate sa fie 0,2,4,6(6 adica 3 perechi de (3,4))
    List<Card> reff2=this.possibleFourty(tromf);//la Count poate sa fie doar 0,2(2 cand avem (3,4) de acceasi culoare si culaorea tromfului ,nu putem avea alta pereche (3,4) de culoarea tromfului daca am avut deja una)
    
    if(reff1.Count==0 && reff2.Count==0)return;//nu exista posib de anunt de 20 sau 40
    
    if(reff2.Count>0){reff1.Remove(reff2[0]);reff1.Remove(reff2[1]);}//eliminarea cartilor de 40 din cele de 20

    int opt=interfaceScreen.choose40or20Screen(cards,reff1,reff2,this);//1 pentru 20 2-pentru 40
    
    twentyAnnounce=(opt==1);//ori anunt de 20 ori de 40
    twentyList=twentyAnnounce?reff1:new List<Card>();

    fourtyAnnounce=(opt==2);//ori anunt de 20ori 40
    fourtyList=fourtyAnnounce?reff2:new List<Card>();
    
    
    if(twentyAnnounce){Console.Clear();
    Console.WriteLine("\x1b[1;31mAtentie ai anuntat 20 de puncte,pentru a se contoriza anuntul\n\nNU FOLOSII ACEASTA RUNDA O CARTE DIN DECK-UL DE 3,4 DIN ANUNT\nCASTIGA RUNDA CURENTA SI ALEGE ATUNCI O CARTE DE 3,4 DIN DECK-UL DIN ANUNT\x1b[0m\n");
    Thread.Sleep(2000);
    Console.WriteLine("\n\x1b[3mplease press any key, to continue the game...\x1b[0m");Console.ReadKey();
    }
    else 
    if(fourtyAnnounce){Console.Clear();
    Console.WriteLine("\x1b[1;31mAtentie ai anuntat 40 de puncte,pentru a se contoriza anuntul\n\nNU FOLOSII ACEASTA RUNDA O CARTE DIN DECK-UL DE 3,4 DIN ANUNT\nCASTIGA RUNDA CURENTA SI ALEGE ATUNCI O CARTE DE 3,4 DIN DECK-UL DIN ANUNT\x1b[0m\n");
    Thread.Sleep(2000);
    Console.WriteLine("\n\x1b[3mplease press any key, to continue the game...\x1b[0m");Console.ReadKey();
    }       
    }
    private List<Card> possibleTwenty(){
        Color? reffColor=null;
        List<Card> twentyList=new List<Card>();
        int k=0,poz=-1;
        List<int> pozList=new List<int>();
        for(int i=0;i<cards.Count;i=k){
        if(cards[i].value==3){
        if(reffColor!=null){//daca avem deja un treiar
            if(cards[i].color==reffColor)
            if(!twentyList.Contains(cards[i])){//daca avem deja un treiar,culorile la doiar sunt la fel ca la treiar si treiarul nu e in lista
            twentyList.Add(cards[i]);reffColor=null;poz=-1;
            k=0;continue;//am gasit prima posibila pereche de (2,3) mergem de la incepuut si incercam sa gasim alta ,care nu se afla deja in lista de perechi
            }
            }
            else 
            if(!twentyList.Contains(cards[i]) && !pozList.Contains(i)){
            twentyList.Add(cards[i]);poz=i;
            reffColor=cards[i].color;}//daca nu avem un treaiar avem prima oara un doiar
        }
        if(cards[i].value==4)
        {
            if(reffColor!=null){//daca avem deja un doiar
            if(cards[i].color==reffColor)
            if(!twentyList.Contains(cards[i])){//daca avem deja un doiar,culorile la treiar sunt la fel ca la doiar si doiarul nu e in lista
            twentyList.Add(cards[i]);reffColor=null;poz=-1;
            k=0;continue;//am gasit prima posibila pereche de (3,2) mergem de la incepuut si incercam sa gasim alta ,care nu se afla deja in lista de perechi
            }
            }
            else
            if(!twentyList.Contains(cards[i]) && !pozList.Contains(i)){
            twentyList.Add(cards[i]);poz=i;
            reffColor=cards[i].color;} //daca nu avem un doiar avem prima oara un treiar
        }
        k=i+1;

        if(k==cards.Count){
            if(twentyList.Count%2!=0){
            twentyList.Remove(twentyList[twentyList.Count-1]);//elimniam el care nu are pereche
            k=0;reffColor=null;
            pozList.Add(poz);
            }
        }
        }
        return twentyList;
    }
    private List<Card> possibleFourty(Color tromf){
        Color? reffColor=null;
        List<Card> fourtyList=new List<Card>();
        foreach(Card reff in cards){
        if(reff.value==3){
        if(reffColor!=null){
            if(reff.color==reffColor && reff.color==tromf){
            fourtyList.Add(reff);return fourtyList;}//exista doar un doiar si un treiar de culoarea tromfului care poate sa vina la un singur jucator dintre toti
            }
            else 
            if(reff.color==tromf){reffColor=tromf;fourtyList.Add(reff);}
        }
        if(reff.value==4)
        {
            if(reffColor!=null){
            if(reff.color==reffColor && reff.color==tromf){
            fourtyList.Add(reff);return fourtyList;}
            }
            else
            if(reff.color==tromf){reffColor=tromf;fourtyList.Add(reff);}
        }
        }
        if(fourtyList.Count>0)
        fourtyList.Remove(fourtyList[fourtyList.Count-1]);
        
        return fourtyList;//cazul cand nu avem pereche deci trebuie returnat o lista care nu are nicun element
    }
    public void addCards(Card newCard){cards.Add(newCard);}
    public void addCards(List<Card> newCard){cards.AddRange(newCard);}//overload
    public void addWonCards(Card newCard){wonCards.Add(newCard);}
    public void addWonCards(List<Card> newCards){wonCards.AddRange(newCards);}//overload
    public Color setTromf(Interface interfaceScreen,ref Card throwCard){
        int cIndex=interfaceScreen.showCardsScreen(cards,new List<Card>(),null,this);//nu e evaluata culaorea tormfului,putem pune orice culoare
        Color reffColor=cards[cIndex].color;
        throwCard=cards[cIndex];
        cards.Remove(cards[cIndex]);//scoaterea cartii dinn stackul de carti al jucatorului
        licitateForAnnounce(interfaceScreen,reffColor);
        return reffColor;
    }
    public Card withdrawCard(Interface interfaceScreen,List<Card> onTableCards,Color tromf,int roundNo){
        if(!twentyAnnounce2 && !fourtyAnnounce2 && roundNo<6)licitateForAnnounce(interfaceScreen,tromf);
        //daca suntem in ultima runda NU SE MAI POATE LICITA pt 20/40 deoarece nu mai avem o runda viitoare in care jucatorul sa poate primii 20/40
        //daca jucatorul este castigatorul rundei si el a licitat 20 sau 40 runda trecuta(ori fourtyAnnounce2 ori twentyAnnounce2 e true) NU SE MAI POATE LICITA
        //(se suprascriu cartile de 20/40 de la runda trecuta cu cele de la runda prezenta si nu mai se poate verifica nimic)
        //in rest daca nu am ajuns la ultima runda sau daca nu este castigatorul rundei sau este castigatorul rundei si nu a facut anunt runda trecuta vom da posibilitatea de a face anunt

        int cIndex=interfaceScreen.showCardsScreen(cards,onTableCards,tromf,this);

        if(twentyAnnounce){
            if(twentyList.Contains(cards[cIndex]))twentyAnnounce=false;
        }
        if(fourtyAnnounce){
            if(fourtyList.Contains(cards[cIndex]))fourtyAnnounce=false;
        }
        
        if(twentyAnnounce2){
        if(twentyList.Contains(cards[cIndex]))//s-a inceput cu cartea de 20 in cazul anuntului de 20
        {
           twentyAnnounce2=false;hadtwentyAnnounce+=20;//pt adaugarea de 20 la final daca se strang punctele anuntate
        }
        else twentyAnnounce2=false;}
        if(fourtyAnnounce2){
        if(fourtyList.Contains(cards[cIndex]))//s-a inceput cu cartea de 40 ikn cazul anuntuului de 40
        {
           fourtyAnnounce2=false;hadfourtyAnnounce+=40;//pt adaugarea de 40 la final daca se strang punctele anuntate
        }
        else fourtyAnnounce2=false;}

        Card cardToRemove=cards[cIndex];
        cards.Remove(cards[cIndex]);//scoaterea cartii dinn stackul de carti al jucatorului
        return cardToRemove;
    }
    public void calculatePoints(){
        if(wonCards.Count==0)return;
        foreach(Card reffCard in wonCards)
        totalPoints+=reffCard.value;

        totalPoints+=hadtwentyAnnounce;//aduagam anuntul de 20 daca a fost respectat
        totalPoints+=hadfourtyAnnounce;//aduagam anuntul de 40 daca a fost respectat
    
        if(isLicitor)
        totalPoints=(totalPoints>=announcedPoints)?totalPoints:totalPoints-announcedPoints;//penalizarea doar pe cel care a condus licitatia
        
        totalPoints=(totalPoints>=0)?totalPoints:0;
        noPoints=(totalPoints>0)?totalPoints/pointsValue.onepoint:0;
    }
    public int calculatePointsCurrent(){
        if(wonCards.Count==0)return 0;
        int totalP=0;
        foreach(Card reffCard in wonCards)
        totalP+=reffCard.value;
        
        return totalP;
    }
}
}