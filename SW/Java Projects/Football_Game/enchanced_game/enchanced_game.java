import java.util.Random;
import java.util.Date;
import java.io.*;
//pentru exceptii avem pachetul java.lang care e inclus automat


class CoordinateGenerator {

private Random randomGenerator;
public CoordinateGenerator() {
Date now = new Date();
long sec = now.getTime();
randomGenerator = new Random(sec);//dam acelasi seed(sec) de ficare data ca sa avem secvente diferite
}
public int generateX(int prob1,int prob2) {
int x = randomGenerator.nextInt(101);int caz=0;
if(x>(100-prob1)){
x=100;caz=1;}
if(x<(prob2)){
x=0;caz=1;}
if(caz==0)x=randomGenerator.nextInt(100);//max-->99
return x;
}
public int generateY(int prob1,int prob2,int x) {
int y = randomGenerator.nextInt(101);
int max=1;if(prob1==prob2)max=0;
int i;

int rlimit=30,llimit=80;//pt sanse mai mici de gol se modifica rlimit se urca si llimit se coboara,pana la max (49,51)
if(y < rlimit-(max*(rlimit/2))) {
	i=(new Random()).nextInt(6);
	if(i<4){
y = randomGenerator.nextInt(20);
	if(y==0)y=1;
	}else
if(i%2==0)y=0;else y=50;

} else if(y >llimit+(max*((llimit-50)/2))) {

	i=(new Random()).nextInt(6);
	if(i<4)
{
	y = randomGenerator.nextInt(30) + 31;
	if(y>50)y-=10;
	if(y==50)y=49;
}
else
if(i%2==0)y=0;else y=50;
} else {
	if(max==1)y=randomGenerator.nextInt(51);
	else
y=25;//max-->49
}
return y;
}
}



class Out extends Exception{
    public String getMessage(){
	return "\n\nOut\n";
    }
}
class Goal extends Exception{
    private String text;
    public Goal(String team){this.text="\n\nGOOOOAL!! for "+team+"\n";}
    public String getMessage(){
	return text;
    }
}

class Corner extends Exception{
     private String text;
    public Corner(String team){this.text="\n\nCOURNER for "+team+"\n";}
    public String getMessage(){
	return text;
    }
}

class Ball{
    private int x,y;
    //constr default se injecteaza si seteaza imlicit coord la 0,0
    Ball(int x,int y){this.x=x;this.y=y;}
     Ball(){this(50,25);}

    public void shoot(int xnew,int ynew,String teams[])throws Goal,Corner,Out{
       x=xnew;y=ynew;
       
       if(y==0 || y==50)throw new Out();
       
       if(y>=20 && y<=30)
	   if(x==0)throw new Goal(teams[1]);
	   else
	       if(x==100)throw new Goal(teams[0]);
      
       if(x==0){
	   if((y>0 && y<20) ||(y>30 && y<50))throw new Corner(teams[1]);
       }
       if(x==100){
            if((y>0 && y<20) ||(y>30 && y<50))throw new Corner(teams[0]);
       }
    }
    public int getX(){return x;}
    public int getY(){return y;}
}

class Game{
    private static final int playTime=90;
    private static final long playTimeUnit=1;//time unit(sec) 
	private int prob1,prob2;
    
    private String teams[]=new String[2];
    private int goals[]=new int[2];
    private int courners[]=new int[2];
    private int outs;
    
   
    Game(String team1,String team2){
	this.teams[0]=team1;this.teams[1]=team2;
    }
    synchronized public void simulate()throws InterruptedException{
	int currentplayTimeUnit=0;
	Ball ball=new Ball();
	CoordinateGenerator coordrand=new CoordinateGenerator();
	int x,y;goals[0]=courners[0]=goals[1]=courners[1]=outs=0;

	
	do{
	    currentplayTimeUnit++;
	try{
	    x=coordrand.generateX(this.prob1,this.prob2);
	    y=coordrand.generateY(this.prob1,this.prob2,x);
	    //System.out.println(teams[0]+"-"+teams[1]+"  : Ball position-->("+x+","+y+")");
	    System.out.print("*");
	    ball.shoot(x,y,teams);
	    
	}catch(Exception e){//sau mai multe catch uri sau unul singur cu downcast
	    System.out.println();
	    if(e instanceof Goal){
		System.out.print(currentplayTimeUnit+"'  "+((Goal)e).getMessage());
		System.out.print(this.toString());
		if(ball.getX()==100)
		    goals[0]++;
		    else
		    goals[1]++;
		ball=new Ball();
	    }
	     if(e instanceof Corner){
		 System.out.print(((Corner)e).getMessage());
		System.out.println(this.toString());
		if(ball.getX()==100)
		    courners[0]++;
		    else
		    courners[1]++;
		ball=new Ball(ball.getX(),ball.getY());
	    }
	      if(e instanceof Out){
		  System.out.print(((Out)e).getMessage());
		System.out.println(this.toString());
		outs++;
		ball=new Ball(ball.getX(),ball.getY());
	      }
	}finally{wait(playTimeUnit*1000);}
	//System.out.print("\n");

	}while(currentplayTimeUnit<playTime);
	System.out.println("\n\n\t\t\t--End of game--\n"+this.toString());
    }
    public String toString(){
	return teams[0]+"-"+teams[1]+" scor: "+goals[0]+":"+goals[1]+"  cornere: "+courners[0]+":"+courners[1]+"  outuri: "+outs+"\n\n";
    }

    public static void main(String args[]){
	try{

	    
	//creeam cele o partida si adaugam totodata si echipele care vor juca
    String team1,team2;

	System.out.println("Introduceti numele echipelor:");
	team1=(new BufferedReader(new InputStreamReader(System.in))).readLine();
	team2=(new BufferedReader(new InputStreamReader(System.in))).readLine();
    Game game1=new Game(team1,team2);
	System.out.println("\nIntroduceti experienta fiecarei echipe  (__max_50):");
    game1.prob1=Integer.parseInt((new BufferedReader(new InputStreamReader(System.in))).readLine());
	game1.prob2=Integer.parseInt((new BufferedReader(new InputStreamReader(System.in))).readLine());System.out.println();
    
	//simulam pe rand partidele
	game1.simulate();
	}
	catch(Throwable e){
	    e.printStackTrace();
	    System.exit(-1);
	}
    }
}
