import java.util.Random;
import java.util.Date;
import java.io.*;
//pentru exceptii avem pachetul java.lang care e inclus automat


class CoordinateGenerator {
private Random randomGenerator;
public CoordinateGenerator() {
Date now = new Date();
long sec = now.getTime();
randomGenerator = new Random(sec+100);
}
public int generateX() {
int x = randomGenerator.nextInt(101);
if(x < 5) {
x = 0;
} else if(x > 95) {
x = 100;
} else {
x = randomGenerator.nextInt(99) + 1;
}
return x;
}
public int generateY() {
int y = randomGenerator.nextInt(101);
if(y < 5) {
y = 0;
} else if(y > 95) {
y = 50;
} else {
y = randomGenerator.nextInt(49) + 1;
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
    //constr default se injecteaza si seteaza imlicit coord la mijlocul terenului
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
    private static final int maxshots=1000;
    
    private String teams[]=new String[2];
    private int goals[]=new int[2];
    private int courners[]=new int[2];
    private int outs;
    
   
    Game(String team1,String team2){
	this.teams[0]=team1;this.teams[1]=team2;
    }
    public void simulate(){
	int nrofshots=0;
	Ball ball=new Ball();
	CoordinateGenerator coordrand=new CoordinateGenerator();
	int x,y;goals[0]=courners[0]=goals[1]=courners[1]=outs=0;

	
	do{
	    nrofshots++;
	try{
	    x=coordrand.generateX();
	    y=coordrand.generateY();
	    System.out.println(teams[0]+"-"+teams[1]+"  : Ball position-->("+x+","+y+")");
	    ball.shoot(x,y,teams);
	    
	}catch(Exception e){//sau mai multe catch uri sau unul singur cu downcast
	    if(e instanceof Goal){
		System.out.print(((Goal)e).getMessage());
		if(ball.getX()==100)
		    goals[0]++;
		    else
		    goals[1]++;
		    
		System.out.print(this.toString());
		ball=new Ball();
	    }
	     if(e instanceof Corner){
		 System.out.print(((Corner)e).getMessage());
		 if(ball.getX()==100)
		    courners[0]++;
		    else
		    courners[1]++;
		    
		System.out.println(this.toString());
		ball=new Ball(ball.getX(),ball.getY());
	    }
	      if(e instanceof Out){
		  System.out.print(((Out)e).getMessage());
		  outs++;
		  
		System.out.println(this.toString());
		ball=new Ball(ball.getX(),ball.getY());
	      }
	}
	System.out.print("\n");

	}while(nrofshots<maxshots);
	System.out.println("\n\n\t\t\t--End of game--\n"+this.toString());
    }
    public String toString(){
	return teams[0]+"-"+teams[1]+" scor: "+this.goals[0]+":"+this.goals[1]+"  cornere: "+this.courners[0]+":"+this.courners[1]+"  outuri: "+this.outs+"\n\n";
    }
    public static void main(String args[]){
	try{

	    
	//creeam cele 2 partide si adaugam totodata si echipele care vor juca
	Game game1;
	game1=new Game("FC Bihor","FCSB Steaua");

	//simulam pe rand partidele
	game1.simulate();

	}
	catch(Throwable e){
	    e.printStackTrace();
	    System.exit(-1);
        }
    }
}
