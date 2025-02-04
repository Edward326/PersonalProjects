using System;
using CruceEduardVesea.entities;
using CruceEduardVesea.ui_interface;
using System.Collections.Generic;


namespace CruceEduardVesea.main{

public class MainGame{
    private Game mainGame{get;set;}
    private List<Player> winnerTeam=new List<Player>();
    private int TeamWinner;
    
    public MainGame(){this.mainGame=new Game();}
    public void start(){
        Interface.getSizesFC();
        mainGame=Game.initialize();
    }
    public void play(){
        int Team=0;
        winnerTeam=mainGame.playGame(ref Team);
        TeamWinner=Team;
    }
    public void end(){
        mainGame.interfaceScreen.showWinners(winnerTeam,TeamWinner);
        mainGame.interfaceScreen.showEndCredidentials();
    }
 }
}