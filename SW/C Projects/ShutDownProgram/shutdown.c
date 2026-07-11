#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>
#define defaultsec 5

void shutDownProgrammed(int sec,int opt){
    if(opt==1)
 printf("Set Automatic ShutDown default_waitime: %dsec\n",sec);
 else if(!opt){
  printf("Set Manual ShutDown default_waitime: %dsec\n",sec);}

  if(opt==1 || !opt){
        for(int i=0;i<sec;i++){
            sleep(1);
            printf(".");
            fflush(stdout);
        }
        printf("\n\nShutting Down...");
        sleep(1);
        if(execl("/sbin/shutdown", "shutdown","-h","now",NULL)==-1){
            printf("\nerror on launching shutdown file\n\n");
            exit(EXIT_FAILURE);
            }
  }
    system("update-grub");
    // Reboot the system
    printf("\nRebooting to boot menu...\n");
    fflush(stdout);
    sleep(2); // Delay to ensure the message is printed before rebooting
    if (execl("/sbin/reboot", "reboot", NULL) == -1) {
        perror("Error on launching reboot command");
        exit(EXIT_FAILURE);
    }
}

int main(int argv,char **argc){
    printf("\n");
    if(argv==1 || argv>2){
       shutDownProgrammed(defaultsec,1);
    }
    else{
        int def_sec;
        if(!strcmp(argc[1],"reboot"))shutDownProgrammed(0,2);
     if((def_sec=atoi(argc[1]))==-1){
        
            printf("Error:Unknown char format,decimal foramt required\nRunning manual shutdown\n\n");
                   shutDownProgrammed(defaultsec,1);
            }
     shutDownProgrammed(def_sec,0);
    }
    return 0;
}
