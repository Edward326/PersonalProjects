#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/ioctl.h>

typedef struct{
  char sign[2];
  int file_size,rez1,offset_start;
  int sizeH,width,height,planes,btc_px,comprss,img_size;
  int px_x,px_y,size_color,size_icolor;
  int **array;
}image;


void show(unsigned int pixel){
  int chunk=sizeof(pixel)*8-1;
  
  for(int i=chunk;i>=0;i--){
    if((i+1)%4==0 && i!=chunk)
      printf(" ");
    
    if(pixel&((1<<(chunk-1))>>(chunk-i-1)))
    printf("1");
    else
      printf("0");
  }
      printf("\n");

}


unsigned int black_white24B(unsigned int nr){
  int red,green,blue,bw_factor,new24=0;
  
  red=(nr>>16)&255;//1111 1111
    green=(nr>>8)&255;//0000 0000 1111 1111
  blue=nr&255;//0000 0000 0000 0000 1111 1111
  bw_factor=(red+green+blue)/3;
  new24|=(bw_factor<<16);
  new24|=(bw_factor<<8);
  new24|=(bw_factor);

  return new24;
}

unsigned int black_white32B(unsigned int nr){
  int red,green,blue,bw_factor,new32=0;
  
  red=(nr>>9)&511;//1111 1111
    green=(nr>>17)&511;//0000 0000 1111 1111
  blue=nr&511;//0000 0000 0000 0000 1111 1111
  bw_factor=(red+green+blue)/3;
  new32|=(bw_factor<<9);
  new32|=(bw_factor<<17);
  new32|=(bw_factor);

  return new32;
}

unsigned int sepia_24B(unsigned int nr){
  int red,green,blue,new24=0;
  int tr,tg,tb;
  
  red=(nr>>16)&255;//1111 1111
    green=(nr>>8)&255;//0000 0000 1111 1111
  blue=nr&255;//0000 0000 0000 0000 1111 1111

  tr=0.393*red+0.769*green+0.189*blue;
  tg=0.349*red+0.686*green+0.168*blue;
  tb=0.272*red+0.534*green+0.131*blue;

    if(tr>255)
      new24|=(255<<16);
    else
      new24|=(tr<<16);
  
   if(tg>255)
      new24|=(255<<8);
    else
      new24|=(tg<<8);
   
    if(tb>255)
      new24|=255;
    else
      new24|=tb;


  return new24;
}

unsigned int sepia_32B(unsigned int nr){
 int red,green,blue,new24=0;
  int tr,tg,tb;
  
  red=(nr>>9)&511;//1111 1111
    green=(nr>>17)&511;//0000 0000 1111 1111
  blue=nr&511;//0000 0000 0000 0000 1111 1111

  tr=0.393*red+0.769*green+0.189*blue;
  tg=0.349*red+0.686*green+0.168*blue;
  tb=0.272*red+0.534*green+0.131*blue;

    if(tr>511)
      new24|=(255<<9);
    else
      new24|=(tr<<9);
  
   if(tg>511)
      new24|=(255<<17);
    else
      new24|=(tg<<17);
   
    if(tb>511)
      new24|=511;
    else
      new24|=tb;


  return new24;
}


void load(image *img,FILE *original){
 fwrite(&img->sign,sizeof(img->sign),1,original);
     fwrite(&img->file_size,sizeof(int),1,original);
    fwrite(&img->rez1,sizeof(int),1,original);
     fwrite(&img->offset_start,sizeof(int),1,original);
     fwrite(&img->sizeH,sizeof(int),1,original);
     fwrite(&img->width,sizeof(int),1,original);
     fwrite(&img->height,sizeof(int),1,original);
    fwrite(&img->planes,sizeof(int)/2,1,original);
     fwrite(&img->btc_px,sizeof(int)/2,1,original);
     fwrite(&img->comprss,sizeof(int),1,original);
     fwrite(&img->img_size,sizeof(int),1,original);
     fwrite(&img->px_x,sizeof(int),1,original);
    fwrite(&img->px_y,sizeof(int),1,original);
     fwrite(&img->size_color,sizeof(int),1,original);
     fwrite(&img->size_icolor,sizeof(int),1,original);
    
    
    for(int i=0;i<img->height;i++){
     
      for(int j=0;j<img->width;j++){
	//show(img->array[i][j]);
	fwrite(&img->array[i][j],(img->btc_px)/8,1,original);
      }
    }
}


image *save(image *img,FILE *original){
  int current=0;
 fread(&img->sign,sizeof(img->sign),1,original);
 if(strcmp(img->sign,"BM"))
   return NULL;
    fread(&img->file_size,sizeof(int),1,original);
    fread(&img->rez1,sizeof(int),1,original);
    fread(&img->offset_start,sizeof(int),1,original);
    fread(&img->sizeH,sizeof(int),1,original);
    fread(&img->width,sizeof(int),1,original);
    fread(&img->height,sizeof(int),1,original);
    fread(&img->planes,sizeof(int)/2,1,original);
    fread(&img->btc_px,sizeof(int)/2,1,original);
    fread(&img->comprss,sizeof(int),1,original);
    fread(&img->img_size,sizeof(int),1,original);
    fread(&img->px_x,sizeof(int),1,original);
    fread(&img->px_y,sizeof(int),1,original);
    fread(&img->size_color,sizeof(int),1,original);
    fread(&img->size_icolor,sizeof(int),1,original);
    
    if((img->array=(int**)malloc(img->height*(sizeof(int*))))==NULL){
      perror("invalid alloc on pixel array");
      exit(-1);}   
    for(int i=0;i<img->height;i++){
      img->array[i]=malloc(img->width*sizeof(int));
      for(int j=0;j<img->width;j++){
	fread(&current,(img->btc_px)/8,1,original);
	img->array[i][j]=current;
      }
    }


  return img;
}

image *bw_save(image *img){
  if(img->btc_px==24){
    
    for(int i=0;i<img->height;i++){
      for(int j=0;j<img->width;j++){
	  img->array[i][j]=black_white24B(img->array[i][j]);
      }
    }
    
  }
  if(img->btc_px==32){
    
for(int i=0;i<img->height;i++){
      for(int j=0;j<img->width;j++){
	  img->array[i][j]=black_white32B(img->array[i][j]);
      }
 }

  }


  return img;
}

image *sepia_save(image *img){
  if(img->btc_px==24){
    
    for(int i=0;i<img->height;i++){
      for(int j=0;j<img->width;j++){
	  img->array[i][j]=sepia_24B(img->array[i][j]);
      }
    }
    
  }
  if(img->btc_px==32){
    
for(int i=0;i<img->height;i++){
      for(int j=0;j<img->width;j++){
	  img->array[i][j]=sepia_32B(img->array[i][j]);
      }
 }

  }


  return img;
}


int options(){
  int nr;
  printf("OPTIUNI:\n\n1.TRANSF BW\n2.TRANSF SEPIA\n\ninput:\t");
  scanf("%d",&nr);
  printf("\n");
  
  return nr;
}

void spacing(int lines){
  for(int i=0;i<lines;i++)
    printf("\n");
}
void help(struct winsize w){
  spacing(w.ws_row);
  printf("programul accepta in linia de comanda ./(fisier executabil) urmat de unul sau mai multe fisiere(format bmp) care vor a le fii aplicate filtre\nalgoritm:aplicatiaia ca arg din linia de coamanda a consolei executabile numele fisierelor pe care le va filtra in modul dorit,fiecare imagine daca ea e binetinteles formatul bmp (BM-offset0') si se afla in calea curenta de unde se executa programul in sine,se va salva intr un vector de structuri(fiecare index de tipul --typedef va retine continutul fiecarei imagini introduse),fiind memorate se va cere optiunea de aplicare asupra imaginilor,odata alesa optiunea se va folosii functia aferenta optiunii pt a trimite fiecare imagine catre editare(ea fiind pe 24 sau 32bits/pixel),  mai apoi prog va creea fisierele modificate ale imaginilor tot in aceeasi cale curenta fiind de structura numeinitial_mod.bmp, bineinteles daca programul primeste o alta optiune decat cea reconscuta nu va efectua nicio operatie\n\n");
  int nr;
  scanf("%d",&nr);

}

void free_up1(image **img,int size){
  for(int i=0;i<size;i++)
    free(img[i]);

}
void free_up2(char **fisier,int size){
  for(int i=0;i<size;i++)
    free(fisier[i]);

}





int main(int argv,char** argc){
  struct winsize w;
  ioctl(0,TIOCGWINSZ,&w);
  
  
 
  if(argv>1){
    printf("v 2.0A(with no memory leaks) \nIMAGE FILTER(SEPIA/BW) //only BITMAP(.bmp) files reconigzed\n\n--rerun with './(executable) help' for help commands\n\n ");
    if(argv==2 && (strcmp(argc[1],"help")==0)){
      help(w);
      spacing(w.ws_row);
      return 0;
    }
    int index=0;
    image **img;
    char **fisier;
    img=malloc(sizeof(image*)*(argv-1));
    
    for(int i=1;i<argv;i++){
      FILE *original;
      
      if((original=fopen(argc[i],"rb"))!=NULL){
	img[index]=malloc(sizeof(image));
	img[index]=save(img[index],original);
	if(img[index]!=NULL){
	index++;
	fisier=realloc(fisier,sizeof(char*)*index);
        fisier[index-1]=malloc((sizeof(char)*strlen(argc[i]))+4);
	strncpy(fisier[index-1],argc[i],strlen(argc[i])-4);
	strcat(fisier[index-1],"_mod");
	strcat(fisier[index-1],".bmp");
	//fclose(original);
	}
      }
    }
    printf("\n%d/%d files found\n\n",index,argv-1);
    img=realloc(img,sizeof(image)*index);

    int opt;
    opt=options();
    if(opt==2){
      for(int i=0;i<index;i++){
	  
	  FILE *secondary=fopen(fisier[i],"wb+");
	  img[i]=sepia_save(img[i]);
          load(img[i],secondary);
	  //fclose(secondary);
	}
      // fclose(secondary);
      spacing(w.ws_row);
      printf("\napp terminated with %d/%d new files created\n\n",index,argv-1);
    }
    if(opt==1){
      
        for(int i=0;i<index;i++){
	  
	  FILE *secondary=fopen(fisier[i],"wb+");
	  img[i]=bw_save(img[i]);
          load(img[i],secondary);
	  //fclose(secondary);
	}
	//fclose(secondary);
	spacing(w.ws_row);
	printf("\napp terminated with %d/%d new files created\n\n",index,argv-1);
    }
    if(opt!=1 && opt!=2){
      spacing(w.ws_row);
      printf("unknown input\n--app terminated with 0/0 files created");
    }
    
    printf("\n\t\t\t\t\t\t\t\t\t\t\t--dev:eduard_vesea\n");
    int a=0;
    scanf("%d",&a);
    spacing(w.ws_row);
    free_up1(img,index);
    free_up2(fisier,index);
  }
  else
    {
      perror("not enough arguments");
      exit(-1);
    }
  
  return 0;
}
