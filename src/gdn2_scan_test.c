/* gdn2_scan_test.c — core GDN-2 scan standalone test. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#define H 4
#define DK 16
#define DV 32
#define L 2
static void gdn2_scan(const float*q,const float*k,const float*v,
                       const float*b,const float*w,const float*decay,
                       float*S,float*o){
    for(int t=0;t<L;t++){for(int h=0;h<H;h++){
        float d=decay[t*H+h],*Sh=S+h*DK*DV,Sn[DK*DV];memset(Sn,0,sizeof(Sn));
        const float *qt=q+t*H*DK+h*DK,*kt=k+t*H*DK+h*DK;
        const float *vt=v+t*H*DV+h*DV,*bt=b+t*H*DK+h*DK,*wt=w+t*H*DV+h*DV;
        float ke[DK],kh[DK],vh[DV],qh[DK];
        for(int i=0;i<DK;i++){ke[i]=kt[i]*bt[i];kh[i]=kt[i];}
        for(int i=0;i<DV;i++)vh[i]=vt[i]*wt[i];
        for(int i=0;i<DK;i++)qh[i]=qt[i];
        for(int i=0;i<DK;i++)for(int j=0;j<DV;j++){
            float es=0;for(int p=0;p<DK;p++)es+=((i==p?1.0f:0)-ke[i]*kh[p])*d*Sh[p*DV+j];
            Sn[i*DV+j]=es+kh[i]*vh[j];}
        memcpy(Sh,Sn,sizeof(Sn));
        float*oh=o+t*H*DV+h*DV;memset(oh,0,DV*sizeof(float));
        for(int j=0;j<DV;j++)for(int p=0;p<DK;p++)oh[j]+=Sn[p*DV+j]*qh[p];
    }}
}
static void tohex(float x, char *buf){ /* buf>=9: 8 hex + NUL */
    uint32_t u; memcpy(&u,&x,4);
    for(int i=0;i<8;i++) buf[i]="0123456789ABCDEF"[(u>>(28-4*i))&0xF];
    buf[8]=0;
}
int main(){
    float q[H*DK*L],k[H*DK*L],v[H*DV*L],b[H*DK*L],w[H*DV*L],dc[H*L],S[H*DK*DV]={0},o[H*DV*L];
    FILE *f=fopen("scan_test.bin","rb");
    fread(q,4,H*DK*L,f);fread(k,4,H*DK*L,f);fread(v,4,H*DV*L,f);
    fread(b,4,H*DK*L,f);fread(w,4,H*DV*L,f);fread(dc,4,H*L,f);fclose(f);
    gdn2_scan(q,k,v,b,w,dc,S,o);
    printf("C:o0..7=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",o[0],o[1],o[2],o[3],o[4],o[5],o[6],o[7]);
    float sn=0;for(int i=0;i<H*DK*DV;i++)sn+=S[i]*S[i];
    printf("C:|S|=%.6f\n",sqrtf(sn));
    char hb[9];
    printf("C_HEX:");
    for(int i=0;i<8;i++){tohex(o[i],hb);printf(" %s",hb);}
    tohex(sn,hb);printf(" SN=%s\n",hb);
    return 0;
}
