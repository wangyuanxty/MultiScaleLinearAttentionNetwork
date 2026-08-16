/* qemu_test.c — bare-metal GDN-2 scan, output to SRAM */
#include "test_data.h"
volatile float *outbuf = (volatile float*)0x20001000;
volatile float *sn_buf = (volatile float*)0x20002000;
volatile unsigned *magic = (volatile unsigned*)0x20000000;
static void gdn2_scan(const float*q,const float*k,const float*v,const float*b,const float*w,const float*dc,float*S,float*o){
    for(int t=0;t<L;t++){for(int h=0;h<H;h++){
        float d=dc[t*H+h],*Sh=S+h*DK*DV;float Sn[DK*DV];for(int i=0;i<DK*DV;i++)Sn[i]=0;
        float ke[DK],kh[DK],vh[DV],qh[DK];for(int i=0;i<DK;i++){ke[i]=k[t*H*DK+h*DK+i]*b[t*H*DK+h*DK+i];kh[i]=k[t*H*DK+h*DK+i];}
        for(int i=0;i<DV;i++)vh[i]=v[t*H*DV+h*DV+i]*w[t*H*DV+h*DV+i];
        for(int i=0;i<DK;i++)qh[i]=q[t*H*DK+h*DK+i];
        for(int i=0;i<DK;i++)for(int j=0;j<DV;j++){float es=0;for(int p=0;p<DK;p++)es+=((i==p?1.0f:0)-ke[i]*kh[p])*d*Sh[p*DV+j];Sn[i*DV+j]=es+kh[i]*vh[j];}
        for(int i=0;i<DK*DV;i++)Sh[i]=Sn[i];
        for(int j=0;j<DV;j++){float s=0;for(int p=0;p<DK;p++)s+=Sn[p*DV+j]*qh[p];o[t*H*DV+h*DV+j]=s;}
    }}
}
void _start(void){
    float S[H*DK*DV]={0},o[H*DV*L];
    gdn2_scan(test_q,test_k,test_v,test_b,test_w,test_dc,S,o);
    for(int i=0;i<8;i++)outbuf[i]=o[i];
    float sn=0;for(int i=0;i<H*DK*DV;i++)sn+=S[i]*S[i];sn_buf[0]=sn;
    float s1=sn*0.5f;for(int k=0;k<5;k++)s1=(s1+sn/s1)*0.5f;sn_buf[1]=s1;
    *magic=0xDEADBEEF;while(1){}
}
