/* perf_main.c — GDN-2 scan performance benchmark, L=64 真实窗口
 * 用 SysTick 测周期, 输出 MACs/step 与总周期 */
#include <stdint.h>
#include <string.h>
#include "perf_data.h"

static void sh_write0(const char *s){
    register uint32_t r0 __asm("r0") = 0x04;
    register const char *r1 __asm("r1") = s;
    __asm volatile("bkpt 0xab" : : "r"(r0), "r"(r1));
}
static uint32_t rd_syst(void){
    volatile uint32_t *cvr = (volatile uint32_t*)0xE000E018;
    return *cvr & 0x00FFFFFF; /* 24-bit only — 高 8 位在 QEMU 中可能非零 */
}
static void utoa(uint32_t x, char *b){ /* 最多 10 位 */
    char tmp[11]; int i=0;
    do{tmp[i++]='0'+x%10; x/=10;}while(x);
    while(i) *b++=tmp[--i]; *b=0;
}
__attribute__((noinline))
static void gdn2_scan(const float*q,const float*k,const float*v,
                      const float*b,const float*w,const float*decay,
                      float*S,float*o){
    for(int t=0;t<L;t++){for(int h=0;h<H;h++){
        float d=decay[t*H+h],*Sh=S+h*DK*DV,Sn[DK*DV];memset(Sn,0,sizeof(Sn));
        const float *qt=q+t*H*DK+h*DK,*kt=k+t*H*DK+h*DK;
        const float *vt=v+t*H*DV+h*DV,*bt=b+t*H*DK+h*DK,*wt=w+t*H*DV+h*DV;
        float ke[DK],kh[DK],vh[DV],qh[DK];
        for(int i=0;i<DK;i++){ke[i]=kt[i]*bt[i];kh[i]=kt[i];}
        for(int i=0;i<DV;i++)vh[i]=vt[i]*wt[i];for(int i=0;i<DK;i++)qh[i]=qt[i];
        for(int i=0;i<DK;i++)for(int j=0;j<DV;j++){
            float es=0;for(int p=0;p<DK;p++)es+=((i==p?1.0f:0)-ke[i]*kh[p])*d*Sh[p*DV+j];
            Sn[i*DV+j]=es+kh[i]*vh[j];}memcpy(Sh,Sn,sizeof(Sn));
        float*oh=o+t*H*DV+h*DV;memset(oh,0,DV*sizeof(float));
        for(int j=0;j<DV;j++)for(int p=0;p<DK;p++)oh[j]+=Sn[p*DV+j]*qh[p];}}}
int main(void){
    volatile uint32_t *syst_csr = (volatile uint32_t*)0xE000E010;
    volatile uint32_t *syst_rvr = (volatile uint32_t*)0xE000E014;
    *syst_rvr = 0x00FFFFFF; *syst_csr = 5; /* clksrc=core, enable, CVR 自动重载 */
    float S[H*DK*DV],o[H*DV*L]; memset(S,0,sizeof(S));
    /* 跑 3 次取中位数, 避免首轮 memset/S 清零开销干扰 */
    uint32_t cyc[3];
    for(int run=0;run<3;run++){
        uint32_t t0 = rd_syst();
        gdn2_scan(test_q,test_k,test_v,test_b,test_w,test_dc,S,o);
        uint32_t t1 = rd_syst();
        cyc[run] = t0 - t1; /* 向下计数, t0>t1 */
    }
    /* bubble sort 3 elements */
    if(cyc[0]>cyc[1]){uint32_t t=cyc[0];cyc[0]=cyc[1];cyc[1]=t;}
    if(cyc[1]>cyc[2]){uint32_t t=cyc[1];cyc[1]=cyc[2];cyc[2]=t;}
    if(cyc[0]>cyc[1]){uint32_t t=cyc[0];cyc[0]=cyc[1];cyc[1]=t;}
    uint32_t mid = cyc[1];
    uint32_t macs = L * H * (DK*DK*DV + 2*DK*DV);
    char msg[128], nb[12]; int p=0;
    char *s="O0 "; for(int i=0;s[i];i++)msg[p++]=s[i];
    s="L="; for(int i=0;s[i];i++)msg[p++]=s[i];
    utoa(L,nb); for(int i=0;nb[i];i++)msg[p++]=nb[i];
    s=" cyc="; for(int i=0;s[i];i++)msg[p++]=s[i];
    utoa(mid,nb); for(int i=0;nb[i];i++)msg[p++]=nb[i];
    s=" per_step="; for(int i=0;s[i];i++)msg[p++]=s[i];
    utoa(mid/L,nb); for(int i=0;nb[i];i++)msg[p++]=nb[i];
    s=" MACs="; for(int i=0;s[i];i++)msg[p++]=s[i];
    utoa(macs,nb); for(int i=0;nb[i];i++)msg[p++]=nb[i];
    s=" eff="; for(int i=0;s[i];i++)msg[p++]=s[i];
    utoa((uint32_t)((float)macs*1000/mid),nb); /* MACs/kCyc * 1000 -> 3 decimal */
    for(int i=0;nb[i];i++)msg[p++]=nb[i];
    msg[p++]='\n';msg[p]=0;
    sh_write0(msg);
    return 0;
}
