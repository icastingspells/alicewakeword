/* Float32 kernels for this pinned, static OpenWakeWord/EVA graph only. */
#include <math.h>
#include <string.h>
#include <stddef.h>
#include <float.h>
/* Explicit four-lane accumulation maps to NEON on the target ARM CPU. */
typedef float eva_vec4 __attribute__((vector_size(16)));
static float dot(const float *a,const float *b,int n) {
    eva_vec4 s0={0},s1={0},s2={0},s3={0};int k=0;
    for(;k+15<n;k+=16) {
        eva_vec4 a0,a1,a2,a3,b0,b1,b2,b3;
        memcpy(&a0,a+k,16);memcpy(&b0,b+k,16);
        memcpy(&a1,a+k+4,16);memcpy(&b1,b+k+4,16);
        memcpy(&a2,a+k+8,16);memcpy(&b2,b+k+8,16);
        memcpy(&a3,a+k+12,16);memcpy(&b3,b+k+12,16);
        s0+=a0*b0;s1+=a1*b1;s2+=a2*b2;s3+=a3*b3;
    }
    eva_vec4 s=s0+s1+s2+s3;float value=s[0]+s[1]+s[2]+s[3];
    for(;k<n;k++)value+=a[k]*b[k];
    return value;
}
static float activation(float v,int kind) {
    if(kind==1) return fmaxf(0,v);
    if(kind==2) return fminf(1,fmaxf(-1,v));
    if(kind==3) return fminf(6,fmaxf(0,v));
    return v;
}
static float binary_value(float a,float b,int op) {
    switch(op) {case 0:return a+b;case 18:return a*b;case 41:return a-b;
    case 55:return fmaxf(a,b);case 57:return fminf(a,b);case 99:return (a-b)*(a-b);}
    return 0;
}
static void binary(const float*a,const float*b,float*out,int op,int an,int bn,int n,
                   const int*d,const int*as,const int*bs,int act) {
    if(an==n&&bn==n) {for(int i=0;i<n;i++)out[i]=activation(binary_value(a[i],b[i],op),act);return;}
    if(an==1) {for(int i=0;i<n;i++)out[i]=activation(binary_value(a[0],b[i],op),act);return;}
    if(bn==1) {for(int i=0;i<n;i++)out[i]=activation(binary_value(a[i],b[0],op),act);return;}
    int i=0;
    for(int x=0;x<d[0];x++)for(int y=0;y<d[1];y++)for(int z=0;z<d[2];z++)for(int w=0;w<d[3];w++)
        out[i++]=activation(binary_value(a[x*as[0]+y*as[1]+z*as[2]+w*as[3]],b[x*bs[0]+y*bs[1]+z*bs[2]+w*bs[3]],op),act);
}
static void unary(const float*a,float*out,int n,int op,float alpha) {
    for(int i=0;i<n;i++) {float x=a[i];switch(op) {
    case 14:out[i]=1.0f/(1.0f+expf(-x));break;
    case 73:out[i]=logf(x);break;
    case 76:out[i]=1.0f/sqrtf(x);break;
    case 98:out[i]=x>=0?x:alpha*x;break;
    }}
}
static void dense(const float*x,const float*w,const float*b,float*y,int rows,int inputs,int outputs,int act) {
    for(int r=0;r<rows;r++)for(int o=0;o<outputs;o++) {
        float v=b?b[o]:0;
        const float *xp=x+r*inputs,*wp=w+o*inputs;
        v+=dot(xp,wp,inputs);
        y[r*outputs+o]=activation(v,act);
    }
}
static void conv_range(const float*x,const float*w,const float*b,float*y,const int*i,const int*o,
                 int kh,int kw,int sh,int sw,int dh,int dw,int ph,int pw,int act,int begin,int end) {
    int ic=i[3],oc=o[3];
    for(int index=begin;index<end;index++) {
        int c=index%oc,v=(index/oc)%o[2],h=(index/(oc*o[2]))%o[1],n=index/(oc*o[2]*o[1]);
        float sum=b?b[c]:0;
        for(int ky=0;ky<kh;ky++) {int iy=h*sh+ky*dh-ph;if(iy<0||iy>=i[1])continue;
            int first=v*sw-pw;
            if(dw==1&&first>=0&&first+kw<=i[2]) {
                sum+=dot(x+((n*i[1]+iy)*i[2]+first)*ic,w+(c*kh+ky)*kw*ic,kw*ic);
                continue;
            }
            for(int kx=0;kx<kw;kx++) {int ix=v*sw+kx*dw-pw;if(ix<0||ix>=i[2])continue;
                const float *xp=x+((n*i[1]+iy)*i[2]+ix)*ic;
                const float *wp=w+((c*kh+ky)*kw+kx)*ic;
                sum+=dot(xp,wp,ic);
            }
        }
        y[((n*o[1]+h)*o[2]+v)*oc+c]=activation(sum,act);
    }
}
#if defined(__aarch64__) && !defined(_WIN32)
#include <pthread.h>
/* Two persistent 64 KiB worker stacks; the calling thread processes the third part. */
struct conv_task {const float*x,*w,*b;float*y;const int*i,*o;int kh,kw,sh,sw,dh,dw,ph,pw,act,total,begin;};
static struct conv_task conv_task;
static pthread_mutex_t conv_mutex=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t conv_start=PTHREAD_COND_INITIALIZER,conv_done=PTHREAD_COND_INITIALIZER;
static pthread_once_t conv_once=PTHREAD_ONCE_INIT;
static unsigned conv_generation,conv_completed;
static int conv_workers;
static void conv_part(int part,int parts) {
    struct conv_task*t=&conv_task;
    conv_range(t->x,t->w,t->b,t->y,t->i,t->o,t->kh,t->kw,t->sh,t->sw,t->dh,t->dw,t->ph,t->pw,t->act,t->begin+t->total*part/parts,t->begin+t->total*(part+1)/parts);
}
static void *conv_worker(void *id) {
    unsigned generation=0;
    pthread_mutex_lock(&conv_mutex);
    for(;;) {
        while(generation==conv_generation)pthread_cond_wait(&conv_start,&conv_mutex);
        generation=conv_generation;
        int parts=conv_workers+1;
        pthread_mutex_unlock(&conv_mutex);
        conv_part((int)(size_t)id,parts);
        pthread_mutex_lock(&conv_mutex);
        conv_completed++;pthread_cond_signal(&conv_done);
    }
    return NULL;
}
static void conv_init_workers(void) {
    pthread_attr_t attr;pthread_attr_init(&attr);pthread_attr_setstacksize(&attr,65536);
    for(int id=1;id<=2;id++) {
        pthread_t thread;
        if(pthread_create(&thread,&attr,conv_worker,(void*)(size_t)id))break;
        conv_workers++;pthread_detach(thread);
    }
    pthread_attr_destroy(&attr);
}
#endif
static void conv_span(const float*x,const float*w,const float*b,float*y,const int*i,const int*o,
                 int kh,int kw,int sh,int sw,int dh,int dw,int ph,int pw,int act,int begin) {
    int total=o[0]*o[1]*o[2]*o[3];
#if defined(__aarch64__) && !defined(_WIN32)
    pthread_once(&conv_once,conv_init_workers);
    if(conv_workers&&total-begin>=4096) {
        pthread_mutex_lock(&conv_mutex);
        conv_task=(struct conv_task){x,w,b,y,i,o,kh,kw,sh,sw,dh,dw,ph,pw,act,total-begin,begin};
        conv_completed=0;conv_generation++;pthread_cond_broadcast(&conv_start);
        pthread_mutex_unlock(&conv_mutex);
        conv_part(0,conv_workers+1);
        pthread_mutex_lock(&conv_mutex);
        while(conv_completed<(unsigned)conv_workers)pthread_cond_wait(&conv_done,&conv_mutex);
        pthread_mutex_unlock(&conv_mutex);
        return;
    }
#endif
    conv_range(x,w,b,y,i,o,kh,kw,sh,sw,dh,dw,ph,pw,act,begin,total);
}
static void conv(const float*x,const float*w,const float*b,float*y,const int*i,const int*o,
                 int kh,int kw,int sh,int sw,int dh,int dw,int ph,int pw,int act) {
    conv_span(x,w,b,y,i,o,kh,kw,sh,sw,dh,dw,ph,pw,act,0);
}
static void conv_cached(const float*x,const float*w,const float*b,float*y,const int*i,const int*o,
                 int kh,int kw,int sh,int sw,int dh,int dw,int ph,int pw,int act,float*cache,int shift,int valid) {
    int total=o[0]*o[1]*o[2]*o[3],row=o[2]*o[3];
    int retained=valid?(o[1]-shift)*row:0;
    if(retained>0) {
        memmove(cache,cache+shift*row,(size_t)retained*sizeof(float));
        memcpy(y,cache,(size_t)retained*sizeof(float));
    }
    conv_span(x,w,b,y,i,o,kh,kw,sh,sw,dh,dw,ph,pw,act,retained);
    memcpy(cache+retained,y+retained,(size_t)(total-retained)*sizeof(float));
}
static void pool(const float*x,float*y,const int*i,const int*o,int kh,int kw,int sh,int sw,int ph,int pw,int act) {
    for(int n=0;n<o[0];n++)for(int h=0;h<o[1];h++)for(int v=0;v<o[2];v++)for(int c=0;c<o[3];c++) {
        float value=-FLT_MAX;
        for(int ky=0;ky<kh;ky++)for(int kx=0;kx<kw;kx++) {
            int iy=h*sh+ky-ph,ix=v*sw+kx-pw;
            if(iy>=0&&iy<i[1]&&ix>=0&&ix<i[2])value=fmaxf(value,x[((n*i[1]+iy)*i[2]+ix)*i[3]+c]);
        }
        y[((n*o[1]+h)*o[2]+v)*o[3]+c]=activation(value,act);
    }
}
static void transpose(const float*x,float*y,int n,const int*od,const int*source_strides) {
    int k=0;
    for(int a=0;a<od[0];a++)for(int b=0;b<od[1];b++)for(int c=0;c<od[2];c++)for(int d=0;d<od[3];d++)
        y[k++]=x[a*source_strides[0]+b*source_strides[1]+c*source_strides[2]+d*source_strides[3]];
}
static void reduce(const float*x,float*y,int n,int outn,const int*d,const int*strides,int maximum) {
    for(int i=0;i<outn;i++)y[i]=maximum?-FLT_MAX:0;
    int k=0;
    for(int a=0;a<d[0];a++)for(int b=0;b<d[1];b++)for(int c=0;c<d[2];c++)for(int e=0;e<d[3];e++) {
        int oi=a*strides[0]+b*strides[1]+c*strides[2]+e*strides[3];
        if(maximum)y[oi]=fmaxf(y[oi],x[k++]);else y[oi]+=x[k++];
    }
    if(!maximum)for(int i=0;i<outn;i++)y[i]/=(float)(n/outn);
}
static void pad(const float*x,float*y,const int*i,const int*o,const int*before) {
    memset(y,0,(size_t)o[0]*o[1]*o[2]*o[3]*sizeof(float));
    int k=0;
    for(int a=0;a<i[0];a++)for(int b=0;b<i[1];b++)for(int c=0;c<i[2];c++)for(int d=0;d<i[3];d++)
        y[(((a+before[0])*o[1]+b+before[1])*o[2]+c+before[2])*o[3]+d+before[3]]=x[k++];
}
static void matmul(const float*a,const float*b,float*y,int batches,int m,int k,int n) {
    for(int batch=0;batch<batches;batch++)for(int i=0;i<m;i++)for(int j=0;j<n;j++) {
        float sum=0;
        for(int t=0;t<k;t++)sum+=a[(batch*m+i)*k+t]*b[t*n+j];
        y[(batch*m+i)*n+j]=sum;
    }
}
