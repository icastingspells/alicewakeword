#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

/* ALSA ABI declarations; the firmware owns every PCM handle and hardware open. */
typedef void snd_pcm_t;
typedef void snd_pcm_hw_params_t;
typedef long frames_t;
static frames_t (*real_readi)(snd_pcm_t*,void*,unsigned long);
static int (*real_close)(snd_pcm_t*);
static int (*stream)(snd_pcm_t*);
static int (*params_malloc)(snd_pcm_hw_params_t**);
static void (*params_free)(snd_pcm_hw_params_t*);
static int (*params_current)(snd_pcm_t*,snd_pcm_hw_params_t*);
static int (*get_rate)(const snd_pcm_hw_params_t*,unsigned*,int*);
static int (*get_channels)(const snd_pcm_hw_params_t*,unsigned*);
static int (*get_format)(const snd_pcm_hw_params_t*,int*);
static snd_pcm_t *watched;
static unsigned rate,channels;
static int usable,fd=-1,initialized;
static uint32_t sequence;
static struct sockaddr_un address;
struct packet {uint32_t magic,sequence,rate,frames;uint16_t channels,sample_bytes;};
/* One process-wide buffer: TLS would multiply this by every Quasar thread. */
static unsigned char payload[sizeof(struct packet)+1536*8*2];
static int sending;

static void init(void) {
    if(initialized)return;
    real_readi=dlsym(RTLD_NEXT,"snd_pcm_readi");real_close=dlsym(RTLD_NEXT,"snd_pcm_close");
    stream=dlsym(RTLD_NEXT,"snd_pcm_stream");params_malloc=dlsym(RTLD_NEXT,"snd_pcm_hw_params_malloc");
    params_free=dlsym(RTLD_NEXT,"snd_pcm_hw_params_free");params_current=dlsym(RTLD_NEXT,"snd_pcm_hw_params_current");
    get_rate=dlsym(RTLD_NEXT,"snd_pcm_hw_params_get_rate");get_channels=dlsym(RTLD_NEXT,"snd_pcm_hw_params_get_channels");
    get_format=dlsym(RTLD_NEXT,"snd_pcm_hw_params_get_format");
    const char *path=getenv("EVA_SOCKET");
    if(path&&strlen(path)<sizeof address.sun_path) {
        address.sun_family=AF_UNIX;strcpy(address.sun_path,path);
        fd=socket(AF_UNIX,SOCK_DGRAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0);
    }
    initialized=1;
}
__attribute__((constructor)) static void construct(void) {init();}

frames_t snd_pcm_readi(snd_pcm_t *pcm,void *buffer,unsigned long requested) {
    if(!real_readi) {initialized=0;init();}
    if(!real_readi)return -ENOSYS;
    frames_t count=real_readi(pcm,buffer,requested);
    int saved=errno;
    if(count<=0||fd<0||!stream||!params_malloc||!params_free||!params_current||!get_rate||!get_channels||!get_format)goto done;
    if(!watched) {
        if(stream(pcm)!=1)goto done;
        snd_pcm_hw_params_t *params=NULL;unsigned r=0,c=0;int format=-1,direction=0;
        if(params_malloc(&params)<0||!params)goto done;
        int ok=params_current(pcm,params)>=0&&get_rate(params,&r,&direction)>=0&&get_channels(params,&c)>=0&&get_format(params,&format)>=0;
        params_free(params);
        /* ALSA S16_LE=2; unsupported formats remain entirely untouched. */
        if(!ok||format!=2||(r!=16000&&r!=48000)||c<1||c>8)goto done;
        snd_pcm_t *empty=NULL;
        if(__atomic_compare_exchange_n(&watched,&empty,pcm,0,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST)) {
            rate=r;channels=c;sequence=0;__atomic_store_n(&usable,1,__ATOMIC_RELEASE);
        }
    }
    if(pcm!=watched||!__atomic_load_n(&usable,__ATOMIC_ACQUIRE))goto done;
    if(__atomic_exchange_n(&sending,1,__ATOMIC_ACQUIRE)) {
        __atomic_fetch_add(&sequence,(uint32_t)(((unsigned long)count+1535)/1536),__ATOMIC_RELAXED);
        goto done;
    }
    for(unsigned long offset=0;offset<(unsigned long)count;) {
        unsigned n=(unsigned long)count-offset>1536?1536:(unsigned)((unsigned long)count-offset);
        struct packet header={0x31415645,__atomic_fetch_add(&sequence,1,__ATOMIC_RELAXED),rate,n,(uint16_t)channels,2};
        memcpy(payload,&header,sizeof header);
        memcpy(payload+sizeof header,(const unsigned char*)buffer+offset*channels*2,n*channels*2);
        /* No retries, waits, files, capture opens, or changes to original samples. */
        sendto(fd,payload,sizeof header+n*channels*2,MSG_DONTWAIT|MSG_NOSIGNAL,(struct sockaddr*)&address,sizeof address);
        offset+=n;
    }
    __atomic_store_n(&sending,0,__ATOMIC_RELEASE);
done:
    errno=saved;return count;
}
int snd_pcm_close(snd_pcm_t *pcm) {
    if(!real_close){initialized=0;init();}
    if(pcm==watched) {__atomic_store_n(&usable,0,__ATOMIC_RELEASE);__atomic_store_n(&watched,NULL,__ATOMIC_SEQ_CST);}
    return real_close?real_close(pcm):-ENOSYS;
}
