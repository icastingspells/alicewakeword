#define _POSIX_C_SOURCE 200809L
#include "models.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifndef _WIN32
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <poll.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/input.h>
#endif

static float arena[MODEL_ARENA_BYTES/4] __attribute__((aligned(16)));
static float raw[1760],mel[76*32],embedding[16*96],mel_out[256];
static unsigned frames,hits;
static float max_score;
static double last_hit=-1000;
static int armed=1,low_count,scores;
static const char *activate_path;
static void trim_log(void) {
#ifndef _WIN32
    struct stat info;
    if(fstat(fileno(stdout),&info)==0&&S_ISREG(info.st_mode)&&info.st_size>65536) {
        fflush(stdout);if(ftruncate(fileno(stdout),0)==0)fseek(stdout,0,SEEK_SET);
    }
#endif
}
static void reset_stream(void) {
    embedding_stream_reset();
    memset(raw,0,sizeof raw);memset(embedding,0,sizeof embedding);
    for(int i=0;i<76*32;i++)mel[i]=1;
    frames=0;armed=1;low_count=0;last_hit=-1000;
}
static void activate(void) {
#ifndef _WIN32
    if(activate_path) {
        pid_t child=fork();
        if(child==0) {execl(activate_path,activate_path,(char*)NULL);_exit(127);}
    }
#endif
}
static float process(const int16_t *pcm) {
    memmove(raw,raw+1280,480*sizeof(float));
    for(int i=0;i<1280;i++)raw[480+i]=pcm[i];
    mel_eval(raw,mel_out,arena);
    memmove(mel,mel+256,(76*32-256)*sizeof(float));
    for(int i=0;i<256;i++)mel[76*32-256+i]=mel_out[i]/10.0f+2.0f;
    memmove(embedding,embedding+96,15*96*sizeof(float));
    embedding_stream_eval(mel,embedding+15*96,arena);
    float score=0;eva_eval(embedding,&score,arena);
    frames++;
    if(!isfinite(score)||score<0||score>1) {fprintf(stderr,"EVA_ERROR nonfinite_score\n");exit(3);}
    if(frames<=20)score=0;
    double t=frames*0.08;
    if(score>max_score)max_score=score;
    if(scores)printf("SCORE %.3f %.9g\n",t,score);
    if(score<MODEL_RELEASE_THRESHOLD) {low_count++;if(low_count>=3&&t-last_hit>=2)armed=1;}else low_count=0;
    if(armed&&score>=MODEL_THRESHOLD&&t-last_hit>=2) {
        hits++;armed=0;last_hit=t;
        printf("EVA_DETECTED time=%.3f score=%.9g mode=%s\n",t,score,activate_path?"active":"observe");
        activate();
        trim_log();
    }
    return score;
}
static uint32_t le32(const unsigned char*p) {return p[0]|p[1]<<8|p[2]<<16|(uint32_t)p[3]<<24;}
static unsigned le16(const unsigned char*p) {return p[0]|p[1]<<8;}
static int wav(const char *path) {
    FILE*f=fopen(path,"rb");if(!f)return 2;
    unsigned char header[12];if(fread(header,1,12,f)!=12||memcmp(header,"RIFF",4)||memcmp(header+8,"WAVE",4))return 2;
    uint32_t remaining=0;int good=0;
    for(;;) {
        unsigned char ch[8];if(fread(ch,1,8,f)!=8)return 2;
        uint32_t n=le32(ch+4);if(n>32000000)return 2;
        if(!memcmp(ch,"fmt ",4)) {
            unsigned char fmt[16];if(n<16||fread(fmt,1,16,f)!=16)return 2;
            good=le16(fmt)==1&&le16(fmt+2)==1&&le32(fmt+4)==16000&&le16(fmt+14)==16;
            if(fseek(f,n-16+(n&1),SEEK_CUR))return 2;
        } else if(!memcmp(ch,"data",4)) {remaining=n;break;}
        else if(fseek(f,n+(n&1),SEEK_CUR))return 2;
    }
    if(!good||(remaining&1))return 2;
    while(remaining) {
        int16_t pcm[1280]={0};size_t bytes=remaining>sizeof pcm?sizeof pcm:remaining;
        if(fread(pcm,1,bytes,f)!=bytes)return 2;
        process(pcm);remaining-=(uint32_t)bytes;
    }
    fclose(f);return 0;
}
#ifndef _WIN32
struct packet {uint32_t magic,sequence,rate,frames;uint16_t channels,sample_bytes;};
static volatile sig_atomic_t stop;
static void stopping(int n) {(void)n;stop=1;}
/* 63-tap low-pass, normalized at startup; evaluated only for each third sample. */
static float coefficients[63],history[63];
static unsigned history_pos,phase;
static void filter_init(void) {
    float sum=0;
    for(int i=0;i<63;i++) {int k=i-31;float value=k?sinf(2*3.14159265358979323846f*0.145f*k)/(3.14159265358979323846f*k):0.29f;
        coefficients[i]=value*(0.54f-0.46f*cosf(2*3.14159265358979323846f*i/62));sum+=coefficients[i];}
    for(int i=0;i<63;i++)coefficients[i]/=sum;
}
static int receive_pcm(const char *path) {
    if(strlen(path)>=sizeof(((struct sockaddr_un*)0)->sun_path))return 2;
    struct stat st;if(lstat(path,&st)==0) {fputs("EVA_ERROR socket_exists\n",stderr);return 2;}
    /* Prime the exact all-ones mel history before accepting PCM packets. */
    embedding_stream_eval(mel,mel_out,arena);
    int fd=socket(AF_UNIX,SOCK_DGRAM,0);if(fd<0)return 2;
    struct sockaddr_un address={.sun_family=AF_UNIX};strcpy(address.sun_path,path);
    if(bind(fd,(struct sockaddr*)&address,sizeof address)||chmod(path,0600))return 2;
    signal(SIGTERM,stopping);signal(SIGINT,stopping);signal(SIGCHLD,SIG_IGN);
    filter_init();puts("EVA_READY");
    unsigned char data[sizeof(struct packet)+1536*8*2];
    int16_t pcm[1280];unsigned have=0,expected=0;int started=0;
    unsigned packets=0,gaps=0;double max_ms=0;
    while(!stop) {
        struct pollfd pollfd={.fd=fd,.events=POLLIN};if(poll(&pollfd,1,1000)<=0)continue;
        ssize_t bytes=recv(fd,data,sizeof data,MSG_TRUNC);
        if(bytes<(ssize_t)sizeof(struct packet)||bytes>(ssize_t)sizeof data)continue;
        struct packet p;memcpy(&p,data,sizeof p);
        if(p.magic!=0x31415645||p.sample_bytes!=2||p.channels<1||p.channels>8||p.frames<1||p.frames>1536||
           (p.rate!=16000&&p.rate!=48000)||bytes!=(ssize_t)(sizeof p+p.frames*p.channels*2))continue;
        if(started&&p.sequence!=expected) {gaps++;have=0;history_pos=phase=0;memset(history,0,sizeof history);reset_stream();}
        expected=p.sequence+1;
        if(!started)printf("EVA_PCM_STARTED rate=%u channels=%u\n",p.rate,p.channels);
        started=1;packets++;
        const int16_t*input=(const int16_t*)(data+sizeof p);
        for(unsigned i=0;i<p.frames;i++) {
            int16_t sample=input[i*p.channels];
            if(p.rate==48000) {
                history[history_pos]=sample;history_pos=(history_pos+1)%63;
                if(++phase<3)continue;phase=0;
                float value=0;for(int k=0;k<63;k++)value+=coefficients[k]*history[(history_pos+62-k)%63];
                if(value>32767)value=32767;if(value< -32768)value=-32768;sample=(int16_t)lrintf(value);
            }
            pcm[have++]=sample;
            if(have==1280) {
                struct timespec a,b;clock_gettime(CLOCK_MONOTONIC,&a);process(pcm);clock_gettime(CLOCK_MONOTONIC,&b);
                double ms=(b.tv_sec-a.tv_sec)*1000+(b.tv_nsec-a.tv_nsec)/1e6;if(ms>max_ms)max_ms=ms;have=0;
                if(frames%375==0) {printf("EVA_HEALTH packets=%u gaps=%u max_frame_ms=%.3f\n",packets,gaps,max_ms);trim_log();}
            }
        }
    }
    printf("EVA_TRANSPORT packets=%u gaps=%u max_frame_ms=%.3f\n",packets,gaps,max_ms);
    close(fd);unlink(path);return 0;
}
#endif
int main(int argc,char**argv) {
    setvbuf(stdout,NULL,_IOLBF,0);reset_stream();
#ifndef _WIN32
    if(argc==2&&!strcmp(argv[1],"--activate-cli")) {
        signal(SIGCHLD,SIG_DFL);
        int channel[2];if(pipe(channel))return 2;
        pid_t child=fork();if(child<0) {close(channel[0]);close(channel[1]);return 2;}
        if(child==0) {
            dup2(channel[0],STDIN_FILENO);close(channel[0]);close(channel[1]);
            int nullfd=open("/dev/null",O_WRONLY);
            if(nullfd<0)_exit(126);
            dup2(nullfd,STDOUT_FILENO);dup2(nullfd,STDERR_FILENO);close(nullfd);
            setenv("LD_LIBRARY_PATH","/system/vendor/quasar:/system/vendor/quasar/libs",1);
            execl("/system/vendor/quasar/maind","maind","--cli",(char*)NULL);_exit(127);
        }
        close(channel[0]);signal(SIGPIPE,SIG_IGN);
        (void)write(channel[1],"s\n",2);
        struct timespec second={1,0},tick={0,100000000};nanosleep(&second,NULL);
        (void)write(channel[1],"x\n",2);close(channel[1]);
        int status=0,finished=0;
        for(int attempt=0;attempt<40;attempt++) {
            if(waitpid(child,&status,WNOHANG)==child) {finished=1;break;}
            nanosleep(&tick,NULL);
        }
        if(!finished) {
            kill(child,SIGTERM);nanosleep(&tick,NULL);
            if(waitpid(child,&status,WNOHANG)!=child) {kill(child,SIGKILL);waitpid(child,&status,0);}
        }
        int code=WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);
        printf("EVA_CLI_EXIT code=%d timed_out=%d\n",code,!finished);
        return code;
    }
    if(argc==2&&!strcmp(argv[1],"--mic-state")) {
        for(int event=0;event<16;event++) {
            char path[64],name[64]={0};unsigned char supported[8]={0},state[8]={0};
            snprintf(path,sizeof path,"/dev/input/event%d",event);
            int fd=open(path,O_RDONLY|O_NONBLOCK|O_CLOEXEC);if(fd<0)continue;
            int valid=ioctl(fd,EVIOCGNAME(sizeof name),name)>=0&&!strcmp(name,"gpio-keys")&&
                ioctl(fd,EVIOCGBIT(EV_SW,sizeof supported),supported)>=0&&(supported[14/8]&(1u<<(14%8)))&&
                ioctl(fd,EVIOCGSW(sizeof state),state)>=0;
            close(fd);
            if(valid) {printf("MICS_MUTED=%d\n",!!(state[14/8]&(1u<<(14%8))));return 0;}
        }
        fputs("EVA_ERROR mic_state_unavailable\n",stderr);return 2;
    }
    if(argc==2&&!strcmp(argv[1],"--publish-link")) {
        struct stat info;char target[80];
        if(getuid()!=0||lstat("/data/eva/current.new",&info)||!S_ISLNK(info.st_mode))return 2;
        ssize_t n=readlink("/data/eva/current.new",target,sizeof target-1);
        if(n!=25||memcmp(target,"releases/",9))return 2;
        for(int i=9;i<25;i++)if(!((target[i]>='0'&&target[i]<='9')||(target[i]>='a'&&target[i]<='f')))return 2;
        if(lstat("/data/eva/current",&info)==0&&!S_ISLNK(info.st_mode))return 2;
        return rename("/data/eva/current.new","/data/eva/current")?2:0;
    }
#endif
    const char *input=NULL,*socket_path=NULL;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--self-test")) {
            int16_t zero[1280]={0};float score=process(zero);
            printf("EVA_SELF_TEST_OK arena_bytes=%d score=%.9g\n",MODEL_ARENA_BYTES,score);return 0;
        } else if(!strcmp(argv[i],"--wav")&&i+1<argc)input=argv[++i];
        else if(!strcmp(argv[i],"--listen")&&i+1<argc)socket_path=argv[++i];
        else if(!strcmp(argv[i],"--activate")&&i+1<argc)activate_path=argv[++i];
        else if(!strcmp(argv[i],"--scores"))scores=1;
        else {fputs("Usage: eva-native --wav FILE [--scores] | --listen SOCKET [--activate SCRIPT] | --self-test\n",stderr);return 2;}
    }
    int result=2;
    if(input)result=wav(input);
#ifndef _WIN32
    else if(socket_path)result=receive_pcm(socket_path);
    struct rusage usage;if(getrusage(RUSAGE_SELF,&usage)==0)printf("EVA_MEMORY max_rss_kb=%ld\n",usage.ru_maxrss);
#endif
    printf("EVA_RESULT exit=%d frames=%u hits=%u max_score=%.9g\n",result,frames,hits,max_score);
    return result;
}
