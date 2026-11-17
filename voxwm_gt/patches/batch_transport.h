// Bounded, lossless per-tick RAW transport. No physics, sampling or token IDs here.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <cerrno>
#include <sys/socket.h>
namespace voxwm_batch {
inline constexpr const char *version="voxwm-advance-batch-v1";
inline constexpr uint32_t cells=48*48*48, raw_bytes=cells*4, max_ticks=48;
inline void u32(std::vector<char> &out,uint32_t value) {
    for(unsigned i=0;i<4;++i) out.push_back(char(value>>(8*i)));
}
inline void u64(std::vector<char> &out,uint64_t value) {
    for(unsigned i=0;i<8;++i) out.push_back(char(value>>(8*i)));
}
struct Clock {uint64_t tick;float dt;};
struct Buffer {
    // Server writes then posts completion; Client reads after sem_wait. The next
    // server tick is requested only after Client::step returns (existing barrier).
    unsigned expected=0,captured=0;
    std::vector<char> payload,previous;
    std::vector<Clock> clocks;
    void begin(unsigned n) {
        if(expected || n<1 || n>max_ticks) throw std::runtime_error("Invalid/overlapping native batch");
        expected=n;captured=0;payload.assign({'V','W','B','1'});
        u32(payload,n);u32(payload,cells);previous.clear();clocks.clear();
    }
    uint32_t push(const char *raw,size_t size) {
        if(!expected || captured>=expected || size!=raw_bytes) throw std::runtime_error("Invalid native batch capture");
        uint32_t changed=0;
        if(!captured) {payload.insert(payload.end(),raw,raw+size);changed=cells;}
        else {
            const size_t at=payload.size();u32(payload,0);
            for(uint32_t i=0;i<cells;++i) if(std::memcmp(raw+4*i,previous.data()+4*i,4)) {
                ++changed;u32(payload,i);payload.insert(payload.end(),raw+4*i,raw+4*i+4);
            }
            for(unsigned i=0;i<4;++i) payload[at+i]=char(changed>>(8*i));
        }
        previous.assign(raw,raw+size);++captured;return changed;
    }
    bool client_complete(uint64_t tick,float dt) {
        if(!expected) return false;
        if(clocks.size()+1!=captured) throw std::runtime_error("Batch lost its server/client completion barrier");
        clocks.push_back({tick,dt});return captured<expected;
    }
    std::vector<char> finish() {
        if(!expected || captured!=expected || clocks.size()!=expected) throw std::runtime_error("Incomplete native batch");
        for(const auto &c:clocks) {
            u64(payload,c.tick);uint32_t bits;std::memcpy(&bits,&c.dt,4);u32(payload,bits);
        }
        std::vector<char> out;out.swap(payload);std::vector<char>().swap(previous);
        clocks.clear();expected=captured=0;return out;
    }
};
inline Buffer buffer;
inline void send_all(int fd,const char *data,size_t size) {
    while(size) {
        const auto n=::send(fd,data,size,0);
        if(n<0 && errno==EINTR) continue;
        if(n<=0) throw std::runtime_error("Native batch socket closed");
        data+=n;size-=n;
    }
}
inline void send_ready(int fd) {
    if(!buffer.expected) return;
    auto data=buffer.finish();std::vector<char> length;u64(length,data.size());
    send_all(fd,length.data(),length.size());send_all(fd,data.data(),data.size());
}
} // namespace voxwm_batch
