// CPU codec/barrier fixture, not a native physics run.
#include "batch_transport.h"
#include <iostream>
#include <cassert>
int main() {
    voxwm_batch::Buffer b;std::vector<char> raw(voxwm_batch::raw_bytes,0);
    bool rejected=false;try{b.begin(49);}catch(const std::runtime_error &){rejected=true;}
    assert(rejected);b.begin(48);
    for(unsigned t=1;t<=48;++t) {
        // Two RAW records change, including light-only and high param2 bits.
        raw[4*123]=char(t);raw[4*123+1]=1;raw[4*123+3]=char(128+t);
        raw[4*901+2]=char(t);
        assert(b.push(raw.data(),raw.size())==(t==1?voxwm_batch::cells:2));
        assert(b.client_complete(100+t,1.0f/24.0f)==(t<48));
    }
    const auto payload=b.finish();assert(!b.expected);
    std::cout.write(payload.data(),payload.size());
}
