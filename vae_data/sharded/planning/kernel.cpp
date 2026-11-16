// Shared-graph sparse 64-source bitset wavefront. No native world states are generated.
#include <algorithm>
#include <cstdint>
#include <vector>
#include <deque>
#include <cmath>
#include <exception>
extern "C" {
// CSR edge type: 0=lateral (-1 range), 1=down (range reset), 2=fire.
int vw_wave(int v,int n,int range,int viscosity,int horizon,int batch,
 const int32_t* ptr,const int32_t* edges,const uint8_t* etype,const uint8_t* center,
 const uint8_t* reverse_ok,const int32_t* sources,const int32_t* cptr,
 const int32_t* products,const uint8_t* partner_time,int32_t* stats,
 uint8_t* details,uint64_t* operations,uint64_t* workspace_bytes) {
 try {
  const bool fire=range<0;const int levels=fire?1:range+2;const int states=v*levels;
  const bool cooling=cptr!=nullptr;const int cols=7;
  std::vector<uint64_t> seen(states),front(states),next(states),voxel(v),product_seen(cooling?v:0);
  std::vector<uint64_t> pending(cooling?(horizon+2)*v:0);
  std::vector<int> active,following,ts,tv,tp,tpend;
  std::vector<std::vector<int>> scheduled(horizon+2);
  *operations=0;*workspace_bytes=(uint64_t)(3LL*states+v+(cooling?v+(horizon+2LL)*v:0))*8;
  std::fill(stats,stats+(int64_t)n*cols,0);
  if(details)std::fill(details,details+(int64_t)n*v,255);
  auto bits_each=[&](uint64_t b,auto fn){while(b){int bit=__builtin_ctzll(b);fn(bit);b&=b-1;}};
  for(int base=0;base<n;base+=batch) {
   active.clear();following.clear();ts.clear();tv.clear();tp.clear();tpend.clear();
   for(auto &bucket:scheduled)bucket.clear();
   int count=std::min(batch,n-base);
   for(int bit=0;bit<count;bit++) {
    int row=base+bit,p=sources[row];stats[row*cols+3]=255;
    if(!reverse_ok[p])continue;
    int s=(fire?0:range+1)*v+p;uint64_t b=1ULL<<bit;
    if(!seen[s])ts.push_back(s);seen[s]|=b;
    if(!front[s])active.push_back(s);front[s]|=b;
    if(!voxel[p])tv.push_back(p);voxel[p]|=b;
    stats[row*cols]++;stats[row*cols+1]+=(center[p]&1)?1:0;stats[row*cols+6]+=(center[p]&2)?1:0;
    if(details)details[(int64_t)row*v+p]=0;
   }
   for(int tick=0;tick<=horizon;tick++) {
    for(int s:active) {
     uint64_t b=front[s];front[s]=0;int r=s/v,p=s%v;(*operations)++;
     uint64_t fresh=b&~voxel[p];
     if(fresh) {
      if(!voxel[p])tv.push_back(p);voxel[p]|=fresh;
      bits_each(fresh,[&](int bit){int row=base+bit;stats[row*cols]++;
       if(details)details[(int64_t)row*v+p]=(uint8_t)tick;
       if(center[p]&2)stats[row*cols+6]++;
       if(center[p]&1){stats[row*cols+1]++;stats[row*cols+2]|=1<<tick;stats[row*cols+3]=std::min(stats[row*cols+3],tick);
        // Level changes are a POTENTIAL for flow-fed viscous lava, never an exact promise.
        if(!fire&&viscosity>1&&tick>=2)for(int t=tick+1;t<=std::min(horizon,tick+viscosity-1);t++)stats[row*cols+2]|=1<<t;
       }
      });
     }
     if(cooling)for(int e=cptr[p];e<cptr[p+1];e++) {
      int t=std::max(tick,(int)partner_time[e])+1,q=products[e];if(t>horizon)continue;
      int k=t*v+q;if(!pending[k]){tpend.push_back(k);scheduled[t].push_back(q);}pending[k]|=b;
     }
     if(tick==horizon)continue;
     for(int e=ptr[p];e<ptr[p+1];e++) {
      int q=edges[e],rr=0;
      if(!fire){if(etype[e]==1)rr=range;else {if(r==0)continue;rr=(r==range+1?range-1:r-1);}}
      // type=3 edges are source-only lateral emission above an open drop.
      if(etype[e]==3&&!fire&&r!=range+1)continue;
      int dest=rr*v+q;uint64_t add=b&~seen[dest];if(!add)continue;
      if(!seen[dest])ts.push_back(dest);seen[dest]|=add;
      if(!next[dest])following.push_back(dest);next[dest]|=add;
     }
    }
    if(cooling)for(int q:scheduled[tick]) {
     uint64_t b=pending[tick*v+q]&~product_seen[q];if(!b)continue;
     if(!product_seen[q])tp.push_back(q);product_seen[q]|=b;
     bits_each(b,[&](int bit){int row=base+bit;stats[row*cols+4]++;stats[row*cols+5]|=1<<tick;stats[row*cols+2]|=1<<tick;stats[row*cols+3]=std::min(stats[row*cols+3],tick);});
    }
    active.swap(following);following.clear();front.swap(next);
   }
   for(int s:ts)seen[s]=front[s]=next[s]=0;
   for(int p:tv)voxel[p]=0;for(int p:tp)product_seen[p]=0;for(int p:tpend)pending[p]=0;
  }
  return 0;
 }catch(...){return 1;}
}
}
