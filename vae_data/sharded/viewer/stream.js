// Reused ABV1 decoding with actual native time indexing. No renderer dependency.
export function timeIndex(rows,time){let lo=0,hi=rows.length;while(lo<hi){const m=(lo+hi)>>1;if(rows[m].metadata.sim_time<=time)lo=m+1;else hi=m;}return Math.max(0,lo-1);}
export function worldToLocal(position,origin){return[position[0]-origin[0]+.5-24,position[2]-origin[2]+.5-24,-(position[1]-origin[1]+.5-24)];}
export class VoxelStream{
 constructor(buffer){const d=new DataView(buffer);if(new TextDecoder().decode(new Uint8Array(buffer,0,4))!=='ABV1')throw Error('Invalid native voxel export');this.frames=d.getUint32(4,true);this.shape=[8,12,16].map(o=>d.getUint32(o,true));this.size=this.shape.reduce((a,b)=>a*b,1);this.initial=new Uint16Array(buffer,20,this.size);let p=20+this.size*2;this.ptr=new Uint32Array(buffer,p,this.frames+1);p+=(this.frames+1)*4;const n=this.ptr[this.frames];this.indices=new Uint32Array(buffer,p,n);this.values=new Uint16Array(buffer,p+n*4,n);this.current=this.initial.slice();this.index=0;}
 frame(t){if(t<0||t>=this.frames)throw Error('Frame index outside native trace');if(t<this.index){this.current=this.initial.slice();this.index=0;}while(this.index<t){this.index++;for(let j=this.ptr[this.index];j<this.ptr[this.index+1];j++)this.current[this.indices[j]]=this.values[j];}return this.current;}
}
