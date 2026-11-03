// PERSIST camera: world-to-CV-camera rotation (first two ROWS), centre-referenced xyz / 48,
// horizontal FOV in radians. Voxel axes are ENU; Three.js axes here are (E, U, -N).
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>{const n=Math.hypot(...a);if(!Number.isFinite(n)||n<1e-8)throw Error('退化的相机旋转');return a.map(x=>x/n);};
export const enuToViewer=([e,n,u])=>[e,u,-n];

export function cropOrigin(view,t,shape){
  // Read the observed crop. Player/camera motion may precede a voxel-window move.
  // Legacy origin is displayed as recorded; it is not a repaired capture.
  return view.crop_origin?.[t] ?? view.origin?.[t] ?? view.obs_voxel_center[t].map((c,i)=>c-shape[i]/2-.5);
}

export function cameraPose(view,t,shape,aspect,{groundTruth=false,referenceOrigin=null}={}){
  const c=view?.camera?.[t];
  if(!c)return null;
  if(c.length!==10||!c.every(Number.isFinite)||!(aspect>0)||!(c[9]>0&&c[9]<Math.PI))throw Error('无效的10D Camera / FOV');
  // Same Gram-Schmidt as official rotation_6d_to_matrix; rows = right, down, forward.
  const right=unit(c.slice(0,3)),a2=c.slice(3,6),projection=dot(right,a2);
  const down=unit(a2.map((x,i)=>x-projection*right[i])),forward=cross(right,down),up=down.map(x=>-x);
  const origin=groundTruth?cropOrigin(view,t,shape):referenceOrigin;
  // Physical display origin = observed crop corner + half grid = voxel center - .5.
  // Model xyz already refers to this point. It must NOT lose another half grid.
  const reference=origin?.map((x,i)=>x+shape[i]/2);
  const localPosition=groundTruth&&view.cam_pos?.[t] ? view.cam_pos[t].map((x,i)=>x-reference[i]) : c.slice(6,9).map((x,i)=>x*shape[i]);
  const worldPosition=groundTruth&&view.cam_pos?.[t] ? view.cam_pos[t] : reference?.map((x,i)=>x+localPosition[i]);
  const encodingOffset=localPosition.map((x,i)=>x-c[6+i]*shape[i]);
  return {origin,worldPosition,localPosition,encodingOffset,forward,up,
    position:enuToViewer(localPosition),
    viewerForward:enuToViewer(forward),viewerUp:enuToViewer(up),
    fovX:c[9],fovY:2*Math.atan(Math.tan(c[9]/2)/aspect),aspect};
}
