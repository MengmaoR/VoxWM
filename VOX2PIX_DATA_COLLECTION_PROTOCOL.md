# Vox2Pix Data Collection Protocol

**NOT NATIVE VERIFIED. Implementation complete != 1K dataset collected.**

Vox2Pix pairs one raw 48³ world region with five camera poses, five three-layer
geometric projections, and five RGB frames. The pilot measures storage costs
and the coverage of the three layers. Biome and sampling metadata support
coverage analysis. Future model input uses node identity, raw `param2`, and
discrete face direction.

## Pilot and formal plan

| Stage | Independent world seeds | Views per scene | RGB frames | Surface / shallow / deep per biome |
| --- | ---: | ---: | ---: | --- |
| Compression pilot | 200 | 5 | 1,000 | 5/2/1, 5/1/2 or 4/2/2 |
| Formal plan | 10,000 | 5 | 50,000 | 240/80/80 |

The pilot uses 640×360. It allocates eight scenes to each of the 25 VAE biome
groups. A permutation determined by the pilot seed assigns the three patterns
to 10, 10, and 5 groups respectively, for totals of 120/40/40. An unfilled slot
remains a gap under its original biome and height. The pilot is train-like and
stores each independent `world_seed`. The formal plan uses world-seed-level
80/10/10 splitting to keep each seed in a single split.

## Geometry and scene selection

- Luanti XYZ uses X east, Y up, Z north. The stored array is ENU
  `[x,z,y]`. Given integer foot anchor `(fx,fy,fz)`, stored index
  `[i,j,k]` refers to the world voxel centered at
  `(fx-24+i, fy-24+k, fz-24+j)` in XYZ. Store this region origin in both
  conventions. The camera is at `(fx,fy+2,fz)`, at ENU index `[24,24,26]`.
  Voxel boundaries lie half a voxel from their centers.
- Reuse Mapgen v7, native biome lookup and the VAE biome grouping. Resolve the
  surface reference `S` from solid native terrain in the sampled column.
  Surface candidates descend from the highest standable floor, including tree
  tops. For the first exposed water surface `y_w`, set the virtual foot to
  `y_w-1` and the camera to `y_w+1` in air.
- The shallow band is `[S-24,S-8]`. The deep band is `[-38,S-32]`.
  In both cases the complete 48³ region lies within `[-62,193]` vertically.
  Search real six-connected air/water cavities in native terrain. Check a
  supporting floor, three air cells from the foot to the eye, and overhead
  terrain. Try other floors of the current component before another component,
  then another location, then another seed.
- The center 16³ must have at least one-third free space and the full 48³ at
  least one-quarter. Air and water are free, and the eye cell is always air.
  The scene must have at least 20 open views. A precomputed angular lookup maps
  occupied voxel offsets of Euclidean length at most 2 to affected regions.
  Air and water leave these regions open. Eight 45° yaw bins and five 36° pitch
  bins form 40 regions. Choose five open bins uniformly without replacement and
  sample one concrete angle pair per bin. Yaw zero is +Z, positive yaw looks
  toward −X, and positive pitch looks downward. Region IDs are
  `8 * pitch_bin + yaw_bin` from the lowest pitch bin upward numerically.
- Capture RGB and actual eye position, direction, and both FOVs from the same
  Craftium observation. Hold the pose until the client reaches the requested
  eye and angle. The white projection uses the actual camera packet and pixel
  centers. Record requested and actual angles, resolution, gamma, time,
  weather, engine/game revisions, and render settings. The fixed capture
  settings request vertical FOV 72°, gamma 3.0, noon and clear weather
  (`mcl_doWeatherCycle=false`, `mcl_weather.change_weather("none")`).
  HUD and camera motion effects, waving, dynamic shadows and auto exposure
  are disabled. Verify the same native node IDs and raw states across all five
  views before committing.

## White model and storage

Trace each pixel ray into the 48³ region. Save its first three distinct
non-air voxel entry surfaces. Skip exits and multiple hits on one voxel.
Transparent and opaque voxels both occupy a layer and the ray continues
through them. Six face codes select the signed dominant axis of the surface
normal in **Luanti XYZ**. The codes are 0=+X, 1=−X, 2=+Y, 3=−Y, 4=+Z,
and 5=−Z. An empty layer is `(65535, 0, 255)`. The format has no depth, UV,
continuous normal, light or precomputed embedding.

The root vocabulary maps stable integer IDs to canonical Node Names.
Each scene contains one `scene_node_id:uint16[48,48,48]` and a separate
`scene_param2:uint8[48,48,48]`. Each view contributes one lossless PNG
and a slice of `white_node_id:uint16`, `white_param2:uint8` and
`white_face:uint8`, all shaped `[5,3,H,W]`. Zarr v2 with Blosc/Zstd stores
the arrays separately. The manifest and scene JSON describe sampling and
geometry. A scene becomes visible as committed after its arrays, PNGs and
metadata are written together. Rerun the same command to fill missing slots.

## Validation and storage gate

`check` is a read-only validator for exact pilot quotas, independent seeds,
camera/region coordinates, raw and white array shapes and dtypes, PNG sizes,
and layer sentinels. The geometry fixtures exercise air, opaque and transparent
hits, adjacent cubes, non-full geometry and underwater rejection. A complete
pilot reports biome/height and view bins, node and node-state visibility,
face codes, layer validity and water/glass frequency by layer.

`compression-report` records RGB size distribution, each compressed array
field, raw-map and metadata bytes, and total GB and GiB. It scales measured
1K bytes by 50. A result of at most decimal 100 GB recommends 640×360.
Larger estimates recommend 320×180. Both remain manual decisions for the
formal dataset.

## Provisional until measured

RGB registration against the live renderer, sampler success for all quotas,
and meaningful layer-2/3 coverage need the user-run pilot. The white projection
queries native connected and rotated nodeboxes. Its plant and flowing-liquid
geometry uses engine definitions and state, and unsupported renderer-specific
draw types use native selection-box proxies. Render texture alpha, animated
materials, and renderer-only mesh details can produce boundary differences.
Scene metadata records approximations. Synthetic success does not establish
pixel-perfect Native alignment or prove the 50K size estimate.

Run and output details are in [the module README](vox2pix_data/README.md).
