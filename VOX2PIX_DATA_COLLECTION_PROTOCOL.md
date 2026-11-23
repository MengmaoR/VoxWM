# Vox2Pix v3 data collection protocol

Protocol identifier: `vox2pix-v3-surface-watercount-cave-underwater-lightfloor`.

Each scene stores one native Mapgen v7 48³ voxel grid and five RGB views from
the same foot anchor. The five camera orientations are independently sampled
without replacement from the active view regions. Future full collection
targets 10,000 independent scene/world seeds and 50,000 views at the existing
seed-level split. All views from a seed belong to one split. The implementation
currently exposes the bounded native smoke entry point for v3.

| Scene kind | Future target | Native smoke |
| --- | ---: | ---: |
| Surface | 60% / 6,000 scenes | 1 scene × 5 views |
| Cave | 30% / 3,000 scenes | 1 scene × 5 views |
| Underwater | 10% / 1,000 scenes | 1 scene × 5 views |

Future Surface and Cave sampling aims to cover the existing 25 biome groups.
Underwater follows available natural water without equal biome quotas. The
three-scene smoke uses natural scenes without imposing biome balance.

## Scene anchors and visibility

- Surface stands on native terrain with the camera in air. Cave stands on a
  native supported floor with body clearance and a roof. There are no depth
  classes. The scene stores both the surface biome and the biome at foot height.
  The reporting label comes from the surface biome.
- Cave candidates prioritize both ordinary and large Mapgen cave notifications.
  Candidate generation reuses connected free-space searches. Membership checks
  require a roof within 12 voxels of the nominal eye, two contiguous natural
  roof cells, nearby roof coverage of at least 5/9, at most 12/32 open upper rays,
  and at least 2/8 lateral rays hitting terrain. These are provisional local
  geometry thresholds. A cave entrance may remain open, and terrain outside
  the crop does not enter the enclosure calculation.
- Underwater starts in a naturally generated water column. The observer sinks
  using native liquid physics, with no target position, preset bottom height
  or active crop during descent. Once the actual player position settles on
  natural support, the recorder constructs the 48³ crop at its matching
  integer cell. It checks the saved eye against the actual liquid volume.
  Water transmits the near-field visibility rays.
- The integer anchor is at index `[24,24,24]` of the ENU array. The default
  initial search band is configurable via `--survey-y -112 240`. The settled
  underwater crop is independently emerged within playable native map bounds.
  Center and full free ratios are diagnostics. There are 40 bins (8 yaw × 5
  pitch), and each scene requires at least 20 active bins using the 1.6 voxel
  visibility radius.
- The nominal eye height is 1.6. Each projection uses the captured RGB camera
  position, direction and both FOVs. Consecutive native captures bracket the
  rendered frame and confirm the scene's Node IDs and raw param2 match the other
  views. Search diagnostics record the attempts, candidate counts and rejection
  reasons.

## Single-surface projection and storage

| Data | Shape per scene | Dtype |
| --- | --- | --- |
| Native RGB | 5 × 640 × 360 × 3, PNG | uint8 |
| Scene Node ID | 48 × 48 × 48, ENU | uint16 |
| Scene raw param2 | 48 × 48 × 48, ENU | uint8 |
| Surface Node ID | 5 × 360 × 640 | uint16 |
| Surface raw param2 | 5 × 360 × 640 | uint8 |
| Surface face direction | 5 × 360 × 640 | uint8 |
| Water Count | 5 × 360 × 640 | uint8 |

Each ray records the first non-Air, non-Water entry hit, including Glass.
The face codes in Luanti XYZ are 0=+X, 1=−X, 2=+Y, 3=−Y, 4=+Z, 5=−Z.
BACKGROUND stores ID 65535, param2 0 and face 255. The projector reuses native
node boxes and the existing approximations for meshes, textured cutouts and
unsupported geometry, with approximation counts in metadata.

Each Water voxel contributes at most one count when a positive-length liquid
intersection is completed before the ray's termination. This includes a
submerged ray origin if the ray exits that liquid. Grazing contacts and the
empty part of a partial voxel contribute zero. Counts saturate at 255, and
BACKGROUND preserves the count. The volume uses the pinned renderer's
source, flow and neighbor corner interpolation rules within the 48³ crop.

The white representation uses five bytes per pixel in separate Zarr v2
Blosc/Zstd fields. No depth, UV, continuous normals or embeddings are saved.
The dataset vocabulary maps Node IDs to canonical names. A directory is
committed atomically after saving its voxel grid, metadata, five RGB frames,
four image fields and any auxiliary lighting frames.

## Rendering and light

The same native settings apply to all three kinds. Capture uses gamma 3.0,
fixed daylight, clear weather, hidden HUD and first-person held model,
`viewing_range=30`, directional colored fog and `fog_start=0.8`. Native sky,
sun and clouds remain in RGB. Distance culling and fog reduce far-field
mismatch. RGB may still depict geometry outside the 48³ projection.

The official RGB uses `display_gamma=3.0` and `lighting_floor=0.35` for all
three kinds. The client maps each Luanti gamma-curve output $L_\gamma$ to
$L_\min+(1-L_\min)L_\gamma$ in its render LUT, leaving the native engine
default at zero. This render transform preserves Node ID, param2 and light
storage and does not affect world light propagation. The fixed recorder
hides the HUD and wield model and leaves the observer's hand empty.

A matching-pose comparison in each scene stores gamma 3.0 with Floor 0.0
and 0.35 outside the official 15 frames. Both views use the same native
world and camera. Validation checks the renderer's applied settings and
stable geometry, and the report uses Rec.709 luma on the saved sRGB bytes
for comparable mean, median and near-black proportions. Native material
shading and fog still affect the final pixel RGB.

## Smoke and compatibility

The locked run is one scene per kind and 15 native RGB views, followed by
`validate` and `report`. The `--scenes-per-kind 3` option selects 45 frames
for a separately requested run. The offline HTML uses relative files and
includes RGB, surface, face, Water Count, hit/background, camera and lighting comparison.
The report separates structural validation and measured lighting behavior.

The camera-1.6 v2 1K pilot used three hit layers and surface/shallow/deep
quotas. It remains readable with its existing Viewer and has separate
[historical results](vox2pix_data/reports/pilot_1k/README.md). Its scenes are
incompatible with v3. See [commands](vox2pix_data/README.md) for the separate
v3 output path.
