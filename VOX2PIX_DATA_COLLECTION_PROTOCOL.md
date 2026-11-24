# Vox2Pix v3 data collection protocol

The sole active v3 rendering profile is `material-ambient-fog20-draw32`, version
`vox2pix-v3-local48-rgb32-absolute-fog20-30`. Native and visual verification
of this final candidate is PENDING until the current calibration and replay complete.
The current authorization covers only that bounded verification; formal 50K
collection and model training have not started.

| Setting | Active value |
| --- | --- |
| Voxel observation / half extent | 48³ / 24 nodes per axis |
| Native RGB map draw range | 32 nodes |
| Independent native linear Fog Start / End | 20 / 30 nodes |
| Gamma / global LUT floor / solid material floor | 3.0 / 0.0 / 0.35 |

RGB draw ranges 24, 30 and 48, ratio Fog `fog_start=0.8` / `2/3`, and global
`lighting_floor=0.35` are deprecated. The 128-node run is a diagnostic preview.
Legacy profiles remain explicit options for reading and replaying their own
versioned data; no historical data or report is overwritten.

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

- Surface uses the top of native terrain or the first air cell above exposed
  native water. The camera stays in air. The observer stays fixed at a water
  anchor with liquid sinking disabled. Cave stands on a native supported
  floor with body clearance and a roof. There are no depth
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
`viewing_range=32`, directional colored fog and independent absolute fog
distances `fog_start_distance=20` and `fog_end_distance=30`. Native sky,
sun and clouds remain in RGB. Distance culling and fog reduce far-field
mismatch. RGB may still depict geometry outside the 48³ projection.

For a new approved smoke, select
`--lighting-profile material-ambient-fog20-draw32` and use the isolated fog-enabled
Craftium build. All three scene kinds use `display_gamma=3.0`,
`lighting_floor=0.0` and `solid_ambient_floor=0.35`. The native LUT still
maps 0 to 0 and sunlight to 255. The world-node shader uses the sampled
texture, material tint and face shading to set a minimum contribution before
fog. Eligible opaque and alpha-cutout terrain receives this contribution.
Liquids, glass, overlays, sky, fog and the underwater post-effect follow
their native rendering paths. Raw light, Node ID, param2 and the White
projection are preserved. Both engine settings default to zero.

The independent-range profile is the smoke CLI default. The explicit legacy
`global-lut`, `material-ambient`, `material-ambient-fog20` and
`material-ambient-fog20-draw48` profiles
retain their recorded settings. Give the new profile a separate dataset path. Each profile
stores its matched-pose zero-floor control outside the main 15 frames.
Validation checks settings, render tick and stable geometry. The report
uses Rec.709 luma on the saved RGB bytes.

The [same-pose native A/B/C trial](vox2pix_data/reports/v3_material_lighting/README.md)
supports selecting the material profile as the candidate for future smoke
tests. The three saved scenes provide a limited test of material coverage.

## Native RGB fog range

The voxel observation is a 48³ cube with half extent 24 on each axis.
The RGB map draw range is 32 blocks. Voxel availability depends on a ray's
intersection with that cube. It has no spherical radius guarantee from
the actual eye position.

The new protocol records the following geometry.

```json
{
  "voxel_observation_shape": [48, 48, 48],
  "voxel_observation_half_extent": 24,
  "rgb_viewing_range": 32,
  "fog_start_distance": 20,
  "fog_end_distance": 30,
  "fog_type": "native_linear"
}
```

The final native fog coefficient is `clamp((d - 20) / 10, 0, 1)`
for Euclidean camera distance in blocks. The pinned native node shader
also adjusts the directional fog target color according to clarity.
The coefficient reaches one at 30, where geometry takes the fog color.
World geometry from 30 through 32 remains eligible for native map
rendering and is fully fogged. MapBlock bounding volumes may retain
faces beyond the nominal 32-block map range. Such retained geometry must
also be hidden by Fog. RGB is not hard-cut at either 24 or 30.

An opt-in native camera setting fixes the map draw range to 32 through
the existing draw-list mechanism. The GPU far plane keeps its original
camera calculation. Native `driver->setFog` reads the independent absolute
start and end settings, converted with BS. The `-1/-1` engine defaults
retain legacy ratio behavior. The observer's Sky overrides are cleared
through the native player API so they cannot restrict block delivery.
The probe records range before and after cloud handling, final driver
color and endpoints, node shader uniforms, FOV and camera planes.

The native client also records loaded and submitted solid-mesh MapBlock
coordinates, draw calls and vertices. Send/generate limits are reported
in MapBlocks and nodes. Existing limits of eight MapBlocks cover 128
nodes. The client requests its wanted range in MapBlocks; the server
adds a one-MapBlock margin. The capture checks that configured server
limits cover its map range. These diagnostics are metadata.

The [final calibration](vox2pix_data/README.md#final-native-rendering-rgb32)
uses a disposable native world with a fixed camera, sequential non-occluding
marker placements at 10/18/20/22/24/25/27/29/30/31/32/34 nodes, and Fog OFF / final
Fog20–30 exposures. At 31–32 it requires actual submission of the marker's
MapBlock. At 34 it reports any bounding-volume retention and checks that Fog
hides the retained geometry. The marker world is independent of production scenes.
The final 15-frame replay reuses the original three world seeds, foot anchors,
actual camera poses, voxel maps and saved White / Water / BACKGROUND arrays.
No world search or new scene generation is part of that replay.

The offline report is `outputs/vox2pix_final_rendering_32/report/index.html`.
It separates Unit Verified, Native Verified, Visual Verified and Dataset
Collection Not Started. Success requires actual native driver/shader probes
and image inspection; configuration alone does not establish verification.
The projection continues to query only the stored 48³ cube, returning
BACKGROUND on exit and counting only water traversed within that cube.

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
