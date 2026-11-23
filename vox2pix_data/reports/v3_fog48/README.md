PASS

# Independent RGB48 and absolute Fog20–30

Unit checks, native renderer submission checks and visual calibration
passed. Production data has not been collected. The [offline report ZIP](fog_range_48_test.zip)
opens at `fog_range_48_test/index.html`. The [measurement JSON](summary.json)
contains all 14 distances and 15 existing-scene B/C pairs.

## Actual native values

| Measurement | B | C |
| --- | ---: | ---: |
| Configured RGB viewing range | 30 | 48 |
| Client wanted range | 30 | 48 |
| Actual ClientMap draw range | 30 | 48 |
| Native runData fog-range context | 30 | 48 |
| Actual driver Fog Start / End | 20 / 30 | 20 / 30 |
| Node Shader fog-distance uniform | 30 | 30 |
| GPU camera near / far plane | 0.1 / 2000 | 0.1 / 2000 |
| Server block send limit | 8 MapBlocks / 128 nodes | 8 MapBlocks / 128 nodes |
| Server generation limit | 8 MapBlocks / 128 nodes | 8 MapBlocks / 128 nodes |
| Active-object send limit | 8 MapBlocks / 128 nodes | 8 MapBlocks / 128 nodes |

The old `runData.fog_range` still records the native draw-range context.
Absolute driver endpoints independently read `fog_start_distance` and
`fog_end_distance`, converting nodes with BS. Actual C shader distance
was 30 and its shading parameter was 3.00000024. This confirms that the
48-block map range did not become a 48-block Fog End.

The client reports wanted range in MapBlocks and the server adds one.
Draw48 therefore requests four MapBlocks, or 64 nodes, before the
configured send/generate cap. The existing eight-MapBlock caps were
sufficient. C calibration frames recorded 283–358 loaded MapBlocks
within the probe's camera-neighborhood window. Their native integer XYZ
coordinates are included in JSON.

## Calibration

One temporary native air world contains the same textured 7×7 wool panel.
The camera moves along a fixed view axis to place its central front face
at 12/18/20/22/24/27/29/30/32/36/40/44/48/52 nodes. Its fixed face center
is engine XYZ `[0,2,32.5]` and its central MapBlock is `[0,0,2]`.
A is Fog OFF/draw48. B and C use absolute Fog20–30 with draw30 and draw48.

| Marker distance | C fog factor | Marker submitted in B | Marker submitted in C | C RGB distance from Fog Color |
| ---: | ---: | --- | --- | ---: |
| 12, 18, 20 | 0% | Yes | Yes | Native marker color |
| 22 | 20% | Yes | Yes | 133.32 |
| 24 | 40% | Yes | Yes | 107.04 |
| 27 | 70% | Yes | Yes | 58.92 |
| 29 | 90% | Yes | Yes | 21.08 |
| 30, 32 | 100% | Yes | Yes | 0.00 |
| 36, 40, 44, 48 | 100% | No | Yes | 0.00 |
| 52 | 100% | No | Yes | 0.00 |

At 24 the panel remains recognizable. At 30 through 52 its sampled RGB is
exactly `[191,215,254]`, the actual Fog Color, with zero sampled texture
variation. The OFF control still shows the colored panel. At 36–48,
the proposed frame records the panel's MapBlock, four native draw calls
and 504 vertices. The old draw30 frame no longer submits that block.
This proves continued native geometry submission beyond the fog endpoint.

The 52-node retention is observed MapBlock bounding-volume leakage
beyond the nominal 48-node draw range. Fog still hides the retained
marker. The B/C marker core colors match exactly at every tested distance
where both modes submit the marker, including 24. Maximum RGB error
against the original native directional-fog equation is 0.45/255.
At 36–52, the retained C panel differs from B's culled background by
1.67/255 in mean RGB. The native sky and Fog Color can leave a faint,
uniform patch even after the marker texture has disappeared.

## Existing scene comparisons and limits

All five original cameras for Surface, Cave and Underwater were replayed
with B/C. Every accepted pair matched the saved 48³ Node IDs and raw
param2 and had identical raw param1 between exposures. White Model,
Face IDs and Water Count read the original arrays. The two new projection
fixtures verify that an in-cube diagonal surface beyond radius 24 remains
available and that water traversal stops at the actual cube exit.

- Cave Views 1, 2, 3 and 5 have pixel-identical B/C RGB. The near rock
  texture remains visible. View 4 changes in the distant opening.
- Surface View 3 has mean absolute RGB difference 16.81/255. Its actual
  native Fog Color changes from `[124,139,165]` in B to `[54,61,72]` in C.
  The center near surface remains unchanged. This pair is not pixel-equivalent.
- Native sky brightness sampling uses the draw-range context and temporal
  interpolation. Native animated water, clouds and bubbles also remain
  present. The image differences include these native effects and cannot
  all be attributed to additional terrain submission.
- Underwater post-effect RGBA is identical in each pair. The original
  water polygons and blue background remain visible. No water, material
  lighting, transparency or GLSL fog formula was changed.

## Failed attempts and replay control

The first native run completed all 14 calibration distances but rejected
the Cave camera after its technical observer moved down 0.5 node.
A subsequent saved-position replay rejected its orientation after native
position updates brought back lower-precision angle feedback.
Both failed outputs remain in their separate server directories.

The final diagnostic holds each saved native Player Position and restores
the original yaw/pitch after position updates. This control requires
`vox2pix_replay_restore` and is limited to temporary RGB replay worlds.
The original camera tolerance and nominal eye height remain in effect.
The replay first performs the original underwater descent, then holds
the saved pose for its image comparison. Production collection continues
to use native underwater sinking.

The successful 15-pair replay used source
`feb0b279ca8495281e44bf8f7f3ad49b75e3fd40` and reused the already verified
14-distance calibration from `41a2278`. All runtime probes and raw PNGs
are retained. The initial calibration was unaffected by the later
saved-scene observer fixes.

## Tests and reproduction

The relevant Python suite passed 45 tests. The compiled native policy
check passed absolute endpoints, invalid settings and legacy defaults.
The isolated Craftium build succeeded. The actual OpenGL rendering path
used Linux Mesa software rendering through Xvfb.

Craftium base commit is `ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49`,
with the repository capture/material/fog-range/fog-absolute patches.
VoxeLibre is `178c71b1e7d6ec35c0d9c481c55eb306d6f330bc`.
The [exact build, test and environment-aware replay commands](../../README.md#independent-rgb48-and-absolute-fog)
use a fresh output path. A full replay runs the 14 markers and 15 pairs.
`--reuse-calibration PATH` optionally reuses completed marker captures.

The active profile is `material-ambient-fog20-draw48`, version
`vox2pix-v3-local48-rgb48-absolute-fog20-30`. Its voxel observation
remains a foot-centered 48³ cube with half extent 24. No formal dataset
collection or Vox2Pix training has run.

## Modified files

- `AGENTS.md`
- `PROJECT_STATE.md`
- `README.md`
- `VOX2PIX_DATA_COLLECTION_PROTOCOL.md`
- `scripts/build_vox2pix_fog48.sh`
- `scripts/test_vox2pix_fog48.py`
- `vox2pix_data/README.md`
- `vox2pix_data/__main__.py`
- `vox2pix_data/fog48_report.py`
- `vox2pix_data/fog_range48.py`
- `vox2pix_data/fog_runtime.py`
- `vox2pix_data/native.py`
- `vox2pix_data/protocol.py`
- `vox2pix_data/reports/v3_fog48/README.md`
- `vox2pix_data/reports/v3_fog48/fog_range_48_test.zip`
- `vox2pix_data/reports/v3_fog48/summary.json`
- `vox2pix_data/scene.lua`
- `vox2pix_data/test_fog48_report.py`
- `vox2pix_data/test_v3_projection.py`
- `vox2pix_data/v3_report.py`
- `voxwm_gt/patches/fog_absolute.patch`
