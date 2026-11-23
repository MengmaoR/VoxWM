PASS

# Native Fog 20–30 block calibration

The proposed profile passed the native marker calibration and visual
inspection of all 15 saved scene poses. The [offline report archive](native_comparison.zip)
opens at `native_comparison/index.html`. Its [JSON measurements](summary.json)
contain every camera, marker coordinate, actual render tick, driver value
and node-shader uniform.

## Actual native rendering

| Value | Measured blocks |
| --- | ---: |
| Configured viewing range | 30 |
| Final map draw-list wanted range | 30 |
| Fog range before / after cloud handling | 30 / 30 |
| Proposed Fog Start / End | 20 / 30 |
| Old Fog Start / End | 24 / 30 |
| Sky fog distance override | −1 |
| GPU camera near / far plane | 0.1 / 2000 |

All 15 proposed scene exposures recorded these distances. The node-shader
fog distance was 30 blocks and its shading parameter was 3.00000024.
Actual vertical FOV was 1.2566371 rad, or 72 degrees. Horizontal FOV was
1.82395399 rad. Gamma was 3.0, global floor 0.0 and solid ambient 0.35
throughout A/B/C. The native LUT endpoints stayed 0 and 255.

The driver used linear fog. The unchanged node Shader computes its
coefficient from Euclidean eye distance. It also adjusts the directional
fog target color with clarity. The report models this original target
expression when fitting observed RGB.

The new opt-in camera control uses the configured map draw radius
without FOV expansion. The observer's native Sky fog distance and start
overrides are reset to client control. The final probe includes any cloud
range reduction and rejects a capture whose final range falls below 30.
The measured scenes had no cloud range reduction.

## Marker images

One disposable singlenode air world contains a 7×7 native colored wool
panel. Its center-facing surface is at engine XYZ `[0, 2, 32.5]`.
The camera translates along its fixed view axis to obtain eight exact
Euclidean distances to that face. This avoids mutual marker occlusion.
Each position captures OFF, old 0.8 and proposed 0.6666667 using the same
640×360 renderer. The report includes full frames, enlarged crops and a
distance montage.

| Distance | Expected coefficient | RGB-fitted coefficient | Proposed RGB distance from Fog Color |
| ---: | ---: | ---: | ---: |
| 20 | 0% | 0.00% | 157.74 |
| 22 | 20% | 20.06% | 133.32 |
| 24 | 40% | 40.06% | 107.04 |
| 25 | 50% | 49.95% | 92.21 |
| 27 | 70% | 70.06% | 58.92 |
| 29 | 90% | 89.97% | 21.08 |
| 30 | 100% | 100.00% | 0.00 |
| 32 | 100% | 100.00% | 0.00 |

RGB distance is the mean absolute channel difference on the central red
marker region, in RGB8 units. Maximum mean error against the native fog
equation was 0.45/255. At 24 blocks the colored panel remains recognizable.
At 30 and 32 the sampled marker RGB is exactly `[191, 215, 254]`, the
actual directional Fog Color. Its sampled texture variation is zero.
The OFF control still shows the panel at 32. This demonstrates
MapBlock bounding-volume retention beyond 30 and fog obscuring that
retained geometry. The camera far plane remains at 2000.

## Saved scenes

The original Surface, Cave and Underwater seeds and five saved cameras
per scene were replayed without a new scene search. Every accepted A/B/C
triplet matched the saved 48³ Node IDs and raw param2. Raw param1 was
identical across the triplet. The White Model, Face IDs, Water Count and
BACKGROUND came directly from the original read-only data.

- Surface distant terrain and horizon become progressively obscured.
  Views looking mainly at sky retain native clouds.
- Near rock textures remain visible in the Cave. Views 3 and 5 have
  pixel-identical OFF/proposed images. More distant walls receive fog.
- Underwater distant terrain receives more fog, while native water
  materials and blue post-effect remain present. Blue polygons in
  View 2 persist. Animated bubbles and water can contribute to
  nearby RGB differences between native exposures.

The temporary reconstruction initially restored 20 Surface cells and 35
Underwater cells. The successful Underwater View 1 restored another 2
saved cells before its matched triplet. Source datasets were read-only.
Earlier attempts exposed an incorrectly scoped uniform probe and a
six-cell mismatch in the disposable underwater world. The probe now
records actual `nodes_shader` calls. Accepted exposures retain strict
saved-state checks.

## Tests and reproduction

The isolated Linux native engine built successfully. Native expressions
for the uniform parameter and GLSL coefficient passed the compiled CPU
check at 10/20/22/24/25/27/29/30/32 blocks. The relevant Python suite
passed 39 tests, including runtime units, stale ticks, profile isolation,
projection and offline report assets.

This native OpenGL run used Mesa software rendering through Xvfb.
Craftium base commit was `ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49`,
with capture, material and fog patches. VoxeLibre was
`178c71b1e7d6ec35c0d9c481c55eb306d6f330bc`. The successful run used
VoxWM `bf15eba`. [Build and test commands](../../README.md#native-fog-calibration)
replay the original dataset into a new report directory.

The active profile is `material-ambient-fog20`, version
`vox2pix-v3-materialambient-linear-fog20-30`. It retains the 48³
observation and its 24-block half extent. These observations cover the
tested panel and three recorded scenes. Formal 50K collection and model
training have not run.

## Changed files

- `AGENTS.md`
- `PROJECT_STATE.md`
- `README.md`
- `VOX2PIX_DATA_COLLECTION_PROTOCOL.md`
- `scripts/build_vox2pix_fog.sh`
- `scripts/test_vox2pix_fog.py`
- `vox2pix_data/README.md`
- `vox2pix_data/__main__.py`
- `vox2pix_data/fog_calibration.lua`
- `vox2pix_data/fog_calibration.py`
- `vox2pix_data/fog_report.py`
- `vox2pix_data/fog_runtime.py`
- `vox2pix_data/native.py`
- `vox2pix_data/protocol.py`
- `vox2pix_data/reports/v3_fog20/README.md`
- `vox2pix_data/reports/v3_fog20/native_comparison.zip`
- `vox2pix_data/reports/v3_fog20/summary.json`
- `vox2pix_data/scene.lua`
- `vox2pix_data/test_fog_report.py`
- `vox2pix_data/v3_native.py`
- `vox2pix_data/v3_report.py`
- `voxwm_gt/patches/fog_range.patch`
