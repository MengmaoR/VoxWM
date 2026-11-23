# Vox2Pix v3 material-aware ambient lighting

Select C as the candidate for the next approved v3 smoke. The saved-scene
trial recovered cave texture while keeping native water, sky, fog and
underwater tint on their existing rendering paths. Open the offline
`lighting_material_comparison/index.html` after extracting the
[report archive](lighting_material_comparison.zip). The [measured JSON](summary.json)
includes all 15 views, four White Model regions, histograms and native probes.

## Native test

The Linux software renderer replayed five recorded 640×360 views in each of
the original Surface, Cave and Underwater scenes. It produced 45 comparison
images from existing world seeds and camera poses. RGB used Gamma 3.0 in all
three modes.

| Mode | Global LUT floor | Solid material ambient floor |
| --- | ---: | ---: |
| A · native control | 0.00 | 0.00 |
| B · legacy global floor | 0.35 | 0.00 |
| C · material ambient | 0.00 | 0.35 |

Within each matched triplet, the 48³ Node ID, raw param1 and raw param2
arrays matched exactly. Camera position, direction and FOV matched the
saved pose. White Model, Face IDs and Water Count came directly from the
original read-only data. Reconstructing the recorded crops restored 20
saved Surface cells and 35 saved Underwater cells in temporary worlds.
The Cave needed no repairs. The original dataset did not retain the full
param1 array. Param1 identity refers to the native A/B/C exposures in each
replayed scene.

The renderer probes matched the RGB render ticks and showed no pending
lighting meshes. A and C both had LUT endpoints 0 and 255. B had 89 and
255. The eligible material shader drew terrain at the Cave poses.
Native fog color, fog range and eye post-effect matched exactly between A
and C at every pose. The reported driver distances 240 and 300 use native
render units, equivalent to the configured 24 and 30 nodes.

## Measured images

Values are Rec.709 luma on RGB8, averaged across the five views for each kind.

| Scene | A | B | C |
| --- | ---: | ---: | ---: |
| Surface | 133.41 | 134.46 | 134.03 |
| Cave | 0.37 | 35.48 | 34.14 |
| Underwater | 87.87 | 95.44 | 87.97 |

- **Cave View 5.** A was completely black. C reached mean luma 38.49
  and P10 27.64 with zero near-black pixels. Median RGB in the projected
  stone, diorite and iron-ore masks was `[37,34,33]`, `[52,52,51]` and
  `[30,27,27]` respectively. Stone texture, ore and step edges are visible in
  the native image. Mean contrast across 4,047 adjacent material or face
  boundaries was 0.00 / 12.00 / 11.93 for A/B/C.
- **Surface.** A and C matched at all directly visible geometry pixels
  identified by the White Model. Across the five views, the largest
  mean absolute RGB difference for local dry background was 0.36 on the
  0–255 scale. Terrain seen through water can change. Surface View 5
  had a mean absolute difference of 4.72 in that group.
- **Underwater View 2.** The local White Model marked 98.65% of pixels as
  BACKGROUND. All blue pixels in the native A/B/C images had nonzero
  Water Count. In the A blue-background mask, A/C mean and P95 absolute
  RGB differences were 0.21 and 1.33, with 91.53% pixel identity.
  B/C mean and P95 were 0.82 and 3.67. The blue polygons and fine ripple
  texture persist in C. The driver recorded the same fog RGBA
  `[191,215,254,255]` and eye post-effect `[24,46,89,60]` in A and C.
  Water Count shows local traversal. It cannot separate the contributions
  from the water surface, distant geometry, sky, fog and the post-effect.
- No image in the 15 triplets had luma at or above 247.

## Limits and checks

The View 5 cross-build comparison at both floors zero matched the true
legacy Cave image pixel for pixel. Surface and Underwater differed across
the independently reconstructed sessions, with mean absolute RGB
differences of 13.15 and 5.33. Cross-build outdoor pixel identity was not
established. The 48³ checks do not cover far-field geometry, and the
outdoor captures do not freeze native clouds and other background state.
The first trial also exposed native fog history after switching away from
B. The reported trial samples A and C consecutively with the global LUT
at zero and requires exact matching fog and post-effect probe values.

The native C++ units cover original zero-floor LUT identity, setting
bounds and exclusion of liquid variants from the ambient branch. The
Python tests cover masked metrics, report references, v3 projection and
capture metadata. The offline package contains only the matched RGB and
diagnostic panels with their measurements.

The actual Craftium base was `ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49`
with the repository capture, legacy LUT and material patches. VoxeLibre
was `178c71b1e7d6ec35c0d9c481c55eb306d6f330bc`. The successful native run
used VoxWM `83b31d51dc785f595d8c56fdf0f7673a94062cab`.
Build and replay commands are in the [Vox2Pix README](../../README.md#material-aware-ambient-comparison).
No new scenes were collected. The new-profile fresh smoke and 50K
collection remain untested.
