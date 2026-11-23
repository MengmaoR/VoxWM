# Vox2Pix v3 lighting-floor smoke

**NATIVE VERIFIED** — 3 independent native scenes, 15 official 640×360 RGB
frames, and 3 same-camera lighting comparisons. Recorded on 2026-11-23
(Asia/Shanghai). All official views use gamma 3.0 and lighting floor 0.35.

| Kind | World seed | Active regions | Views | Search and capture |
| --- | ---: | ---: | ---: | ---: |
| Surface | 42 | 40/40 | 5 | 32.0 s |
| Cave | 1908133298 | 28/40 | 5 | 32.1 s |
| Underwater | 1668782906 | 40/40 | 5 | 63.9 s on resume |

The first floor-enabled run committed Surface and Cave, then rejected the
underwater metadata because Lua represented the integer terrain coordinates
as floating-point JSON numbers. The corrected recorder preserved those
coordinates exactly as integer lattice indices. The resume captured the
remaining Underwater scene. The runs lasted 62.9 s and 67.5 s, excluding the
interval between runs and validation. The older v2 pilot and the earlier
lighting trial use separate directories.

The accepted cave had a roof distance of 0.9 voxels, a 21-cell thick roof at
the probe column, 100% nearby roof coverage, 6.25% open sky rays, and 87.5%
enclosed lateral rays. The enclosure thresholds are provisional. The
underwater observer descended naturally from Y −1 to −7.500 in 44 native
ticks. The terrain support top was Y −7.5, and the recorder then anchored
the output grid at integer Y −7 using the settled position.

## Same-camera native RGB

Luma is Rec.709 weighted sRGB byte intensity on the 0–255 scale. Near-black
means luma below 8. The client reported the applied LUT and no pending
lighting-remesh work for each compared render.

| Kind | Mean luma 0.0 → 0.35 | Median luma 0.0 → 0.35 | Near-black 0.0 → 0.35 |
| --- | ---: | ---: | ---: |
| Surface | 88.97 → 85.63 | 87.72 → 82.30 | 0% → 0% |
| Cave | 0.00 → 39.78 | 0.00 → 39.07 | 100% → 0% |
| Underwater | 89.13 → 95.08 | 75.49 → 81.69 | 0% → 0% |

The cave's native eye param1 was 0 in both renders. The proposed image
shows block edges and distinct surface textures. The projection for the
paired cave pose identifies stone, diorite and iron ore. All three compared
pairs preserved native Node ID, raw param2, raw param1 and the complete
white-model output. Near-white fraction was zero in all comparison frames.
The highest near-white fraction over the fifteen official views was 0.59%.
These comparisons characterize the three measured scenes.

All five submerged views had nonzero Water Count at every pixel. The second
underwater view had 98.65% BACKGROUND pixels with nonzero Water Count. The
HTML flags the sparse local surface hits and shows the water still crossed
along those rays. Native fog and sky can show a different background beyond
the local crop.

## Provenance and artifacts

- Craftium source `ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49`, built in
  `third_party/craftium_vox2pix_floor` with the repository fixed24 and
  [lighting-floor patch](../../../voxwm_gt/patches/lighting_floor.patch).
  The runtime renderer probe reported `voxwm-light-floor-v1`.
- VoxeLibre `178c71b1e7d6ec35c0d9c481c55eb306d6f330bc`.
- Surface and Cave capture source `99f21e49d0e19f92c549e48a19876f2c1dd00411`.
  Underwater capture and validation source
  `32246d858f523be94b14abd5896d265d06be7add`.
- Data and offline HTML on the server:
  `outputs/vox2pix_v3_lightfloor_15/` and its `report/index.html`.
  Logs: `runs/vox2pix_v3_lightfloor_15/collect-1795425602.log` and
  `collect-1795425771.log`. There are six additional comparison PNGs.
- [Machine-readable measurements](summary.json) and
  [reproduction commands](../../README.md#v3-native-smoke).

The C++ LUT unit test covers exact legacy restoration at floor 0,
floor 0.35 (zero → 89), floor 0.5 (zero → 128), maximum 255, monotonicity
and invalid inputs. 32 Python unit and synthetic tests covered native data
interfaces, white projection and natural-depth sampling. Both native CLI
validation and offline report export returned `NATIVE VERIFIED`.
