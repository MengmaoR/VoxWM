# Vox2Pix 1K pilot collection report

The camera-1.6 pilot completed and passed validation with **200 scenes, 1,000 views at 640×360, and 25 biomes**. It contains 120 surface, 40 shallow cave, and 40 deep cave scenes. The initial pass committed 168 scenes. Four follow-up passes filled the remaining 32 in the same dataset, retaining the terrain, clearance, camera, and quota criteria.

## Collection throughput

| Pass | Workers | Worlds started | New S / shallow / deep | Committed | Elapsed | New scenes/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial | 8 | 572 | 120 / 30 / 18 | 168 / 200 | 1 h 08 m 10 s | 147.9 |
| Wider survey | 24 | 147 | 0 / 9 / 17 | 194 / 200 | 13 m 28 s | 115.8 |
| Parallel rounds and land ranking | 24 | 195 | 0 / 1 / 1 | 196 / 200 | 13 m 03 s | 9.2 |
| Native cave hints | 28 | 256 | 0 / 0 / 3 | 199 / 200 | 10 m 59 s | 16.4 |
| Last slot with batched biome checks | 28 | 137 | 0 / 0 / 1 | 200 / 200 | 3 m 59 s | 15.1 |
| **Total** | | **1,307** | **120 / 40 / 40** | **200 / 200** | **1 h 49 m 40 s** | **109.4** |

Elapsed time sums active runs through each pass's final log entry and excludes inter-pass gaps. Initial timing starts at manifest creation. Follow-up timing starts at launch. World starts count `Creating Minetest run directory` entries, including attempts that failed or were abandoned once another worker filled the slot. The follow-up passes took **41 m 30 s**, about 38% of total active runtime. Full validation and compression reporting took another 179 seconds.

Workers ran independent Craftium worlds. The last pass searched 28 independent seed rounds for one remaining slot. The host had a 57-CPU cgroup quota. Each world also used its own server, client, and Xvfb. Difficulty and search settings varied across passes, so the table measures production progress. Peak throughput by worker count would require a matched workload with fixed seed and survey settings. See [phase metrics](phase_metrics.csv) for the per-pass search settings.

## Search and write bottlenecks

The initial limit of eight worlds and four sites per world left 10 shallow and 22 deep scenes unfilled. For the six slots still missing after the wider survey, the final rejection summaries recorded 766 surveyed sites without an accepted anchor. Of those, 702 failed one of the first three cave geometry checks.

| Site-level rejection in those six slots | Sites |
| --- | ---: |
| No standing cave floor with a roof in the requested height band | 328 |
| Standing floors failed center 16³ free-space requirement | 283 |
| Standing floors passed center clearance but failed full 48³ clearance | 91 |
| Unaccepted sites that reached later checks | 64 |

The same six summaries recorded only two `biome_not_found` lookups. Candidate-level checks recorded **3,289 native anchor or biome mismatches** and **278 failures to provide at least 20 active view regions**. These counts can include many candidate anchors from one site. The logs combined anchor and biome failures at that point. The last-pass implementation batched the exact surface biome checks and logged those reasons separately. The last MangroveSwamp deep scene passed in seed round 35 at site 8.

The search generates terrain, exports it, and checks candidates before RGB drawing begins. Each 48-column survey exports up to 4.62 MB of temporary native IDs because the search also needs a perimeter and samples 256 vertical cells. The land-priority pass ended after a server/client/Lua clock mismatch. Subsequent passes discarded the affected world on that error and kept the clock check for accepted frames.

Successful scenes also incur camera settling, software RGB rendering, white-model projection, and serial writer work. File timestamps give a **1.816 s median and 1.976 s p95** from Zarr group creation to scene metadata write across 200 scenes, with 375 s summed across scenes. This interval includes encoding and writing, can overlap worker search, and excludes writer validation, process communication, and final rename. The logs lack separate render and projection timings.

## Storage and 50K projection

| Payload | Stored bytes |
| --- | ---: |
| RGB PNG | 85,329,780 |
| Three white-model layers | 37,537,023 |
| Original 48³ voxel maps | 2,640,125 |
| Metadata and array descriptors | 2,212,048 |
| **Total** | **127,718,976** |

The 1K data occupies 177.2 MB in allocated file blocks. At the same resolution, terrain mix, and observed throughput, 50K views project to **91.4 collection hours**, **6.39 GB of stored data**, and **8.86 GB of allocated file blocks**. Temporary worlds and validation time are outside this projection. Rare-cave yield and the production worker setting can change the runtime.

The published [log excerpt](search_log_excerpt.txt) records stage boundaries, representative rejections, the clock mismatch, and the final commit. The complete collection logs are on the data server under `/user-data/txj/VoxWM/runs/vox2pix_pilot_1k_640x360/` with the filenames listed in the CSV. The validation and compression summary is `/user-data/txj/VoxWM/outputs/vox2pix_pilot_1k_640x360/compression_report.json`.
