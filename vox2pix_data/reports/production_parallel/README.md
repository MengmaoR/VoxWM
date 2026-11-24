# Parallel production collector

The final RGB32 / absolute Fog20–30 collection path is implemented. Formal
50K collection has not started. [Small-test measurements](small_test.json)
record two successful local Native worlds with two workers: Surface and
Underwater, five RGB views each, zero failed tasks, 28.724 seconds total.
All ten views recorded actual draw32 and Fog20/30. The shared parent writer
validated and atomically committed the original 48³ / single-surface schema.

The bounded test used a ten-scene 60/30/10 plan, selected only slots 0 and 3,
and stopped at `LIMIT_REACHED` with 2/10 scenes. Its HTML report is intentionally
`INCOMPLETE` because eight unrequested slots were not collected. This is not
a failed validation of the two committed scenes. The test ran on macOS with
native window rendering; Cave collection and Linux software rendering at high
parallelism were not tested this round. No full-run speed estimate is inferred.

```bash
.venv/bin/python -B -u -m vox2pix_data collect \
  --mode production --dataset outputs/vox2pix_production_local2 \
  --scenes 10 --scene-ids 0 3 --max-scenes 2 --workers 2 \
  --worlds-per-task 1 --task-retries 0 --sites-per-world 1 \
  --scene-timeout 120 --renderer window \
  --craftium-dir third_party/craftium_vox2pix_local32
.venv/bin/python -m vox2pix_data dashboard \
  --dataset outputs/vox2pix_production_local2 --once
.venv/bin/python -m vox2pix_data report \
  --dataset outputs/vox2pix_production_local2
```

The independent local engine build uses Craftium
`ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49` with the existing final-rendering
patches. The old native checkout and Python environment were preserved.
The sample report is `outputs/vox2pix_production_local2/report/index.html`.
The run log remains at `runs/vox2pix_production_local2/collect.log` locally.

44 relevant Python tests passed in 1.951 seconds. They cover production
coordination, bounded task counts, retry/resume cursors, cleanup, loaded biome
checks, quotas, seed independence, storage, monitoring and existing sampling.
Six coordinator tests passed again after adding source-version recording and
the finite task deadline. Shell syntax and CLI help passed. No additional
Native world was started after the authorized two.

The production plan is 6,000 Surface / 3,000 Cave / 1,000 Underwater scenes,
five views per scene. Surface and Cave each actively locate their assigned
biome, then verify the actual loaded surface biome. Their 25 groups receive
240 and 120 scenes each; Underwater has no biome quota. Seeds are unique
across slots and search rounds, and each world has a shared 80/10/10 split
assignment. Completed scenes remain untouched on resume.

Workers never write dataset scenes directly. At most `workers` tasks are
in flight, the parent writes complete scenes, and incomplete slots retain
their next seed cursor. SIGINT/SIGTERM/SIGHUP cleanup targets only this run's
workers and tagged native/display processes. An independent read-only dashboard
shows category/biome progress, failures, active tasks, measured rate and ETA.
Use a dedicated tmux session to survive SSH disconnections.

Production retains actual per-view rendering probes, without repeating the
already completed lighting-control experiment at every scene. Models, native
physics, White Model traversal, view sampling and rendering parameters were
not changed. Two omitted `viewing_range` arguments in acquisition/validation
were fixed: they otherwise applied the old default48 check to final32 data.

See [production commands](../../README.md#parallel-production-collection) for
the launcher, environment, monitoring, stopping and resuming instructions.
