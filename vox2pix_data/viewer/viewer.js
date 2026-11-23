import { createScene } from "./scene.js";

const $ = (id) => document.getElementById(id);
const heightLabels = { surface: "地表", shallow: "浅洞", deep: "深洞" };
const faceNames = ["+X", "−X", "+Y", "−Y", "+Z", "−Z"];
const faceShades = [201, 156, 240, 116, 183, 137];
const canvas = $("white");
const context = canvas.getContext("2d");
const rgb = $("rgb");
const layerButtons = [...$("layers").querySelectorAll("button")];
const markers = [...document.querySelectorAll(".pixel-marker")];

let catalog, voxelScene, metadata, white, pixel;
let sceneRequest, viewRequest, rgbURL;
let layer = 0;
let following = false;

async function get(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`请求失败（${response.status}）：${url}`);
  return response;
}

function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}

function setFollowing(value) {
  following = value;
  $("follow").setAttribute("aria-pressed", String(value));
}

function populateFilters() {
  const heights = new Set(catalog.scenes.map((row) => row.height));
  for (const [value, label] of Object.entries(heightLabels)) {
    if (heights.has(value)) $("height").add(new Option(label, value));
  }
  const biomes = [...new Set(catalog.scenes.map((row) => row.biome))].sort();
  for (const biome of biomes) $("biome").add(new Option(biome, biome));
  $("height").disabled = $("biome").disabled = !catalog.scenes.length;
}

function filterScenes() {
  const height = $("height").value;
  const biome = $("biome").value;
  const previous = $("scene").value;
  const rows = catalog.scenes.filter((row) =>
    (!height || row.height === height) && (!biome || row.biome === biome));
  $("scene").replaceChildren(...rows.map((row) => {
    const id = String(row.scene_id).padStart(4, "0");
    return new Option(`${id} · ${heightLabels[row.height] ?? row.height} · ${row.biome}`, row.scene_id);
  }));
  $("scene").disabled = rows.length === 0;
  if (rows.some((row) => String(row.scene_id) === previous)) $("scene").value = previous;
  void loadScene();
}

function clearImages() {
  white = pixel = null;
  if (rgbURL) URL.revokeObjectURL(rgbURL);
  rgbURL = null;
  rgb.removeAttribute("src");
  context.clearRect(0, 0, canvas.width, canvas.height);
  $("coverage").textContent = "";
  $("pose").replaceChildren();
  $("follow").disabled = true;
  for (const button of layerButtons) button.disabled = true;
  showPixel();
}

function clearVolume() {
  if (voxelScene) {
    const emptyGrid = new Uint16Array(48 ** 3).fill(catalog.vocabulary.indexOf("air"));
    voxelScene.setCamera(null);
    voxelScene.setVoxels(emptyGrid, catalog.vocabulary);
  }
  $("reset").disabled = $("cutaway").disabled = true;
}

async function loadScene() {
  sceneRequest?.abort();
  viewRequest?.abort();
  clearImages();
  metadata = null;
  $("view").replaceChildren();
  $("view").disabled = true;
  if (!$("scene").value) {
    clearVolume();
    message("当前筛选条件下没有场景。");
    return;
  }
  message("正在读取场景…");
  const url = `/api/scenes/${$("scene").value}`;
  const request = sceneRequest = new AbortController();
  try {
    const [meta, data] = await Promise.all([
      get(`${url}/meta`, request.signal).then((response) => response.json()),
      get(`${url}/voxels`, request.signal).then((response) => response.arrayBuffer()),
    ]);
    if (request.signal.aborted) return;
    if (data.byteLength !== 2 * 48 ** 3) throw new Error("Voxel 数据长度不匹配。");
    metadata = meta;
    voxelScene?.setVoxels(new Uint16Array(data), catalog.vocabulary);
    $("view").replaceChildren(...meta.views.map((pose, index) =>
      new Option(`${index + 1} · 区域 ${pose.region_id}`, index)));
    $("view").disabled = false;
    $("reset").disabled = !voxelScene;
    $("cutaway").disabled = !voxelScene;
    await loadView();
  } catch (error) {
    if (!request.signal.aborted) {
      clearVolume();
      message(error.message, true);
    }
  }
}

async function loadView() {
  if (!metadata) return;
  viewRequest?.abort();
  clearImages();
  const index = Number($("view").value);
  const pose = metadata.views[index];
  const url = `/api/scenes/${$("scene").value}/views/${index}`;
  const request = viewRequest = new AbortController();
  message("正在读取 RGB 和白模…");
  try {
    const [png, data] = await Promise.all([
      get(`${url}/rgb`, request.signal).then((response) => response.blob()),
      get(`${url}/white`, request.signal).then((response) => response.arrayBuffer()),
    ]);
    if (request.signal.aborted) return;
    const cells = 3 * canvas.width * canvas.height;
    if (data.byteLength !== 4 * cells) throw new Error("白模数据长度不匹配。");
    white = {
      nodes: new DataView(data, 0, 2 * cells),
      states: new Uint8Array(data, 2 * cells, cells),
      faces: new Uint8Array(data, 3 * cells, cells),
    };
    rgbURL = URL.createObjectURL(png);
    rgb.src = rgbURL;
    voxelScene?.setCamera(pose);
    if (following) voxelScene?.followCamera();
    renderPose(pose);
    drawWhite();
    for (const button of layerButtons) button.disabled = false;
    $("follow").disabled = !voxelScene;
    message(`场景 ${$("scene").value.padStart(4, "0")} · 视角 ${index + 1} / ${metadata.views.length}`);
  } catch (error) {
    if (!request.signal.aborted) {
      clearImages();
      voxelScene?.setCamera(null);
      message(error.message, true);
    }
  }
}

function renderPose(pose) {
  const vector = (xyz) => xyz.map((value) => value.toFixed(3)).join(", ");
  const fieldOfView = [pose.fov_x_rad, pose.fov_y_rad].map((angle) =>
    `${(angle * 180 / Math.PI).toFixed(1)}°`).join(" × ");
  const fields = [
    ["世界坐标（Luanti XYZ）", vector(pose.actual_camera_world_xyz)],
    ["朝向向量（Luanti XYZ）", vector(pose.cam_dir_xyz)],
    ["Yaw / Pitch", `${pose.yaw_deg.toFixed(1)}° / ${pose.pitch_deg.toFixed(1)}°`],
    ["水平 × 垂直 FOV", fieldOfView],
    ["RGB 采集 Tick", pose.render_simulation_tick],
    ["世界种子", metadata.world_seed],
  ];
  $("pose").replaceChildren(...fields.map(([key, value]) => {
    const group = document.createElement("div");
    const label = document.createElement("dt");
    const reading = document.createElement("dd");
    label.textContent = key;
    reading.textContent = value;
    group.append(label, reading);
    return group;
  }));
}

function drawWhite() {
  const image = context.createImageData(canvas.width, canvas.height);
  const pixels = canvas.width * canvas.height;
  const offset = layer * pixels;
  let occupied = 0;
  for (let i = 0; i < pixels; i++) {
    const cell = offset + i;
    const empty = white.nodes.getUint16(2 * cell, true) === 65535;
    const shade = empty ? 20 : faceShades[white.faces[cell]] ?? 190;
    image.data[4 * i] = shade;
    image.data[4 * i + 1] = empty ? 29 : shade;
    image.data[4 * i + 2] = empty ? 41 : shade;
    image.data[4 * i + 3] = 255;
    if (!empty) occupied++;
  }
  context.putImageData(image, 0, 0);
  $("coverage").textContent = `第 ${layer + 1} 层命中 ${(100 * occupied / pixels).toFixed(1)}% 的像素 · 灰阶表示面方向`;
}

function pickPixel(event) {
  if (!white) return;
  const { left, top, width, height } = event.currentTarget.getBoundingClientRect();
  const x = Math.min(canvas.width - 1, Math.max(0, Math.floor((event.clientX - left) * canvas.width / width)));
  const y = Math.min(canvas.height - 1, Math.max(0, Math.floor((event.clientY - top) * canvas.height / height)));
  pixel = { x, y };
  showPixel();
}

function showPixel() {
  for (const marker of markers) marker.hidden = !pixel;
  $("hits").replaceChildren();
  $("pixel").textContent = pixel
    ? `像素 (${pixel.x}, ${pixel.y})`
    : "点击 RGB 或白模，查看同一像素的三层原始数据。";
  if (!pixel) return;
  for (const marker of markers) {
    marker.style.left = `${100 * (pixel.x + 0.5) / canvas.width}%`;
    marker.style.top = `${100 * (pixel.y + 0.5) / canvas.height}%`;
  }
  for (let l = 0; l < 3; l++) {
    const cell = (l * canvas.height + pixel.y) * canvas.width + pixel.x;
    const id = white.nodes.getUint16(2 * cell, true);
    const hit = document.createElement("li");
    hit.textContent = id === 65535 ? "无命中"
      : `${catalog.vocabulary[id] ?? "未知节点"} · ID ${id} · param2 ${white.states[cell]} · 面 ${faceNames[white.faces[cell]] ?? white.faces[cell]}`;
    $("hits").append(hit);
  }
}

$("height").addEventListener("change", filterScenes);
$("biome").addEventListener("change", filterScenes);
$("scene").addEventListener("change", loadScene);
$("view").addEventListener("change", loadView);
$("follow").addEventListener("click", () => {
  setFollowing(!following);
  if (following) voxelScene.followCamera();
  else voxelScene.resetView();
});
$("reset").addEventListener("click", () => {
  setFollowing(false);
  voxelScene.resetView();
});
$("cutaway").addEventListener("input", (event) => {
  const top = Number(event.currentTarget.value);
  $("cutaway-level").textContent = top === 47 ? "完整区域" : `相对 Y ≤ ${top - 24}`;
});
$("cutaway").addEventListener("change", (event) => {
  voxelScene.setTopLayer(Number(event.currentTarget.value));
});
for (const button of layerButtons) {
  button.addEventListener("click", () => {
    layer = Number(button.dataset.layer);
    for (const tab of layerButtons) tab.setAttribute("aria-pressed", String(tab === button));
    drawWhite();
  });
}
rgb.addEventListener("click", pickPixel);
canvas.addEventListener("click", pickPixel);

async function start() {
  try {
    voxelScene = createScene($("voxel-view"));
  } catch (error) {
    $("voxel-view").textContent = `3D 视图无法启动。${error.message}`;
  }
  try {
    catalog = await get("/api/catalog").then((response) => response.json());
    const { width, height, views_per_scene: views } = catalog.config;
    canvas.width = width;
    canvas.height = height;
    document.documentElement.style.setProperty("--image-ratio", `${width} / ${height}`);
    $("dataset-summary").textContent =
      `${catalog.scenes.length} 个场景 · 每场景 ${views} 个视角 · ${width}×${height}`;
    populateFilters();
    filterScenes();
  } catch (error) {
    message(error.message, true);
  }
}

void start();
