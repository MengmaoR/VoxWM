import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';

const SIDE = 48;
const HALF = SIDE / 2;
const BACKGROUND = 0x16232b;
const MARKER = 0x76eaff;

// Neighbor offsets are ENU. The face corners are in Three.js XYZ, outward CCW.
const FACES = [
    { step: [1, 0, 0], shade: .81, corners: [[1,-1,-1],[1,1,-1],[1,1,1],[1,-1,1]] },
    { step: [-1, 0, 0], shade: .68, corners: [[-1,-1,1],[-1,1,1],[-1,1,-1],[-1,-1,-1]] },
    { step: [0, 0, 1], shade: 1, corners: [[-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]] },
    { step: [0, 0, -1], shade: .53, corners: [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]] },
    { step: [0, -1, 0], shade: .87, corners: [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]] },
    { step: [0, 1, 0], shade: .74, corners: [[1,-1,-1],[-1,-1,-1],[-1,1,-1],[1,1,-1]] },
];

function appearance(name) {
    if (name === 'air' || name === 'ignore') return null;
    let opacity = 1;
    let rgb;
    if (/water/.test(name)) { rgb = 0x398acf; opacity = .68; }
    else if (/glass/.test(name)) { rgb = 0xa2dae3; opacity = .38; }
    else if (/lava/.test(name)) rgb = 0xff711c;
    else if (/leaves|grass|moss|fern|vine/.test(name)) rgb = 0x69a04f;
    else if (/dirt|mud|wood|tree|log/.test(name)) rgb = 0x92633f;
    else if (/sand/.test(name)) rgb = 0xd6bf86;
    else if (/snow|ice/.test(name)) rgb = 0xd6e5ea;
    else if (/stone|deepslate|gravel/.test(name)) rgb = 0x8a929c;
    if (rgb !== undefined) return { color: new THREE.Color(rgb), opacity };
    // Keep unclassified nodes distinguishable across scenes without a texture dependency.
    let hash = 0;
    for (const c of name) hash = (Math.imul(hash, 31) + c.charCodeAt(0)) >>> 0;
    return { color: new THREE.Color().setHSL(hash % 360 / 360, .42, .53), opacity };
}

function index(east, north, up) {
    return (east * SIDE + north) * SIDE + up;
}

function inGrid(coord) {
    return coord >= 0 && coord < SIDE;
}

function voxelMeshes(voxels, vocabulary, topLayer = SIDE - 1) {
    const nodes = vocabulary.map(appearance);
    const batches = new Map();
    for (let east = 0; east < SIDE; east++) {
        for (let north = 0; north < SIDE; north++) {
            for (let up = 0; up <= topLayer; up++) {
                const node = nodes[voxels[index(east, north, up)]];
                if (!node) continue;
                for (const { step: [de, dn, du], shade, corners } of FACES) {
                    const e = east + de, n = north + dn, u = up + du;
                    const neighbor = inGrid(e) && inGrid(n) && u >= 0 && u <= topLayer
                        ? nodes[voxels[index(e, n, u)]] : null;
                    if (neighbor && neighbor.opacity >= node.opacity) continue;
                    if (!batches.has(node.opacity)) {
                        batches.set(node.opacity, { positions: [], colors: [], indices: [] });
                    }
                    const batch = batches.get(node.opacity);
                    const first = batch.positions.length / 3;
                    for (const [dx, dy, dz] of corners) {
                        batch.positions.push(east - HALF + dx / 2, up - HALF + dy / 2, HALF - north + dz / 2);
                        const { r, g, b } = node.color;
                        batch.colors.push(r * shade, g * shade, b * shade);
                    }
                    batch.indices.push(first, first+1, first+2, first, first+2, first+3);
                }
            }
        }
    }
    return [...batches].map(([opacity, batch]) => {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(batch.positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(batch.colors, 3));
        geometry.setIndex(batch.indices);
        const material = new THREE.MeshBasicMaterial({
            vertexColors: true, transparent: opacity < 1, opacity, depthWrite: opacity === 1,
        });
        return new THREE.Mesh(geometry, material);
    });
}

function fromEngine([x, y, z]) {
    return new THREE.Vector3(x, y, -z);
}

export function createScene(container) {
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(BACKGROUND);
    renderer.domElement.setAttribute('aria-label', '3D voxel volume and recorded camera');
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const overview = new THREE.PerspectiveCamera(42, 1, .1, 400);
    const recorded = new THREE.PerspectiveCamera(45, 1, .1, 130);
    const controls = new OrbitControls(overview, renderer.domElement);
    const volume = new THREE.Group();
    const box = new THREE.BoxGeometry(SIDE, SIDE, SIDE);
    const bounds = new THREE.LineSegments(new THREE.EdgesGeometry(box),
        new THREE.LineBasicMaterial({ color: 0x82959e, transparent: true, opacity: .5 }));
    box.dispose();
    bounds.position.set(-.5, -.5, .5);

    const marker = new THREE.Group();
    const dot = new THREE.Mesh(new THREE.SphereGeometry(.32, 14, 10),
        new THREE.MeshBasicMaterial({ color: MARKER, depthTest: false, depthWrite: false }));
    const frustum = new THREE.LineSegments(new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ color: MARKER, depthTest: false, depthWrite: false }));
    dot.renderOrder = frustum.renderOrder = 2;
    marker.add(dot, frustum);
    scene.add(volume, bounds, marker);

    let hasCamera = false;
    let following = false;
    let grid = null, vocabulary = null;
    let topLayer = SIDE - 1;
    let width = 1, height = 1;

    function draw() {
        marker.visible = hasCamera && !following;
        bounds.visible = !following;
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, width, height);
        if (!following) {
            renderer.render(scene, overview);
            return;
        }

        // Keep both recorded FOV axes exact when this pane and the RGB differ in aspect ratio.
        const w = Math.min(width, height * recorded.aspect);
        const h = Math.min(height, width / recorded.aspect);
        const x = (width - w) / 2, y = (height - h) / 2;
        renderer.clear();
        renderer.setViewport(x, y, w, h);
        renderer.setScissor(x, y, w, h);
        renderer.setScissorTest(true);
        renderer.render(scene, recorded);
        renderer.setScissorTest(false);
    }

    function resize() {
        width = Math.max(1, container.clientWidth);
        height = Math.max(1, container.clientHeight);
        renderer.setSize(width, height);
        overview.aspect = width / height;
        overview.updateProjectionMatrix();
        draw();
    }

    function resetView() {
        following = false;
        controls.enabled = true;
        controls.target.set(0, 0, 0);
        overview.position.set(79, 68, 88);
        controls.update();
        draw();
    }

    function followCamera() {
        if (!hasCamera) return;
        following = true;
        controls.enabled = false;
        draw();
    }

    function setCamera(view) {
        hasCamera = Boolean(view);
        if (!hasCamera) {
            resetView();
            return;
        }
        const { fov_x_rad: fovX, fov_y_rad: fovY } = view;
        const forward = fromEngine(view.cam_dir_xyz).normalize();
        // Matches Irrlicht's unrolled basis in trace.cpp after the Z sign change.
        const right = forward.clone().cross(new THREE.Vector3(0, 1, 0)).normalize();
        recorded.up.copy(right.cross(forward)).normalize();
        recorded.position.copy(fromEngine(view.projection_origin_local_xyz));
        recorded.lookAt(recorded.position.clone().add(forward));
        recorded.fov = THREE.MathUtils.radToDeg(fovY);
        recorded.aspect = Math.tan(fovX / 2) / Math.tan(fovY / 2);
        recorded.updateProjectionMatrix();

        marker.position.copy(recorded.position);
        marker.quaternion.copy(recorded.quaternion);
        const length = 5;
        const x = length * Math.tan(fovX / 2), y = length * Math.tan(fovY / 2);
        const corners = [[-x,-y,-length],[x,-y,-length],[x,y,-length],[-x,y,-length]];
        const points = [0,0,0, 0,0,-length];
        for (let i = 0; i < corners.length; i++) {
            points.push(0,0,0, ...corners[i], ...corners[i], ...corners[(i + 1) % corners.length]);
        }
        frustum.geometry.dispose();
        frustum.geometry = new THREE.BufferGeometry();
        frustum.geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
        draw();
    }

    function redrawVolume() {
        if (!grid) return;
        for (const mesh of [...volume.children]) {
            volume.remove(mesh);
            mesh.geometry.dispose();
            mesh.material.dispose();
        }
        for (const mesh of voxelMeshes(grid, vocabulary, topLayer)) volume.add(mesh);
        draw();
    }

    function setVoxels(voxels, names) {
        if (voxels.length !== SIDE ** 3) throw new Error('Expected a 48³ voxel grid');
        grid = voxels;
        vocabulary = names;
        redrawVolume();
    }

    function setTopLayer(index) {
        if (!Number.isInteger(index) || !inGrid(index) || index === topLayer) return;
        topLayer = index;
        redrawVolume();
    }

    new ResizeObserver(resize).observe(container);
    controls.addEventListener('change', draw);
    resetView();
    resize();
    return { setVoxels, setTopLayer, setCamera, followCamera, resetView };
}
