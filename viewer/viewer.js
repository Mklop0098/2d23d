import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const stage = document.querySelector('#stage');
const status = document.querySelector('#status');
const materialSelect = document.querySelector('#material');
const roughness = document.querySelector('#roughness');
const metalness = document.querySelector('#metalness');
const exposure = document.querySelector('#exposure');
const rotate = document.querySelector('#rotate');
document.querySelector('#retry').addEventListener('click', () => window.location.reload());
let renderer;
let controls;
let environment;
let resizeObserver;
let disposed = false;

function disposeViewer() {
  disposed = true;
  resizeObserver?.disconnect();
  controls?.dispose();
  environment?.dispose();
  resizeObserver = undefined;
  controls = undefined;
  environment = undefined;
  if (renderer) {
    renderer.setAnimationLoop(null);
    renderer.dispose();
    renderer.forceContextLoss();
    renderer = undefined;
  }
}
window.addEventListener('pagehide', disposeViewer, { once: true });

async function createRenderer() {
  const profiles = [
    { antialias: true, powerPreference: 'default' },
    { antialias: false, powerPreference: 'low-power' },
  ];
  let reason = '';
  for (const profile of profiles) {
    const canvas = document.createElement('canvas');
    canvas.addEventListener('webglcontextcreationerror', (event) => { reason = event.statusMessage; });
    const context = canvas.getContext('webgl2', { ...profile, failIfMajorPerformanceCaveat: false });
    if (context) return new THREE.WebGLRenderer({ canvas, context, ...profile });
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  const error = new Error(reason || 'WebGL 2 context unavailable');
  error.name = 'WebGLUnavailableError';
  throw error;
}
const presets = {
  clay: { color: 0xd3c7b7, roughness: 0.85, metalness: 0, texture: false },
  wood: { color: 0xffffff, roughness: 0.68, metalness: 0, texture: true },
  bronze: { color: 0xc89758, roughness: 0.38, metalness: 0.9, texture: false },
  ceramic: { color: 0xffffff, roughness: 0.18, metalness: 0, texture: true },
};

try {
  renderer = await createRenderer();
  if (disposed) throw new DOMException('Viewer closed', 'AbortError');
  const antialias = renderer.getContext().getContextAttributes().antialias;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, antialias ? 2 : 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  stage.appendChild(renderer.domElement);
  renderer.domElement.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    if (!disposed) {
      status.textContent = '';
      document.querySelector('#error-panel').hidden = false;
      document.querySelector('#error-message').textContent = 'Kết nối đồ họa bị gián đoạn. Bạn có thể thử lại khung 3D.';
      document.querySelector('#graphics-help').hidden = false;
      document.querySelector('#error-detail').textContent = 'WebGL context lost';
    }
  });
  renderer.domElement.addEventListener('webglcontextrestored', () => {
    if (disposed) return;
    document.querySelector('#error-panel').hidden = true;
    status.textContent = document.body.dataset.loaded === 'true' ? '' : 'Kết nối đồ họa đã khôi phục.';
  });
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xedf0f3);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  environment = pmrem.fromScene(room, 0.04);
  scene.environment = environment.texture;
  room.dispose();
  pmrem.dispose();

  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minDistance = 0.5;
  controls.maxDistance = 12;
  controls.maxPolarAngle = Math.PI * 0.94;
  const key = new THREE.DirectionalLight(0xfff1dd, 3);
  key.position.set(3, 5, 4);
  key.castShadow = true;
  const shadowSize = antialias ? 2048 : 1024;
  key.shadow.mapSize.set(shadowSize, shadowSize);
  Object.assign(key.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.1, far: 20 });
  key.shadow.normalBias = 0.015;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xd9e7ff, 1.3);
  fill.position.set(-4, 2, -3);
  scene.add(fill);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200),
    new THREE.ShadowMaterial({ opacity: 0.2 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.005;
  floor.receiveShadow = true;
  scene.add(floor);

  let model;
  const materials = [];
  function resetCamera() {
    const aspect = stage.clientWidth / Math.max(stage.clientHeight, 1);
    const vertical = THREE.MathUtils.degToRad(camera.fov / 2);
    const horizontal = Math.atan(Math.tan(vertical) * aspect);
    const distance = 1.45 / Math.sin(Math.min(vertical, horizontal));
    camera.position.set(distance, 1 + distance * 0.15, distance * 0.35);
    controls.target.set(0, 1, 0);
    controls.update();
  }
  function updateMaterial(resetValues = false) {
    for (const { material, original } of materials) {
      const preset = presets[materialSelect.value];
      material.copy(original);
      if (preset) {
        material.color.setHex(preset.color);
        material.roughness = preset.roughness;
        material.metalness = preset.metalness;
        if (!preset.texture) {
          material.map = null;
          material.vertexColors = false;
        }
        material.roughnessMap = null;
        material.metalnessMap = null;
      }
      if (!resetValues) {
        material.roughness = Number(roughness.value);
        material.metalness = Number(metalness.value);
        material.roughnessMap = null;
        material.metalnessMap = null;
      }
      material.needsUpdate = true;
    }
    if (resetValues && materials.length) {
      roughness.value = materials[0].material.roughness;
      metalness.value = materials[0].material.metalness;
    }
  }
  materialSelect.addEventListener('change', () => updateMaterial(true));
  roughness.addEventListener('input', () => updateMaterial());
  metalness.addEventListener('input', () => updateMaterial());
  exposure.addEventListener('input', () => { renderer.toneMappingExposure = Number(exposure.value); });
  rotate.addEventListener('change', () => { controls.autoRotate = rotate.checked; });
  document.querySelector('#reset').addEventListener('click', resetCamera);
  resizeObserver = new ResizeObserver(() => {
    camera.aspect = stage.clientWidth / Math.max(stage.clientHeight, 1);
    camera.updateProjectionMatrix();
    renderer.setSize(stage.clientWidth, stage.clientHeight);
  });
  resizeObserver.observe(stage);
  resetCamera();
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });

  const params = new URLSearchParams(window.location.search);
  const url = params.get('model');
  if (url) {
    // Assets are served by the same local app, including uploaded GLB files.
    const asset = new URL(url, window.location.origin);
    if (asset.origin !== window.location.origin) throw new Error('Đường dẫn mô hình phải thuộc ứng dụng này.');
    status.textContent = 'Đang tải mô hình…';
    const gltf = await new GLTFLoader().loadAsync(asset.href);
    if (disposed) throw new DOMException('Viewer closed', 'AbortError');
    model = gltf.scene;
    const bounds = new THREE.Box3().setFromObject(model);
    const size = bounds.getSize(new THREE.Vector3());
    const largest = Math.max(size.x, size.y, size.z);
    if (!Number.isFinite(largest) || largest <= 0) throw new Error('Mô hình có kích thước không hợp lệ.');
    const center = bounds.getCenter(new THREE.Vector3());
    const scale = 2 / largest;
    const group = new THREE.Group();
    group.add(model);
    group.scale.setScalar(scale);
    group.position.set(-center.x * scale, -bounds.min.y * scale, -center.z * scale);
    model.traverse((object) => {
      if (!object.isMesh) return;
      object.castShadow = true;
      object.receiveShadow = true;
      const originals = Array.isArray(object.material) ? object.material : [object.material];
      const copies = originals.map((original) => {
        const material = original.clone();
        if (params.get('spot') === '1') {
          material.metalness = 0;
          material.roughness = 0.85;
        }
        materials.push({ material, original: material.clone() });
        return material;
      });
      object.material = Array.isArray(object.material) ? copies : copies[0];
      if (params.get('spot') === '1') {
        const colors = object.geometry.getAttribute('color');
        const positions = object.geometry.getAttribute('position');
        const flagged = [];
        for (let index = 0; index < positions.count; index++) {
          if (colors && colors.getX(index) > 0.8 && colors.getY(index) < 0.4) {
            flagged.push(positions.getX(index), positions.getY(index), positions.getZ(index));
          }
        }
        const markers = new THREE.BufferGeometry();
        markers.setAttribute('position', new THREE.Float32BufferAttribute(flagged, 3));
        object.add(new THREE.Points(markers, new THREE.PointsMaterial({
          color: 0xf03728, size: 4, sizeAttenuation: false, depthWrite: false,
        })));
      }
    });
    scene.add(group);
    materialSelect.value = params.get('clay') === '1' ? 'clay' : 'original';
    updateMaterial(true);
    status.textContent = '';
    document.body.dataset.loaded = 'true';
  }
} catch (error) {
  disposeViewer();
  if (error.name !== 'AbortError') {
    console.error(error);
    status.textContent = '';
    document.querySelector('#error-panel').hidden = false;
    const webglError = error.name === 'WebGLUnavailableError';
    document.querySelector('#error-message').textContent = webglError
      ? 'Trình duyệt chưa tạo được kết nối đồ họa WebGL 2 để xem mô hình.'
      : `Không mở được khung 3D. ${error.message}`;
    document.querySelector('#graphics-help').hidden = !webglError;
    document.querySelector('#error-detail').textContent = error.message;
    document.querySelector('.toolbar').hidden = true;
    document.querySelector('.hint').hidden = true;
    document.body.dataset.error = error.message;
  }
}
