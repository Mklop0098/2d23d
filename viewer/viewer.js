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
const presets = {
  clay: { color: 0xd3c7b7, roughness: 0.85, metalness: 0, texture: false },
  wood: { color: 0xffffff, roughness: 0.68, metalness: 0, texture: true },
  bronze: { color: 0xc89758, roughness: 0.38, metalness: 0.9, texture: false },
  ceramic: { color: 0xffffff, roughness: 0.18, metalness: 0, texture: true },
};

try {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  stage.appendChild(renderer.domElement);
  renderer.domElement.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    status.textContent = 'Khung 3D bị mất kết nối đồ họa. Tải lại trang để tiếp tục.';
  });
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xedf0f3);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const environment = pmrem.fromScene(room, 0.04);
  scene.environment = environment.texture;
  room.dispose();
  pmrem.dispose();

  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minDistance = 0.5;
  controls.maxDistance = 12;
  controls.maxPolarAngle = Math.PI * 0.94;
  const key = new THREE.DirectionalLight(0xfff1dd, 3);
  key.position.set(3, 5, 4);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
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
  new ResizeObserver(() => {
    camera.aspect = stage.clientWidth / Math.max(stage.clientHeight, 1);
    camera.updateProjectionMatrix();
    renderer.setSize(stage.clientWidth, stage.clientHeight);
  }).observe(stage);
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
        materials.push({ material, original: original.clone() });
        return material;
      });
      object.material = Array.isArray(object.material) ? copies : copies[0];
    });
    scene.add(group);
    materialSelect.value = params.get('clay') === '1' ? 'clay' : 'original';
    updateMaterial(true);
    status.textContent = '';
    document.body.dataset.loaded = 'true';
  }
  window.addEventListener('pagehide', () => {
    renderer.setAnimationLoop(null);
    controls.dispose();
    environment.dispose();
    renderer.dispose();
  }, { once: true });
} catch (error) {
  console.error(error);
  status.textContent = `Không mở được khung 3D. ${error.message}`;
  document.body.dataset.error = error.message;
}
