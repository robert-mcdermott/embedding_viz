import './styles.css';
import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createIcons, icons } from 'lucide';

const DATA_URL = '/data/embedding-map.json';
const DEFAULT_CAMERA = new THREE.Vector3(0, 7, 82);
const BASE_POINT_SIZE = 0.135;

const els = {
  host: document.querySelector('#scene-host'),
  loading: document.querySelector('#loading'),
  loadingStatus: document.querySelector('#loading-status'),
  tooltip: document.querySelector('#tooltip'),
  wordCount: document.querySelector('#word-count'),
  clusterCount: document.querySelector('#cluster-count'),
  rendererMode: document.querySelector('#renderer-mode'),
  search: document.querySelector('#word-search'),
  searchResults: document.querySelector('#search-results'),
  clusterList: document.querySelector('#cluster-list'),
  detailWord: document.querySelector('#detail-word'),
  detailCluster: document.querySelector('#detail-cluster'),
  neighborList: document.querySelector('#neighbor-list'),
  resetView: document.querySelector('#reset-view'),
  toggleMotion: document.querySelector('#toggle-motion'),
  fitSelection: document.querySelector('#fit-selection'),
  clearFocus: document.querySelector('#clear-focus'),
  pointSize: document.querySelector('#point-size'),
  glowSize: document.querySelector('#glow-size'),
};

createIcons({ icons });

let renderer;
let scene;
let camera;
let controls;
let points;
let glowPoints;
let edgeLines;
let selectionMarker;
let atlas;
let colors;
let glowColors;
let baseColors;
let positions;
let activeCluster = null;
let selectedIndex = null;
let hoveredIndex = null;
let motionEnabled = true;
let cameraGoal = null;
let targetGoal = null;

const pointer = new THREE.Vector2();
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.42;

init().catch((error) => {
  console.error(error);
  els.loading.classList.add('is-error');
  els.loadingStatus.textContent = error.message;
});

async function init() {
  setLoading('Loading embedding-map.json');
  atlas = await fetchAtlas();

  setLoading('Starting WebGPU renderer');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050505);
  scene.fog = new THREE.FogExp2(0x050505, 0.0085);

  camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.05, 1200);
  camera.position.copy(DEFAULT_CAMERA);

  renderer = new THREE.WebGPURenderer({
    antialias: true,
    alpha: false,
    logarithmicDepthBuffer: true,
  });
  await renderer.init();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setAnimationLoop(animate);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.25;
  els.host.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.055;
  controls.rotateSpeed = 0.34;
  controls.zoomSpeed = 0.72;
  controls.panSpeed = 0.46;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.38;
  controls.minDistance = 4;
  controls.maxDistance = 170;

  buildPointCloud();
  buildSceneGuides();
  buildClusterPanel();
  updateStats();
  bindEvents();
  setRendererMode();
  setLoading('Rendering');
  requestAnimationFrame(() => els.loading.classList.add('is-hidden'));
}

async function fetchAtlas() {
  const response = await fetch(DATA_URL);
  if (!response.ok) {
    throw new Error(`Missing ${DATA_URL}. Run npm run build:data first.`);
  }
  return response.json();
}

function buildPointCloud() {
  const pointCount = atlas.points.length;
  positions = new Float32Array(pointCount * 3);
  colors = new Float32Array(pointCount * 3);
  glowColors = new Float32Array(pointCount * 3);
  baseColors = new Float32Array(pointCount * 3);

  const clusterColors = new Map(atlas.clusters.map((cluster) => [cluster.id, new THREE.Color(cluster.color)]));
  for (let index = 0; index < pointCount; index += 1) {
    const point = atlas.points[index];
    const offset = index * 3;
    positions[offset] = point.x;
    positions[offset + 1] = point.y;
    positions[offset + 2] = point.z;

    const color = clusterColors.get(point.c) ?? new THREE.Color('#ffffff');
    const depthLift = 0.82 + Math.min(Math.abs(point.z) / 60, 0.28);
    baseColors[offset] = Math.min(color.r * depthLift, 1);
    baseColors[offset + 1] = Math.min(color.g * depthLift, 1);
    baseColors[offset + 2] = Math.min(color.b * depthLift, 1);
  }
  colors.set(baseColors);
  glowColors.set(baseColors.map((value) => value * 0.62));

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();

  const glowGeometry = geometry.clone();
  glowGeometry.setAttribute('color', new THREE.BufferAttribute(glowColors, 3));

  points = new THREE.Points(
    geometry,
    new THREE.PointsMaterial({
      size: BASE_POINT_SIZE,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );

  glowPoints = new THREE.Points(
    glowGeometry,
    new THREE.PointsMaterial({
      size: Number(els.glowSize.value),
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.18,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );

  edgeLines = new THREE.LineSegments(
    createEmptyLineGeometry(),
    new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.78,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );
  edgeLines.visible = false;

  selectionMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.48, 24, 16),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );

  scene.add(glowPoints);
  scene.add(points);
  scene.add(edgeLines);
  scene.add(selectionMarker);
}

function buildSceneGuides() {
  const grid = new THREE.PolarGridHelper(48, 10, 18, 96, 0x34302b, 0x1e1d1a);
  grid.rotation.x = Math.PI / 2;
  grid.material.transparent = true;
  grid.material.opacity = 0.24;
  scene.add(grid);

  const shell = new THREE.LineSegments(
    new THREE.WireframeGeometry(new THREE.SphereGeometry(43, 32, 16)),
    new THREE.LineBasicMaterial({
      color: 0x4a423a,
      transparent: true,
      opacity: 0.1,
      depthWrite: false,
    }),
  );
  scene.add(shell);
}

function bindEvents() {
  window.addEventListener('resize', resize);
  renderer.domElement.addEventListener('pointermove', onPointerMove);
  renderer.domElement.addEventListener('pointerleave', clearHover);
  renderer.domElement.addEventListener('click', () => {
    if (hoveredIndex !== null) {
      selectPoint(hoveredIndex, true);
    }
  });

  els.resetView.addEventListener('click', resetView);
  els.fitSelection.addEventListener('click', () => {
    if (selectedIndex !== null) flyToPoint(selectedIndex);
  });
  els.clearFocus.addEventListener('click', clearFocus);
  els.toggleMotion.addEventListener('click', toggleMotion);
  els.pointSize.addEventListener('input', () => {
    points.material.size = Number(els.pointSize.value);
  });
  els.glowSize.addEventListener('input', () => {
    glowPoints.material.size = Number(els.glowSize.value);
  });
  els.search.addEventListener('input', renderSearchResults);
  els.search.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      const first = els.searchResults.querySelector('[data-index]');
      if (first) selectPoint(Number(first.dataset.index), true);
    }
    if (event.key === 'Escape') {
      els.search.value = '';
      renderSearchResults();
    }
  });
}

function buildClusterPanel() {
  const fragment = document.createDocumentFragment();
  atlas.clusters
    .slice()
    .sort((a, b) => b.count - a.count)
    .forEach((cluster) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'cluster-button';
      button.dataset.cluster = String(cluster.id);
      button.innerHTML = `
        <span class="cluster-swatch" style="--cluster-color:${cluster.color}"></span>
        <span class="cluster-copy">
          <strong>${escapeHtml(cluster.label)}</strong>
          <small>${cluster.count.toLocaleString()} words</small>
        </span>
      `;
      button.addEventListener('click', () => setActiveCluster(cluster.id));
      fragment.appendChild(button);
    });
  els.clusterList.appendChild(fragment);
}

function updateStats() {
  els.wordCount.textContent = atlas.meta.count.toLocaleString();
  els.clusterCount.textContent = atlas.clusters.length.toLocaleString();
}

function setRendererMode() {
  const mode = renderer.backend?.isWebGPUBackend ? 'WebGPU' : 'WebGL2';
  els.rendererMode.textContent = mode;
  document.body.dataset.renderer = mode.toLowerCase();
}

function onPointerMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(points, false)[0];
  if (!hit) {
    clearHover();
    return;
  }

  hoveredIndex = hit.index;
  showTooltip(hoveredIndex, event.clientX, event.clientY);
  updateEdges(hoveredIndex, true);
}

function showTooltip(index, x, y) {
  const point = atlas.points[index];
  const cluster = atlas.clusters[point.c];
  els.tooltip.innerHTML = `
    <strong>${escapeHtml(point.w)}</strong>
    <span>${escapeHtml(cluster?.label ?? `Cluster ${point.c + 1}`)}</span>
  `;
  els.tooltip.style.transform = `translate(${x + 16}px, ${y + 16}px)`;
  els.tooltip.classList.add('is-visible');
}

function clearHover() {
  hoveredIndex = null;
  els.tooltip.classList.remove('is-visible');
  if (selectedIndex !== null) {
    updateEdges(selectedIndex, false);
  } else {
    updateEdges(null);
  }
}

function selectPoint(index, frame = false) {
  selectedIndex = index;
  const point = atlas.points[index];
  const cluster = atlas.clusters[point.c];
  setActiveCluster(point.c, false);
  updateDetails(index);
  updateSelectionMarker(index);
  updateEdges(index, false);
  if (frame) flyToPoint(index);
  els.search.value = point.w;
  renderSearchResults();
  if (cluster) highlightClusterButton(cluster.id);
}

function updateDetails(index) {
  const point = atlas.points[index];
  const cluster = atlas.clusters[point.c];
  els.detailWord.textContent = point.w;
  els.detailCluster.textContent = cluster ? cluster.label : `Cluster ${point.c + 1}`;
  els.neighborList.innerHTML = '';

  for (const [neighborIndex, weight] of point.n.slice(0, 5)) {
    const neighbor = atlas.points[neighborIndex];
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'neighbor-chip';
    button.textContent = `${neighbor.w} ${(weight * 100).toFixed(0)}`;
    button.addEventListener('click', () => selectPoint(neighborIndex, true));
    els.neighborList.appendChild(button);
  }
}

function updateSelectionMarker(index) {
  const point = atlas.points[index];
  selectionMarker.position.set(point.x, point.y, point.z);
  selectionMarker.material.opacity = 0.78;
}

function updateEdges(index, isHover = false) {
  if (index === null || index === undefined) {
    edgeLines.geometry.dispose();
    edgeLines.geometry = createEmptyLineGeometry();
    edgeLines.visible = false;
    return;
  }

  const point = atlas.points[index];
  const origin = new THREE.Vector3(point.x, point.y, point.z);
  const linePositions = [];
  const lineColors = [];
  const originColor = new THREE.Color('#ffffff');
  const clusterColor = new THREE.Color(atlas.clusters[point.c]?.color ?? '#ffffff');
  const neighborLimit = isHover ? 4 : 6;

  for (const [neighborIndex, weight] of point.n.slice(0, neighborLimit)) {
    const neighbor = atlas.points[neighborIndex];
    const target = new THREE.Vector3(neighbor.x, neighbor.y, neighbor.z);
    const lift = origin.clone().add(target).multiplyScalar(0.5).normalize().multiplyScalar(2.5 + weight * 4);
    const midpoint = origin.clone().add(target).multiplyScalar(0.5).add(lift);
    addSegment(linePositions, lineColors, origin, midpoint, originColor, clusterColor);
    addSegment(linePositions, lineColors, midpoint, target, clusterColor, new THREE.Color(atlas.clusters[neighbor.c]?.color ?? '#ffffff'));
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));
  edgeLines.geometry.dispose();
  edgeLines.geometry = geometry;
  edgeLines.visible = true;
}

function addSegment(linePositions, lineColors, start, end, startColor, endColor) {
  linePositions.push(start.x, start.y, start.z, end.x, end.y, end.z);
  lineColors.push(startColor.r, startColor.g, startColor.b, endColor.r, endColor.g, endColor.b);
}

function createEmptyLineGeometry() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute([], 3));
  return geometry;
}

function setActiveCluster(clusterId, allowToggle = true) {
  activeCluster = allowToggle && activeCluster === clusterId ? null : clusterId;
  recolorPoints();
  highlightClusterButton(activeCluster);
}

function highlightClusterButton(clusterId) {
  for (const button of els.clusterList.querySelectorAll('.cluster-button')) {
    button.classList.toggle('is-active', clusterId !== null && Number(button.dataset.cluster) === clusterId);
  }
}

function recolorPoints() {
  for (let index = 0; index < atlas.points.length; index += 1) {
    const offset = index * 3;
    const isActive = activeCluster === null || atlas.points[index].c === activeCluster;
    const boost = isActive ? 1.14 : 0.16;
    colors[offset] = Math.min(baseColors[offset] * boost, 1);
    colors[offset + 1] = Math.min(baseColors[offset + 1] * boost, 1);
    colors[offset + 2] = Math.min(baseColors[offset + 2] * boost, 1);
    glowColors[offset] = Math.min(baseColors[offset] * (isActive ? 0.72 : 0.04), 1);
    glowColors[offset + 1] = Math.min(baseColors[offset + 1] * (isActive ? 0.72 : 0.04), 1);
    glowColors[offset + 2] = Math.min(baseColors[offset + 2] * (isActive ? 0.72 : 0.04), 1);
  }
  points.geometry.attributes.color.needsUpdate = true;
  glowPoints.geometry.attributes.color.needsUpdate = true;
}

function renderSearchResults() {
  const query = els.search.value.trim().toLocaleLowerCase();
  els.searchResults.innerHTML = '';
  if (!query || (selectedIndex !== null && atlas.points[selectedIndex]?.w.toLocaleLowerCase() === query)) {
    els.searchResults.classList.remove('is-visible');
    return;
  }

  const matches = [];
  for (let index = 0; index < atlas.points.length && matches.length < 9; index += 1) {
    const word = atlas.points[index].w;
    const lower = word.toLocaleLowerCase();
    if (lower === query || lower.startsWith(query) || lower.includes(query)) {
      matches.push([index, word]);
    }
  }

  for (const [index, word] of matches) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.index = String(index);
    button.textContent = word;
    button.addEventListener('click', () => selectPoint(index, true));
    els.searchResults.appendChild(button);
  }
  els.searchResults.classList.toggle('is-visible', matches.length > 0);
}

function flyToPoint(index) {
  const point = atlas.points[index];
  const target = new THREE.Vector3(point.x, point.y, point.z);
  const direction = camera.position.clone().sub(controls.target).normalize();
  if (direction.lengthSq() < 0.1) direction.set(0.2, 0.16, 1).normalize();
  targetGoal = target;
  cameraGoal = target.clone().add(direction.multiplyScalar(15));
}

function resetView() {
  activeCluster = null;
  selectedIndex = null;
  els.search.value = '';
  renderSearchResults();
  recolorPoints();
  highlightClusterButton(null);
  updateEdges(null);
  selectionMarker.material.opacity = 0;
  els.detailWord.textContent = 'none';
  els.detailCluster.textContent = '...';
  els.neighborList.innerHTML = '';
  targetGoal = new THREE.Vector3(0, 0, 0);
  cameraGoal = DEFAULT_CAMERA.clone();
}

function clearFocus() {
  activeCluster = null;
  selectedIndex = null;
  recolorPoints();
  highlightClusterButton(null);
  updateEdges(null);
  selectionMarker.material.opacity = 0;
  els.search.value = '';
  renderSearchResults();
  els.detailWord.textContent = 'none';
  els.detailCluster.textContent = '...';
  els.neighborList.innerHTML = '';
}

function toggleMotion() {
  motionEnabled = !motionEnabled;
  controls.autoRotate = motionEnabled;
  els.toggleMotion.setAttribute('aria-label', motionEnabled ? 'Pause motion' : 'Resume motion');
  els.toggleMotion.setAttribute('title', motionEnabled ? 'Pause motion' : 'Resume motion');
  els.toggleMotion.innerHTML = `<i data-lucide="${motionEnabled ? 'pause' : 'play'}" aria-hidden="true"></i>`;
  createIcons({ icons });
}

function resize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate(time) {
  if (points) {
    const pulse = Math.sin(time * 0.0016) * 0.018;
    glowPoints.material.opacity = motionEnabled ? 0.18 + pulse : 0.16;
    edgeLines.material.opacity = 0.66 + Math.sin(time * 0.002) * 0.08;
  }

  if (targetGoal && cameraGoal) {
    controls.target.lerp(targetGoal, 0.08);
    camera.position.lerp(cameraGoal, 0.08);
    if (controls.target.distanceTo(targetGoal) < 0.02 && camera.position.distanceTo(cameraGoal) < 0.04) {
      targetGoal = null;
      cameraGoal = null;
    }
  }

  if (selectionMarker.material.opacity > 0) {
    const scale = 1 + Math.sin(time * 0.004) * 0.16;
    selectionMarker.scale.setScalar(scale);
  }

  controls?.update();
  renderer.render(scene, camera);
}

function setLoading(message) {
  els.loadingStatus.textContent = message;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const entities = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;',
    };
    return entities[char];
  });
}
