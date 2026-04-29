import './styles.css';
import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createIcons, icons } from 'lucide';

const DATA_URL = '/data/embedding-map.json';
const DEFAULT_CAMERA = new THREE.Vector3(0, 7, 82);
const DEFAULT_POINT_SIZE = 0.14;
const DEFAULT_GLOW_SIZE = 0.44;
const MAX_GLOW_SIZE = 1.25;

const els = {
  host: document.querySelector('#scene-host'),
  loading: document.querySelector('#loading'),
  loadingStatus: document.querySelector('#loading-status'),
  tooltip: document.querySelector('#tooltip'),
  selectedLabel: document.querySelector('#selected-label'),
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
  neighborLinks: document.querySelector('#neighbor-links'),
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
const instanceDummy = new THREE.Object3D();
const instanceColor = new THREE.Color();
const projectedPosition = new THREE.Vector3();
let activeCluster = null;
let selectedIndex = null;
let hoveredIndex = null;
let motionEnabled = true;
let neighborLinksEnabled = true;
let controlsInteracting = false;
let selectedLabelActive = false;
let pointScale = DEFAULT_POINT_SIZE;
let glowScale = DEFAULT_GLOW_SIZE;
let selectedRingScale = 1;
let cameraGoal = null;
let targetGoal = null;

const pointer = new THREE.Vector2();
const raycaster = new THREE.Raycaster();

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
  controls.addEventListener('start', handleControlsStart);
  controls.addEventListener('end', handleControlsEnd);

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

  const pointGeometry = new THREE.OctahedronGeometry(1, 0);
  const glowGeometry = new THREE.IcosahedronGeometry(1, 1);

  points = new THREE.InstancedMesh(
    pointGeometry,
    new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
      blending: THREE.NormalBlending,
      toneMapped: false,
    }),
    pointCount,
  );

  glowPoints = new THREE.InstancedMesh(
    glowGeometry,
    new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.055,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    }),
    pointCount,
  );
  points.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  glowPoints.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  points.frustumCulled = false;
  glowPoints.frustumCulled = false;
  updatePointScale(Number(els.pointSize.value) || DEFAULT_POINT_SIZE);
  updateGlowScale(Number(els.glowSize.value));
  updateInstanceColors();

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
    new THREE.TorusGeometry(0.36, 0.018, 8, 54),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      toneMapped: false,
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
    updatePointScale(Number(els.pointSize.value));
  });
  els.glowSize.addEventListener('input', () => {
    updateGlowScale(Number(els.glowSize.value));
  });
  els.neighborLinks.addEventListener('change', toggleNeighborLinks);
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

  hoveredIndex = hit.instanceId;
  if (hoveredIndex === null || hoveredIndex === undefined) return;
  showTooltip(hoveredIndex, event.clientX, event.clientY);
  if (neighborLinksEnabled) updateEdges(hoveredIndex, true);
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
  if (neighborLinksEnabled && selectedIndex !== null) {
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
  if (neighborLinksEnabled) {
    updateEdges(index, false);
  } else {
    updateEdges(null);
  }
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
  const color = new THREE.Color(atlas.clusters[point.c]?.color ?? '#ffffff');
  selectionMarker.position.set(point.x, point.y, point.z);
  selectionMarker.material.color.copy(color);
  selectionMarker.material.opacity = 0.86;
  selectedRingScale = Math.max(0.72, pointScale * 4.4);
  selectionMarker.scale.setScalar(selectedRingScale);
  updatePointScale(pointScale);
  updateSelectedLabel(index, color);
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
  updateInstanceColors();
}

function updatePointScale(scale) {
  pointScale = Math.max(Number(scale) || DEFAULT_POINT_SIZE, 0.001);
  updateInstanceScale(points, pointScale);
  if (selectedIndex !== null) {
    selectedRingScale = Math.max(0.72, pointScale * 4.4);
  }
}

function updateGlowScale(scale) {
  glowScale = Math.max(Number(scale) || 0, 0);
  glowPoints.visible = glowScale > 0;
  if (glowScale > 0) updateInstanceScale(glowPoints, glowScale);
}

function updateInstanceScale(mesh, scale) {
  const safeScale = Math.max(Number(scale) || DEFAULT_POINT_SIZE, 0.001);
  for (let index = 0; index < atlas.points.length; index += 1) {
    const point = atlas.points[index];
    const selectedBoost = mesh === points && selectedIndex === index ? 1.8 : 1;
    instanceDummy.position.set(point.x, point.y, point.z);
    instanceDummy.scale.setScalar(safeScale * selectedBoost);
    instanceDummy.updateMatrix();
    mesh.setMatrixAt(index, instanceDummy.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
}

function updateInstanceColors() {
  for (let index = 0; index < atlas.points.length; index += 1) {
    const offset = index * 3;
    instanceColor.setRGB(colors[offset], colors[offset + 1], colors[offset + 2]);
    points.setColorAt(index, instanceColor);
    instanceColor.setRGB(glowColors[offset], glowColors[offset + 1], glowColors[offset + 2]);
    glowPoints.setColorAt(index, instanceColor);
  }
  points.instanceColor.needsUpdate = true;
  glowPoints.instanceColor.needsUpdate = true;
}

function updateSelectedLabel(index, color) {
  selectedLabelActive = true;
  els.selectedLabel.textContent = atlas.points[index].w;
  els.selectedLabel.style.setProperty('--selected-color', color.getStyle());
  els.selectedLabel.classList.add('is-visible');
  updateSelectedLabelPosition();
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
  controls.autoRotate = false;
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
  selectedLabelActive = false;
  els.selectedLabel.classList.remove('is-visible');
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
  selectedLabelActive = false;
  els.selectedLabel.classList.remove('is-visible');
  els.search.value = '';
  renderSearchResults();
  els.detailWord.textContent = 'none';
  els.detailCluster.textContent = '...';
  els.neighborList.innerHTML = '';
}

function toggleMotion() {
  motionEnabled = !motionEnabled;
  controls.autoRotate = motionEnabled && !controlsInteracting && !cameraGoal;
  els.toggleMotion.setAttribute('aria-label', motionEnabled ? 'Pause motion' : 'Resume motion');
  els.toggleMotion.setAttribute('title', motionEnabled ? 'Pause motion' : 'Resume motion');
  els.toggleMotion.innerHTML = `<i data-lucide="${motionEnabled ? 'pause' : 'play'}" aria-hidden="true"></i>`;
  createIcons({ icons });
}

function handleControlsStart() {
  controlsInteracting = true;
  cancelCameraFlight();
  controls.autoRotate = false;
}

function handleControlsEnd() {
  controlsInteracting = false;
  controls.autoRotate = motionEnabled && !cameraGoal;
}

function cancelCameraFlight() {
  targetGoal = null;
  cameraGoal = null;
}

function toggleNeighborLinks() {
  neighborLinksEnabled = els.neighborLinks.checked;
  if (!neighborLinksEnabled) {
    updateEdges(null);
    return;
  }
  if (hoveredIndex !== null) {
    updateEdges(hoveredIndex, true);
  } else if (selectedIndex !== null) {
    updateEdges(selectedIndex, false);
  }
}

function resize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate(time) {
  if (points) {
    const glowAmount = Math.min(glowScale / MAX_GLOW_SIZE, 1);
    const pulse = Math.sin(time * 0.0016) * 0.012 * glowAmount;
    glowPoints.material.opacity = glowScale > 0 ? 0.015 + glowAmount * 0.105 + pulse : 0;
    edgeLines.material.opacity = 0.66 + Math.sin(time * 0.002) * 0.08;
  }

  if (targetGoal && cameraGoal) {
    controls.target.lerp(targetGoal, 0.08);
    camera.position.lerp(cameraGoal, 0.08);
    if (controls.target.distanceTo(targetGoal) < 0.02 && camera.position.distanceTo(cameraGoal) < 0.04) {
      targetGoal = null;
      cameraGoal = null;
      controls.autoRotate = motionEnabled && !controlsInteracting;
    }
  }

  if (selectionMarker.material.opacity > 0) {
    selectionMarker.lookAt(camera.position);
    const scale = selectedRingScale * (1 + Math.sin(time * 0.004) * 0.045);
    selectionMarker.scale.setScalar(scale);
    updateSelectedLabelPosition();
  }

  controls?.update();
  renderer.render(scene, camera);
}

function updateSelectedLabelPosition() {
  if (selectedIndex === null || !selectedLabelActive) return;
  const point = atlas.points[selectedIndex];
  projectedPosition.set(point.x, point.y, point.z).project(camera);
  const isVisible =
    projectedPosition.z > -1 &&
    projectedPosition.z < 1 &&
    Math.abs(projectedPosition.x) < 1.25 &&
    Math.abs(projectedPosition.y) < 1.25;
  els.selectedLabel.classList.toggle('is-visible', isVisible);
  if (!isVisible) return;
  const x = (projectedPosition.x * 0.5 + 0.5) * window.innerWidth;
  const y = (-projectedPosition.y * 0.5 + 0.5) * window.innerHeight;
  els.selectedLabel.style.transform = `translate(${x}px, ${y}px) translate(-50%, -130%)`;
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
