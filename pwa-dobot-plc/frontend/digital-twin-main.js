import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OPERATION_MODE, MODE } from './digital-twin-io-config.js';

// Debug mode flag - set to true to enable coordinate display and hover tile
const DEBUG_MODE = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x151515);

// Simple in-app logger
const logBuffer = [];
function log(msg) {
 const line = `[${new Date().toISOString()}] ${msg}`;
 logBuffer.push(line);
 if (logBuffer.length > 2000) logBuffer.shift();
 console.log(line);
 // send to local server log
 fetch('/log', {
 method: 'POST',
 headers: { 'Content-Type': 'application/json' },
 body: JSON.stringify({ message: msg })
 }).catch(() => {});
}
window.__getLog = () => logBuffer.join('\n');
log('App start');

// ============================================
// Digital Twin Metrics & Data Tracking System
// ============================================

class MetricsTracker {
 constructor() {
 this.startTime = Date.now();
 this.boxesProcessed = 0;
 this.boxStartTimes = new Map(); // track individual box cycle times
 this.cycleTimes = [];
 this.throughputHistory = []; // { timestamp, count }
 this.lastThroughputUpdate = Date.now();
 this.boxesInLastMinute = 0;

 // Equipment status
 this.equipment = {
 gantry: { status: 'idle', utilization: 0, lastActive: 0 },
 conveyor1: { status: 'running', utilization: 100 },
 conveyor2: { status: 'running', utilization: 100 },
 robot: { status: 'ready', utilization: 0, lastActive: 0 }
 };

 // Queue tracking
 this.queueStats = {
 waiting: 0,
 inTransit: 0,
 onPallet: 0
 };
 }

 // Called when a box starts its journey
 boxStarted(boxId) {
 this.boxStartTimes.set(boxId, Date.now());
 }

 // Called when a box completes its journey
 boxCompleted(boxId) {
 const startTime = this.boxStartTimes.get(boxId);
 if (startTime) {
 const cycleTime = (Date.now() - startTime) / 1000; // seconds
 this.cycleTimes.push(cycleTime);
 if (this.cycleTimes.length > 100) this.cycleTimes.shift(); // keep last 100
 this.boxStartTimes.delete(boxId);
 }
 this.boxesProcessed++;
 this.boxesInLastMinute++;
 }

 // Update throughput calculations
 updateThroughput() {
 const now = Date.now();
 const timeSince = (now - this.lastThroughputUpdate) / 1000;

 if (timeSince >= 1.0) { // update every second
 const rate = this.boxesInLastMinute / (timeSince / 60); // boxes per minute
 this.throughputHistory.push({ timestamp: now, rate });

 // keep last 60 seconds
 const cutoff = now - 60000;
 this.throughputHistory = this.throughputHistory.filter(h => h.timestamp > cutoff);

 this.lastThroughputUpdate = now;
 this.boxesInLastMinute = 0;
 }
 }

 // Get average cycle time
 getAvgCycleTime() {
 if (this.cycleTimes.length === 0) return 0;
 const sum = this.cycleTimes.reduce((a, b) => a + b, 0);
 return sum / this.cycleTimes.length;
 }

 // Get current throughput (boxes/min)
 getCurrentThroughput() {
 if (this.throughputHistory.length === 0) return 0;
 // average of last few readings
 const recent = this.throughputHistory.slice(-10);
 const sum = recent.reduce((a, b) => a + b.rate, 0);
 return sum / recent.length;
 }

 // Update equipment status
 setEquipmentStatus(equipmentId, status) {
 if (this.equipment[equipmentId]) {
 this.equipment[equipmentId].status = status;
 if (status === 'running' || status === 'active') {
 this.equipment[equipmentId].lastActive = Date.now();
 }
 }
 }

 // Update queue statistics
 updateQueueStats(waiting, inTransit, onPallet) {
 this.queueStats.waiting = waiting;
 this.queueStats.inTransit = inTransit;
 this.queueStats.onPallet = onPallet;
 }

 // API for external data injection (for hybrid mode)
 injectExternalData(data) {
 // This allows real sensor data to override simulated data
 if (data.equipment) {
 Object.keys(data.equipment).forEach(key => {
 if (this.equipment[key]) {
 this.equipment[key] = { ...this.equipment[key], ...data.equipment[key] };
 }
 });
 }
 if (data.throughput !== undefined) {
 this.throughputHistory.push({
 timestamp: Date.now(),
 rate: data.throughput
 });
 }
 log('External data injected: ' + JSON.stringify(data));
 }
}

const metrics = new MetricsTracker();
window.__metrics = metrics; // expose for external access

// Make metrics available globally for real data injection
window.injectFactoryData = (data) => {
 metrics.injectExternalData(data);
};

// ============================================
// WebSocket Connection for Real-Time Data
// ============================================

let ws = null;
let wsReconnectTimeout = null;

function connectWebSocket() {
 try {
 const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
 const wsUrl = `${protocol}//${window.location.host}`;

 ws = new WebSocket(wsUrl);

 ws.onopen = () => {
 log('WebSocket connected - ready for real-time data');
 console.log('[WebSocket] Connected to server');
 };

 ws.onmessage = (event) => {
 try {
 const message = JSON.parse(event.data);

 if (message.type === 'factoryData') {
 // Inject external data into metrics
 metrics.injectExternalData(message.data);
 console.log('[WebSocket] Data injected:', message.data);
 } else if (message.type === 'connected') {
 console.log('[WebSocket]', message.message);
 }
 } catch (e) {
 console.error('[WebSocket] Message parse error:', e);
 }
 };

 ws.onerror = (error) => {
 console.log('[WebSocket] Connection error (expected if no WebSocket server)');
 };

 ws.onclose = () => {
 console.log('[WebSocket] Connection closed');
 ws = null;

 // Attempt to reconnect after 5 seconds
 if (wsReconnectTimeout) clearTimeout(wsReconnectTimeout);
 wsReconnectTimeout = setTimeout(() => {
 console.log('[WebSocket] Attempting to reconnect...');
 connectWebSocket();
 }, 5000);
 };
 } catch (e) {
 console.log('[WebSocket] Not available - running in simulation mode');
 }
}

// Connect to WebSocket if available
connectWebSocket();

log('Metrics tracker initialized');

// ============================================
// Dashboard UI Update Functions
// ============================================

function updateDashboard() {
 // Check if dashboard elements exist before updating
 const throughputEl = document.getElementById('throughput');
 if (!throughputEl) return; // Dashboard not present in UI
 
 // Update throughput
 const throughput = metrics.getCurrentThroughput();
 throughputEl.textContent = throughput.toFixed(1) + ' boxes/min';

 // Update boxes processed
 const boxesEl = document.getElementById('boxesProcessed');
 if (boxesEl) boxesEl.textContent = metrics.boxesProcessed;

 // Update average cycle time
 const avgCycle = metrics.getAvgCycleTime();
 const cycleEl = document.getElementById('cycleTime');
 if (cycleEl) cycleEl.textContent = avgCycle.toFixed(1) + 's';

 // Update equipment status
 updateEquipmentUI('gantry', metrics.equipment.gantry.status);
 updateEquipmentUI('conv1', metrics.equipment.conveyor1.status);
 updateEquipmentUI('conv2', metrics.equipment.conveyor2.status);
 updateEquipmentUI('robot', metrics.equipment.robot.status);

 // Update queue stats
 const queueWaitingEl = document.getElementById('queueWaiting');
 const inTransitEl = document.getElementById('inTransit');
 const onPalletEl = document.getElementById('onPallet');
 if (queueWaitingEl) queueWaitingEl.textContent = metrics.queueStats.waiting;
 if (inTransitEl) inTransitEl.textContent = metrics.queueStats.inTransit;
 if (onPalletEl) onPalletEl.textContent = metrics.queueStats.onPallet;

 // Update throughput chart
 updateThroughputChart();
}

function updateEquipmentUI(equipmentId, status) {
 const indicator = document.getElementById(equipmentId + '-status');
 const stateText = document.getElementById(equipmentId + '-state');
 
 if (!indicator || !stateText) return; // Elements don't exist

 // Update indicator class
 indicator.className = 'status-indicator';
 if (status === 'running' || status === 'active') {
 indicator.classList.add('running');
 stateText.textContent = status === 'running' ? 'Running' : 'Active';
 } else if (status === 'idle' || status === 'ready') {
 indicator.classList.add('idle');
 stateText.textContent = status === 'idle' ? 'Idle' : 'Ready';
 } else {
 indicator.classList.add('stopped');
 stateText.textContent = 'Stopped';
 }
}

function updateThroughputChart() {
 const canvas = document.getElementById('throughputChart');
 if (!canvas) return;

 const ctx = canvas.getContext('2d');
 const width = canvas.width;
 const height = canvas.height;

 // Clear canvas
 ctx.clearRect(0, 0, width, height);

 const history = metrics.throughputHistory;
 if (history.length < 2) return;

 // Find max for scaling
 const maxRate = Math.max(...history.map(h => h.rate), 1);

 // Draw chart
 ctx.strokeStyle = '#4CAF50';
 ctx.lineWidth = 2;
 ctx.beginPath();

 history.forEach((point, i) => {
 const x = (i / (history.length - 1)) * width;
 const y = height - (point.rate / maxRate) * height;
 if (i === 0) {
 ctx.moveTo(x, y);
 } else {
 ctx.lineTo(x, y);
 }
 });

 ctx.stroke();

 // Draw baseline
 ctx.strokeStyle = 'rgba(255,255,255,0.1)';
 ctx.lineWidth = 1;
 ctx.beginPath();
 ctx.moveTo(0, height);
 ctx.lineTo(width, height);
 ctx.stroke();
}

// Get the canvas container
const container = document.getElementById('canvas-container');
const containerWidth = container.clientWidth;
const containerHeight = container.clientHeight;

const camera = new THREE.PerspectiveCamera(45, containerWidth/containerHeight, 0.1, 1000);
camera.position.set(0, 8, 14);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(containerWidth, containerHeight);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.5, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI / 2.1;

// Lights
const hemi = new THREE.HemisphereLight(0xffffff, 0x222222, 1.0);
scene.add(hemi);
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(5, 10, 5);
scene.add(dir);

// Floor
const floorGeo = new THREE.PlaneGeometry(30, 20);
const floorMat = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.9 });
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI / 2;
scene.add(floor);

// Dimensions (metres)
const conveyorLength = 5.0; // 0.5m units * 10 = 5m for visibility
const conveyorWidth = 0.6;
const conveyorHeight = 0.4;
const conveyorGap = 3.0;

// Conveyor geometry
function makeConveyor(x, z) {
 const base = new THREE.Mesh(
 new THREE.BoxGeometry(conveyorLength, 0.15, conveyorWidth),
 new THREE.MeshStandardMaterial({ color: 0x3a3a3a, metalness: 0.1, roughness: 0.7 })
 );
 base.position.set(x, conveyorHeight, z);
 scene.add(base);

 const belt = new THREE.Mesh(
 new THREE.BoxGeometry(conveyorLength * 0.98, 0.06, conveyorWidth * 0.92),
 new THREE.MeshStandardMaterial({ color: 0x1f1f1f, metalness: 0.05, roughness: 0.9 })
 );
 belt.position.set(x, conveyorHeight + 0.11, z);
 scene.add(belt);

 return { base, belt };
}

const conveyor1 = makeConveyor(0, -3); // north/top
const conveyor2 = makeConveyor(0, 3); // south/bottom

// Defect bin + piston near end of conveyor 1 (west end)
const defectBin = new THREE.Mesh(
 new THREE.BoxGeometry(0.6, 0.4, 0.6),
 new THREE.MeshStandardMaterial({ color: 0x444444 })
);
defectBin.position.set(-2.5, 0.25, -4.0);
scene.add(defectBin);

const piston = new THREE.Mesh(
 new THREE.BoxGeometry(0.2, 0.2, 1.0),
 new THREE.MeshStandardMaterial({ color: 0x888888 })
);
piston.position.set(-2.5, 0.6, -2.5);
scene.add(piston);

// Label for defect bin
const binLabel = new THREE.Mesh(
 new THREE.PlaneGeometry(0.9, 0.25),
 new THREE.MeshStandardMaterial({ color: 0x111111, transparent: true, opacity: 0.8 })
);
binLabel.position.set(-2.5, 0.9, -4.0);
binLabel.rotation.y = Math.PI / 4;
scene.add(binLabel);

// Piston animation - only moves when rejecting
let pistonActive = false;
let pistonT = 0;
function updatePiston(delta) {
 if (pistonActive) {
 pistonT += delta * 3;
 const push = Math.sin(pistonT * 4) > 0.5 ? 0.3 : 0;
 piston.position.z = -2.5 - push;
 } else {
 piston.position.z = -2.5;
 pistonT = 0;
 }
}

// Labels removed

// Gantry
const gantryY = 2.2; // Height above ground
let gantryX = -2.5; // left end default
let gantryZ = 0;

const gantry = new THREE.Group();
const gantryBeam = new THREE.Mesh(
 new THREE.BoxGeometry(0.4, 0.2, conveyorGap + 6.0),
 new THREE.MeshStandardMaterial({ color: 0x777777 })
);

const gantryHead = new THREE.Mesh(
 new THREE.BoxGeometry(0.6, 0.4, 0.6),
 new THREE.MeshStandardMaterial({ color: 0x888888 })
);

const gantryGrip = new THREE.Mesh(
 new THREE.BoxGeometry(0.3, 0.2, 0.3),
 new THREE.MeshStandardMaterial({ color: 0xaaaaaa })
);

// Gantry tower legs at X=-3, Z=4 and X=-3, Z=-4
const gantryLegMaterial = new THREE.MeshStandardMaterial({ color: 0x666666 });
const gantryLeg1 = new THREE.Mesh(
 new THREE.BoxGeometry(0.3, gantryY, 0.3),
 gantryLegMaterial
);
gantryLeg1.position.set(-3, gantryY / 2, 4);

const gantryLeg2 = new THREE.Mesh(
 new THREE.BoxGeometry(0.3, gantryY, 0.3),
 gantryLegMaterial
);
gantryLeg2.position.set(-3, gantryY / 2, -4);

scene.add(gantryLeg1);
scene.add(gantryLeg2);

gantryBeam.position.set(gantryX, gantryY, gantryZ);

const gantryCart = new THREE.Group();
gantryCart.add(gantryHead);

const gripOffsetY = -0.6;
gantryGrip.position.set(0, gripOffsetY, 0);
gantryCart.add(gantryGrip);

scene.add(gantryBeam);
scene.add(gantryCart);

// Keyboard control for gantry base
window.addEventListener('keydown', (e) => {
 const step = e.shiftKey ? 0.5 : 0.2;
 if (e.key === 'ArrowLeft') gantryX -= step;
 if (e.key === 'ArrowRight') gantryX += step;
 if (e.key === 'ArrowUp') gantryZ -= step;
 if (e.key === 'ArrowDown') gantryZ += step;
 gantryBeam.position.set(gantryX, gantryY, gantryZ);
 log(`Gantry base moved to X:${gantryX.toFixed(2)} Z:${gantryZ.toFixed(2)}`);
});

// Robot Arm (simple stylized)
const robot = new THREE.Group();
const robotBaseX = 3.0;
const robotBaseZ = 1.0;

const base = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.4, 0.3, 20), new THREE.MeshStandardMaterial({ color: 0xbfbfbf }));
base.position.set(0, 0.15, 0);
robot.add(base);

const arm1 = new THREE.Mesh(new THREE.BoxGeometry(0.4, 1.2, 0.4), new THREE.MeshStandardMaterial({ color: 0xd0d0d0 }));
arm1.position.set(0, 0.9, 0);
robot.add(arm1);

const arm2 = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.25, 0.25), new THREE.MeshStandardMaterial({ color: 0xd0d0d0 }));
arm2.position.set(0.7, 1.35, 0);
robot.add(arm2);

const gripper = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.1, 0.3), new THREE.MeshStandardMaterial({ color: 0x999999 }));
gripper.position.set(1.4, 1.25, 0);
robot.add(gripper);

robot.position.set(robotBaseX, 0, robotBaseZ);
scene.add(robot);

// Robot animation state
let robotAnimT = 0;
let robotState = 'idle'; // idle, reaching, grabbing, returning, placing

// ============================================
// 3D Status Indicators for Equipment
// ============================================

// Create status lights for equipment
function createStatusLight(x, y, z, label) {
 const light = new THREE.Mesh(
 new THREE.SphereGeometry(0.15, 16, 16),
 new THREE.MeshStandardMaterial({
 color: 0x00ff00,
 emissive: 0x00ff00,
 emissiveIntensity: 0.5
 })
 );
 light.position.set(x, y, z);
 scene.add(light);

 // Add point light for glow effect
 const pointLight = new THREE.PointLight(0x00ff00, 0.5, 3);
 pointLight.position.set(x, y, z);
 scene.add(pointLight);

 return { mesh: light, light: pointLight, label };
}

// Status indicators for each equipment
const statusIndicators = {
 gantry: createStatusLight(gantryX, gantryY + 0.8, gantryZ, 'Gantry'),
 conveyor1: createStatusLight(0, conveyorHeight + 0.6, -3, 'Conv1'),
 conveyor2: createStatusLight(0, conveyorHeight + 0.6, 3, 'Conv2'),
 robot: createStatusLight(3.0, 2.0, 1.0, 'Robot')
};

// Function to update 3D status indicator colors
function update3DStatusIndicators() {
 Object.keys(statusIndicators).forEach(key => {
 const indicator = statusIndicators[key];
 const status = metrics.equipment[key]?.status || 'idle';

 let color;
 let intensity;

 switch (status) {
 case 'running':
 case 'active':
 color = 0x00ff00; // green
 intensity = 0.8;
 break;
 case 'idle':
 case 'ready':
 color = 0xffaa00; // yellow/orange
 intensity = 0.4;
 break;
 default:
 color = 0xff0000; // red
 intensity = 0.2;
 }

 indicator.mesh.material.color.setHex(color);
 indicator.mesh.material.emissive.setHex(color);
 indicator.mesh.material.emissiveIntensity = intensity;
 indicator.light.color.setHex(color);
 indicator.light.intensity = intensity;
 });
}

// ============================================
// Sorting Bays (4 bays for different materials)
// ============================================

const PLATFORM_HEIGHT = 1.0; // Height of raised platform (1 tile)

const sortingBays = [
 { name: 'Steel', position: new THREE.Vector3(1.5, PLATFORM_HEIGHT + 0.35, -3.0), count: 0, maxStack: 4, stackOffsetZ: 0.4 },
 { name: 'Aluminum', position: new THREE.Vector3(2.0, PLATFORM_HEIGHT + 0.35, -3.0), count: 0, maxStack: 4, stackOffsetZ: 0.4 },
 { name: 'Plastic Yellow', position: new THREE.Vector3(2.5, PLATFORM_HEIGHT + 0.35, -3.0), count: 0, maxStack: 4, stackOffsetZ: 0.4 },
 { name: 'Plastic Purple', position: new THREE.Vector3(3.0, PLATFORM_HEIGHT + 0.35, -3.0), count: 0, maxStack: 4, stackOffsetZ: 0.4 }
];

function getSortingBayPosition(bayIndex) {
 const bay = sortingBays[bayIndex];
 if (!bay) return null;

 // Stack horizontally (along Z axis) instead of vertically
 const z = bay.position.z + (bay.count * bay.stackOffsetZ);
 bay.count++;

 return { x: bay.position.x, y: bay.position.y, z };
}

// ============================================
// Sorting Bay 3D Models
// ============================================

// Create thin platform top for sorting bays
const platformTop = new THREE.Mesh(
 new THREE.BoxGeometry(2.0, 0.08, 2.0),
 new THREE.MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.6, metalness: 0.3 })
);
platformTop.position.set(2.25, PLATFORM_HEIGHT, -2.2);
scene.add(platformTop);

// Platform support poles (cylindrical)
const poleMaterial = new THREE.MeshStandardMaterial({ color: 0x555555, roughness: 0.5, metalness: 0.4 });

// Single central column
const centralPole = new THREE.Mesh(
 new THREE.CylinderGeometry(0.1, 0.1, PLATFORM_HEIGHT, 16),
 poleMaterial
);
centralPole.position.set(2.25, PLATFORM_HEIGHT / 2, -2.2);
scene.add(centralPole);

// Create visual bays for sorted materials
sortingBays.forEach((bay, index) => {
 const bayBase = new THREE.Mesh(
 new THREE.BoxGeometry(0.4, 0.1, 0.4),
 new THREE.MeshStandardMaterial({ color: 0x6b4e2e, roughness: 0.8 })
 );
 bayBase.position.set(bay.position.x, PLATFORM_HEIGHT + 0.05, bay.position.z);
 scene.add(bayBase);

 // Bay label (tiny indicator light matching material)
 const materialColors = [0xbfbfbf, 0x999999, 0xd4b000, 0x7a4cff];
 const label = new THREE.Mesh(
 new THREE.SphereGeometry(0.08, 12, 12),
 new THREE.MeshStandardMaterial({
 color: materialColors[index],
 emissive: materialColors[index],
 emissiveIntensity: 0.5
 })
 );
 label.position.set(bay.position.x, PLATFORM_HEIGHT + 0.5, bay.position.z - 0.3);
 scene.add(label);
});

// Central dividers (2 lines creating 4 quadrants)
const dividerMat = new THREE.MeshStandardMaterial({ color: 0x4a4a4a, roughness: 0.7 });

// Vertical divider (separating columns)
const verticalDivider1 = new THREE.Mesh(
 new THREE.BoxGeometry(0.05, 0.3, 2.0),
 dividerMat
);
verticalDivider1.position.set(1.75, PLATFORM_HEIGHT + 0.15, -2.2);
scene.add(verticalDivider1);

const verticalDivider2 = new THREE.Mesh(
 new THREE.BoxGeometry(0.05, 0.3, 2.0),
 dividerMat
);
verticalDivider2.position.set(2.75, PLATFORM_HEIGHT + 0.15, -2.2);
scene.add(verticalDivider2);

// ============================================
// Coordinate Grid Helper (for debugging)
// ============================================

const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
gridHelper.position.y = 0.01;
scene.add(gridHelper);

// Hover tile highlight
const hoverTile = new THREE.Mesh(
 new THREE.PlaneGeometry(1, 1),
 new THREE.MeshBasicMaterial({ 
 color: 0x00ff00, 
 transparent: true, 
 opacity: 0.3,
 side: THREE.DoubleSide
 })
);
hoverTile.rotation.x = -Math.PI / 2;
hoverTile.position.y = 0.02;
hoverTile.visible = false;
scene.add(hoverTile);

// Raycaster for mouse position
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let mouseWorldPos = null;

// Mouse move handler
window.addEventListener('mousemove', (event) => {
 // Calculate mouse position in normalized device coordinates (-1 to +1)
 mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
 mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
 
 // Raycast to floor
 raycaster.setFromCamera(mouse, camera);
 const intersects = raycaster.intersectObject(floor);
 
 if (intersects.length > 0) {
 const point = intersects[0].point;
 mouseWorldPos = point;
 
 // Snap to grid (1m tiles)
 const gridX = Math.round(point.x);
 const gridZ = Math.round(point.z);
 
 // Update hover tile
 hoverTile.position.set(gridX, 0.02, gridZ);
 hoverTile.visible = DEBUG_MODE; // Only show in debug mode
 
 // Update UI if elements exist
 const mouseXEl = document.getElementById('mouseX');
 const mouseZEl = document.getElementById('mouseZ');
 if (mouseXEl) mouseXEl.textContent = gridX.toFixed(1);
 if (mouseZEl) mouseZEl.textContent = gridZ.toFixed(1);
 } else {
 hoverTile.visible = false;
 const mouseXEl = document.getElementById('mouseX');
 const mouseZEl = document.getElementById('mouseZ');
 if (mouseXEl) mouseXEl.textContent = '-';
 if (mouseZEl) mouseZEl.textContent = '-';
 }
});

// Click to log coordinates
window.addEventListener('click', (event) => {
 if (mouseWorldPos) {
 const gridX = Math.round(mouseWorldPos.x);
 const gridZ = Math.round(mouseWorldPos.z);
 log(`CLICK at X:${gridX} Z:${gridZ}`);
 console.log(`%c Clicked Coordinates: X=${gridX}, Z=${gridZ}`, 'color: #4CAF50; font-weight: bold; font-size: 14px;');
 }
});

// Add coordinate markers
function addCoordinateMarker(x, z, label) {
 const markerGeo = new THREE.SphereGeometry(0.1, 8, 8);
 const markerMat = new THREE.MeshBasicMaterial({ color: 0xffff00 });
 const marker = new THREE.Mesh(markerGeo, markerMat);
 marker.position.set(x, 0.1, z);
 scene.add(marker);
 log(`Marker "${label}" at X:${x.toFixed(2)} Z:${z.toFixed(2)}`);
}

// Mark key positions
addCoordinateMarker(2.3, 3, 'Conv2End'); // Where box stops on conveyor
addCoordinateMarker(3.0, 1.0, 'RobotBase'); // Robot base
addCoordinateMarker(2.0, -1.5, 'Bay1'); // First sorting bay

// ============================================
// Sensor System
// ============================================

const sensors = {
 conv1End: { position: -2.5, active: false }, // End of conveyor 1
 conv2Start: { position: -2.5, active: false }, // Start of conveyor 2 (gantry drop)
 metalSensor1: { position: 0, active: false }, // Steel detector on conv2
 metalSensor2: { position: 1.5, active: false }, // Aluminum detector on conv2
 conv2End: { position: 2.3, active: false } // End of conveyor 2 (before robot)
};

// Visual sensor indicators
function createSensorIndicator(x, y, z, label) {
 const indicator = new THREE.Mesh(
 new THREE.BoxGeometry(0.1, 0.1, 0.8),
 new THREE.MeshStandardMaterial({ color: 0xff0000, emissive: 0xff0000, emissiveIntensity: 0.3 })
 );
 indicator.position.set(x, y, z);
 scene.add(indicator);
 return indicator;
}

const sensorIndicators = {
 conv1End: createSensorIndicator(-2.5, conveyorHeight + 0.3, -3, 'Conv1 End'),
 conv2Start: createSensorIndicator(-2.5, conveyorHeight + 0.3, 3, 'Conv2 Start'),
 metalSensor1: createSensorIndicator(0, conveyorHeight + 0.3, 3, 'Metal 1'),
 metalSensor2: createSensorIndicator(1.5, conveyorHeight + 0.3, 3, 'Metal 2'),
 conv2End: createSensorIndicator(2.3, conveyorHeight + 0.3, 3, 'Conv2 End')
};

function updateSensorIndicator(sensorKey, active) {
 const indicator = sensorIndicators[sensorKey];
 if (indicator) {
 indicator.material.color.setHex(active ? 0x00ff00 : 0xff0000);
 indicator.material.emissive.setHex(active ? 0x00ff00 : 0xff0000);
 indicator.material.emissiveIntensity = active ? 0.8 : 0.3;
 }
}

// ============================================
// Material & Defect System
// ============================================

const MATERIALS = {
 STEEL: { color: 0xbfbfbf, name: 'Steel', isMetal: true, bay: 0 },
 ALUMINUM: { color: 0x999999, name: 'Aluminum', isMetal: true, bay: 1 },
 PLASTIC_YELLOW: { color: 0xd4b000, name: 'Plastic Yellow', isMetal: false, bay: 2 },
 PLASTIC_PURPLE: { color: 0x7a4cff, name: 'Plastic Purple', isMetal: false, bay: 3 }
};

const DEFECT_RATE = 0.15; // 15% defect rate

// Boxes
const boxes = [];
const hopperQueue = [];

// Create hopper queue with mixed materials (16 cubes total)
function initializeHopper() {
 hopperQueue.length = 0;
 const materialTypes = Object.values(MATERIALS);

 for (let i = 0; i < 16; i++) {
 const material = materialTypes[i % 4];
 const isDefect = Math.random() < DEFECT_RATE;
 hopperQueue.push({
 material,
 isDefect,
 id: `cube_${Date.now()}_${i}`
 });
 }

 // Shuffle for realistic mixed hopper
 for (let i = hopperQueue.length - 1; i > 0; i--) {
 const j = Math.floor(Math.random() * (i + 1));
 [hopperQueue[i], hopperQueue[j]] = [hopperQueue[j], hopperQueue[i]];
 }

 log(`Hopper initialized with ${hopperQueue.length} cubes (${Math.round(DEFECT_RATE * 100)}% defect rate)`);
}

// ============================================
// Cube Spawning & Hopper System
// ============================================

let autoReleaseTimer = 0;
const AUTO_RELEASE_INTERVAL = 3.0; // seconds between auto-releases
let autoReleaseEnabled = false; // manual release by default

function spawnCubeFromHopper() {
 if (hopperQueue.length === 0) {
 log('Hopper empty - cycle complete');
 return null;
 }

 const cubeData = hopperQueue.shift();
 const box = new THREE.Mesh(
 new THREE.BoxGeometry(0.35, 0.35, 0.35),
 new THREE.MeshStandardMaterial({ color: 0x1a1a1a }) // Start as dark/unknown
 );

 // Add white "?" marks on all 6 sides using canvas textures
 const questionMarkGroup = new THREE.Group();
 
 // Create a canvas with "?" text
 const canvas = document.createElement('canvas');
 canvas.width = 128;
 canvas.height = 128;
 const ctx = canvas.getContext('2d');
 ctx.fillStyle = 'white';
 ctx.font = 'bold 100px Arial';
 ctx.textAlign = 'center';
 ctx.textBaseline = 'middle';
 ctx.fillText('?', 64, 64);
 
 const texture = new THREE.CanvasTexture(canvas);
 const qMaterial = new THREE.MeshBasicMaterial({ 
 map: texture, 
 transparent: true, 
 side: THREE.DoubleSide,
 depthWrite: false
 });
 
 // Top
 const qTop = new THREE.Mesh(new THREE.PlaneGeometry(0.25, 0.25), qMaterial);
 qTop.rotation.x = -Math.PI / 2;
 qTop.position.y = 0.176;
 questionMarkGroup.add(qTop);
 
 // Bottom
 const qBottom = new THREE.Mesh(new THREE.PlaneGeometry(0.25, 0.25), qMaterial);
 qBottom.rotation.x = Math.PI / 2;
 qBottom.position.y = -0.176;
 questionMarkGroup.add(qBottom);
 
 // Front (+Z)
 const qFront = new THREE.Mesh(new THREE.PlaneGeometry(0.25, 0.25), qMaterial);
 qFront.position.z = 0.176;
 questionMarkGroup.add(qFront);
 
 // Back (-Z)
 const qBack = new THREE.Mesh(new THREE.PlaneGeometry(0.25, 0.25), qMaterial);
 qBack.rotation.y = Math.PI;
 qBack.position.z = -0.176;
 questionMarkGroup.add(qBack);
 
 // Right (+X)
 const qRight = new THREE.Mesh(new THREE.PlaneGeometry(0.25, 0.25), qMaterial);
 qRight.rotation.y = Math.PI / 2;
 qRight.position.x = 0.176;
 questionMarkGroup.add(qRight);
 
 // Left (-X)
 const qLeft = new THREE.Mesh(new THREE.PlaneGeometry(0.25, 0.25), qMaterial);
 qLeft.rotation.y = -Math.PI / 2;
 qLeft.position.x = -0.176;
 questionMarkGroup.add(qLeft);
 
 box.add(questionMarkGroup);

 // Start position at hopper/beginning of conveyor 1
 box.position.set(2.5, conveyorHeight + 0.35, -3);
 box.userData = {
 state: 'on_conv1',
 material: cubeData.material,
 isDefect: cubeData.isDefect,
 id: cubeData.id,
 t: 0,
 identityRevealed: false, // Track if identity is revealed
 questionMark: questionMarkGroup // Store reference to remove later
 };

 scene.add(box);
 boxes.push(box);
 metrics.boxStarted(box.uuid);

 log(`Released: ${cubeData.material.name}${cubeData.isDefect ? ' [DEFECT]' : ''}`);
 return box;
}

// Function to reveal box identity (callable from PLC or sensors)
function revealBoxIdentity(box) {
 if (!box.userData.identityRevealed) {
 // Change box color to true material
 box.material.color.setHex(box.userData.material.color);
 
 // Remove question mark
 if (box.userData.questionMark) {
 box.remove(box.userData.questionMark);
 box.userData.questionMark = null;
 }
 
 box.userData.identityRevealed = true;
 log(`Identity revealed: ${box.userData.material.name}`);
 }
}

// Animation state
let paused = false;
let gantryT = 0; // 0 = at conveyor 1 (north), 1 = at conveyor 2 (south)
let gantryDir = 1;
let gantryHolding = false;
let gantryActive = false;

function updateGantry(delta) {
 // Check if any cube is ready for pickup
 if (!gantryHolding && !gantryActive) {
 for (const box of boxes) {
 if (box.userData.state === 'ready_for_gantry') {
 gantryActive = true;
 gantryT = 0;
 gantryDir = 1;
 box.userData.state = 'gantry_lift';
 break;
 }
 }
 }

 // Move gantry
 if (gantryActive) {
 gantryT += delta * gantryDir * 0.3;

 if (gantryT >= 1) {
 gantryT = 1;
 gantryDir = -1;
 }

 if (gantryT <= 0 && gantryDir === -1) {
 gantryT = 0;
 gantryDir = 1;
 gantryActive = false;
 metrics.setEquipmentStatus('gantry', 'idle');
 }
 }

 // Move straight along Z between conveyors at fixed X
 const pickup = new THREE.Vector3(gantryX, gantryY, -3);
 const drop = new THREE.Vector3(gantryX, gantryY, 3);
 const pos = pickup.clone().lerp(drop, gantryT);
 gantryCart.position.copy(pos);

 // Bob the gripper
 gantryGrip.position.y = gripOffsetY + Math.sin(performance.now() * 0.002) * 0.05;
 gantryGrip.material.color.set(gantryHolding ? 0xaaaaaa : 0x555555);
}

// ============================================
// Robot Arm Animation
// ============================================

let currentRobotBox = null;
const conveyorEndPos = new THREE.Vector3(2.3, conveyorHeight + 0.35, 3); // Where box waits on conveyor

function updateRobotArm(delta) {
 // Animate robot arm based on state
 if (robotState === 'idle') {
 // Reset to home position
 robot.rotation.y = Math.PI;
 arm2.rotation.z = 0;
 return;
 }
 
 if (robotState === 'reaching') {
 robotAnimT += delta * 2;
 const t = Math.min(robotAnimT, 1);
 
 // Rotate 90 degrees LEFT from home position to reach conveyor
 const conveyorAngle = Math.PI + Math.PI / 2; // 270 = -90 (left turn)
 robot.rotation.y = Math.PI * (1 - t) + conveyorAngle * t;
 arm2.rotation.z = -Math.PI * 0.2 * t; // Extend arm
 
 if (robotAnimT >= 1) {
 robotState = 'grabbing';
 robotAnimT = 0;
 log(`Robot reaching conveyor at 90 left`);
 }
 }
 
 if (robotState === 'grabbing') {
 robotAnimT += delta * 2;
 
 // Attach box to gripper after grab delay
 if (currentRobotBox && robotAnimT >= 0.3) {
 const gripperWorldPos = new THREE.Vector3();
 gripper.getWorldPosition(gripperWorldPos);
 currentRobotBox.position.set(gripperWorldPos.x, conveyorHeight + 0.35, gripperWorldPos.z);
 
 if (robotAnimT === 0.3) {
 log(`Box grabbed at X:${gripperWorldPos.x.toFixed(2)} Z:${gripperWorldPos.z.toFixed(2)}`);
 }
 }
 
 if (robotAnimT >= 1) {
 robotState = 'returning';
 robotAnimT = 0;
 }
 }
 
 if (robotState === 'returning') {
 robotAnimT += delta * 1.5;
 const t = Math.min(robotAnimT, 1);
 
 // Rotate 90 degrees RIGHT from home position to reach pallet
 if (currentRobotBox) {
 const conveyorAngle = Math.PI + Math.PI / 2; // Where we picked up (270)
 const palletAngle = Math.PI - Math.PI / 2; // 90 (right turn from home)
 
 // Interpolate from conveyor angle to pallet angle
 robot.rotation.y = conveyorAngle * (1 - t) + palletAngle * t;
 arm2.rotation.z = -Math.PI * 0.2 * (1 - t);
 
 // Keep box attached to gripper
 const gripperWorldPos = new THREE.Vector3();
 gripper.getWorldPosition(gripperWorldPos);
 currentRobotBox.position.set(gripperWorldPos.x, conveyorHeight + 0.35, gripperWorldPos.z);
 }
 
 if (robotAnimT >= 1) {
 robotState = 'placing';
 robotAnimT = 0;
 }
 }
 
 if (robotState === 'placing') {
 robotAnimT += delta * 2;
 
 if (robotAnimT >= 0.5 && currentRobotBox) {
 // Place the box in sorting bay
 const bayIndex = currentRobotBox.userData.material.bay;
 const bayPos = getSortingBayPosition(bayIndex);
 
 if (bayPos) {
 currentRobotBox.position.set(bayPos.x, bayPos.y, bayPos.z);
 currentRobotBox.userData.state = 'done';
 
 if (!currentRobotBox.userData.completed) {
 metrics.boxCompleted(currentRobotBox.uuid);
 currentRobotBox.userData.completed = true;
 }
 
 log(`Robot: Sorted to ${sortingBays[bayIndex].name} bay at X:${bayPos.x.toFixed(2)} Z:${bayPos.z.toFixed(2)}`);
 }
 
 currentRobotBox = null;
 robotState = 'idle';
 robotAnimT = 0;
 metrics.setEquipmentStatus('robot', 'ready');
 }
 }
}

// ============================================
// Main Process State Machine
// ============================================

function updateBoxes(delta) {
 // Count boxes in different states for metrics
 let waiting = hopperQueue.length;
 let inTransit = 0;
 let onPallet = 0;
 let defectsRejected = 0;

 // Check if conveyors should be running
 let conv1Running = true;
 let conv2Running = false; // Only runs when start sensor is triggered
 
 // Stop conveyor 1 if any box is at the sensor or being inspected
 for (const box of boxes) {
 if (box.userData.state === 'inspecting' || box.userData.state === 'ready_for_gantry' || box.userData.state === 'rejecting') {
 conv1Running = false;
 break;
 }
 }
 
 // Conveyor 2 runs only if boxes are on it and none are waiting at end sensor
 let boxesOnConv2 = false;
 let boxAtConv2End = false;
 
 for (const box of boxes) {
 if (box.userData.state === 'on_conv2') {
 boxesOnConv2 = true;
 }
 if (box.userData.state === 'waiting_robot') {
 boxAtConv2End = true;
 break;
 }
 }
 
 conv2Running = boxesOnConv2 && !boxAtConv2End;

 for (const box of boxes) {
 const state = box.userData.state;

 // Track stats
 if (state === 'done') onPallet++;
 else if (state === 'rejected') defectsRejected++;
 else inTransit++;

 switch (state) {
 case 'on_conv1':
 // Move on conveyor 1 toward sensor at end
 const conv1Target = sensors.conv1End.position;
 
 // Check if any other box is too close ahead
 let canMove1 = conv1Running;
 if (canMove1) {
 for (const other of boxes) {
 if (other === box || other.userData.state !== 'on_conv1') continue;
 // Check if other box is ahead (smaller X) and too close
 if (other.position.x < box.position.x && (box.position.x - other.position.x) < 0.5) {
 canMove1 = false;
 break;
 }
 }
 }
 
 // Only move if conveyor is running and no collision
 if (canMove1 && box.position.x > conv1Target) {
 box.position.x -= delta * 0.8; // conveyor speed
 }
 
 metrics.setEquipmentStatus('conveyor1', conv1Running ? 'running' : 'idle');

 // Reached sensor at end
 if (box.position.x <= conv1Target + 0.05) {
 box.position.x = conv1Target;
 box.userData.state = 'inspecting';
 box.userData.t = 0;
 sensors.conv1End.active = true;
 updateSensorIndicator('conv1End', true);
 metrics.setEquipmentStatus('conveyor1', 'idle');
 log(`Sensor 1: Cube detected, inspection starting`);
 }
 break;

 case 'inspecting':
 // Camera inspecting cube for color and defects
 box.userData.t += delta;

 if (box.userData.t > 1.5) { // inspection takes 1.5 seconds
 sensors.conv1End.active = false;
 updateSensorIndicator('conv1End', false);

 if (box.userData.isDefect) {
 box.userData.state = 'rejecting';
 box.userData.t = 0;
 pistonActive = true; // Activate piston for rejection
 log(`Vision: DEFECT detected - ${box.userData.material.name}`);
 } else {
 box.userData.state = 'ready_for_gantry';
 // Gantry will be activated by updateGantry function
 log(`Vision: OK - ${box.userData.material.name}`);
 }
 }
 break;

 case 'rejecting':
 // Piston pushes defect off conveyor toward bin
 box.userData.t += delta;
 const pushAmount = Math.min(box.userData.t * 2, 2.0);
 box.position.z = -3 - pushAmount; // Push from conveyor at Z=-3 toward bin at Z=-4
 box.position.x = -2.5; // Move toward bin X position (aligned with sensor)
 box.position.y = Math.max(0.2, conveyorHeight + 0.35 - box.userData.t * 0.5);

 if (box.userData.t > 1.5) {
 // Place box in bin at X=-2.5, Z=-4.0
 box.position.set(-2.5, 0.3, -4.0);
 box.userData.state = 'rejected';
 pistonActive = false; // Deactivate piston after rejection
 log(`Defect rejected into bin`);
 }
 break;

 case 'ready_for_gantry':
 // Waiting for gantry pickup
 break;

 case 'gantry_lift':
 // Gantry picking up cube
 gantryHolding = true;
 metrics.setEquipmentStatus('gantry', 'active');
 box.position.copy(gantryCart.position).add(new THREE.Vector3(0, -0.6, 0));

 if (gantryT > 0.95) {
 box.userData.state = 'on_conv2';
 box.position.set(-2.5, conveyorHeight + 0.35, 3);
 gantryHolding = false;
 // Trigger start sensor when cube is placed
 sensors.conv2Start.active = true;
 updateSensorIndicator('conv2Start', true);
 log(`Gantry: Placed on conveyor 2, start sensor triggered`);
 }
 break;

 case 'on_conv2':
 // Move on conveyor 2, passing metal sensors
 
 // Clear start sensor when box moves away from drop point
 if (box.position.x > sensors.conv2Start.position + 0.3) {
 sensors.conv2Start.active = false;
 updateSensorIndicator('conv2Start', false);
 }

 // Check metal sensors
 if (Math.abs(box.position.x - sensors.metalSensor1.position) < 0.2) {
 const detected = box.userData.material.isMetal && box.userData.material.name === 'Steel';
 sensors.metalSensor1.active = detected;
 updateSensorIndicator('metalSensor1', detected);
 } else {
 sensors.metalSensor1.active = false;
 updateSensorIndicator('metalSensor1', false);
 }

 if (Math.abs(box.position.x - sensors.metalSensor2.position) < 0.2) {
 const detected = box.userData.material.isMetal && box.userData.material.name === 'Aluminum';
 sensors.metalSensor2.active = detected;
 updateSensorIndicator('metalSensor2', detected);
 
 // Reveal box identity after passing metalSensor2 (simulation mode)
 if (!box.userData.identityRevealed) {
 revealBoxIdentity(box);
 }
 } else {
 sensors.metalSensor2.active = false;
 updateSensorIndicator('metalSensor2', false);
 }

 const conv2Target = sensors.conv2End.position;

 // Check if any other box is too close ahead
 let canMove2 = conv2Running;
 if (canMove2) {
 for (const other of boxes) {
 if (other === box || other.userData.state !== 'on_conv2') continue;
 // Check if other box is ahead (larger X) and too close
 if (other.position.x > box.position.x && (other.position.x - box.position.x) < 0.5) {
 canMove2 = false;
 break;
 }
 }
 }

 // Only move if conveyor is running and no collision
 if (canMove2 && box.position.x < conv2Target) {
 box.position.x += delta * 0.8;
 }
 
 metrics.setEquipmentStatus('conveyor2', conv2Running ? 'running' : 'idle');

 // Reached end sensor
 if (box.position.x >= conv2Target - 0.05) {
 box.position.x = conv2Target;
 box.userData.state = 'waiting_robot';
 box.userData.t = 0;
 sensors.conv2End.active = true;
 updateSensorIndicator('conv2End', true);
 metrics.setEquipmentStatus('conveyor2', 'idle');
 log(`Sensor 2: Cube at end, robot picking`);
 }
 break;

 case 'waiting_robot':
 // Robot picking and sorting with animation
 box.userData.t += delta;
 metrics.setEquipmentStatus('robot', 'active');

 // Start robot animation once
 if (box.userData.t < 0.1 && robotState === 'idle') {
 robotState = 'reaching';
 robotAnimT = 0;
 currentRobotBox = box;
 sensors.conv2End.active = false;
 updateSensorIndicator('conv2End', false);
 }
 
 // Box stays at conveyor until grabbed (during 'grabbing' state)
 // After that, the robot animation handles the box position
 
 break;

 case 'done':
 case 'rejected':
 // Nothing to do
 break;
 }
 }

 // Update metrics
 metrics.updateQueueStats(waiting, inTransit, onPallet);
}

let cycleInitialized = false;

function animate() {
 requestAnimationFrame(animate);

 if (!paused) {
 const delta = 0.016;

 // Initialize hopper on first run
 if (!cycleInitialized) {
 initializeHopper();
 cycleInitialized = true;
 }

 // Auto-release cubes from hopper (disabled by default)
 if (autoReleaseEnabled) {
 autoReleaseTimer += delta;
 if (autoReleaseTimer >= AUTO_RELEASE_INTERVAL && hopperQueue.length > 0) {
 // Only release if no cube is currently being inspected or rejected
 const canRelease = !boxes.some(b =>
 b.userData.state === 'inspecting' ||
 b.userData.state === 'rejecting' ||
 b.userData.state === 'ready_for_gantry'
 );

 if (canRelease) {
 spawnCubeFromHopper();
 autoReleaseTimer = 0;
 }
 }
 }

 updateGantry(delta);
 updateRobotArm(delta);
 updateBoxes(delta);
 updatePiston(delta);

 // Update metrics and dashboard
 metrics.updateThroughput();
 updateDashboard();
 update3DStatusIndicators();
 }

 controls.update();
 renderer.render(scene, camera);
}

animate();

// UI
const resetBtn = document.getElementById('reset');
const releaseBtn = document.getElementById('release');

// Hide reset button in real PLC mode
if (OPERATION_MODE === MODE.REAL_PLC) {
 resetBtn.style.display = 'none';
}

resetBtn.addEventListener('click', () => {
 // Clear all boxes
 for (const b of boxes) scene.remove(b);
 boxes.length = 0;

 // Reset gantry
 gantryT = 0;
 gantryDir = 1;
 gantryHolding = false;
 gantryActive = false;

 // Reset bays
 sortingBays.forEach(bay => bay.count = 0);

 // Reset sensors
 Object.keys(sensors).forEach(key => {
 sensors[key].active = false;
 updateSensorIndicator(key, false);
 });

 // Reinitialize hopper
 cycleInitialized = false;
 autoReleaseTimer = 0;

 // Reset metrics
 window.location.reload();
 log('System reset');
});

releaseBtn.addEventListener('click', () => {
 // Only release if no cube is currently being inspected or rejected
 const canRelease = !boxes.some(b =>
 b.userData.state === 'inspecting' ||
 b.userData.state === 'rejecting' ||
 b.userData.state === 'ready_for_gantry'
 );
 if (canRelease) {
 spawnCubeFromHopper();
 autoReleaseTimer = 0;
 log('Manual release');
 }
});

// Resize
window.addEventListener('resize', () => {
 const container = document.getElementById('canvas-container');
 const containerWidth = container.clientWidth;
 const containerHeight = container.clientHeight;
 camera.aspect = containerWidth / containerHeight;
 camera.updateProjectionMatrix();
 renderer.setSize(containerWidth, containerHeight);
});
