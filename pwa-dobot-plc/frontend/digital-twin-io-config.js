// ============================================
// IO Configuration & PLC Integration Layer
// ============================================

// OPERATION MODE - Switch between simulation and real PLC
export const MODE = {
 SIMULATION: 'simulation',
 REAL_PLC: 'real_plc'
};

export const OPERATION_MODE = MODE.SIMULATION; // Change to MODE.REAL_PLC for production

// ============================================
// PLC Connection Configuration (Siemens S7)
// ============================================
export const PLC_CONFIG = {
 ip: '192.168.1.100', // PLC IP address
 rack: 0, // S7 rack number
 slot: 1, // S7 slot number
 pollInterval: 100, // Polling interval in ms
 timeout: 5000 // Connection timeout in ms
};

// ============================================
// IO Mapping Tables
// ============================================

// Digital Inputs (Sensors from field)
export const DIGITAL_INPUTS = {
 conv1End: {
 db: 1,
 offset: 0,
 bit: 0,
 description: 'Conveyor 1 end sensor',
 position: -2.5
 },
 conv2Start: {
 db: 1,
 offset: 0,
 bit: 1,
 description: 'Conveyor 2 start sensor',
 position: -2.5
 },
 conv2End: {
 db: 1,
 offset: 0,
 bit: 2,
 description: 'Conveyor 2 end sensor',
 position: 2.3
 },
 metalSensor1: {
 db: 1,
 offset: 0,
 bit: 3,
 description: 'Metal detector 1 (Steel)',
 position: 0
 },
 metalSensor2: {
 db: 1,
 offset: 0,
 bit: 4,
 description: 'Metal detector 2 (Aluminum)',
 position: 1.5
 },
 visionDefect: {
 db: 1,
 offset: 0,
 bit: 5,
 description: 'Vision system defect signal'
 },
 robotGripperClosed: {
 db: 1,
 offset: 0,
 bit: 6,
 description: 'Robot gripper closed sensor'
 },
 robotAtHome: {
 db: 1,
 offset: 0,
 bit: 7,
 description: 'Robot at home position'
 }
};

// Digital Outputs (Commands to field)
export const DIGITAL_OUTPUTS = {
 conveyor1Run: {
 db: 2,
 offset: 0,
 bit: 0,
 description: 'Conveyor 1 motor run'
 },
 conveyor2Run: {
 db: 2,
 offset: 0,
 bit: 1,
 description: 'Conveyor 2 motor run'
 },
 gantryPickup: {
 db: 2,
 offset: 0,
 bit: 2,
 description: 'Gantry pickup command'
 },
 rejectPistonExtend: {
 db: 2,
 offset: 0,
 bit: 3,
 description: 'Reject piston extend'
 },
 robotPickup: {
 db: 2,
 offset: 0,
 bit: 4,
 description: 'Robot pickup command'
 },
 robotPlace: {
 db: 2,
 offset: 0,
 bit: 5,
 description: 'Robot place command'
 }
};

// Analog Inputs
export const ANALOG_INPUTS = {
 gantryPositionX: {
 db: 3,
 offset: 0,
 type: 'REAL',
 description: 'Gantry X position (meters)',
 scale: { min: -5.0, max: 5.0 }
 },
 gantryPositionZ: {
 db: 3,
 offset: 4,
 type: 'REAL',
 description: 'Gantry Z position (meters)',
 scale: { min: -5.0, max: 5.0 }
 },
 robotAngle: {
 db: 3,
 offset: 8,
 type: 'REAL',
 description: 'Robot rotation angle (radians)',
 scale: { min: -Math.PI, max: Math.PI }
 }
};

// Analog Outputs
export const ANALOG_OUTPUTS = {
 gantryTargetX: {
 db: 4,
 offset: 0,
 type: 'REAL',
 description: 'Gantry target X position (meters)'
 },
 gantryTargetZ: {
 db: 4,
 offset: 4,
 type: 'REAL',
 description: 'Gantry target Z position (meters)'
 },
 robotTargetAngle: {
 db: 4,
 offset: 8,
 type: 'REAL',
 description: 'Robot target angle (radians)'
 }
};

// ============================================
// IO Interface Class
// ============================================
export class IOInterface {
 constructor() {
 this.mode = OPERATION_MODE;
 this.connected = false;
 this.plcClient = null;
 
 // Simulated IO state (used in simulation mode)
 this.simulatedInputs = {};
 this.simulatedOutputs = {};
 
 // Initialize simulated states
 Object.keys(DIGITAL_INPUTS).forEach(key => {
 this.simulatedInputs[key] = false;
 });
 Object.keys(ANALOG_INPUTS).forEach(key => {
 this.simulatedInputs[key] = 0;
 });
 
 if (this.mode === MODE.REAL_PLC) {
 this.initializePLC();
 }
 }
 
 async initializePLC() {
 console.log(`Connecting to PLC at ${PLC_CONFIG.ip}...`);
 // TODO: Initialize actual PLC connection here
 // Example: this.plcClient = new S7Client(PLC_CONFIG);
 // await this.plcClient.connect();
 this.connected = false; // Set to true when real connection succeeds
 }
 
 // Read digital input
 readDigitalInput(name) {
 if (this.mode === MODE.SIMULATION) {
 return this.simulatedInputs[name] || false;
 } else {
 const config = DIGITAL_INPUTS[name];
 if (!config) return false;
 // TODO: Read from actual PLC
 // return this.plcClient.readBit(config.db, config.offset, config.bit);
 return false;
 }
 }
 
 // Write digital output
 writeDigitalOutput(name, value) {
 if (this.mode === MODE.SIMULATION) {
 this.simulatedOutputs[name] = value;
 } else {
 const config = DIGITAL_OUTPUTS[name];
 if (!config) return;
 // TODO: Write to actual PLC
 // this.plcClient.writeBit(config.db, config.offset, config.bit, value);
 }
 }
 
 // Read analog input
 readAnalogInput(name) {
 if (this.mode === MODE.SIMULATION) {
 return this.simulatedInputs[name] || 0;
 } else {
 const config = ANALOG_INPUTS[name];
 if (!config) return 0;
 // TODO: Read from actual PLC
 // return this.plcClient.readReal(config.db, config.offset);
 return 0;
 }
 }
 
 // Write analog output
 writeAnalogOutput(name, value) {
 if (this.mode === MODE.SIMULATION) {
 this.simulatedOutputs[name] = value;
 } else {
 const config = ANALOG_OUTPUTS[name];
 if (!config) return;
 // TODO: Write to actual PLC
 // this.plcClient.writeReal(config.db, config.offset, value);
 }
 }
 
 // Simulate sensor activation (only works in simulation mode)
 setSimulatedInput(name, value) {
 if (this.mode === MODE.SIMULATION) {
 this.simulatedInputs[name] = value;
 }
 }
 
 // Get connection status
 isConnected() {
 if (this.mode === MODE.SIMULATION) {
 return true; // Always connected in simulation
 }
 return this.connected;
 }
 
 // Get current mode
 getMode() {
 return this.mode;
 }
}
