Technical Plan: PWA + Headless Raspberry Pi Gateway for Dobot ↔ S7-1200
1. System Overview

A headless Node.js backend runs on a Raspberry Pi, handling all communication between:

Dobot Magician (via USB serial or TCP 29999 API)

Siemens S7-1200 PLC (via Snap7 or Modbus TCP)

Web browser clients running a Progressive Web App (PWA)

[Web UI (tablet/PC)]
        ⇅  WebSocket / HTTPS
[Node.js server on Raspberry Pi]
        ⇅
   USB or TCP (29999)
        ⇅
       Dobot Magician
        ⇅
     Ethernet (S7Comm)
        ⇅
      Siemens S7-1200

2. Core Components
Layer	Technology	Purpose
Backend Server	Node.js (Express + Socket.io)	Central logic & message broker
Dobot Interface	serialport or net	Communicate via USB or TCP
PLC Interface	node-snap7 or jsmodbus	Read/write PLC registers
Frontend (PWA)	React / Vue / Svelte + Vite	Real-time dashboard & controls
Database (optional)	SQLite / NeDB	Log poses, events, faults
Runtime Manager	systemd / PM2	Auto-start & monitor Node service
3. Backend Architecture
Directories
/gateway
 ├─ /server
 │   ├─ app.js            → Express + Socket.io entrypoint
 │   ├─ routes/
 │   │   └─ api.js        → REST endpoints
 │   ├─ services/
 │   │   ├─ dobot.js      → USB/TCP communication
 │   │   ├─ plc.js        → S7-1200 communication
 │   │   └─ bridge.js     → PLC↔Dobot data mapping
 │   ├─ utils/
 │   │   └─ logger.js
 │   └─ config.json       → IPs, ports, tag mapping
 ├─ /client               → PWA source (React/Vue)
 ├─ /public               → Built static files served by Express
 └─ package.json

Main Process Flow

Startup

Load config (PLC IP, Dobot port, mapping).

Connect to Dobot (USB /dev/ttyUSB0 or TCP 29999).

Connect to S7-1200 (rack 0, slot 1).

Start WebSocket server for UI clients.

Runtime Loop

Poll PLC memory → detect commands (Start, Stop, Reset).

Send mapped Dobot commands.

Poll Dobot pose & status → update PLC DB and broadcast to UI.

Log events.

Shutdown

Close all sockets gracefully.

Save session log.

4. Communication Details
A. Dobot Layer (dobot.js)
Action	API Command
Home	Home()
Run routine	RunScript("PickAndPlace")
Stop	Pause()
Clear errors	ClearError()
Pose feedback	GetPose()

Implementation

import net from "net";
const client = new net.Socket();
client.connect(29999, "192.168.0.30");

function send(cmd) {
  client.write(cmd + "\n");
}


For USB:

import { SerialPort } from "serialport";
const port = new SerialPort({ path: "/dev/ttyUSB0", baudRate: 115200 });

B. PLC Layer (plc.js)

Using node-snap7:

import snap7 from "node-snap7";
const plc = new snap7.S7Client();
plc.ConnectTo("192.168.0.10", 0, 1);

PLC Address	Type	Description
M0.0	BOOL	Start Dobot
M0.1	BOOL	Stop
M0.2	BOOL	Reset
DB1.DBD0–DBD8	REALs	X, Y, Z poses
DB1.DBW20	INT	Dobot status code
C. Bridge Layer (bridge.js)

Reads bits from PLC → triggers Dobot commands.

Writes Dobot pose → PLC DB values.

Maintains state cache for UI.

Example logic

if (plc.getBit("M0.0") && !state.running) {
  dobot.send("RunScript('PickAndPlace')");
  plc.resetBit("M0.0");
}

const pose = await dobot.getPose();
plc.writeReals("DB1.DBD0", [pose.x, pose.y, pose.z]);
socket.emit("pose", pose);

5. Frontend (PWA)
Structure
/client
 ├─ /src
 │   ├─ App.jsx
 │   ├─ components/
 │   │   ├─ ConnectionStatus.jsx
 │   │   ├─ PoseDisplay.jsx
 │   │   ├─ Controls.jsx
 │   │   └─ LogPanel.jsx
 │   ├─ api/socket.js
 │   └─ manifest.json

Core Features

Real-time pose & status display (WebSocket subscription).

Manual buttons (Home, Start, Stop, Vacuum).

PLC I/O monitor (mirror bits/words).

Fault log viewer.

Offline caching via service worker.

6. API & WebSocket Specification
REST (Express)
Endpoint	Method	Description
/api/status	GET	Returns Dobot & PLC state
/api/run/:task	POST	Executes a Dobot script
/api/clear	POST	Clears Dobot errors
WebSocket (Socket.io)
Event	Direction	Payload
pose	server → client	{x,y,z,r}
status	server → client	{dobot:"Running", plc:"Connected"}
command	client → server	{action:"home"}
7. Deployment Setup
On Raspberry Pi

Install Node 20 LTS:

curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs git


Clone and build:

git clone https://github.com/YourRepo/dobot-gateway.git
cd dobot-gateway && npm install
npm run build


Create systemd service:

[Unit]
Description=Dobot Gateway
After=network.target
[Service]
ExecStart=/usr/bin/node /home/pi/dobot-gateway/server/app.js
Restart=always
[Install]
WantedBy=multi-user.target


Enable & start:

sudo systemctl enable dobot-gateway
sudo systemctl start dobot-gateway


Access via browser: http://pi.local:8080

8. Logging & Safety

Logs: /var/log/dobot-gateway/YYYYMMDD.log

Auto-reconnects on Dobot or PLC disconnect.

Emergency stop: watch a PLC bit or GPIO input.

Command validation: reject moves outside safe X/Y/Z ranges.

9. Extensibility

MQTT Broker → publish pose & status for higher-level factory dashboards.

Grafana/InfluxDB → record cycle metrics.

Multiple Dobots → extend mapping for additional IPs.

Edge AI module → anomaly detection on vibration/current data.

10. Summary

Headless Pi backend: Node.js handles Dobot & PLC comms.

PWA frontend: served from same Pi, accessible on LAN.

Clean separation: hardware logic in Node services, UI via WebSocket.

Future-proof: compatible with additional smart-factory devices and dashboards.