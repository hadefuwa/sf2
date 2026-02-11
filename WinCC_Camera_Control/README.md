# How to Install Your Raspberry Pi Camera Control on WinCC Unified

## 0. Prerequisites
- **Device**: Must be a **WinCC Unified Basic Panel** or **Unified Comfort Panel**. (Older KTP Basic panels are NOT supported).
- **File**: You need `webcc.min.js`. This is a proprietary Siemens file found in their "Custom Web Control" example projects (Siemens Support ID: 109779176).

---

## 1. Prepare the Control
1.  **Get `webcc.min.js`**:
    - Download it from Siemens Support.
    - Copy the file into this folder: `c:\Users\HamedA\Documents\sf2\WinCC_Camera_Control\`
    - **Overwrite** the placeholder file that is currently there.

2.  **Run the Packager**:
    - I have included a script to zip everything correctly for you.
    - Open a terminal in this folder and run:
      ```powershell
      .\package_control.ps1
      ```
    - **Result**: You will see a new file named `{551BF148-7B62-436A-8A4F-9C1D1E2F3A4B}.zip`.

---

## 2. Install into TIA Portal
TIA Portal requires a specific folder structure to recognize custom controls.

1.  **Locate your TIA Portal Project**:
    - Go to the Windows folder where your `.ap18` (or `.ap17`, etc.) project file lives.

2.  **Create Folders** (if they don't exist):
    - Create a folder named `UserFiles`.
    - Inside that, create a folder named `CustomControls`.
    - Path should look like: `[ProjectFolder]\UserFiles\CustomControls\`

3.  **Copy the Zip**:
    - Copy the `{551...}.zip` file you generated in Step 1.
    - Paste it into the `\UserFiles\CustomControls\` folder.

---

## 3. Usage in TIA Portal
1.  Open your project in TIA Portal.
2.  Open a Screen editor.
3.  Open the **Toolbox** sidebar (right side).
4.  Expand the **"My controls"** section.
5.  Click the small **Refresh icon** (circular arrows) in the My controls header.
6.  You should see **"PiCameraControl"**.
7.  **Drag and drop** it onto your screen.

## 4. Configuration
- Select the control on the screen.
- Go to **Properties** -> **Miscellaneous (or Custom)**.
- Find the **CameraURL** property.
- Enter your Raspberry Pi stream URL.

### HTTPS (required for WinCC Unified)
WinCC Unified HMI requires **HTTPS** for embedded content (mixed content policy).

**URL format:** `https://<RASPBERRY_PI_IP>:8080/api/camera/stream`

**Examples:**
- `https://192.168.7.5:8080/api/camera/stream`
- `https://rpi:8080/api/camera/stream`

### Digital Twin Stream (HMI cannot run WebGL)
Basic panels lack the RAM/GPU for Three.js. The Pi renders the 3D view and streams it as MJPEG (same as camera).

**Simple HTML page (recommended):**  
`https://192.168.7.5:8080/digital-twin-stream.html`  

This page polls single frames and displays them. No MJPEG stream – works reliably.

**MJPEG stream URL** (for WinCC embed):  
`https://192.168.7.5:8080/api/digital-twin/stream`

**First-time setup on Pi:** Install Playwright and Chromium:
```bash
pip install playwright
playwright install chromium
```
Restart the app afterward. If Playwright is missing, the digital twin stream is disabled (camera stream still works).

**First-time setup on the Pi:** Run the certificate generator, then restart the app:
```bash
cd ~/sf2/pwa-dobot-plc
chmod +x deploy/generate_ssl_cert.sh
./deploy/generate_ssl_cert.sh 192.168.7.5   # use your Pi's IP
pm2 restart pwa-dobot-plc
```
WinCC may show a certificate warning (self-signed) – accept/trust it to enable the camera.

---

## 5. Reference

- **Siemens Support** – Custom Web Control example (webcc.min.js): Support ID 109779176
- **WinCC Unified** – Web control supports `https://`, `http://`, and `file://` on Basic/Comfort panels
- **Note** – `http://` and `file://` do not work in simulation; use `https://` for embedded camera in production
