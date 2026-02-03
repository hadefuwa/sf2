# Simple Camera Display in WinCC Unified (Built-in Web Control)

## The Easy Way - No Custom Control Needed!

WinCC Unified has a **built-in "Web control"** that can display your camera stream directly. You don't need the custom control for basic camera viewing.

---

## Steps to Display Camera Feed

### 1. In TIA Portal

1. Open your WinCC Unified screen in the editor
2. From the **Toolbox** (right sidebar), find the **"Web control"** object (also called "Browser")
3. **Drag and drop** it onto your screen
4. Resize it to your desired camera view size

### 2. Configure the URL

1. Select the Web control object on your screen
2. In the **Inspector window** (Properties panel), go to:
   - **Properties → General → URL**
3. Enter the camera stream URL:
   ```
   http://192.168.0.86:8080/api/camera/stream
   ```
   Or use your Pi's hostname:
   ```
   http://rpi:8080/api/camera/stream
   ```

### 3. Optional: Configure Toolbar

Under **Properties → Miscellaneous → Toolbar**, you can:
- Enable/disable browser navigation buttons
- Show/hide the URL bar
- Configure refresh button

For a clean camera view, disable all toolbar buttons.

---

## What I Fixed in Your Backend

Your backend now sends these headers with the camera stream:
- `X-Frame-Options: ALLOWALL` - Allows embedding in iFrames
- `Access-Control-Allow-Origin: *` - Allows cross-origin requests

This means the WinCC Unified Web control (which uses iFrames) can now display your camera stream.

---

## Important Notes

### Protocol Limitations
- **On actual hardware**: http://, https://, file:// all work
- **In TIA Portal simulation**: Only https:// works
  - This means you **cannot test the camera in simulation**
  - You must test on the actual HMI panel

### Network Requirements
- The HMI panel must be able to reach your Raspberry Pi
- Ensure both devices are on the same network
- Test the URL in a web browser on the HMI panel first

---

## Testing

1. **Test in browser first** (from HMI panel or your PC):
   ```
   http://192.168.0.86:8080/api/camera/stream
   ```
   You should see the live camera feed as an MJPEG stream

2. **Deploy to HMI panel** (not simulation)

3. The camera stream should appear in the Web control

---

## When to Use the Custom Control

The custom control in your other README is only needed if you want:
- Additional camera controls (zoom, pan, settings)
- Custom UI elements overlaid on the video
- JavaScript interaction with the camera
- Advanced features beyond just displaying the stream

For **simple camera viewing**, the built-in Web control is sufficient and much easier!

---

## Troubleshooting

### "Cannot display page" or blank screen
- Verify the URL is correct
- Check network connectivity from HMI to Pi
- Ensure you're testing on hardware, not in simulation
- Test the URL in a browser first

### Frame rate issues
- The stream runs at ~20 FPS by default
- Network bandwidth may affect performance
- Consider lowering resolution in camera_service.py if needed

### iFrame blocked
- This should now work with the backend fix deployed
- If still blocked, check your HMI panel's web browser security settings
