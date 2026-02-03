# How to Install Your Raspberry Pi Camera Control on WinCC Unified

## 0. Prerequisites
- **Device**: Must be a **WinCC Unified Basic Panel** or **Unified Comfort Panel**. (Older KTP Basic panels are NOT supported).
- **File**: You need `webcc.min.js`. This is a proprietary Siemens file found in their "Custom Web Control" example projects (Siemens Support ID: 109779176).

---

## 1. Prepare the Control
1.  **Get `webcc.min.js`**:
    - Download it from Siemens Support.
    - Copy the file into this folder: `c:\Users\Hamed\Documents\sf2\WinCC_Camera_Control\`
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
  - If using the Smart Factory PWA backend: `http://<RASPBERRY_PI_IP>:8080/api/camera/stream`
  - Example: `http://192.168.1.50:8080/api/camera/stream`
  http://rpi:8080/api/camera/stream




  ^^^^^^^^ Ignore above information

  SIMATIC HMI WinCC Unified V19
 
Document:	WinCC Unified V19 (02/2024, English)			
Type of topic:	Action

Web control

Use
You use the "Web control" object to display basic HTML pages and documents in PDF format.

You have access to the data of the local user management in runtime via a "Browser".



Layout
In the Inspector window, you can change the settings for the position, geometry, style, and color of the object. You can adapt the following properties in particular:

"URL": Specifies which Internet address is opened in the HTML Browser.
"Toolbar": Specifies the buttons of the browser.
Defining the URL
The "browser" object supports the following protocols:

On a Unified Basic Panel and Unified Comfort Panel:
- https://
- http://
- file://
On a Unified PC:
- https://
To define the URL, follow these steps:

Define the Internet address in the Inspector window under "Properties > Properties > General > URL".

Note

Please note that the "http://" and "file://" protocols do not function in the simulation.


Displaying HTML pages
Please note the following when using the object:

The "Web control" object only displays content that is supported by the web browser in which runtime is open.
The object is implemented as an iFrame. Pages with X-frame option settings that prevent the display in an iFrame are not displayed in the object. 
Limitations
The "Web control" object has a limited range of functions compared to a standard browser:

Navigation from the "Web control" object is not supported (top-level navigation).
Calls of queries and dialogs (popups and modal dialogs) are only supported if they were activated in the file <Path for the WinCC Unified installation directory>WinCCUnified\WebRH\public\content\custom\CustomSettings.json:
{"CustomSettings": {"HmiWebControl" : {"AllowPopups" : true,"AllowModals" : true}}}

Note

Popups and modal dialogs stop the update.


Displaying PDF files in the "Browser" on a Unified PC
The "Browser" object displays PDF files that are available:

Locally on the HMI device
On the Internet
You can view a PDF file in the following ways:

Copy the PDF files to the directory "C:\Program Files\Siemens\Automation\WinCCUnified\WebRH\public".
Under "Properties > URL", enter the address "https://localhost/WebRH/<pdfname.pdf>".

Note

You cannot display any PDF files that are saved locally in a different directory on your PC.


You can also use the IP address or the PC name instead of "localhost".

If you operate runtime on a different PC than the TIA Portal, also save the PDF files on the runtime PC.

Enter a valid Internet address under "Properties > Properties > URL".
Influencing how the document is displayed on a Unified PC
The "Browser" object supports a large number of default parameters with which you can influence how a PDF file is displayed.

Examples of parameters when opening the PDF file:

Jump to specific page: https://winccunified/WebRH/UCPManual.pdf#page=18
Jump to table of contents: https://winccunified/WebRH/UCPManual.pdf#lnhaltsverzeichnis
Zoom in on page: https://winccunified/WebRH/UCPManual.pdf#zoom=200
Displaying PDF files in the "Browser" on a Unified Comfort Panel
The "Browser" object displays PDF files that are available:

Locally on the HMI device
On a Unified Comfort Panel, the download directory of the browser is: "/home/industrial/download".

On an external storage medium
You can view a PDF file in the following ways:

Enter path and file name in the URL input field of the "Browser" operating object.
In the configuration of the "Browser" operating object under "Properties", link the URL with a tag of the type WString which contains path and file name.
Syntax: file:///<path>/<filename>.pdf

Pay attention to uppercase/lowercase spelling.

Examples:

Open file from the data memory card: file:///media/simatic/X51/UCPManual.pdf
Open locally saved file: file:///home/industrial/UCPManual.pdf
Influencing how the document is displayed on a Unified Comfort Panel
The "Browser" object supports a large number of default parameters with which you can influence how a PDF file is displayed.

Examples of parameters when opening the PDF file:

Open file on page 20: file:///media/simatic/X51/UCPManual.pdf?20#page=20
Open file with zoom factor 150%: file:///media/simatic/X51/UCPManual.pdf?150#zoom=150
Open file on page 20 with zoom factor 150%: file:///media/simatic/X51/UCPManual.pdf?(20,150)#page=20&zoom=150
Dynamization of graphic properties with tags or scripts
You can dynamize the following properties containing a graphic with a tag or with a script:

Graphic
Icon
Toolbar
You can define the buttons of the browser in runtime and their operator authorizations in the Inspector window under "Properties > Properties > Miscellaneous > Toolbar > Elements". The buttons are enabled by default.

The following buttons are available for the process control: