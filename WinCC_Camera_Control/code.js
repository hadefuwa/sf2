var webcc = null;

// Initialize the control
function init() {
    console.log("Initializing Camera Control...");
    
    // Check if WebCC is available (it should be in TIA Portal Runtime)
    if (typeof WebCC !== 'undefined') {
        WebCC.start(
            function(result) {
                if (result) {
                    console.log("WebCC connected successfully.");
                    webcc = result;
                    
                    // Subscribe to property changes from TIA Portal
                    webcc.onPropertyChanged.subscribe(onPropertyChanged);
                    
                    // Initial load of the property
                    if (webcc.Properties && webcc.Properties.CameraURL) {
                        updateCameraUrl(webcc.Properties.CameraURL);
                    }
                } else {
                    console.error("WebCC connection failed!");
                    document.getElementById('loading-msg').innerText = "Error: WebCC connection failed.";
                }
            },
            {
                // define the contract
                methods: [],
                events: [],
                properties: {
                    CameraURL: "" // Default value
                }
            }
        );
    } else {
        // Fallback for testing in a normal browser without WebCC
        console.warn("WebCC not found. Running in standalone mode.");
        // Simulate a URL for testing if you open index.html directly
        // updateCameraUrl("http://localhost:8080/stream");
        document.getElementById('loading-msg').innerText = "WebCC not found (Standalone Mode)";
    }
}

function onPropertyChanged(propertyName, propertyValue) {
    console.log("Property Changed: " + propertyName + " = " + propertyValue);
    if (propertyName === "CameraURL") {
        updateCameraUrl(propertyValue);
    }
}

function updateCameraUrl(url) {
    var img = document.getElementById("camera-feed");
    var loading = document.getElementById("loading-msg");
    
    if (url && url.length > 0) {
        console.log("Setting Camera URL to: " + url);
        // Force reload by appending timestamp if needed to break cache, 
        // but for streams it's usually not needed.
        img.src = url;
        img.style.display = "block";
        if (loading) loading.style.display = "none";
    } else {
        img.style.display = "none";
        if (loading) {
            loading.style.display = "block";
            loading.innerText = "No Camera URL provided";
        }
    }
}

// Start when window loads
window.onload = init;
