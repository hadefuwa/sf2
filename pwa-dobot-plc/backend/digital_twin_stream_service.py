"""
Digital Twin Stream Service - Renders the 3D view on the Pi and streams as MJPEG.

The HMI (Basic Panel) cannot run WebGL/Three.js. This service:
1. Runs headless Chromium with the digital-twin-embed page
2. Captures screenshots of the rendered 3D view
3. Serves them as MJPEG - same format as the camera stream

The HMI displays the video feed (which it can do) instead of rendering 3D.
"""

import logging
import threading
import time
import io
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import Playwright
PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    logger.warning("Playwright not installed. Digital twin stream disabled. Run: pip install playwright && playwright install chromium")


class DigitalTwinStreamService:
    """Captures digital twin page in headless browser and provides JPEG frames for streaming."""

    def __init__(self, port: int = 8080, width: int = 640, height: int = 480):
        self.port = port
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        self._frame_time = 0.0
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._playwright = None
        self._browser = None
        self._page = None
        self._error: Optional[str] = None

    def start(self) -> bool:
        """Start the capture thread."""
        if not PLAYWRIGHT_AVAILABLE:
            self._error = "Playwright not installed. Run: pip install playwright && playwright install chromium"
            return False

        with self._lock:
            if self._running:
                return True
            self._running = True

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Digital twin stream service started (capture thread)")
        return True

    def stop(self):
        """Stop the capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def get_frame_jpeg(self, quality: int = 70) -> Optional[bytes]:
        """Get the latest captured frame as JPEG bytes."""
        with self._lock:
            return self._latest_frame

    def _capture_loop(self):
        """Background loop: launch browser, load page, capture screenshots."""
        # Wait for Flask server to be ready
        time.sleep(10)
        url = f"http://127.0.0.1:{self.port}/digital-twin-embed.html"
        try:
            with sync_playwright() as p:
                self._playwright = p
                self._browser = p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
                )
                self._page = self._browser.new_page(
                    viewport={"width": self.width, "height": self.height},
                    ignore_https_errors=True
                )
                self._page.goto(url, wait_until="networkidle", timeout=30000)
                # Wait for Three.js to render - give it a moment
                time.sleep(2)

                capture_interval = 0.1  # ~10 FPS - 3D is heavier than camera
                last_capture = 0.0

                while self._running:
                    now = time.time()
                    if now - last_capture >= capture_interval:
                        try:
                            screenshot_bytes = self._page.screenshot(type="jpeg", quality=70)
                            if screenshot_bytes:
                                with self._lock:
                                    self._latest_frame = screenshot_bytes
                                    self._frame_time = now
                            last_capture = now
                        except Exception as e:
                            logger.debug(f"Screenshot error: {e}")
                            time.sleep(0.5)

                    time.sleep(0.05)

        except Exception as e:
            self._error = str(e)
            logger.error(f"Digital twin stream error: {e}", exc_info=True)
        finally:
            try:
                if self._browser:
                    self._browser.close()
            except Exception:
                pass

    @property
    def is_available(self) -> bool:
        return PLAYWRIGHT_AVAILABLE
