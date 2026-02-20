"""
Camera Service for USB Camera Access and Defect Detection
Uses OpenCV for camera capture and image processing
"""

import cv2
import numpy as np
import logging
import threading
import time
import hashlib
from typing import Optional, Dict, List, Tuple
import io
import os

logger = logging.getLogger(__name__)

# Try to import YOLO (optional)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("YOLO (Ultralytics) loaded successfully")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available - install with: pip install ultralytics")

class CameraService:
    """Service for managing USB camera and defect detection"""
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera service
        
        Args:
            camera_index: Camera device index (usually 0 for first USB camera)
            width: Frame width
            height: Frame height
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_streaming = False
        self.lock = threading.Lock()
        self.last_frame = None
        self.frame_time = 0
        # Analyzed frame caching for PLC HMI stream
        self.last_analyzed_frame = None
        self.analyzed_frame_time = 0
        # Object detection model
        self.object_net = None
        self.object_classes = []
        self.object_detection_enabled = False
        # Background subtractor for moving object detection
        self.bg_subtractor: Optional[cv2.BackgroundSubtractor] = None
        self.bg_learning_frames = 0
        self.bg_initialized = False
        # YOLO model
        self.yolo_model = None
        self.yolo_model_path = None
        self.yolo_lock = threading.Lock()  # Lock to prevent concurrent YOLO calls (YOLO is not thread-safe)
        self.last_yolo_call_time = 0
        self.min_yolo_interval = 3.0  # Minimum 3 seconds between YOLO calls to prevent crashes
        self.cached_yolo_result = None  # Cache last YOLO detection result
        self.cached_yolo_result_time = 0
        self.cached_yolo_frame_hash = None  # Hash of frame to detect if frame changed
        self.yolo_crash_count = 0  # Track consecutive crashes
        self.yolo_disabled_until = 0  # Timestamp when YOLO can be re-enabled after crashes
        self.max_crashes = 2  # Disable YOLO after 2 consecutive crashes (more aggressive)
        self.disable_duration = 60  # Disable for 60 seconds after crashes (longer cooldown)
        # Crop/zoom settings
        self.crop_enabled = False
        self.crop_x = 0  # Top-left X (as percentage 0-100)
        self.crop_y = 0  # Top-left Y (as percentage 0-100)
        self.crop_width = 100  # Width (as percentage 0-100)
        self.crop_height = 100  # Height (as percentage 0-100)
        
    def initialize_camera(self) -> bool:
        """Initialize and open camera"""
        try:
            with self.lock:
                if self.camera is not None:
                    try:
                        self.camera.release()
                    except Exception:
                        pass  # Ignore errors when releasing
                
                self.camera = cv2.VideoCapture(self.camera_index)
                
                if not self.camera.isOpened():
                    logger.error(f"Failed to open camera at index {self.camera_index}")
                    self.camera = None
                    return False
                
                # Set camera properties (these may fail silently, which is okay)
                try:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.camera.set(cv2.CAP_PROP_FPS, 30)
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for faster response
                except Exception as e:
                    logger.warning(f"Could not set some camera properties: {e}")

                # Quick warm up camera (reduced from 5 to 2 frames)
                # If read() fails, camera may not be ready - that's okay, we'll try again later
                try:
                    for _ in range(2):
                        ret, _ = self.camera.read()
                        if not ret:
                            logger.warning(f"Camera read() failed during warm-up - camera may not be ready")
                            break
                except Exception as e:
                    logger.warning(f"Camera read() error during initialization: {e} - camera may not be ready")
                    # Don't fail completely - camera might work later
                    # Keep self.camera set so we can try again
                
                logger.info(f"Camera initialized at index {self.camera_index} (may need warm-up)")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing camera: {e}", exc_info=True)
            # Make sure camera is set to None on error
            try:
                if self.camera is not None:
                    self.camera.release()
            except Exception:
                pass
            self.camera = None
            return False
    
    def release_camera(self):
        """Release camera resources"""
        with self.lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                logger.info("Camera released")
            # Reset background subtractor
            if self.bg_subtractor is not None:
                self.bg_subtractor = None
                self.bg_initialized = False
                self.bg_learning_frames = 0
                logger.info("Background subtractor reset")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from camera and apply crop if enabled"""
        try:
            with self.lock:
                if self.camera is None:
                    return None
                
                # Check if camera is still opened
                if not self.camera.isOpened():
                    logger.warning("Camera is no longer opened")
                    return None
                
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    # Apply crop if enabled
                    if self.crop_enabled:
                        frame = self._apply_crop(frame)
                    self.last_frame = frame
                    self.frame_time = time.time()
                    return frame
                else:
                    return None
        except Exception as e:
            logger.warning(f"Error reading camera frame: {e}")
            return None
    
    def _apply_crop(self, frame: np.ndarray) -> np.ndarray:
        """Apply crop/zoom to frame based on crop settings"""
        if frame is None or frame.size == 0:
            return frame
        
        try:
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate crop region in pixels
            x_px = int(frame_width * self.crop_x / 100)
            y_px = int(frame_height * self.crop_y / 100)
            width_px = int(frame_width * self.crop_width / 100)
            height_px = int(frame_height * self.crop_height / 100)
            
            # Ensure crop region is within frame bounds
            x_px = max(0, min(x_px, frame_width - 1))
            y_px = max(0, min(y_px, frame_height - 1))
            width_px = max(1, min(width_px, frame_width - x_px))
            height_px = max(1, min(height_px, frame_height - y_px))
            
            # Crop the frame
            cropped = frame[y_px:y_px + height_px, x_px:x_px + width_px]
            
            return cropped
        except Exception as e:
            logger.warning(f"Error applying crop: {e}")
            return frame
    
    def set_crop(self, enabled: bool, x: float = 0, y: float = 0, width: float = 100, height: float = 100):
        """
        Set crop/zoom region for camera feed
        
        Args:
            enabled: Enable/disable crop
            x: Top-left X position as percentage (0-100)
            y: Top-left Y position as percentage (0-100)
            width: Crop width as percentage (0-100)
            height: Crop height as percentage (0-100)
        """
        with self.lock:
            self.crop_enabled = enabled
            self.crop_x = max(0, min(100, x))
            self.crop_y = max(0, min(100, y))
            self.crop_width = max(1, min(100, width))
            self.crop_height = max(1, min(100, height))
            logger.info(f"Crop settings updated: enabled={enabled}, x={self.crop_x}%, y={self.crop_y}%, width={self.crop_width}%, height={self.crop_height}%")
    
    def get_crop(self) -> Dict:
        """Get current crop settings"""
        with self.lock:
            return {
                'enabled': self.crop_enabled,
                'x': self.crop_x,
                'y': self.crop_y,
                'width': self.crop_width,
                'height': self.crop_height
            }
    
    def set_analyzed_frame(self, frame: np.ndarray):
        """Store the last analyzed frame for streaming"""
        with self.lock:
            self.last_analyzed_frame = frame.copy() if frame is not None else None
            self.analyzed_frame_time = time.time()
    
    def get_frame_jpeg(self, quality: int = 85, use_cache: bool = True, max_cache_age: float = 0.5, prefer_analyzed: bool = False, analyzed_max_age: float = 5.0) -> Optional[bytes]:
        """
        Get current frame as JPEG bytes
        
        Args:
            quality: JPEG quality (0-100)
            use_cache: If True, use cached frame if recent enough (reduces camera reads)
            max_cache_age: Maximum age of cached frame in seconds before reading new one
            prefer_analyzed: If True, prefer analyzed frame when available and recent
            analyzed_max_age: Maximum age of analyzed frame in seconds before falling back to raw
            
        Returns:
            JPEG bytes or None if frame not available
        """
        frame = None
        
        # Check if we should prefer analyzed frame
        if prefer_analyzed:
            with self.lock:
                if self.last_analyzed_frame is not None:
                    analyzed_age = time.time() - self.analyzed_frame_time
                    if analyzed_age < analyzed_max_age:
                        # Use analyzed frame if it's recent enough
                        frame = self.last_analyzed_frame.copy()
        
        # If no analyzed frame (or not preferring it), get raw frame
        if frame is None:
            # Use cached frame if available and recent enough (optimization for snapshot mode)
            if use_cache and self.last_frame is not None:
                cache_age = time.time() - self.frame_time
                if cache_age < max_cache_age:
                    frame = self.last_frame.copy()
                else:
                    # Cache is too old, read new frame
                    frame = self.read_frame()
            else:
                # No cache or cache disabled, read new frame
                frame = self.read_frame()
        
        if frame is None:
            return None
        
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            if ret:
                return buffer.tobytes()
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
        
        return None
    
    def load_yolo_model(self, model_path: str = 'yolov8n.pt') -> bool:
        """
        Load YOLO model for object detection

        Args:
            model_path: Path to YOLO model file (.pt)
                       Use 'yolov8n.pt' for pretrained nano model
                       Use custom path for trained counter detection model

        Returns:
            True if model loaded successfully
        """
        if not YOLO_AVAILABLE:
            logger.error("YOLO not available")
            return False

        try:
            logger.info(f"Loading YOLO model: {model_path}")
            self.yolo_model = YOLO(model_path)
            self.yolo_model_path = model_path
            logger.info(f"YOLO model loaded successfully: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            return False

    def detect_objects(self, frame: np.ndarray, method: str = 'yolo', params: Optional[Dict] = None) -> Dict:
        """
        Detect circular counters on conveyor belt.

        Args:
            frame: Input image frame (BGR format)
            method: Detection method - 'yolo' (default), 'blob', 'color'
            params: Optional detection parameters:
                YOLO:
                - conf: Confidence threshold (default: 0.25)
                - iou: IOU threshold for NMS (default: 0.45)
                - classes: List of class IDs to detect (default: None = all)

                SimpleBlobDetector:
                - min_area: Minimum counter area in pixels (default: 500)
                - max_area: Maximum counter area in pixels (default: 50000)
                - min_circularity: Minimum circularity (0-1, default: 0.6)
                - min_convexity: Minimum convexity (0-1, default: 0.7)
                - min_inertia_ratio: Minimum inertia ratio (0-1, default: 0.3)
                
                Color Filter (HSV):
                - min_area: Minimum object area in pixels (default: 500)
                - max_area: Maximum object area in pixels (default: 50000)
                - detect_yellow: Detect yellow cubes (default: True)
                - detect_white: Detect white cubes (default: True)
                - detect_metal: Detect metal/grey cubes (default: True)

        Returns:
            Dictionary with counter detection results
        """
        if frame is None:
            return {
                'objects_found': False,
                'object_count': 0,
                'objects': [],
                'error': 'No frame provided'
            }

        if params is None:
            params = {}

        try:
            # YOLO Detection
            if method == 'yolo':
                return self._detect_with_yolo(frame, params)

            # SimpleBlobDetector (fallback)
            elif method == 'blob':
                return self._detect_with_blob(frame, params)

            # Color Filter (HSV-based)
            elif method == 'color':
                return self._detect_with_color(frame, params)

            else:
                return {
                    'objects_found': False,
                    'object_count': 0,
                    'objects': [],
                    'error': f'Unknown detection method: {method}'
                }

        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return {
                'objects_found': False,
                'object_count': 0,
                'objects': [],
                'error': str(e)
            }

    def _detect_with_yolo(self, frame: np.ndarray, params: Dict) -> Dict:
        """Detect objects using YOLO - thread-safe with locking, rate limiting, and result caching"""
        if not YOLO_AVAILABLE:
            error_msg = 'YOLO library not available. Install with: pip install ultralytics'
            logger.error(error_msg)
            return {
                'objects_found': False,
                'object_count': 0,
                'objects': [],
                'error': error_msg
            }
        
        if self.yolo_model is None:
            error_msg = 'YOLO model not loaded. Call load_yolo_model() first.'
            logger.error(error_msg)
            return {
                'objects_found': False,
                'object_count': 0,
                'objects': [],
                'error': error_msg
            }

        # Calculate frame hash to detect if frame changed
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
        
        # Rate limiting and caching: return cached result if called too soon or same frame
        current_time = time.time()
        time_since_last_call = current_time - self.last_yolo_call_time
        cache_age = current_time - self.cached_yolo_result_time
        
        # Circuit breaker: Check if YOLO is temporarily disabled due to crashes
        if current_time < self.yolo_disabled_until:
            remaining = self.yolo_disabled_until - current_time
            logger.warning(f"YOLO temporarily disabled due to crashes (re-enable in {remaining:.1f}s)")
            # Return cached result if available, otherwise empty
            if self.cached_yolo_result is not None:
                cached = self.cached_yolo_result.copy()
                cached['timestamp'] = current_time
                cached['cached'] = True
                cached['error'] = f'YOLO disabled (crashed {self.yolo_crash_count} times)'
                return cached
            return {
                'objects_found': False,
                'object_count': 0,
                'objects': [],
                'method': 'yolo',
                'timestamp': current_time,
                'error': f'YOLO disabled due to crashes (re-enable in {remaining:.1f}s)'
            }

        # ALWAYS return cached result if available and recent (prevents crashes)
        # Return cached result if:
        # 1. Cache exists and is less than 5 seconds old, OR
        # 2. Called too soon (less than min interval)
        if self.cached_yolo_result is not None:
            if cache_age < 5.0 or time_since_last_call < self.min_yolo_interval:
                logger.debug(f"YOLO returning cached result (cache age: {cache_age:.3f}s, time since last: {time_since_last_call:.3f}s)")
                # Return cached result with updated timestamp
                cached = self.cached_yolo_result.copy()
                cached['timestamp'] = current_time
                cached['cached'] = True
                return cached

        # Rate limiting: prevent YOLO calls too close together (prevents crashes)
        if time_since_last_call < self.min_yolo_interval:
            # Return cached result if available, otherwise empty
            if self.cached_yolo_result is not None:
                logger.debug(f"YOLO call rate-limited, returning cached result")
                cached = self.cached_yolo_result.copy()
                cached['timestamp'] = current_time
                cached['cached'] = True
                return cached
            else:
                logger.debug(f"YOLO call rate-limited (last call {time_since_last_call:.3f}s ago, min {self.min_yolo_interval}s)")
                return {
                    'objects_found': False,
                    'object_count': 0,
                    'objects': [],
                    'method': 'yolo',
                    'timestamp': current_time,
                    'error': 'Rate limited - too many calls'
                }

        # Use lock to prevent concurrent YOLO calls (YOLO is NOT thread-safe)
        with self.yolo_lock:
            try:
                self.last_yolo_call_time = time.time()
                
                # Extract YOLO parameters
                conf_threshold = params.get('conf', 0.25)  # Default confidence threshold (0.25 = 25%)
                iou = params.get('iou', 0.45)
                classes = params.get('classes', None)
                crop_top_percent = params.get('crop_top_percent', 0)  # No cropping by default
                crop_bottom_percent = params.get('crop_bottom_percent', 0)  # No cropping by default
                
                logger.debug(f"YOLO detection params: conf={conf_threshold}, iou={iou}, crop_top={crop_top_percent}%, crop_bottom={crop_bottom_percent}%")

                # Crop the frame to remove top and bottom regions
                original_height = frame.shape[0]
                original_width = frame.shape[1]

                crop_top = int(original_height * crop_top_percent / 100)
                crop_bottom = int(original_height * (100 - crop_bottom_percent) / 100)

                cropped_frame = frame[crop_top:crop_bottom, :]
                logger.debug(f"Cropped frame from {original_height}x{original_width} to {cropped_frame.shape[0]}x{cropped_frame.shape[1]}")

                # Run inference on cropped frame - wrap VERY defensively to catch C++ crashes
                logger.debug(f"Running YOLO inference on frame shape: {cropped_frame.shape}")
                
                # Reset crash count on successful call
                self.yolo_crash_count = 0
                
                # Try to run YOLO inference - wrap everything in try-except
                try:
                    # Validate frame before calling YOLO
                    if cropped_frame.size == 0 or cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
                        raise ValueError("Invalid frame dimensions")
                    
                    # Call YOLO model - this is where crashes happen
                    results = self.yolo_model(cropped_frame, conf=conf_threshold, iou=iou, classes=classes, verbose=False)
                    
                    # Process results - also wrap in try-except in case results are corrupted
                    try:
                        # Log raw results for debugging
                        total_detections = sum(len(r.boxes) for r in results)
                        logger.debug(f"YOLO raw detection count: {total_detections}")

                        objects = []
                        for result in results:
                            try:
                                boxes = result.boxes
                                for box in boxes:
                                    try:
                                        # Extract box coordinates (relative to cropped frame)
                                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                        box_confidence = float(box.conf[0])  # Individual box confidence
                                        cls = int(box.cls[0])
                                        class_name = result.names[cls]

                                        # Adjust coordinates back to original frame
                                        x1_original = x1
                                        y1_original = y1 + crop_top
                                        x2_original = x2
                                        y2_original = y2 + crop_top

                                        # Calculate center and dimensions
                                        x = int(x1_original)
                                        y = int(y1_original)
                                        w = int(x2_original - x1_original)
                                        h = int(y2_original - y1_original)
                                        center_x = int(x1_original + w / 2)
                                        center_y = int(y1_original + h / 2)
                                        area = w * h

                                        objects.append({
                                            'type': 'counter',
                                            'class': class_name,
                                            'class_id': cls,
                                            'x': x,
                                            'y': y,
                                            'width': w,
                                            'height': h,
                                            'area': float(area),
                                            'center': (center_x, center_y),
                                            'confidence': round(box_confidence, 2),
                                            'method': 'yolo'
                                        })
                                    except Exception as box_error:
                                        logger.warning(f"Error processing YOLO box: {box_error}")
                                        continue  # Skip this box, continue with others
                            except Exception as result_error:
                                logger.warning(f"Error processing YOLO result: {result_error}")
                                continue  # Skip this result, continue with others

                        logger.debug(f"YOLO detected {len(objects)} objects")
                        if len(objects) == 0:
                            logger.debug(f"No objects detected - conf threshold may be too high (current: {conf_threshold})")

                        # Cache the result
                        result = {
                            'objects_found': len(objects) > 0,
                            'object_count': len(objects),
                            'objects': objects,
                            'method': 'yolo',
                            'timestamp': time.time(),
                            'cached': False
                        }
                        self.cached_yolo_result = result
                        self.cached_yolo_result_time = time.time()
                        self.cached_yolo_frame_hash = frame_hash
                        
                        return result
                    except Exception as process_error:
                        logger.error(f"Error processing YOLO results: {process_error}", exc_info=True)
                        raise  # Re-raise to be caught by outer exception handler
                        
                except (Exception, SystemError, RuntimeError, MemoryError) as e:
                    # Catch ALL exceptions including C++ exceptions that might bypass normal Python exceptions
                    self.yolo_crash_count += 1
                    logger.error(f"YOLO detection crashed (crash #{self.yolo_crash_count}): {e}", exc_info=True)
                    
                    # If too many crashes, disable YOLO temporarily
                    if self.yolo_crash_count >= self.max_crashes:
                        self.yolo_disabled_until = time.time() + self.disable_duration
                        logger.error(f"YOLO disabled for {self.disable_duration}s after {self.yolo_crash_count} consecutive crashes")
                    
                    # Return cached result if available, otherwise empty
                    if self.cached_yolo_result is not None:
                        cached = self.cached_yolo_result.copy()
                        cached['timestamp'] = time.time()
                        cached['cached'] = True
                        cached['error'] = f'YOLO crashed (attempt {self.yolo_crash_count})'
                        return cached
                    
                    return {
                        'objects_found': False,
                        'object_count': 0,
                        'objects': [],
                        'method': 'yolo',
                        'timestamp': time.time(),
                        'error': f'YOLO detection failed: {str(e)} (crash #{self.yolo_crash_count})'
                    }
            finally:
                # Ensure lock is released even if there's an error
                pass

    def _detect_with_blob(self, frame: np.ndarray, params: Dict) -> Dict:
        """Detect objects using SimpleBlobDetector"""
        # Extract parameters
        min_area = params.get('min_area', 500)
        max_area = params.get('max_area', 50000)
        min_circularity = params.get('min_circularity', 0.6)
        min_convexity = params.get('min_convexity', 0.7)
        min_inertia_ratio = params.get('min_inertia_ratio', 0.3)

        try:
            # Step 1: Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Step 2: Apply Gaussian blur to reduce noise and reflections
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Step 3: Setup SimpleBlobDetector parameters (very relaxed)
            blob_params = cv2.SimpleBlobDetector_Params()

            # Filter by Area
            blob_params.filterByArea = True
            blob_params.minArea = min_area
            blob_params.maxArea = max_area

            # Filter by Circularity (relaxed for imperfect circles)
            blob_params.filterByCircularity = True
            blob_params.minCircularity = min_circularity

            # Filter by Convexity (relaxed)
            blob_params.filterByConvexity = True
            blob_params.minConvexity = min_convexity

            # Filter by Inertia (relaxed to accept ovals)
            blob_params.filterByInertia = True
            blob_params.minInertiaRatio = min_inertia_ratio

            # Detect both dark and light blobs
            blob_params.filterByColor = False

            # Set threshold values for better detection
            blob_params.minThreshold = 10
            blob_params.maxThreshold = 200
            blob_params.thresholdStep = 10

            # Step 4: Create detector and detect blobs
            detector = cv2.SimpleBlobDetector_create(blob_params)
            keypoints = detector.detect(blurred)

            logger.info(f"SimpleBlobDetector found {len(keypoints)} keypoints")

            objects = []

            # Step 5: Extract blob information
            for kp in keypoints:
                x, y = kp.pt
                size = kp.size
                radius = size / 2
                area = np.pi * radius * radius

                logger.debug(f"Blob at ({x:.0f},{y:.0f}), size={size:.0f}, area={area:.0f}")

                # Calculate bounding box
                x_int = int(x - radius)
                y_int = int(y - radius)
                w = h = int(size)

                objects.append({
                    'type': 'counter',
                    'x': x_int,
                    'y': y_int,
                    'width': w,
                    'height': h,
                    'area': float(area),
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'circularity': 1.0,  # SimpleBlobDetector filters by circularity already
                    'confidence': 0.9,  # High confidence for blob detection
                    'method': 'blob'
                })

            logger.info(f"Returning {len(objects)} detected counters")

            return {
                'objects_found': len(objects) > 0,
                'object_count': len(objects),
                'objects': objects,
                'method': 'blob',
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Error in blob detection: {e}")
            return {
                'objects_found': False,
                'object_count': 0,
                'objects': [],
                'error': str(e)
            }
    
    def _detect_with_color(self, frame: np.ndarray, params: Dict) -> Dict:
        """
        Detect cubes using HSV color filtering (Yellow, White, Metal/Grey)
        
        This is a simple color-based detection method that doesn't use AI.
        It converts the image to HSV color space and looks for specific color ranges.
        HSV is better for computer vision because Hue stays consistent even with shadows.
        """
        # Extract parameters with defaults
        min_area = params.get('min_area', 500)
        max_area = params.get('max_area', 50000)
        detect_yellow = params.get('detect_yellow', True)
        detect_white = params.get('detect_white', True)
        detect_metal = params.get('detect_metal', True)
        
        try:
            # Debug: Log frame info
            frame_height, frame_width = frame.shape[:2]
            logger.info(f"🔍 Color Detection Debug - Frame size: {frame_width}x{frame_height}, Min area: {min_area}, Max area: {max_area}")
            
            # Convert BGR to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Debug: Log HSV statistics
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            v_mean = np.mean(hsv[:, :, 2])
            logger.info(f"🔍 HSV Statistics - H: {h_mean:.1f}, S: {s_mean:.1f}, V: {v_mean:.1f}")
            
            all_objects = []
            debug_info = {}
            
            # Define HSV color ranges (may need tuning based on lighting)
            # Yellow cubes
            if detect_yellow:
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([30, 255, 255])
                mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
                yellow_pixels = np.sum(mask_yellow > 0)
                logger.info(f"🔍 Yellow mask: {yellow_pixels} pixels matched (range: H[20-30], S[100-255], V[100-255])")
                objects_yellow = self._find_objects_from_mask(mask_yellow, min_area, max_area, 'yellow', debug_info)
                all_objects.extend(objects_yellow)
                debug_info['yellow'] = {'pixels': yellow_pixels, 'objects': len(objects_yellow)}
            
            # White cubes (low saturation, high brightness)
            if detect_white:
                lower_white = np.array([0, 0, 180])
                upper_white = np.array([180, 40, 255])
                mask_white = cv2.inRange(hsv, lower_white, upper_white)
                white_pixels = np.sum(mask_white > 0)
                logger.info(f"🔍 White mask: {white_pixels} pixels matched (range: H[0-180], S[0-40], V[180-255])")
                objects_white = self._find_objects_from_mask(mask_white, min_area, max_area, 'white', debug_info)
                all_objects.extend(objects_white)
                debug_info['white'] = {'pixels': white_pixels, 'objects': len(objects_white)}
            
            # Metal/Grey cubes (low saturation, mid-range brightness)
            if detect_metal:
                lower_metal = np.array([0, 0, 50])
                upper_metal = np.array([180, 50, 150])
                mask_metal = cv2.inRange(hsv, lower_metal, upper_metal)
                metal_pixels = np.sum(mask_metal > 0)
                logger.info(f"🔍 Metal mask: {metal_pixels} pixels matched (range: H[0-180], S[0-50], V[50-150])")
                objects_metal = self._find_objects_from_mask(mask_metal, min_area, max_area, 'metal', debug_info)
                all_objects.extend(objects_metal)
                debug_info['metal'] = {'pixels': metal_pixels, 'objects': len(objects_metal)}
            
            logger.info(f"🔍 Color filter detected {len(all_objects)} objects (yellow: {len([o for o in all_objects if o.get('color') == 'yellow'])}, white: {len([o for o in all_objects if o.get('color') == 'white'])}, metal: {len([o for o in all_objects if o.get('color') == 'metal'])})")
            
            return {
                'objects_found': len(all_objects) > 0,
                'object_count': len(all_objects),
                'objects': all_objects,
                'method': 'color',
                'timestamp': time.time(),
                'debug': debug_info
            }
            
        except Exception as e:
            logger.error(f"Error in color detection: {e}")
            return {
                'objects_found': False,
                'object_count': 0,
                'objects': [],
                'method': 'color',
                'error': str(e)
            }
    
    def _find_objects_from_mask(self, mask: np.ndarray, min_area: int, max_area: int, color_name: str, debug_info: Dict = None) -> List[Dict]:
        """
        Find objects from a binary mask using contour detection
        
        Args:
            mask: Binary mask (white = object, black = background)
            min_area: Minimum object area in pixels
            max_area: Maximum object area in pixels
            color_name: Name of the color being detected (for labeling)
            debug_info: Optional dict to store debug information
            
        Returns:
            List of detected objects with bounding boxes
        """
        objects = []
        
        # Clean up the mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask_before = np.sum(mask > 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # Remove small noise
        mask_after = np.sum(mask > 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug: Log contour info
        total_contours = len(contours)
        rejected_small = 0
        rejected_large = 0
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area:
                rejected_small += 1
                continue
            if area > max_area:
                rejected_large += 1
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Calculate circularity (how close to a circle)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Create object dictionary
            obj = {
                'type': 'cube',
                'color': color_name,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'area': float(area),
                'center': (int(center_x), int(center_y)),
                'circularity': round(circularity, 2),
                'confidence': 0.85,  # Color detection is fairly reliable
                'method': 'color'
            }
            
            objects.append(obj)
            logger.info(f"🔍 {color_name.capitalize()} object found: area={area:.0f}, center=({center_x},{center_y}), size={w}x{h}")
        
        # Debug logging
        logger.info(f"🔍 {color_name.capitalize()} contours: {total_contours} total, {rejected_small} too small (<{min_area}), {rejected_large} too large (>{max_area}), {len(objects)} accepted")
        if debug_info is not None:
            debug_info[f'{color_name}_contours'] = {
                'total': total_contours,
                'rejected_small': rejected_small,
                'rejected_large': rejected_large,
                'accepted': len(objects),
                'mask_pixels_before_morph': mask_before,
                'mask_pixels_after_morph': mask_after
            }
        
        return objects
    
    def _merge_nearby_objects(self, objects: List[Dict], threshold: int = 30) -> List[Dict]:
        """Merge objects that are close to each other"""
        if len(objects) == 0:
            return []
        
        merged = []
        used = set()
        
        for i, obj in enumerate(objects):
            if i in used:
                continue
            
            center = obj['center']
            group = [obj]
            used.add(i)
            
            for j, other_obj in enumerate(objects):
                if j in used or j == i:
                    continue
                
                other_center = other_obj['center']
                dist = np.sqrt((center[0] - other_center[0])**2 + (center[1] - other_center[1])**2)
                
                if dist < threshold:
                    group.append(other_obj)
                    used.add(j)
            
            # Merge group into single object
            if len(group) > 1:
                xs = [o['x'] for o in group]
                ys = [o['y'] for o in group]
                ws = [o['width'] for o in group]
                hs = [o['height'] for o in group]
                
                x, y = min(xs), min(ys)
                w = max(x + w for x, w in zip(xs, ws)) - x
                h = max(y + h for y, h in zip(ys, hs)) - y
                
                merged.append({
                    'type': 'object',
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': sum(o['area'] for o in group),
                    'center': (x + w//2, y + h//2),
                    'confidence': max(o['confidence'] for o in group),
                    'method': 'merged'
                })
            else:
                merged.append(obj)
        
        return merged
    
    def _extract_circle_roi(self, frame: np.ndarray, x: int, y: int, radius: int) -> Optional[np.ndarray]:
        """
        Extract region of interest around a circle for classification
        
        Args:
            frame: Full image frame
            x: Circle center X coordinate
            y: Circle center Y coordinate
            radius: Circle radius
            
        Returns:
            ROI image or None if extraction fails
        """
        y1 = max(0, y - radius)
        y2 = min(frame.shape[0], y + radius)
        x1 = max(0, x - radius)
        x2 = min(frame.shape[1], x + radius)
        
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else None
    
    def _create_circle_object(self, x: int, y: int, radius: int, area: float, 
                             confidence: float) -> Dict:
        """
        Create a standardized circle object dictionary
        
        Args:
            x: Circle center X coordinate
            y: Circle center Y coordinate
            radius: Circle radius
            area: Circle area
            confidence: Detection confidence (0-1)
            
        Returns:
            Object dictionary
        """
        return {
            'type': 'circle',
            'x': int(x - radius),
            'y': int(y - radius),
            'width': int(2 * radius),
            'height': int(2 * radius),
            'area': float(area),
            'center': (int(x), int(y)),
            'radius': int(radius),
            'circularity': 1.0,
            'confidence': round(confidence, 2),
            'method': 'circle',
            'aspect_ratio': 1.0
        }
    
    def classify_disc(self, roi: np.ndarray) -> str:
        """
        Classify a disc (counter) as white, black, silver, or grey
        Uses color analysis in HSV space
        
        Args:
            roi: Region of interest (the disc area) in BGR format
            
        Returns:
            Classification string: 'white', 'black', 'silver', or 'grey'
        """
        if roi is None or roi.size == 0:
            return 'unknown'
        
        try:
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Calculate mean HSV values
            mean_hsv = hsv.mean(axis=(0, 1))  # H, S, V mean values
            H, S, V = mean_hsv
            
            # Calculate variance in BGR space for silver vs grey detection
            grey_var = np.var(roi)
            
            # Classification logic based on brightness, saturation, and variance
            if V > 180 and S < 40:
                return 'white'
            elif V < 60:
                return 'black'
            elif grey_var > 200:
                return 'silver'
            else:
                return 'grey'
                
        except Exception as e:
            logger.error(f"Error classifying disc: {e}")
            return 'unknown'
    
    def _detect_circles_hough(self, frame: np.ndarray, params: Dict, 
                              min_object_area: int, max_object_area: int, 
                              min_confidence: float) -> List[Dict]:
        """
        Detect circles using HoughCircles on grayscale image
        
        Args:
            frame: Input frame in BGR format
            params: Detection parameters
            min_object_area: Minimum object area
            max_object_area: Maximum object area
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detected circle objects
        """
        objects = []
        
        # Convert to grayscale for circle detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise (larger kernel for better smoothing)
        blurred = cv2.GaussianBlur(gray, (11, 11), 2)
        
        # HoughCircles parameters - optimized for ultra-reliable circle detection
        dp = 1  # Inverse ratio of accumulator resolution
        min_dist = params.get('min_dist_between_circles', 50)
        param1 = 100  # Upper threshold for edge detection (higher = better edge detection)
        param2 = params.get('hough_circle_threshold', 30)  # Higher = fewer false positives, more reliable
        min_radius = max(5, int(np.sqrt(min_object_area / np.pi)))  # Ensure minimum radius is at least 5 pixels
        max_radius = int(np.sqrt(max_object_area / np.pi))
        
        logger.info(f"Circle detection params: min_area={min_object_area}, max_area={max_object_area}, "
                   f"min_radius={min_radius}, max_radius={max_radius}, param2={param2}")
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            logger.info(f"HoughCircles found {len(circles)} potential circles")
            
            for (x, y, r) in circles:
                # Calculate area
                area = np.pi * r * r
                
                logger.debug(f"Circle candidate: center=({x},{y}), radius={r}, area={area:.0f}")
                
                # Basic area filtering: Only accept circles within the size range
                if area < min_object_area:
                    logger.debug(f"  Rejected: area {area:.0f} < min {min_object_area}")
                    continue
                if area > max_object_area:
                    logger.debug(f"  Rejected: area {area:.0f} > max {max_object_area}")
                    continue
                
                # Simple confidence calculation based on size
                # Normalize area to 0-1 range within min/max bounds
                if max_object_area > min_object_area:
                    normalized_area = (area - min_object_area) / (max_object_area - min_object_area)
                    # Higher confidence for circles in the middle range
                    confidence = 0.5 + (normalized_area * 0.5)  # Range: 0.5 to 1.0
                else:
                    confidence = 0.7  # Default confidence
                
                # Only add if meets confidence threshold
                if confidence >= min_confidence:
                    logger.info(f"  Accepted circle: center=({x},{y}), radius={r}, area={area:.0f}, confidence={confidence:.2f}")
                    obj = self._create_circle_object(x, y, r, area, confidence)
                    objects.append(obj)
                else:
                    logger.debug(f"  Rejected: confidence {confidence:.2f} < min {min_confidence}")
        
        logger.info(f"Total circles detected after filtering: {len(objects)}")
        
        return objects
    
    def _detect_circles_hsv_fallback(self, frame: np.ndarray, params: Dict,
                                     min_object_area: int, max_object_area: int,
                                     min_confidence: float) -> List[Dict]:
        """
        Fallback circle detection using HSV color masking (for when HoughCircles fails)
        
        Args:
            frame: Input frame in BGR format
            params: Detection parameters
            min_object_area: Minimum object area
            max_object_area: Maximum object area
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detected circle objects
        """
        objects = []
        
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_hue_min = params.get('hsv_hue_min', 90)
        hsv_hue_max = params.get('hsv_hue_max', 130)
        
        # Create blue background mask
        lower_blue = np.array([hsv_hue_min, 50, 50])
        upper_blue = np.array([hsv_hue_max, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        object_mask = cv2.bitwise_not(blue_mask)
        
        # Clean up mask
        kernel_size = params.get('morphological_kernel_size', 7)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_object_area < area < max_object_area:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                min_circularity = params.get('min_circularity', 0.6)
                if circularity >= min_circularity:
                    x_int, y_int, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    if 0.7 < aspect_ratio < 1.3:
                        # Calculate confidence based on circularity and size
                        size_confidence = min(area / max_object_area, 1.0)
                        confidence = (circularity * 0.7 + size_confidence * 0.3)
                        
                        if confidence >= min_confidence:
                            obj = self._create_circle_object(int(x), int(y), int(radius), area, confidence)
                            obj['circularity'] = round(circularity, 2)
                            obj['aspect_ratio'] = round(aspect_ratio, 2)
                            objects.append(obj)
        
        return objects
    
    def draw_objects(self, frame: np.ndarray, objects: List[Dict], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw detected counters on frame with bounding box and info overlay.

        Args:
            frame: Input frame
            objects: List of detected counter objects
            color: Color for annotations (default: green)

        Returns:
            Annotated frame with visual overlays
        """
        annotated = frame.copy()

        for obj in objects:
            x, y = obj['x'], obj['y']
            w, h = obj['width'], obj['height']
            center = obj['center']

            # Draw bounding box (rectangle)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw center point (small dot)
            cv2.circle(annotated, center, 5, color, -1)

            # Draw label with counter number
            counter_number = obj.get('counterNumber')
            confidence = obj.get('confidence', 0)
            
            if counter_number:
                label = f"Counter {counter_number}"
                if confidence > 0:
                    label += f" ({confidence*100:.0f}%)"
            else:
                # Fallback if no counter number assigned
                circularity = obj.get('circularity', 0)
                label = f"Counter ({confidence*100:.0f}%, C:{circularity:.2f})"

            # Draw label background for better visibility
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, y - text_height - 10), (x + text_width + 4, y), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x + 2, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated
    
    def detect_defects(self, frame: np.ndarray, method: str = 'blob', params: Optional[Dict] = None) -> Dict:
        """
        Detect defects in an image frame
        
        Note: Defect detection is currently disabled - focus on object/counter detection first
        """
        return {
            'defects_found': False,
            'defect_count': 0,
            'defects': [],
            'confidence': 0.0,
            'method': method,
            'timestamp': time.time(),
            'note': 'Defect detection is currently disabled'
        }
    def _detect_blobs(self, gray: np.ndarray, min_area: int = 10, max_area: int = 5000,
                     adaptive_block: int = 11, adaptive_c: int = 2, kernel_size: int = 3) -> List[Dict]:
        """Stub - defect detection disabled"""
        return []
    
    def _detect_contours(self, gray: np.ndarray, min_area: int = 50, max_area: int = 10000,
                        canny_low: int = 50, canny_high: int = 150, aspect_ratio_min: float = 0.2,
                        aspect_ratio_max: float = 5.0, dilation_iterations: int = 1, kernel_size: int = 3) -> List[Dict]:
        """Stub - defect detection disabled"""
        return []
    
    def _detect_edges(self, gray: np.ndarray, canny_low: int = 30, canny_high: int = 100,
                     hough_threshold: int = 50, min_line_length: int = 30, max_line_gap: int = 10,
                     line_grouping_distance: int = 30, min_lines_per_defect: int = 2, min_defect_size: int = 10) -> List[Dict]:
        """Stub - defect detection disabled"""
        return []
    
    def _merge_nearby_defects(self, defects: List[Dict], threshold: int = 20) -> List[Dict]:
        """Stub - defect detection disabled"""
        return defects if defects else []
    
    def _calculate_confidence(self, defects: List[Dict], frame_shape: Tuple[int, int, int]) -> float:
        """Stub - defect detection disabled, always returns 0.0"""
        return 0.0
    
    def draw_defects(self, frame: np.ndarray, defects: List[Dict]) -> np.ndarray:
        """
        Stub - defect detection disabled, returns frame unchanged
        """
        # Defect detection is disabled, just return the frame as-is
        return frame.copy()

