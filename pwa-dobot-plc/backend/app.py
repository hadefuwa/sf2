"""
PWA Dobot-PLC Control Backend
Flask API with WebSocket support for real-time PLC monitoring
"""

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import logging
import os
import time
import threading
import json
import subprocess
import sys
import cv2
import numpy as np
import requests
import base64
from typing import Dict, List, Optional
from datetime import datetime
from plc_client import PLCClient
from dobot_client import DobotClient
from camera_service import CameraService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory for saving counter images
COUNTER_IMAGES_DIR = os.path.expanduser('~/counter_images')
COUNTER_POSITIONS_FILE = os.path.join(COUNTER_IMAGES_DIR, 'counter_positions.json')
COUNTER_DEFECTS_FILE = os.path.join(COUNTER_IMAGES_DIR, 'counter_defects.json')

# Track last save time for each counter (to enforce 15-second interval)
counter_last_save_time = {}  # counter_number -> timestamp

# Create directory if it doesn't exist
os.makedirs(COUNTER_IMAGES_DIR, exist_ok=True)

def cleanup_all_counter_images():
    """Delete all counter images - only call this when 16 counters have been detected"""
    try:
        if os.path.exists(COUNTER_IMAGES_DIR):
            deleted_count = 0
            for filename in os.listdir(COUNTER_IMAGES_DIR):
                if filename.startswith('counter_') and filename.endswith('.jpg'):
                    filepath = os.path.join(COUNTER_IMAGES_DIR, filename)
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {filename}: {e}")
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} counter images (16 counters detected)")
    except Exception as e:
        logger.error(f"Error cleaning up counter images: {e}")

def count_existing_counter_images() -> int:
    """Count how many unique counter images exist"""
    try:
        if not os.path.exists(COUNTER_IMAGES_DIR):
            return 0
        counter_numbers = set()
        for filename in os.listdir(COUNTER_IMAGES_DIR):
            if filename.startswith('counter_') and filename.endswith('.jpg'):
                parts = filename.split('_')
                if len(parts) >= 2:
                    try:
                        counter_numbers.add(int(parts[1]))
                    except ValueError:
                        pass
        return len(counter_numbers)
    except Exception as e:
        logger.error(f"Error counting counter images: {e}")
        return 0

# Delete all counter images on startup for a fresh start
existing_counter_count = count_existing_counter_images()
if existing_counter_count > 0:
    logger.info(f"Found {existing_counter_count} counter images - cleaning up on startup")
    cleanup_all_counter_images()
else:
    logger.info("No counter images found - starting fresh")

# Also clean up counter positions file on startup
try:
    if os.path.exists(COUNTER_POSITIONS_FILE):
        os.remove(COUNTER_POSITIONS_FILE)
        logger.info("Cleaned up counter positions file on startup")
except Exception as e:
    logger.warning(f"Error cleaning up counter positions file: {e}")

logger.info(f"Counter images will be saved to: {COUNTER_IMAGES_DIR}")

# Global counter tracking - tracks the highest counter number ever assigned
# This ensures counters keep their numbers even when they move off-screen
_counter_tracker = {'max_counter_number': 0}

# Reset counter tracker to start fresh (after definition)
_counter_tracker['max_counter_number'] = 0
logger.info("Counter tracker reset to 0 on startup")

def get_next_counter_number() -> int:
    """Get the next available counter number, incrementing sequentially"""
    # Simply increment from the tracker (don't check existing images to avoid jumps)
    _counter_tracker['max_counter_number'] += 1
    
    # Safety cap: reset if somehow we exceed 20 (shouldn't happen with proper cleanup)
    if _counter_tracker['max_counter_number'] > 20:
        logger.warning(f"Counter tracker exceeded 20 ({_counter_tracker['max_counter_number']}), resetting to 0")
        _counter_tracker['max_counter_number'] = 0
        _counter_tracker['max_counter_number'] += 1
    
    return _counter_tracker['max_counter_number']

def get_max_counter_number() -> int:
    """Get the maximum counter number that has been assigned"""
    return _counter_tracker['max_counter_number']

def initialize_counter_tracker():
    """Initialize counter tracker - start fresh since images are deleted on startup"""
    # Don't check existing images - start from 0 since we delete images on startup
    # This ensures sequential numbering: 1, 2, 3, 4, etc.
    _counter_tracker['max_counter_number'] = 0
    logger.info("Initialized counter tracker: starting from 0 (images deleted on startup)")

# Initialize counter tracker on startup
initialize_counter_tracker()

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend')
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize clients
plc_client = None  # Will be None if snap7 fails
dobot_client = None
camera_service = None

# Vision service configuration
VISION_SERVICE_URL = os.getenv('VISION_SERVICE_URL', 'http://127.0.0.1:5001')
VISION_SERVICE_TIMEOUT = 5.0  # 5 second timeout

# Polling state
poll_thread = None
poll_running = False
poll_interval = 0.1  # 100ms

# Vision handshaking state
vision_handshake_processing = False
vision_handshake_last_start_state = False

def call_vision_service(frame: np.ndarray, params: Dict) -> Dict:
    """
    Call the vision service for YOLO detection
    
    Args:
        frame: Image frame (BGR format)
        params: Detection parameters
    
    Returns:
        Detection results dictionary
    """
    try:
        # Encode frame as JPEG then base64
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        if not ret:
            return {
                'objects_found': False,
                'object_count': 0,
                'objects': [],
                'error': 'Failed to encode frame'
            }
        
        frame_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        # Call vision service
        response = requests.post(
            f"{VISION_SERVICE_URL}/detect",
            json={
                'frame_base64': frame_base64,
                'params': params
            },
            timeout=VISION_SERVICE_TIMEOUT
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
                # Validate result structure
                if not isinstance(result, dict):
                    raise ValueError("Vision service returned invalid response format")
                return result
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing vision service response: {e}")
                return {
                    'objects_found': False,
                    'object_count': 0,
                    'objects': [],
                    'error': 'Invalid response from vision service'
                }
        else:
            logger.error(f"Vision service returned error: {response.status_code} - {response.text}")
            return {
                'objects_found': False,
                'object_count': 0,
                'objects': [],
                'error': f'Vision service error: {response.status_code}'
            }
            
    except requests.exceptions.Timeout:
        logger.warning("Vision service timeout - service may be down or overloaded")
        return {
            'objects_found': False,
            'object_count': 0,
            'objects': [],
            'error': 'Vision service timeout'
        }
    except requests.exceptions.ConnectionError:
        logger.warning("Vision service connection error - service may be down")
        return {
            'objects_found': False,
            'object_count': 0,
            'objects': [],
            'error': 'Vision service unavailable'
        }
    except Exception as e:
        logger.error(f"Error calling vision service: {e}", exc_info=True)
        return {
            'objects_found': False,
            'object_count': 0,
            'objects': [],
            'error': f'Vision service error: {str(e)}'
        }

def load_config():
    """Load configuration from config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return defaults if config doesn't exist
        return {
            "dobot": {
                "usb_path": "/dev/ttyACM0",
                "home_position": {"x": 200.0, "y": 0.0, "z": 150.0, "r": 0.0},
                "use_usb": True
            },
            "plc": {
                "ip": "192.168.7.2",
                "rack": 0,
                "slot": 1,
                "db_number": 123,
                "poll_interval": 2.0
            },
            "server": {"port": 8080}
        }

def delete_old_counter_images(counter_number: int):
    """
    Delete all old images for a specific counter number, keeping only the most recent one
    
    Args:
        counter_number: Counter number to clean up
    """
    try:
        if not os.path.exists(COUNTER_IMAGES_DIR):
            return
        
        # Find all images for this counter
        prefix = f"counter_{counter_number}_"
        counter_images = []
        
        for filename in os.listdir(COUNTER_IMAGES_DIR):
            if filename.startswith(prefix) and filename.endswith('.jpg'):
                filepath = os.path.join(COUNTER_IMAGES_DIR, filename)
                stat = os.stat(filepath)
                counter_images.append((filepath, stat.st_mtime, filename))
        
        # Sort by modification time (most recent first)
        counter_images.sort(key=lambda x: x[1], reverse=True)
        
        # Delete all except the most recent one
        if len(counter_images) > 1:
            for filepath, _, filename in counter_images[1:]:  # Skip first (most recent)
                try:
                    os.remove(filepath)
                    logger.debug(f"Deleted old counter {counter_number} image: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to delete {filename}: {e}")
    
    except Exception as e:
        logger.error(f"Error deleting old counter images: {e}", exc_info=True)

def find_most_central_counter(detected_objects: List[Dict], frame_shape: tuple, 
                              selection_method: str = 'most_central') -> Optional[Dict]:
    """
    Find a single counter from multiple detected objects based on selection method
    
    Args:
        detected_objects: List of detected counter objects
        frame_shape: Tuple of (height, width) of the frame
        selection_method: Method to select counter - 'most_central', 'largest', 'smallest', 
                         'leftmost', 'rightmost', 'topmost', 'bottommost'
    
    Returns:
        The selected counter object, or None if no objects detected
    """
    if not detected_objects:
        return None
    
    if len(detected_objects) == 1:
        return detected_objects[0]
    
    frame_height, frame_width = frame_shape[:2]
    image_center_x = frame_width // 2
    image_center_y = frame_height // 2
    
    best_counter = None
    
    if selection_method == 'most_central':
        # Find counter closest to image center
        min_distance = float('inf')
        for obj in detected_objects:
            obj_center = obj.get('center')
            if obj_center:
                center_x, center_y = obj_center
            else:
                center_x = obj.get('x', 0) + obj.get('width', 0) // 2
                center_y = obj.get('y', 0) + obj.get('height', 0) // 2
            
            distance = ((center_x - image_center_x) ** 2 + (center_y - image_center_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                best_counter = obj
    
    elif selection_method == 'largest':
        # Find counter with largest area
        max_area = 0
        for obj in detected_objects:
            area = obj.get('area', 0)
            if area > max_area:
                max_area = area
                best_counter = obj
    
    elif selection_method == 'smallest':
        # Find counter with smallest area
        min_area = float('inf')
        for obj in detected_objects:
            area = obj.get('area', 0)
            if area < min_area:
                min_area = area
                best_counter = obj
    
    elif selection_method == 'leftmost':
        # Find counter with leftmost X position
        min_x = float('inf')
        for obj in detected_objects:
            x = obj.get('x', 0)
            if x < min_x:
                min_x = x
                best_counter = obj
    
    elif selection_method == 'rightmost':
        # Find counter with rightmost X position
        max_x = -1
        for obj in detected_objects:
            x = obj.get('x', 0) + obj.get('width', 0)
            if x > max_x:
                max_x = x
                best_counter = obj
    
    elif selection_method == 'topmost':
        # Find counter with topmost Y position
        min_y = float('inf')
        for obj in detected_objects:
            y = obj.get('y', 0)
            if y < min_y:
                min_y = y
                best_counter = obj
    
    elif selection_method == 'bottommost':
        # Find counter with bottommost Y position
        max_y = -1
        for obj in detected_objects:
            y = obj.get('y', 0) + obj.get('height', 0)
            if y > max_y:
                max_y = y
                best_counter = obj
    
    else:
        # Default to most_central if unknown method
        logger.warning(f"Unknown selection method: {selection_method}, using 'most_central'")
        return find_most_central_counter(detected_objects, frame_shape, 'most_central')
    
    return best_counter

def find_matching_counter(obj: Dict, existing_counters: Dict[int, Dict]) -> int:
    """
    Try to match a detected object to an existing counter based on position similarity
    
    Args:
        obj: Detected object with x, y, center coordinates
        existing_counters: Dictionary mapping counter_number -> {x, y, center}
    
    Returns:
        Matching counter number if found, or None
    """
    if not existing_counters:
        return None
    
    obj_center = obj.get('center', (obj.get('x', 0) + obj.get('width', 0) // 2, 
                                    obj.get('y', 0) + obj.get('height', 0) // 2))
    obj_x, obj_y = obj_center
    
    # Position matching threshold (pixels) - counters within this distance are considered the same
    # Load from config if available
    config = load_config()
    vision_config = config.get('vision', {})
    POSITION_THRESHOLD = vision_config.get('position_matching_threshold', 100)  # Default 100 pixels tolerance
    
    best_match = None
    best_distance = float('inf')
    
    for counter_num, counter_info in existing_counters.items():
        counter_center = counter_info.get('center', (counter_info.get('x', 0), counter_info.get('y', 0)))
        counter_x, counter_y = counter_center
        
        # Calculate distance between centers
        distance = ((obj_x - counter_x) ** 2 + (obj_y - counter_y) ** 2) ** 0.5
        
        if distance < POSITION_THRESHOLD and distance < best_distance:
            best_match = counter_num
            best_distance = distance
    
    return best_match

def load_existing_counter_positions() -> Dict[int, Dict]:
    """
    Load positions of existing counters from JSON file
    Stores counter_number -> {x, y, center, last_seen_timestamp}
    """
    existing = {}
    try:
        if os.path.exists(COUNTER_POSITIONS_FILE):
            with open(COUNTER_POSITIONS_FILE, 'r') as f:
                existing = json.load(f)
                # Convert keys back to int
                existing = {int(k): v for k, v in existing.items()}
    except Exception as e:
        logger.warning(f"Error loading counter positions: {e}")
    
    # Also check for counters that have images but no position data
    if os.path.exists(COUNTER_IMAGES_DIR):
        for filename in os.listdir(COUNTER_IMAGES_DIR):
            if filename.startswith('counter_') and filename.endswith('.jpg'):
                parts = filename.split('_')
                if len(parts) >= 2:
                    try:
                        counter_num = int(parts[1])
                        if counter_num not in existing:
                            existing[counter_num] = {'has_image': True}
                    except ValueError:
                        pass
    return existing

def save_counter_positions(counter_positions: Dict[int, Dict]):
    """Save counter positions to JSON file"""
    try:
        os.makedirs(COUNTER_IMAGES_DIR, exist_ok=True)
        with open(COUNTER_POSITIONS_FILE, 'w') as f:
            json.dump(counter_positions, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving counter positions: {e}")

def load_counter_defect_results() -> Dict[str, Dict]:
    """Load stored defect detection results for counters"""
    try:
        if os.path.exists(COUNTER_DEFECTS_FILE):
            with open(COUNTER_DEFECTS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading counter defect results: {e}")
    return {}

def save_counter_defect_results(results: Dict[str, Dict]):
    """Persist defect detection results"""
    try:
        with open(COUNTER_DEFECTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving counter defect results: {e}")

def record_counter_defect_result(counter_number: int, image_path: str, defect_results: Dict):
    """Store defect detection results for a counter"""
    results = load_counter_defect_results()
    results[str(counter_number)] = {
        'counter_number': counter_number,
        'image_path': image_path,
        'timestamp': time.time(),
        'defect_results': defect_results
    }
    save_counter_defect_results(results)

def counter_image_exists(counter_number: int) -> bool:
    """
    Check if an image already exists for a counter number
    
    Args:
        counter_number: Counter number to check
    
    Returns:
        True if image exists, False otherwise
    """
    try:
        prefix = f"counter_{counter_number}_"
        if os.path.exists(COUNTER_IMAGES_DIR):
            for filename in os.listdir(COUNTER_IMAGES_DIR):
                if filename.startswith(prefix) and filename.endswith('.jpg'):
                    return True
        return False
    except Exception as e:
        logger.error(f"Error checking if counter image exists: {e}")
        return False

def save_counter_image(frame: np.ndarray, obj: Dict, counter_number: int, timestamp: float) -> str:
    """
    Crop and save a detected counter image with timestamp
    Only saves if 15 seconds have passed since last save for this counter
    Deletes the previous image for this counter before saving the new one
    
    Args:
        frame: Original camera frame
        obj: Detected object dictionary with x, y, width, height
        counter_number: Counter number (1, 2, 3, etc.)
        timestamp: Detection timestamp
    
    Returns:
        Path to saved image file, or None if failed or too soon since last save
    """
    try:
        # Check if 15 seconds have passed since last save for this counter
        SAVE_INTERVAL_SECONDS = 15
        last_save_time = counter_last_save_time.get(counter_number, 0)
        time_since_last_save = timestamp - last_save_time
        
        if time_since_last_save < SAVE_INTERVAL_SECONDS:
            # Too soon, skip saving
            logger.debug(f"Counter {counter_number}: Only {time_since_last_save:.1f}s since last save, skipping (need {SAVE_INTERVAL_SECONDS}s)")
            return None
        
        # Delete previous image(s) for this counter
        prefix = f"counter_{counter_number}_"
        if os.path.exists(COUNTER_IMAGES_DIR):
            deleted_count = 0
            for filename in os.listdir(COUNTER_IMAGES_DIR):
                if filename.startswith(prefix) and filename.endswith('.jpg'):
                    filepath = os.path.join(COUNTER_IMAGES_DIR, filename)
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                        logger.debug(f"Deleted previous counter {counter_number} image: {filename}")
                    except Exception as e:
                        logger.warning(f"Error deleting previous image {filename}: {e}")
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} previous image(s) for counter {counter_number}")
        
        # Get bounding box coordinates
        x = obj.get('x', 0)
        y = obj.get('y', 0)
        w = obj.get('width', 0)
        h = obj.get('height', 0)
        
        # Add minimal padding around the counter (reduced from 20 to 5 for tighter crop)
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Crop the image
        cropped = frame[y1:y2, x1:x2]
        
        if cropped.size == 0:
            logger.warning(f"Empty crop for counter {counter_number}")
            return None
        
        # Create filename with timestamp
        dt = datetime.fromtimestamp(timestamp)
        filename = f"counter_{counter_number}_{dt.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
        filepath = os.path.join(COUNTER_IMAGES_DIR, filename)
        
        # Save the cropped image
        cv2.imwrite(filepath, cropped)
        logger.info(f"Saved counter {counter_number} image: {filename} (after {time_since_last_save:.1f}s)")

        # Update last save time
        counter_last_save_time[counter_number] = timestamp

        # Automatically analyze the saved counter image for defects
        auto_analyze_counter_image(counter_number, filepath)

        return filepath
    except Exception as e:
        logger.error(f"Error saving counter image: {e}", exc_info=True)
        return None

def auto_analyze_counter_image(counter_number: int, image_path: str):
    """Automatically analyze a saved counter image for defects and store the result"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Auto defect analysis skipped for Counter {counter_number} - could not read image")
            return
        defect_results = detect_color_defects(image)
        record_counter_defect_result(counter_number, image_path, defect_results)
        logger.info(f"Auto defect analysis completed for Counter {counter_number}")
    except Exception as e:
        logger.error(f"Error auto-analyzing counter {counter_number}: {e}", exc_info=True)

def save_config(config):
    """Save configuration to config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def write_vision_to_plc(object_count: int, defect_count: int, object_ok: bool, defect_detected: bool, 
                       busy: bool = False, completed: bool = False):
    """Write vision detection results to PLC DB123 tags
    
    This is a wrapper function that uses the unified PLC client methods.
    All S7 communication is handled in plc_client.py.
    
    Args:
        object_count: Number of objects detected
        defect_count: Number of defects found
        object_ok: Whether objects are OK (no defects)
        defect_detected: Whether any defects were detected
        busy: Whether vision system is currently processing (default: False)
        completed: Whether vision processing is completed (default: False)
    """
    if plc_client is None:
        return False
    
    try:
        config = load_config()
        db123_config = config.get('plc', {}).get('db123', {})
        
        # Check if DB123 communication is enabled
        if not db123_config.get('enabled', False):
            return False
        
        db_number = db123_config.get('db_number', 123)
        
        # Use unified PLC client method for all S7 communication
        return plc_client.write_vision_detection_results(
            object_count=object_count,
            defect_count=defect_count,
            object_ok=object_ok,
            defect_detected=defect_detected,
            busy=busy,
            completed=completed,
            db_number=db_number
        )
    except Exception as e:
        logger.error(f"Error writing vision tags to PLC: {e}")
        return False

def init_clients():
    """Initialize PLC and Dobot clients from config"""
    global plc_client, dobot_client, camera_service

    config = load_config()

    # PLC settings - only create if snap7 is available (gracefully handle if not)
    plc_config = config['plc']
    try:
        plc_client = PLCClient(
            plc_config['ip'],
            plc_config['rack'],
            plc_config['slot']
        )
        # Check if snap7 client was actually created
        if plc_client.client is None:
            logger.warning("PLC client created but snap7 not available - PLC features disabled")
    except Exception as e:
        logger.error(f"Failed to initialize PLC client: {e} - PLC features will be disabled")
        plc_client = None

    # Dobot settings
    dobot_config = config['dobot']
    dobot_client = DobotClient(
        use_usb=dobot_config.get('use_usb', True),
        usb_path=dobot_config.get('usb_path', '/dev/ttyACM0')
    )
    
    # Update home position if specified
    if 'home_position' in dobot_config:
        dobot_client.HOME_POSITION = dobot_config['home_position']

    # Camera settings
    camera_config = config.get('camera', {})
    camera_service = CameraService(
        camera_index=camera_config.get('index', 0),
        width=camera_config.get('width', 640),
        height=camera_config.get('height', 480)
    )
    # Load crop settings if available
    crop_config = camera_config.get('crop', {})
    if crop_config:
        camera_service.set_crop(
            enabled=crop_config.get('enabled', False),
            x=crop_config.get('x', 0),
            y=crop_config.get('y', 0),
            width=crop_config.get('width', 100),
            height=crop_config.get('height', 100)
        )
    # Initialize camera and keep it always active
    try:
        success = camera_service.initialize_camera()
        if success:
            logger.info("📷 Camera initialized and will stay always active")
        else:
            logger.warning("📷 Camera initialization failed - will retry automatically")
            # Retry in background
            def retry_camera_init():
                while True:
                    time.sleep(5)  # Retry every 5 seconds
                    if camera_service is not None:
                        try:
                            if camera_service.camera is None or (camera_service.camera is not None and not camera_service.camera.isOpened()):
                                success = camera_service.initialize_camera()
                                if success:
                                    logger.info("📷 Camera initialized successfully (retry)")
                                    break
                        except Exception:
                            pass
            threading.Thread(target=retry_camera_init, daemon=True).start()
    except Exception as e:
        logger.warning(f"Camera initialization failed (may not be connected): {e}")
        # Retry in background
        def retry_camera_init():
            while True:
                time.sleep(5)
                if camera_service is not None:
                    try:
                        if camera_service.camera is None or (camera_service.camera is not None and not camera_service.camera.isOpened()):
                            camera_service.initialize_camera()
                    except Exception:
                        pass
        threading.Thread(target=retry_camera_init, daemon=True).start()

    # YOLO model is now loaded in the separate vision-service process
    # No need to load it here - all YOLO calls go through vision service
    logger.info("YOLO detection will be handled by vision-service (separate process)")

    logger.info(f"Clients initialized - PLC: {plc_config['ip']}, Dobot USB: {dobot_config.get('usb_path', 'auto-detect')}")

# ==================================================
# REST API Endpoints
# ==================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time()
    })

@app.route('/api/data', methods=['GET'])
def get_all_data():
    """Get all data in a single request to minimize PLC load"""
    # Default values - don't try to connect to PLC
    target_pose = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    control_bits = {}
    plc_ip = 'unknown'
    if plc_client and hasattr(plc_client, 'ip'):
        plc_ip = plc_client.ip
    plc_status = {'connected': False, 'ip': plc_ip, 'last_error': 'PLC not available'}
    
    # Only try PLC operations if snap7 is available and client exists
    if plc_client and hasattr(plc_client, 'client') and plc_client.client is not None:
        try:
            plc_status = plc_client.get_status()
            # Only read if already connected - don't try to connect
            if plc_status.get('connected', False):
                try:
                    # Read from configured DB number
                    config = load_config()
                    db_number = config.get('plc', {}).get('db_number', 123)
                    target_pose = plc_client.read_target_pose(db_number)
                    time.sleep(0.15)  # 150ms delay to avoid job pending with S7-1200
                    control_bits = plc_client.read_control_bits()
                except Exception as e:
                    logger.debug(f"PLC read error: {e}")
                    target_pose = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                    control_bits = {}
        except Exception as e:
            logger.debug(f"PLC status check failed: {e}")
            plc_ip = 'unknown'
            if plc_client and hasattr(plc_client, 'ip'):
                plc_ip = plc_client.ip
            plc_status = {'connected': False, 'ip': plc_ip, 'last_error': str(e)}

    # Get Dobot data
    dobot_status_data = {
        'connected': dobot_client.connected,
        'last_error': dobot_client.last_error
    }
    dobot_pose = dobot_client.get_pose() if dobot_client.connected else {'x': 0.0, 'y': 0.0, 'z': 0.0, 'r': 0.0}

    return jsonify({
        'plc': {
            'status': plc_status,
            'pose': target_pose,
            'control': control_bits
        },
        'dobot': {
            'status': dobot_status_data,
            'pose': dobot_pose
        }
    })

@app.route('/api/plc/status', methods=['GET'])
def plc_status():
    """Get PLC connection status"""
    try:
        if plc_client is None:
            return jsonify({'connected': False, 'ip': 'unknown', 'last_error': 'PLC client not initialized'})
        status = plc_client.get_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in plc_status endpoint: {e}")
        return jsonify({'connected': False, 'ip': 'unknown', 'last_error': str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint to verify backend is responding"""
    return jsonify({
        'success': True,
        'message': 'Backend is responding',
        'timestamp': time.time()
    })

@app.route('/api/plc/connect', methods=['POST'])
def plc_connect():
    """Connect to PLC"""
    if plc_client is None:
        return jsonify({
            'success': False,
            'connected': False,
            'error': 'PLC client not initialized'
        })
    success = plc_client.connect()
    return jsonify({
        'success': success,
        'connected': plc_client.is_connected(),
        'error': plc_client.last_error if not success else None
    })

@app.route('/api/plc/disconnect', methods=['POST'])
def plc_disconnect():
    """Disconnect from PLC"""
    if plc_client is not None:
        plc_client.disconnect()
    return jsonify({'success': True})

@app.route('/api/plc/pose', methods=['GET'])
def get_plc_pose():
    """Get target pose from PLC"""
    # Don't try to connect - just return default if not connected
    try:
        if plc_client and hasattr(plc_client, 'client') and plc_client.client is not None:
            if plc_client.is_connected():
                config = load_config()
                db_number = config.get('plc', {}).get('db_number', 123)
                pose = plc_client.read_target_pose(db_number)
                return jsonify(pose)
        return jsonify({'x': 0.0, 'y': 0.0, 'z': 0.0})
    except Exception as e:
        logger.debug(f"PLC pose read error: {e}")
        return jsonify({'x': 0.0, 'y': 0.0, 'z': 0.0})

@app.route('/api/plc/pose', methods=['POST'])
def set_plc_pose():
    """Write current pose to PLC"""
    try:
        data = request.json
        if not all(k in data for k in ['x', 'y', 'z']):
            return jsonify({'error': 'Missing x, y, or z'}), 400

        # Don't try to connect - only write if already connected
        if plc_client and hasattr(plc_client, 'client') and plc_client.client is not None:
            if plc_client.is_connected():
                config = load_config()
                db_number = config.get('plc', {}).get('db_number', 123)
                success = plc_client.write_current_pose(data, db_number)
                return jsonify({'success': success})
        return jsonify({'success': False, 'error': 'PLC not available'})
    except Exception as e:
        logger.debug(f"PLC pose write error: {e}")
        return jsonify({'success': False, 'error': 'PLC not available'})

@app.route('/api/plc/control', methods=['GET'])
def get_control_bits():
    """Get all control bits"""
    # Default values - don't try to connect
    default_bits = {
        'start': False, 'stop': False, 'home': False, 'estop': False,
        'suction': False, 'ready': False, 'busy': False, 'error': False
    }
    try:
        if plc_client and hasattr(plc_client, 'client') and plc_client.client is not None:
            if plc_client.is_connected():
                bits = plc_client.read_control_bits()
                return jsonify(bits)
        return jsonify(default_bits)
    except Exception as e:
        logger.debug(f"PLC control bits read error: {e}")
        return jsonify(default_bits)

@app.route('/api/plc/control/<bit_name>', methods=['POST'])
def set_control_bit(bit_name):
    """Set a single control bit"""
    try:
        data = request.json
        value = data.get('value', False)

        # Don't try to connect - only write if already connected
        if plc_client and hasattr(plc_client, 'client') and plc_client.client is not None:
            if plc_client.is_connected():
                success = plc_client.write_control_bit(bit_name, value)
                return jsonify({'success': success})
        return jsonify({'success': False, 'error': 'PLC not available'})
    except Exception as e:
        logger.debug(f"PLC control bit write error: {e}")
        return jsonify({'success': False, 'error': 'PLC not available'})

@app.route('/api/dobot/status', methods=['GET'])
def dobot_status():
    """Get Dobot connection status"""
    return jsonify({
        'connected': dobot_client.connected,
        'last_error': dobot_client.last_error
    })

@app.route('/api/dobot/debug', methods=['GET'])
def dobot_debug():
    """Get detailed Dobot debug information"""
    import os
    import glob
    
    # Get available USB ports
    available_ports = dobot_client.find_dobot_ports()
    
    # Check if pydobot is available
    try:
        from pydobot import Dobot as PyDobot
        pydobot_available = True
    except ImportError:
        pydobot_available = False
    
    # Check port permissions
    port_info = []
    for port in available_ports:
        try:
            import stat
            port_stat = os.stat(port)
            permissions = oct(port_stat.st_mode)[-3:]
            port_info.append({
                'port': port,
                'exists': True,
                'permissions': permissions,
                'readable': bool(port_stat.st_mode & stat.S_IRUSR),
                'writable': bool(port_stat.st_mode & stat.S_IWUSR)
            })
        except Exception as e:
            port_info.append({
                'port': port,
                'exists': False,
                'error': str(e)
            })
    
    return jsonify({
        'pydobot_available': pydobot_available,
        'use_usb': dobot_client.use_usb,
        'configured_port': dobot_client.usb_path,
        'actual_port': dobot_client.actual_port,
        'connected': dobot_client.connected,
        'last_error': dobot_client.last_error,
        'available_ports': available_ports,
        'port_details': port_info
    })

@app.route('/api/dobot/connect', methods=['POST'])
def dobot_connect():
    """Connect to Dobot"""
    logger.info("🔌 Manual Dobot connection requested")
    success = dobot_client.connect()
    if success:
        logger.info("✅ Manual Dobot connection successful")
    else:
        logger.error(f"❌ Manual Dobot connection failed: {dobot_client.last_error}")
    return jsonify({
        'success': success,
        'connected': dobot_client.connected,
        'error': dobot_client.last_error if not success else None
    })

@app.route('/api/dobot/home', methods=['POST'])
def dobot_home():
    """Home Dobot robot"""
    if not dobot_client.connected:
        return jsonify({'error': 'Dobot not connected'}), 503

    logger.info("🏠 Home command received from web interface")
    success = dobot_client.home(wait=True)  # Wait=True for immediate execution
    logger.info(f"✅ Home command result: {success}")
    return jsonify({'success': success})

@app.route('/api/dobot/move', methods=['POST'])
def dobot_move():
    """Move Dobot to position"""
    data = request.json
    if not all(k in data for k in ['x', 'y', 'z']):
        return jsonify({'error': 'Missing x, y, or z'}), 400

    if not dobot_client.connected:
        return jsonify({'error': 'Dobot not connected'}), 503

    # Get position before move
    pos_before = dobot_client.get_pose()
    logger.info(f"▶️ Move command: ({data['x']}, {data['y']}, {data['z']}, {data.get('r', 0)}) - Current: ({pos_before['x']:.1f}, {pos_before['y']:.1f}, {pos_before['z']:.1f})")

    success = dobot_client.move_to(
        data['x'],
        data['y'],
        data['z'],
        data.get('r', 0),
        wait=True  # Wait=True for immediate execution
    )

    if success:
        # Verify robot actually moved
        time.sleep(0.3)  # Brief delay to ensure movement settled
        pos_after = dobot_client.get_pose()

        # Calculate distance moved
        distance = ((pos_after['x'] - pos_before['x'])**2 +
                   (pos_after['y'] - pos_before['y'])**2 +
                   (pos_after['z'] - pos_before['z'])**2)**0.5

        if distance > 1.0:  # Moved more than 1mm
            logger.info(f"✅ ACTUAL MOVEMENT: Moved {distance:.1f}mm to ({pos_after['x']:.1f}, {pos_after['y']:.1f}, {pos_after['z']:.1f})")
            return jsonify({'success': True, 'executed': True, 'distance_moved': round(distance, 1)})
        else:
            logger.error(f"⚠️ ROBOT DID NOT MOVE! Distance: {distance:.1f}mm - Position: ({pos_after['x']:.1f}, {pos_after['y']:.1f}, {pos_after['z']:.1f})")
            return jsonify({'success': False, 'error': f'Robot did not move (only {distance:.1f}mm)', 'distance_moved': round(distance, 1)}), 500
    else:
        error_msg = dobot_client.last_error or 'Movement failed'
        logger.error(f"❌ Move command failed: {error_msg}")
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/api/dobot/pose', methods=['GET'])
def get_dobot_pose():
    """Get current Dobot pose"""
    if not dobot_client.connected:
        return jsonify({'error': 'Dobot not connected'}), 503

    pose = dobot_client.get_pose()
    return jsonify(pose)

@app.route('/api/dobot/suction', methods=['POST'])
def dobot_suction():
    """Control suction cup"""
    if not dobot_client.connected:
        return jsonify({'error': 'Dobot not connected'}), 503

    data = request.json
    enable = data.get('enable', False)
    
    try:
        logger.info(f"💨 Suction cup: {'ON' if enable else 'OFF'}")
        dobot_client.set_suction(enable)
        return jsonify({'success': True, 'enabled': enable})
    except Exception as e:
        logger.error(f"❌ Suction control failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dobot/gripper', methods=['POST'])
def dobot_gripper():
    """Control gripper (if available)"""
    if not dobot_client.connected:
        return jsonify({'error': 'Dobot not connected'}), 503

    data = request.json
    open_gripper = data.get('open', True)
    
    try:
        # Check if gripper control method exists
        if hasattr(dobot_client, 'set_gripper'):
            logger.info(f"✋ Gripper: {'OPEN' if open_gripper else 'CLOSE'}")
            dobot_client.set_gripper(open_gripper)
            return jsonify({'success': True, 'open': open_gripper})
        else:
            logger.warning("⚠️ Gripper not available on this Dobot model")
            return jsonify({
                'success': False,
                'message': 'Gripper not available. This Dobot model only has suction cup.'
            })
    except Exception as e:
        logger.error(f"❌ Gripper control failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/emergency-stop', methods=['POST'])
def emergency_stop():
    """Emergency stop - stop both Dobot and signal PLC"""
    logger.error("🛑 EMERGENCY STOP TRIGGERED")

    results = {}

    # Stop Dobot
    try:
        if dobot_client.connected:
            dobot_client.stop_queue()  # Stop queue execution first
            dobot_client.clear_queue()  # Then clear queued commands
            results['dobot'] = 'stopped'
    except Exception as e:
        logger.error(f"Dobot emergency stop error: {e}")
        results['dobot'] = 'error'

    # Signal PLC (gracefully handle if PLC is offline)
    try:
        if plc_client and hasattr(plc_client, 'client') and plc_client.client is not None:
            if plc_client.is_connected():
                plc_client.write_control_bit('estop', True)
                results['plc'] = 'signaled'
            else:
                results['plc'] = 'not_connected'
        else:
            results['plc'] = 'not_available'
    except Exception as e:
        logger.debug(f"PLC emergency stop error: {e}")
        results['plc'] = 'error'

    return jsonify({'success': True, **results})

@app.route('/api/dobot/test', methods=['POST'])
def dobot_test():
    """Run comprehensive Dobot test sequence"""
    if not dobot_client.connected:
        return jsonify({'error': 'Dobot not connected', 'steps': []}), 503

    results = []
    success = True

    try:
        # Step 1: Get current position
        logger.info("🧪 Test Step 1: Getting current position...")
        pos = dobot_client.get_pose()
        results.append({
            'step': 1,
            'name': 'Get Current Position',
            'success': True,
            'message': f"X: {pos['x']:.2f}, Y: {pos['y']:.2f}, Z: {pos['z']:.2f}, R: {pos['r']:.2f}"
        })
        time.sleep(0.5)

        # Step 2: Move to home position
        logger.info("🧪 Test Step 2: Moving to HOME position...")
        if dobot_client.home(wait=True):
            results.append({
                'step': 2,
                'name': 'Move to Home',
                'success': True,
                'message': f"Moved to ({dobot_client.HOME_POSITION['x']}, {dobot_client.HOME_POSITION['y']}, {dobot_client.HOME_POSITION['z']})"
            })
        else:
            results.append({'step': 2, 'name': 'Move to Home', 'success': False, 'message': 'Failed to move'})
            success = False
        time.sleep(1)

        # Step 3: Verify home position
        logger.info("🧪 Test Step 3: Verifying position...")
        pos = dobot_client.get_pose()
        results.append({
            'step': 3,
            'name': 'Verify Position',
            'success': True,
            'message': f"X: {pos['x']:.2f}, Y: {pos['y']:.2f}, Z: {pos['z']:.2f}"
        })
        time.sleep(0.5)

        # Step 4: Small movement test (20mm forward)
        logger.info("🧪 Test Step 4: Small movement test...")
        home = dobot_client.HOME_POSITION
        if dobot_client.move_to(home['x'] + 20, home['y'], home['z'], home['r'], wait=True):
            results.append({
                'step': 4,
                'name': 'Small Movement (forward 20mm)',
                'success': True,
                'message': 'Movement completed successfully'
            })
            time.sleep(1)
            
            # Move back
            logger.info("🧪 Test Step 4b: Moving back...")
            dobot_client.home(wait=True)
            time.sleep(0.5)
        else:
            results.append({'step': 4, 'name': 'Small Movement', 'success': False, 'message': 'Failed to move'})
            success = False

        # Step 5: Suction test
        logger.info("🧪 Test Step 5: Testing suction cup...")
        try:
            dobot_client.set_suction(True)
            time.sleep(2)
            dobot_client.set_suction(False)
            results.append({
                'step': 5,
                'name': 'Suction Cup Test',
                'success': True,
                'message': 'ON/OFF cycle completed'
            })
        except Exception as e:
            results.append({'step': 5, 'name': 'Suction Cup Test', 'success': False, 'message': str(e)})
            success = False

        logger.info("✅ Dobot test sequence completed!")
        return jsonify({
            'success': success,
            'steps': results,
            'message': 'All tests passed!' if success else 'Some tests failed'
        })

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return jsonify({
            'success': False,
            'steps': results,
            'error': str(e)
        }), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    try:
        config = load_config()
        # Ensure vision config exists
        if 'vision' not in config:
            config['vision'] = {
                'fault_bit_enabled': False,
                'fault_bit_byte': 1,
                'fault_bit_bit': 0
            }
        return jsonify(config)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration (for vision config and DB123)"""
    try:
        new_config = request.json
        current_config = load_config()
        
        # Update vision config if provided
        if 'vision' in new_config:
            current_config.setdefault('vision', {})
            current_config['vision'].update(new_config['vision'])
        
        # Update DB123 config if provided
        if 'plc' in new_config and 'db123' in new_config['plc']:
            current_config.setdefault('plc', {})
            current_config['plc'].setdefault('db123', {})
            current_config['plc']['db123'].update(new_config['plc']['db123'])
        
        save_config(current_config)
        return jsonify({'success': True, 'message': 'Configuration saved'})
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current configuration"""
    try:
        config = load_config()
        
        # Add available USB ports to the response
        available_ports = dobot_client.find_dobot_ports() if dobot_client else []
        config['available_usb_ports'] = available_ports
        
        return jsonify(config)
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update configuration"""
    try:
        new_config = request.json
        
        # Validate required fields
        if 'dobot' not in new_config or 'plc' not in new_config:
            return jsonify({'error': 'Missing required config sections'}), 400
        
        # Load current config and merge
        current_config = load_config()
        current_config['dobot'].update(new_config['dobot'])
        current_config['plc'].update(new_config['plc'])
        
        # Update vision config if provided
        if 'vision' in new_config:
            current_config.setdefault('vision', {})
            current_config['vision'].update(new_config['vision'])
        
        # Save to file
        save_config(current_config)
        
        logger.info("⚙️ Settings updated - restart required to apply changes")
        return jsonify({
            'success': True,
            'message': 'Settings saved. Restart server to apply changes.'
        })
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/restart', methods=['POST'])
def restart_server():
    """Restart the server"""
    try:
        logger.info("🔄 Server restart requested")
        
        # Try PM2 restart first (if running under PM2)
        try:
            result = subprocess.run(['pm2', 'restart', 'pwa-dobot-plc'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ PM2 restart successful")
                return jsonify({
                    'success': True,
                    'message': 'Server restarting via PM2...'
                })
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Fallback: try systemctl restart (if running as service)
        try:
            result = subprocess.run(['sudo', 'systemctl', 'restart', 'pwa-dobot-plc'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ Systemctl restart successful")
                return jsonify({
                    'success': True,
                    'message': 'Server restarting via systemctl...'
                })
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Last resort: exit the process (will be restarted by supervisor/PM2)
        logger.info("⚠️ No restart method available, exiting process")
        threading.Timer(2.0, lambda: sys.exit(0)).start()
        return jsonify({
            'success': True,
            'message': 'Server will restart in 2 seconds...'
        })
        
    except Exception as e:
        logger.error(f"Error restarting server: {e}")
        return jsonify({'error': str(e)}), 500

# ==================================================
# WebSocket Events
# ==================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connection_status', {'connected': True})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_polling')
def handle_start_polling():
    """Start real-time polling"""
    global poll_running
    if not poll_running:
        start_polling_thread()
    emit('polling_status', {'running': True})

@socketio.on('stop_polling')
def handle_stop_polling():
    """Stop real-time polling"""
    global poll_running
    poll_running = False
    emit('polling_status', {'running': False})

# ==================================================
# Background Polling Thread
# ==================================================

def start_polling_thread():
    """Start background polling thread"""
    global poll_thread, poll_running

    if poll_thread and poll_thread.is_alive():
        return

    poll_running = True
    poll_thread = threading.Thread(target=poll_loop, daemon=True)
    poll_thread.start()
    logger.info("Polling thread started")

def process_vision_handshake():
    """Process vision detection when Start command is received from PLC
    
    This function:
    1. Detects objects
    2. Saves images
    3. Checks for defects
    4. Writes results to PLC
    5. Sets Completed flag when done
    """
    global vision_handshake_processing
    
    if camera_service is None:
        logger.warning("Camera service not available for handshake processing")
        return False
    
    try:
        vision_handshake_processing = True
        logger.info("🔄 Vision handshake: Starting processing (Start command received)")
        logger.info("🔄 Vision handshake: Step 1 - Setting Busy flag")
        
        # Set busy flag
        write_vision_to_plc(0, 0, True, False, busy=True, completed=False)
        logger.info("🔄 Vision handshake: Step 2 - Busy flag set, reading frame")
        
        # Read current frame
        frame = camera_service.read_frame()
        if frame is None:
            logger.error("Vision handshake: Failed to read frame")
            write_vision_to_plc(0, 0, True, False, busy=False, completed=True)
            return False
        
        logger.info("🔄 Vision handshake: Step 3 - Frame read successfully, running YOLO detection")
        
        # Run object detection using YOLO
        object_params = {}
        object_results = call_vision_service(frame, object_params)
        logger.info(f"🔄 Vision handshake: Step 4 - YOLO detection completed")
        
        if 'error' in object_results:
            logger.error(f"Vision handshake: Object detection error: {object_results['error']}")
            write_vision_to_plc(0, 0, True, False, busy=False, completed=True)
            return False
        
        detected_objects = object_results.get('objects', [])
        
        # Find the selected counter (only process 1 counter)
        config = load_config()
        vision_config = config.get('vision', {})
        single_counter_enabled = vision_config.get('single_counter_enabled', True)
        selection_method = vision_config.get('counter_selection_method', 'most_central')
        
        if single_counter_enabled:
            central_counter = find_most_central_counter(detected_objects, frame.shape, selection_method)
        else:
            # Process all counters (use first one for now, but could be extended)
            central_counter = detected_objects[0] if detected_objects else None
        
        if central_counter:
            # Only process the most central counter
            logger.info(f"🎯 Processing most central counter (out of {len(detected_objects)} detected)")
            
            # Load existing counter positions
            existing_counters = load_existing_counter_positions()
            existing_counter_numbers = set(existing_counters.keys())
            
            detection_timestamp = time.time()
            
            obj_center = central_counter.get('center', (central_counter.get('x', 0) + central_counter.get('width', 0) // 2,
                                                        central_counter.get('y', 0) + central_counter.get('height', 0) // 2))
            
            # Try to match this object to an existing counter by position
            matched_counter_num = find_matching_counter(central_counter, existing_counters)
            
            if matched_counter_num:
                # Matched to an existing counter - save image
                saved_path = save_counter_image(frame, central_counter, matched_counter_num, detection_timestamp)
                if saved_path:
                    central_counter['counterNumber'] = matched_counter_num
                    central_counter['saved_image_path'] = saved_path
            else:
                # Assign new number and save image
                counter_num = get_next_counter_number()
                saved_path = save_counter_image(frame, central_counter, counter_num, detection_timestamp)
                if saved_path:
                    central_counter['counterNumber'] = counter_num
                    existing_counter_numbers.add(counter_num)
                    central_counter['saved_image_path'] = saved_path
            
            # Update object_count to reflect only the central counter
            object_count = 1
        else:
            object_count = 0
            logger.info("No counters detected")
        
        # Check for defects (from stored results)
        defect_count = 0
        defect_detected = False
        object_ok = True
        
        try:
            if os.path.exists(COUNTER_DEFECTS_FILE):
                with open(COUNTER_DEFECTS_FILE, 'r') as f:
                    defect_data = json.load(f)
                    # Count defects with significant issues
                    defect_count = sum(1 for item in defect_data.values() 
                                     if item.get('defect_results', {}).get('defects_found', False))
                    defect_detected = defect_count > 0
                    object_ok = not defect_detected
        except Exception as e:
            logger.debug(f"Error reading defect data: {e}")
        
        # Write final results to PLC (busy=False, completed=True)
        write_vision_to_plc(object_count, defect_count, object_ok, defect_detected, 
                          busy=False, completed=True)
        
        logger.info(f"✅ Vision handshake: Completed - {object_count} objects, {defect_count} defects")
        return True
        
    except Exception as e:
        logger.error(f"Error in vision handshake processing: {e}", exc_info=True)
        write_vision_to_plc(0, 0, True, False, busy=False, completed=True)
        return False
    finally:
        vision_handshake_processing = False

def poll_loop():
    """Background polling loop for real-time data"""
    global poll_running, vision_handshake_last_start_state, vision_handshake_processing

    while poll_running:
        try:
            # Skip PLC operations entirely if snap7 is not available
            # This prevents snap7 crashes from killing the app
            control_bits = {
                'start': False, 'stop': False, 'home': False, 'estop': False,
                'suction': False, 'ready': False, 'busy': False, 'error': False
            }
            target_pose = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            
            # Vision handshaking: Check Start command from PLC
            start_command = False
            if plc_client and hasattr(plc_client, 'client') and plc_client.client is not None:
                try:
                    if plc_client.is_connected():
                        try:
                            config = load_config()
                            db123_config = config.get('plc', {}).get('db123', {})
                            if db123_config.get('enabled', False):
                                db_number = db123_config.get('db_number', 123)
                                # SIMPLE: Read start command
                                start_command = plc_client.read_vision_start_command(db_number)
                                
                                # SIMPLE: If start is TRUE, camera on. If FALSE, camera off.
                                if start_command:
                                    # Start is TRUE - make sure camera is on
                                    if camera_service is not None:
                                        if camera_service.camera is None or (camera_service.camera is not None and not camera_service.camera.isOpened()):
                                            try:
                                                camera_service.initialize_camera()
                                                logger.info("📷 Camera turned ON (Start command is TRUE)")
                                            except Exception as e:
                                                logger.warning(f"Error initializing camera: {e}")
                                    
                                    # Trigger vision processing if not already processing
                                    if not vision_handshake_processing:
                                        logger.info("📸 Start command is TRUE - triggering vision processing")
                                        threading.Thread(target=process_vision_handshake, daemon=True).start()
                                else:
                                    # Start is FALSE - turn off camera
                                    if camera_service is not None:
                                        try:
                                            camera_service.release_camera()
                                            logger.info("📷 Camera turned OFF (Start command is FALSE)")
                                        except Exception as e:
                                            logger.warning(f"Error releasing camera: {e}")
                                    
                                    # Reset flags
                                    if vision_handshake_last_start_state:
                                        write_vision_to_plc(0, 0, True, False, busy=False, completed=False)
                                
                                vision_handshake_last_start_state = start_command
                        except Exception as e:
                            logger.debug(f"PLC vision handshake read error: {e}")
                except Exception as e:
                    logger.debug(f"PLC check error in polling: {e}")
            
            # Only try PLC operations if snap7 is available and client exists
            if plc_client and hasattr(plc_client, 'client') and plc_client.client is not None:
                try:
                    # Don't try to connect - just check if already connected
                    if plc_client.is_connected():
                        try:
                            # Read from configured DB number
                            config = load_config()
                            db_number = config.get('plc', {}).get('db_number', 123)
                            control_bits = plc_client.read_control_bits()
                            target_pose = plc_client.read_target_pose(db_number)
                        except Exception as e:
                            logger.debug(f"PLC read error in polling: {e}")
                except Exception as e:
                    logger.debug(f"PLC check error in polling: {e}")

            # Read Dobot data
            dobot_pose = None
            try:
                if dobot_client.connected:
                    dobot_pose = dobot_client.get_pose()
            except Exception as e:
                logger.debug(f"Dobot read error in polling: {e}")

            # Emit data to all connected clients
            try:
                socketio.emit('plc_data', {
                    'control_bits': control_bits,
                    'target_pose': target_pose,
                    'timestamp': time.time()
                })

                if dobot_pose:
                    socketio.emit('dobot_data', {
                        'pose': dobot_pose,
                        'timestamp': time.time()
                    })
            except Exception as e:
                logger.debug(f"Socket emit error: {e}")

        except Exception as e:
            logger.error(f"Polling error: {e}")

        # Add delay to prevent flooding the PLC (minimum 50ms between cycles)
        # This gives the S7-1200 time to process requests
        time.sleep(max(poll_interval, 0.05))  # At least 50ms between polling cycles

    logger.info("Polling thread stopped")

# ==================================================
# Start Command Polling Loop (Lightweight - Camera Control Only)
# ==================================================

def start_command_poll_loop():
    """Lightweight polling loop that checks start command and enables/disables vision analysis
    Camera stays always active - only analysis is controlled by start command
    """
    global vision_handshake_last_start_state, vision_handshake_processing
    
    while True:
        try:
            # Only check if PLC is connected and DB123 is enabled
            if plc_client and hasattr(plc_client, 'client') and plc_client.client is not None:
                if plc_client.is_connected():
                    try:
                        config = load_config()
                        db123_config = config.get('plc', {}).get('db123', {})
                        if db123_config.get('enabled', False):
                            db_number = db123_config.get('db_number', 123)
                            
                            # SIMPLE: Read start command
                            start_command = plc_client.read_vision_start_command(db_number)
                            
                            # SIMPLE: If start is TRUE, enable analysis. If FALSE, disable analysis.
                            # Camera stays always active
                            if start_command:
                                # Start is TRUE - trigger vision processing if not already processing
                                if not vision_handshake_processing and not vision_handshake_last_start_state:
                                    logger.info("📸 Start command is TRUE - enabling vision analysis")
                                    threading.Thread(target=process_vision_handshake, daemon=True).start()
                            else:
                                # Start is FALSE - reset flags (analysis disabled, camera stays on)
                                if vision_handshake_last_start_state:
                                    logger.info("📸 Start command is FALSE - disabling vision analysis (camera stays active)")
                                    write_vision_to_plc(0, 0, True, False, busy=False, completed=False)
                            
                            vision_handshake_last_start_state = start_command
                    except Exception as e:
                        logger.debug(f"Start command polling error: {e}")
        except Exception as e:
            logger.debug(f"Start command polling loop error: {e}")
        
        # Poll every 500ms (2 times per second) - fast enough to be responsive
        time.sleep(0.5)

# ==================================================
# Camera & Vision System Endpoints
# ==================================================

def write_plc_fault_bit(defects_found: bool):
    """Write vision fault status to PLC memory bit
    
    This is a wrapper function that uses the unified PLC client methods.
    All S7 communication is handled in plc_client.py.
    """
    if plc_client is None:
        return {'written': False, 'reason': 'plc_not_available'}
    
    try:
        config = load_config()
        vision_config = config.get('vision', {})
        
        # Check if fault bit is enabled
        if not vision_config.get('fault_bit_enabled', False):
            return {'written': False, 'reason': 'disabled'}
        
        byte_offset = vision_config.get('fault_bit_byte', 1)
        bit_offset = vision_config.get('fault_bit_bit', 0)
        
        # Use unified PLC client method for all S7 communication
        return plc_client.write_vision_fault_bit(defects_found, byte_offset, bit_offset)
    except Exception as e:
        logger.debug(f"Error in write_plc_fault_bit: {e}")
        return {'written': False, 'reason': str(e)}

def generate_frames():
    """Generator function for MJPEG streaming"""
    while True:
        if camera_service is None:
            break
        
        frame_bytes = camera_service.get_frame_jpeg(quality=70)  # Reduced quality for faster streaming
        if frame_bytes is None:
            time.sleep(0.05)  # Reduced sleep time when no frame available
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)  # ~20 FPS - reduced for faster initial load

@app.route('/api/camera/stream')
def camera_stream():
    """MJPEG video stream endpoint"""
    if camera_service is None:
        return jsonify({'error': 'Camera service not initialized'}), 503
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    """Get camera connection status"""
    if camera_service is None:
        return jsonify({
            'initialized': False,
            'connected': False,
            'error': 'Camera service not initialized'
        })
    
    try:
        # SIMPLE: Check if camera is opened, not just if we can read a frame
        # Camera might be initialized but still warming up
        with camera_service.lock:
            camera_opened = camera_service.camera is not None and camera_service.camera.isOpened()
        
        # Try to read a frame to confirm it's working
        frame = camera_service.read_frame()
        can_read = frame is not None
        
        # Camera is connected if it's opened (even if we can't read yet - might be warming up)
        connected = camera_opened
        
        return jsonify({
            'initialized': True,
            'connected': connected,
            'can_read': can_read,  # Additional info: can we actually read frames?
            'camera_index': camera_service.camera_index,
            'resolution': {
                'width': camera_service.width,
                'height': camera_service.height
            },
            'last_frame_time': camera_service.frame_time
        })
    except Exception as e:
        logger.error(f"Error checking camera status: {e}")
        return jsonify({
            'initialized': True,
            'connected': False,
            'error': str(e)
        }), 500

@app.route('/api/camera/connect', methods=['POST'])
def camera_connect():
    """Initialize and connect to camera"""
    global camera_service
    
    try:
        data = request.json or {}
        camera_index = data.get('index', 0)
        width = data.get('width', 640)
        height = data.get('height', 480)
        
        if camera_service is None:
            camera_service = CameraService(
                camera_index=camera_index,
                width=width,
                height=height
            )
        
        success = camera_service.initialize_camera()
        
        if success:
            # Update config
            config = load_config()
            config['camera'] = {
                'index': camera_index,
                'width': width,
                'height': height
            }
            save_config(config)
        
        return jsonify({
            'success': success,
            'connected': success,
            'error': None if success else 'Failed to initialize camera'
        })
    except Exception as e:
        logger.error(f"Error connecting camera: {e}")
        return jsonify({
            'success': False,
            'connected': False,
            'error': str(e)
        }), 500

@app.route('/api/camera/disconnect', methods=['POST'])
def camera_disconnect():
    """Disconnect and release camera"""
    global camera_service
    
    try:
        if camera_service is not None:
            camera_service.release_camera()
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error disconnecting camera: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera/capture', methods=['GET'])
def camera_capture():
    """Capture a single frame as JPEG - uses cached frame if recent to reduce camera load"""
    if camera_service is None:
        return jsonify({'error': 'Camera service not initialized'}), 503
    
    try:
        # SIMPLE: Check if camera is actually opened before trying to capture
        with camera_service.lock:
            if camera_service.camera is None or not camera_service.camera.isOpened():
                return jsonify({'error': 'Camera not opened'}), 503
        
        # Use cached frame if less than 0.5 seconds old (optimization for 1-second snapshot updates)
        frame_bytes = camera_service.get_frame_jpeg(quality=85, use_cache=True, max_cache_age=0.5)
        if frame_bytes is None:
            return jsonify({'error': 'Failed to capture frame - camera may still be warming up'}), 500
        
        return Response(
            frame_bytes,
            mimetype='image/jpeg',
            headers={'Content-Disposition': 'inline; filename=capture.jpg'}
        )
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/crop', methods=['GET'])
def get_camera_crop():
    """Get current camera crop settings"""
    if camera_service is None:
        return jsonify({'error': 'Camera service not initialized'}), 503
    
    try:
        crop_settings = camera_service.get_crop()
        return jsonify(crop_settings)
    except Exception as e:
        logger.error(f"Error getting crop settings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/crop', methods=['POST'])
def set_camera_crop():
    """Set camera crop settings"""
    if camera_service is None:
        return jsonify({'error': 'Camera service not initialized'}), 503
    
    try:
        data = request.json or {}
        enabled = data.get('enabled', False)
        x = data.get('x', 0)
        y = data.get('y', 0)
        width = data.get('width', 100)
        height = data.get('height', 100)
        
        camera_service.set_crop(enabled, x, y, width, height)
        
        # Save to config
        config = load_config()
        if 'camera' not in config:
            config['camera'] = {}
        config['camera']['crop'] = {
            'enabled': enabled,
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }
        save_config(config)
        
        return jsonify({'success': True, 'crop': camera_service.get_crop()})
    except Exception as e:
        logger.error(f"Error setting crop: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/vision/detect-objects', methods=['POST'])
def vision_detect_objects():
    """Run object detection on current frame"""
    if camera_service is None:
        return jsonify({'error': 'Camera service not initialized'}), 503
    
    try:
        data = request.json or {}
        method = data.get('method', 'contour')  # 'contour', 'blob', 'combined'
        
        # Read current frame
        frame = camera_service.read_frame()
        if frame is None:
            return jsonify({'error': 'Failed to read frame from camera'}), 500
        
        # Extract detection parameters
        detection_params = data.get('params', {})
        
        # Run object detection
        results = camera_service.detect_objects(frame, method=method, params=detection_params)
        
        # Optionally draw objects on frame
        if data.get('annotate', False) and results['objects_found']:
            annotated_frame = camera_service.draw_objects(frame, results['objects'])
            # Encode annotated frame
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            ret, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            if ret:
                results['annotated_image'] = buffer.tobytes().hex()
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in object detection: {e}")
        return jsonify({'error': str(e)}), 500

# Removed duplicate vision_detect function - using the one at line 1140 instead

@app.route('/api/vision/analyze', methods=['POST'])
def vision_analyze():
    """Analyze frame and return annotated image (with optional object detection)
    
    SIMPLE: No start command check - polling loop handles camera control.
    """
    if camera_service is None:
        return jsonify({'error': 'Camera service not initialized'}), 503
    
    try:
        
        data = request.json or {}
        method = data.get('method', 'combined')
        use_object_detection = data.get('use_object_detection', False)
        object_method = data.get('object_method', 'contour')
        
        # Read current frame
        frame = camera_service.read_frame()
        if frame is None:
            return jsonify({'error': 'Failed to read frame from camera'}), 500
        
        # Extract detection parameters
        detection_params = data.get('params', {})
        object_params = data.get('object_params', {})
        
        detected_objects = []
        roi_regions = []
        
        # Run object detection first if enabled
        detected_objects = []
        if use_object_detection:
            # If using YOLO, call vision service
            if object_method == 'yolo':
                object_results = call_vision_service(frame, object_params)
                detected_objects = object_results.get('objects', [])
            else:
                # Non-YOLO methods use camera_service directly
                object_results = camera_service.detect_objects(frame, method=object_method, params=object_params)
                detected_objects = object_results.get('objects', [])
            
            # Find the most central counter (only process 1 counter)
            central_counter = find_most_central_counter(detected_objects, frame.shape)
            
            if central_counter:
                # Only process the most central counter
                logger.info(f"🎯 Processing most central counter (out of {len(detected_objects)} detected)")
                
                # Load existing counter positions (from JSON file and saved images)
                existing_counters = load_existing_counter_positions()
                existing_counter_numbers = set(existing_counters.keys())
                
                # Track which detected objects have been matched in this frame
                detection_timestamp = time.time()
                matched_counters = {}  # Maps counter_number -> obj for position tracking
                updated_positions = {}  # Track positions to save at end
                
                obj_center = central_counter.get('center', (central_counter.get('x', 0) + central_counter.get('width', 0) // 2,
                                                            central_counter.get('y', 0) + central_counter.get('height', 0) // 2))
                
                # Try to match this object to an existing counter by position
                matched_counter_num = find_matching_counter(central_counter, existing_counters)
                
                if matched_counter_num:
                    # Matched to an existing counter - use that number
                    central_counter['counterNumber'] = matched_counter_num
                    # Update position for future matching
                    updated_positions[matched_counter_num] = {
                        'x': central_counter.get('x', 0),
                        'y': central_counter.get('y', 0),
                        'center': obj_center,
                        'last_seen_timestamp': detection_timestamp
                    }
                    matched_counters[matched_counter_num] = updated_positions[matched_counter_num]
                    # Always save a new image with timestamp (allows multiple images per counter)
                    saved_path = save_counter_image(frame, central_counter, matched_counter_num, detection_timestamp)
                    if saved_path:
                        central_counter['saved_image_path'] = saved_path
                else:
                    # No match found - assign new number and save image
                    counter_num = get_next_counter_number()
                    saved_path = save_counter_image(frame, central_counter, counter_num, detection_timestamp)
                    if saved_path:
                        central_counter['counterNumber'] = counter_num
                        existing_counter_numbers.add(counter_num)
                        # Track position for future matching
                        updated_positions[counter_num] = {
                            'x': central_counter.get('x', 0),
                            'y': central_counter.get('y', 0),
                            'center': obj_center,
                            'last_seen_timestamp': detection_timestamp
                        }
                        matched_counters[counter_num] = updated_positions[counter_num]
                        central_counter['saved_image_path'] = saved_path
                
                # Save updated positions for next detection cycle
                if updated_positions:
                    # Merge with existing positions
                    all_positions = existing_counters.copy()
                    all_positions.update(updated_positions)
                    save_counter_positions(all_positions)
                
                # Extract ROI region from the central counter
                x, y = central_counter['x'], central_counter['y']
                w, h = central_counter['width'], central_counter['height']
                padding = object_params.get('roi_padding', 10)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                roi_regions.append((x1, y1, x2, y2))
                
                # Update detected_objects to only include the central counter
                detected_objects = [central_counter]
            else:
                detected_objects = []
                logger.info("No counters detected")
        
        # Return object detection results only (defect detection disabled)
        results = {
            'defects_found': False,
            'defect_count': 0,
            'defects': [],
            'confidence': 0.0,
            'method': method,
            'objects_detected': len(detected_objects),
            'timestamp': time.time()
        }
        
        results['detected_objects'] = detected_objects
        
        # Draw objects on frame
        annotated_frame = frame.copy()
        if detected_objects:
            annotated_frame = camera_service.draw_objects(annotated_frame, detected_objects, color=(0, 255, 0))
        
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        ret, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
        
        if not ret:
            return jsonify({'error': 'Failed to encode annotated image'}), 500
        
        # Return both JSON results and image
        return Response(
            buffer.tobytes(),
            mimetype='image/jpeg',
            headers={
                'X-Defect-Count': str(results['defect_count']),
                'X-Defects-Found': str(results['defects_found']).lower(),
                'X-Confidence': str(results['confidence']),
                'X-Objects-Detected': str(results.get('objects_detected', 0)),
                'Content-Disposition': 'inline; filename=analyzed.jpg'
            }
        )
    except Exception as e:
        logger.error(f"Error in vision analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/vision/detect', methods=['POST'])
def vision_detect():
    """Detect objects/defects and return JSON results (no image)
    
    SIMPLE: No start command check - polling loop handles camera control.
    """
    if camera_service is None:
        return jsonify({'error': 'Camera service not initialized'}), 503

    try:
        
        data = request.json or {}
        object_detection_enabled = data.get('object_detection_enabled', True)
        defect_detection_enabled = data.get('defect_detection_enabled', False)
        object_method = data.get('object_method', 'yolo')  # Default to YOLO for counter detection
        defect_method = data.get('method', 'combined')

        # Read current frame
        frame = camera_service.read_frame()
        if frame is None:
            return jsonify({'error': 'Failed to read frame from camera'}), 500

        # Extract detection parameters
        detection_params = data.get('params', {})
        object_params = data.get('object_params', {})

        results = {
            'object_detection_enabled': object_detection_enabled,
            'defect_detection_enabled': defect_detection_enabled,
            'timestamp': time.time()
        }

        detected_objects = []

        # Set busy flag at start of detection
        write_vision_to_plc(0, 0, True, False, busy=True)
        
        # Run object detection if enabled
        if object_detection_enabled:
            # If using YOLO, call vision service instead of direct YOLO
            if object_method == 'yolo':
                object_results = call_vision_service(frame, object_params)
            else:
                # Non-YOLO methods use camera_service directly
                object_results = camera_service.detect_objects(frame, method=object_method, params=object_params)
            
            # Check for errors in detection
            if 'error' in object_results:
                logger.error(f"Object detection error: {object_results['error']}")
                results['detection_error'] = object_results['error']
            
            detected_objects = object_results.get('objects', [])
            
            # Assign counter numbers (images are saved in /api/vision/analyze endpoint to avoid duplicates)
            # Use the same counter tracking logic as /api/vision/analyze
            if detected_objects:
                # Check which counters already have images (have been seen before)
                existing_counter_numbers = set()
                if os.path.exists(COUNTER_IMAGES_DIR):
                    for filename in os.listdir(COUNTER_IMAGES_DIR):
                        if filename.startswith('counter_') and filename.endswith('.jpg'):
                            parts = filename.split('_')
                            if len(parts) >= 2:
                                try:
                                    existing_counter_numbers.add(int(parts[1]))
                                except ValueError:
                                    pass
                
                # Sort by x position (left to right) for consistent ordering
                detected_objects.sort(key=lambda obj: obj.get('x', 0))
                
                # Assign new numbers incrementally
                for obj in detected_objects:
                    if 'counterNumber' not in obj:
                        obj['counterNumber'] = get_next_counter_number()
            
            results['object_count'] = len(detected_objects)
            results['objects'] = detected_objects
            results['objects_found'] = len(detected_objects) > 0
            results['object_method'] = object_method
            
            # Log detection results for debugging
            logger.info(f"Detection completed: {len(detected_objects)} objects found using {object_method} method")

        # Run defect detection if enabled (currently disabled)
        defect_count = 0
        defect_detected = False
        object_ok = True
        
        if defect_detection_enabled:
            results['defects_found'] = False
            results['defect_count'] = 0
            results['defects'] = []
            results['confidence'] = 0.0
            results['defect_method'] = defect_method
            results['note'] = 'Defect detection is currently disabled'
        else:
            # Check stored defect results to get defect count
            try:
                if os.path.exists(COUNTER_DEFECTS_FILE):
                    with open(COUNTER_DEFECTS_FILE, 'r') as f:
                        defect_data = json.load(f)
                        # Count defects with significant issues
                        defect_count = sum(1 for item in defect_data.values() 
                                         if item.get('defect_results', {}).get('defects_found', False))
                        defect_detected = defect_count > 0
                        object_ok = not defect_detected
            except Exception as e:
                logger.debug(f"Error reading defect data: {e}")

        # Write vision results to PLC DB123 (busy=False since detection is complete)
        object_count = results.get('object_count', 0)
        write_vision_to_plc(object_count, defect_count, object_ok, defect_detected, busy=False)

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in vision detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/plc/db123/read', methods=['GET'])
def read_db123_tags():
    """Read current vision tags from PLC DB123 (ultra-simple version)"""
    # Always return immediately - no exceptions that could cause timeouts
    default_tags = {
        'start': False,
        'connected': False,
        'busy': False,
        'completed': False,
        'object_detected': False,
        'object_ok': False,
        'defect_detected': False,
        'object_number': 0,
        'defect_number': 0
    }
    
    if plc_client is None:
        return jsonify({
            'success': False,
            'error': 'PLC client not initialized',
            'db_number': 123,
            'tags': default_tags,
            'plc_connected': False
        }), 503
    
    try:
        # Get config
        config = load_config()
        db123_config = config.get('plc', {}).get('db123', {})
        db_number = db123_config.get('db_number', 123)
        
        # Try to read tags (returns immediately with cached values if lock busy)
        tags = plc_client.read_vision_tags(db_number)
        
        # Always return success - even if we got cached values
        return jsonify({
            'success': True,
            'db_number': db_number,
            'tags': tags,
            'plc_connected': plc_client.is_connected()
        })
    except Exception as e:
        logger.error(f"Error reading DB123 tags: {e}")
        # Return default tags on error - never timeout
        return jsonify({
            'success': False,
            'error': str(e),
            'db_number': 123,
            'tags': default_tags,
            'plc_connected': False
        }), 500

@app.route('/api/counter-images', methods=['GET'])
def list_counter_images():
    """List all saved counter images"""
    try:
        if not os.path.exists(COUNTER_IMAGES_DIR):
            return jsonify({'images': [], 'count': 0})
        
        # Get all counter image files
        image_files = []
        for filename in sorted(os.listdir(COUNTER_IMAGES_DIR), reverse=True):  # Most recent first
            if filename.startswith('counter_') and filename.endswith('.jpg'):
                filepath = os.path.join(COUNTER_IMAGES_DIR, filename)
                stat = os.stat(filepath)
                
                # Parse filename: counter_1_20241124_141530_123.jpg
                parts = filename.replace('.jpg', '').split('_')
                if len(parts) >= 5:
                    counter_num = parts[1]
                    date_str = parts[2]
                    time_str = parts[3]
                    ms_str = parts[4] if len(parts) > 4 else '000'
                    
                    # Parse timestamp
                    try:
                        dt = datetime.strptime(f"{date_str}_{time_str}_{ms_str}", "%Y%m%d_%H%M%S_%f")
                        timestamp = dt.timestamp()
                    except:
                        timestamp = stat.st_mtime
                else:
                    counter_num = '?'
                    timestamp = stat.st_mtime
                
                image_files.append({
                    'filename': filename,
                    'counter_number': int(counter_num) if counter_num.isdigit() else 0,
                    'timestamp': timestamp,
                    'formatted_time': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    'size': stat.st_size,
                    'url': f'/api/counter-images/{filename}'
                })
        
        return jsonify({
            'images': image_files,
            'count': len(image_files)
        })
    except Exception as e:
        logger.error(f"Error listing counter images: {e}")
        return jsonify({'error': str(e), 'images': [], 'count': 0}), 500

@app.route('/api/counter-images/<filename>', methods=['GET'])
def serve_counter_image(filename):
    """Serve a specific counter image"""
    try:
        # Security: only allow counter_*.jpg files
        if not filename.startswith('counter_') or not filename.endswith('.jpg'):
            return jsonify({'error': 'Invalid filename'}), 400
        
        filepath = os.path.join(COUNTER_IMAGES_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        return send_from_directory(COUNTER_IMAGES_DIR, filename)
    except Exception as e:
        logger.error(f"Error serving counter image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/counter-images/<int:counter_number>/analyze-defects', methods=['POST'])
def analyze_counter_defects(counter_number: int):
    """Analyze a saved counter image for defects (color changes on surface)"""
    try:
        if not os.path.exists(COUNTER_IMAGES_DIR):
            return jsonify({'error': 'Counter images directory not found'}), 404
        
        # Find the image file for this counter
        prefix = f"counter_{counter_number}_"
        image_file = None
        for filename in os.listdir(COUNTER_IMAGES_DIR):
            if filename.startswith(prefix) and filename.endswith('.jpg'):
                image_file = os.path.join(COUNTER_IMAGES_DIR, filename)
                break
        
        if not image_file or not os.path.exists(image_file):
            return jsonify({'error': f'Counter {counter_number} image not found'}), 404
        
        # Read the image
        image = cv2.imread(image_file)
        if image is None:
            return jsonify({'error': 'Failed to read image file'}), 500
        
        # Analyze for color variations (defects)
        defect_results = detect_color_defects(image)
        
        # Store the latest results for quick access
        record_counter_defect_result(counter_number, image_file, defect_results)

        return jsonify({
            'counter_number': counter_number,
            'image_file': os.path.basename(image_file),
            'defects_found': defect_results.get('defects_found', False),
            'defect_count': defect_results.get('defect_count', 0),
            'defects': defect_results.get('defects', []),
            'confidence': defect_results.get('confidence', 0.0),
            'dominant_color': defect_results.get('dominant_color', {'b': 0, 'g': 0, 'r': 0}),
            'total_defect_area_percentage': defect_results.get('total_defect_area_percentage', 0.0),
            'color_variance': defect_results.get('total_defect_area_percentage', 0.0),  # Keep for backward compatibility
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Error analyzing counter defects: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def detect_color_defects(image: np.ndarray) -> Dict:
    """
    Detect defects on counter surface by finding large areas with significantly different colors
    Only analyzes the circular counter, excluding conveyor belt and background
    
    Args:
        image: Counter image (BGR format)
    
    Returns:
        Dictionary with defect detection results
    """
    try:
        h, w = image.shape[:2]
        
        # Convert to grayscale for circle detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect circular counter using HoughCircles with more lenient parameters
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(h, w) // 3,  # More lenient - allow circles closer together
            param1=50,
            param2=20,  # Lower threshold for circle detection
            minRadius=min(h, w) // 5,  # Smaller minimum radius
            maxRadius=int(min(h, w) * 0.48)  # Slightly larger max radius
        )
        
        # Create mask for circular counter area only
        counter_mask = np.zeros((h, w), dtype=np.uint8)
        if circles is not None and len(circles[0]) > 0:
            # Use the largest circle found
            circles = np.uint16(np.around(circles))
            # Sort by radius and use the largest
            circles_sorted = sorted(circles[0], key=lambda c: c[2], reverse=True)
            largest_circle = circles_sorted[0]
            center_x, center_y, radius = largest_circle[0], largest_circle[1], largest_circle[2]
            # Shrink radius slightly (90%) to exclude edge effects and conveyor belt
            radius = int(radius * 0.9)
            # Draw filled circle on mask
            cv2.circle(counter_mask, (center_x, center_y), radius, 255, -1)
        else:
            # Fallback: use center region as circular area (smaller to exclude edges)
            center_x, center_y = w // 2, h // 2
            radius = int(min(w, h) * 0.35)  # Smaller radius to exclude conveyor belt
            cv2.circle(counter_mask, (center_x, center_y), radius, 255, -1)
        
        # Extract only the counter region (mask out conveyor belt and background)
        counter_region = cv2.bitwise_and(image, image, mask=counter_mask)
        
        # Find the dominant/main color of the counter (only within the mask)
        counter_pixels = counter_region[counter_mask > 0].reshape(-1, 3).astype(np.float32)
        
        if len(counter_pixels) == 0:
            return {
                'defects_found': False,
                'defect_count': 0,
                'defects': [],
                'confidence': 0.0,
                'error': 'Could not extract counter region'
            }
        
        # Use k-means to find dominant colors (try 3 clusters)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = min(3, len(counter_pixels))
        if k < 2:
            return {
                'defects_found': False,
                'defect_count': 0,
                'defects': [],
                'confidence': 0.0,
                'error': 'Not enough pixels for analysis'
            }
        
        _, labels, centers = cv2.kmeans(counter_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Find the most common color (dominant color)
        unique, counts = np.unique(labels, return_counts=True)
        dominant_idx = unique[np.argmax(counts)]
        dominant_color = centers[dominant_idx].astype(np.uint8)
        
        # Calculate color difference threshold - defects must be significantly different
        COLOR_DIFFERENCE_THRESHOLD = 110  # Minimum color difference to be considered a defect
        MIN_DEFECT_AREA_PERCENT = 2.0  # Defect must be at least 2% of counter area
        counter_area = np.sum(counter_mask > 0)  # Total counter area in pixels
        min_defect_area = int(counter_area * (MIN_DEFECT_AREA_PERCENT / 100))
        
        # Create mask for pixels that differ significantly from dominant color (only within counter mask)
        counter_region_float = counter_region.astype(np.float32)
        color_diff = np.linalg.norm(counter_region_float - dominant_color.astype(np.float32), axis=2)
        defect_mask = (color_diff > COLOR_DIFFERENCE_THRESHOLD) & (counter_mask > 0)
        
        # Convert to uint8 for morphological operations
        defect_mask_uint8 = (defect_mask * 255).astype(np.uint8)
        
        # Apply morphological operations to connect nearby defect pixels and remove noise
        kernel = np.ones((5, 5), np.uint8)
        defect_mask_uint8 = cv2.morphologyEx(defect_mask_uint8, cv2.MORPH_CLOSE, kernel)  # Connect nearby defects
        defect_mask_uint8 = cv2.morphologyEx(defect_mask_uint8, cv2.MORPH_OPEN, kernel)   # Remove small noise
        
        # Find contours of defect regions
        contours, _ = cv2.findContours(defect_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects = []
        total_defect_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_defect_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Get the defect region
                defect_region = image[y:y+h_rect, x:x+w_rect]
                defect_color_avg = np.mean(defect_region.reshape(-1, 3), axis=0)
                
                # Calculate color difference from dominant color
                color_diff_value = np.linalg.norm(defect_color_avg - dominant_color)
                confidence = min(100, (color_diff_value / 255.0) * 100)
                
                # Calculate percentage of counter covered by this defect
                defect_percentage = (area / counter_area) * 100 if counter_area > 0 else 0
                
                defects.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w_rect),
                    'height': int(h_rect),
                    'area': float(area),
                    'area_percentage': round(defect_percentage, 2),
                    'confidence': round(confidence, 2),
                    'color_difference': round(float(color_diff_value), 2),
                    'type': 'color_variation'
                })
                total_defect_area += area
        
        # Determine if defects were found
        defects_found = len(defects) > 0
        total_defect_percentage = (total_defect_area / counter_area) * 100 if defects_found and counter_area > 0 else 0
        overall_confidence = min(100, total_defect_percentage * 2) if defects_found else 0
        
        return {
            'defects_found': defects_found,
            'defect_count': len(defects),
            'defects': defects,
            'confidence': round(overall_confidence, 2),
            'dominant_color': {
                'b': int(dominant_color[0]),
                'g': int(dominant_color[1]),
                'r': int(dominant_color[2])
            },
            'total_defect_area_percentage': round(total_defect_percentage, 2),
            'method': 'color_variation'
        }
    except Exception as e:
        logger.error(f"Error in color defect detection: {e}", exc_info=True)
        return {
            'defects_found': False,
            'defect_count': 0,
            'defects': [],
            'confidence': 0.0,
            'error': str(e)
        }

@app.route('/api/counter-images/delete-all', methods=['POST'])
def delete_all_counter_images():
    """Delete all counter images and reset counter tracker"""
    try:
        deleted_count = 0
        
        if os.path.exists(COUNTER_IMAGES_DIR):
            for filename in os.listdir(COUNTER_IMAGES_DIR):
                if filename.startswith('counter_') and filename.endswith('.jpg'):
                    filepath = os.path.join(COUNTER_IMAGES_DIR, filename)
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {filename}: {e}")
        
        # Reset counter tracker
        _counter_tracker['max_counter_number'] = 0
        
        # Delete counter positions file
        if os.path.exists(COUNTER_POSITIONS_FILE):
            try:
                os.remove(COUNTER_POSITIONS_FILE)
            except Exception as e:
                logger.warning(f"Failed to delete counter positions file: {e}")

        # Delete stored defect results
        if os.path.exists(COUNTER_DEFECTS_FILE):
            try:
                os.remove(COUNTER_DEFECTS_FILE)
            except Exception as e:
                logger.warning(f"Failed to delete counter defect results file: {e}")
        
        logger.info(f"Deleted all counter images ({deleted_count} images) and reset counter tracker")
        return jsonify({
            'message': f'Deleted all counter images and reset timeline',
            'deleted': deleted_count
        })
    except Exception as e:
        logger.error(f"Error deleting all counter images: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/counter-images/defects', methods=['GET'])
def get_counter_defect_results():
    """Return stored defect detection results for all counters"""
    try:
        results = load_counter_defect_results()
        return jsonify({
            'defects': list(results.values()),
            'count': len(results)
        })
    except Exception as e:
        logger.error(f"Error fetching counter defect results: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/counter-images/cleanup', methods=['POST'])
def cleanup_counter_images():
    """Clean up duplicate counter images - keep only most recent per counter"""
    try:
        if not os.path.exists(COUNTER_IMAGES_DIR):
            return jsonify({'message': 'No images directory found', 'deleted': 0})
        
        deleted_count = 0
        
        # Group images by counter number
        counter_groups = {}
        for filename in os.listdir(COUNTER_IMAGES_DIR):
            if filename.startswith('counter_') and filename.endswith('.jpg'):
                # Parse counter number from filename: counter_1_20241124_141530_123.jpg
                parts = filename.replace('.jpg', '').split('_')
                if len(parts) >= 2:
                    try:
                        counter_num = int(parts[1])
                        if counter_num not in counter_groups:
                            counter_groups[counter_num] = []
                        filepath = os.path.join(COUNTER_IMAGES_DIR, filename)
                        stat = os.stat(filepath)
                        counter_groups[counter_num].append((filepath, stat.st_mtime, filename))
                    except ValueError:
                        continue
        
        # For each counter, keep only the most recent image
        for counter_num, images in counter_groups.items():
            if len(images) > 1:
                # Sort by modification time (most recent first)
                images.sort(key=lambda x: x[1], reverse=True)
                # Delete all except the first (most recent)
                for filepath, _, filename in images[1:]:
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                        logger.info(f"Cleaned up old counter {counter_num} image: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {filename}: {e}")
        
        logger.info(f"Cleanup complete: Deleted {deleted_count} duplicate counter images")
        return jsonify({
            'message': f'Cleanup complete: Deleted {deleted_count} duplicate images',
            'deleted': deleted_count
        })
    except Exception as e:
        logger.error(f"Error cleaning up counter images: {e}")
        return jsonify({'error': str(e)}), 500

# ==================================================
# Serve PWA Frontend
# ==================================================

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_pwa(path):
    """Serve PWA frontend"""
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# ==================================================
# Application Startup
# ==================================================

if __name__ == '__main__':
    init_clients()

    # Auto-connect to PLC on startup (with retry logic)
    if plc_client:
        plc_ip = plc_client.ip if hasattr(plc_client, 'ip') else 'unknown'
        logger.info(f"🔌 Attempting to connect to PLC at {plc_ip}...")
        plc_connected = plc_client.connect()
        if plc_connected:
            logger.info(f"✅ PLC connected successfully to {plc_ip}")
        else:
            logger.warning(f"⚠️ PLC connection failed: {plc_client.last_error}")
            logger.info("💡 PLC will retry connection automatically, or use /api/plc/connect endpoint")
    else:
        logger.info("PLC client not initialized - PLC features disabled")

    # Auto-connect to Dobot
    logger.info("🤖 Attempting to connect to Dobot robot...")
    dobot_connected = dobot_client.connect()
    if dobot_connected:
        logger.info("✅ Dobot connected successfully")
    else:
        logger.error(f"❌ Dobot connection failed: {dobot_client.last_error}")
        logger.error("💡 Check the debug logs above for detailed troubleshooting steps")

    # Start lightweight polling for start command (camera control only)
    # This is separate from the main polling loop to avoid lock contention
    start_command_polling_thread = threading.Thread(target=start_command_poll_loop, daemon=True)
    start_command_polling_thread.start()
    logger.info("Start command polling thread started")

    # Start server
    port = int(os.getenv('PORT', 8080))
    logger.info(f"Starting server on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
