"""
PLC Communication Module using python-snap7
Handles S7 protocol communication with Siemens S7-1200/1500 PLCs
"""

import time
import threading
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import snap7, but handle gracefully if it fails or crashes
snap7_available = False
snap7 = None
try:
    import snap7
    from snap7.util import get_bool, get_real, get_int, set_bool, set_real, set_int
    snap7_available = True
except Exception as e:
    logger.warning(f"snap7 library not available or failed to import: {e}")
    snap7_available = False

class PLCClient:
    """S7 PLC Communication Client"""

    def __init__(self, ip: str = '192.168.1.150', rack: int = 0, slot: int = 1):
        """
        Initialize PLC client

        Args:
            ip: PLC IP address
            rack: PLC rack number (0 for S7-1200)
            slot: PLC slot number (1 for S7-1200)
        """
        self.ip = ip
        self.rack = rack
        self.slot = slot
        self.connected = False
        self.last_error = ""
        self.client = None
        
        # Start command stability tracking (prevents flickering from read errors)
        self.start_command_history = []  # History of recent reads (max 3)
        self.start_command_stable_value = None  # Last stable value
        
        # Thread safety: Snap7 client is NOT thread-safe - all operations must be serialized
        self.plc_lock = threading.Lock()
        
        # Only create snap7 client if library is available
        if snap7_available:
            try:
                self.client = snap7.client.Client()
            except Exception as e:
                logger.error(f"Failed to create snap7 client: {e}")
                self.client = None
                self.last_error = f"snap7 client creation failed: {str(e)}"
        else:
            self.last_error = "snap7 library not available"

        # Connection retry settings
        self.max_retries = 3
        self.retry_delay = 1.0
        self.last_connection_attempt = 0
        self.connection_attempt_interval = 5.0

    def connect(self) -> bool:
        """Connect to PLC with retry logic - gracefully handles failures without crashing"""
        # If snap7 is not available, don't try to connect
        if not snap7_available or self.client is None:
            self.connected = False
            self.last_error = "snap7 library not available"
            return False
            
        try:
            current_time = time.time()

            # Don't attempt connection too frequently
            if (current_time - self.last_connection_attempt) < self.connection_attempt_interval:
                return self.connected

            self.last_connection_attempt = current_time

            # Check if already connected (with error handling)
            try:
                if self.connected and self.client and self.client.get_connected():
                    return True
            except Exception:
                # If get_connected() fails, assume disconnected
                self.connected = False

            logger.info(f"Connecting to PLC at {self.ip}, rack {self.rack}, slot {self.slot}")

            # Try to connect with retries
            for attempt in range(self.max_retries):
                try:
                    if self.client:
                        self.client.connect(self.ip, self.rack, self.slot)

                        # Check connection status with error handling
                        try:
                            if self.client.get_connected():
                                self.connected = True
                                self.last_error = ""
                                logger.info(f"✅ Connected to S7 PLC at {self.ip}")
                                return True
                        except Exception as check_error:
                            logger.warning(f"Connection check failed: {check_error}")
                            self.connected = False

                except Exception as e:
                    self.last_error = f"Connection error: {str(e)}"
                    logger.error(f"{self.last_error} (attempt {attempt + 1}/{self.max_retries})")
                    self.connected = False

                # Wait before retry
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

            self.connected = False
            logger.warning(f"PLC unreachable at {self.ip} - continuing without PLC connection")
            return False

        except Exception as e:
            self.last_error = f"Connection error: {str(e)}"
            logger.error(f"PLC connection failed: {self.last_error}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from PLC"""
        if not snap7_available or self.client is None:
            return
        try:
            if self.connected and self.client:
                self.client.disconnect()
                self.connected = False
                logger.info("Disconnected from PLC")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")

    def is_connected(self) -> bool:
        """Check if connected to PLC - gracefully handles errors"""
        if not snap7_available or self.client is None:
            return False
        try:
            return self.connected and self.client.get_connected()
        except Exception as e:
            # If check fails, assume disconnected
            logger.debug(f"PLC connection check failed: {e}")
            self.connected = False
            return False

    def read_db_real(self, db_number: int, offset: int) -> Optional[float]:
        """Read REAL (float) value from data block"""
        if not snap7_available or self.client is None:
            return None
        try:
            if not self.is_connected():
                return None

            data = self.client.db_read(db_number, offset, 4)
            return get_real(data, 0)
        except Exception as e:
            self.last_error = f"Error reading DB{db_number}.DBD{offset}: {str(e)}"
            logger.error(self.last_error)
            return None

    def write_db_real(self, db_number: int, offset: int, value: float) -> bool:
        """Write REAL (float) value to data block"""
        if not snap7_available or self.client is None:
            return False
        try:
            if not self.is_connected():
                return False

            data = bytearray(4)
            set_real(data, 0, value)
            self.client.db_write(db_number, offset, data)
            return True
        except Exception as e:
            self.last_error = f"Error writing DB{db_number}.DBD{offset}: {str(e)}"
            logger.error(self.last_error)
            return False

    def read_db_bool(self, db_number: int, byte_offset: int, bit_offset: int) -> Optional[bool]:
        """Read BOOL value from data block (thread-safe)"""
        if not snap7_available or self.client is None:
            return None
        try:
            if not self.is_connected():
                return None

            # Thread-safe: Only one Snap7 operation at a time
            with self.plc_lock:
                time.sleep(0.02)  # 20ms delay to avoid flooding
                data = self.client.db_read(db_number, byte_offset, 1)
                return get_bool(data, 0, bit_offset)
        except Exception as e:
            self.last_error = f"Error reading DB{db_number}.DBX{byte_offset}.{bit_offset}: {str(e)}"
            logger.error(self.last_error)
            return None

    def write_db_bool(self, db_number: int, byte_offset: int, bit_offset: int, value: bool) -> bool:
        """Write BOOL value to data block (thread-safe)"""
        if not snap7_available or self.client is None:
            return False
        try:
            if not self.is_connected():
                return False

            # Thread-safe: Only one Snap7 operation at a time
            with self.plc_lock:
                time.sleep(0.02)  # 20ms delay to avoid flooding
                # Read-modify-write for bit operations
                data = bytearray(self.client.db_read(db_number, byte_offset, 1))
                set_bool(data, 0, bit_offset, value)
                self.client.db_write(db_number, byte_offset, data)
                return True
        except Exception as e:
            self.last_error = f"Error writing DB{db_number}.DBX{byte_offset}.{bit_offset}: {str(e)}"
            logger.error(self.last_error)
            return False

    def read_m_bit(self, byte_offset: int, bit_offset: int) -> Optional[bool]:
        """Read Merker (M memory) bit"""
        if not snap7_available or self.client is None:
            return None
        try:
            if not self.is_connected():
                return None

            data = self.client.mb_read(byte_offset, 1)
            return get_bool(data, 0, bit_offset)
        except Exception as e:
            self.last_error = f"Error reading M{byte_offset}.{bit_offset}: {str(e)}"
            logger.error(self.last_error)
            return None

    def write_m_bit(self, byte_offset: int, bit_offset: int, value: bool) -> bool:
        """Write Merker (M memory) bit"""
        if not snap7_available or self.client is None:
            return False
        try:
            if not self.is_connected():
                return False

            # Read-modify-write
            data = bytearray(self.client.mb_read(byte_offset, 1))
            set_bool(data, 0, bit_offset, value)
            self.client.mb_write(byte_offset, data)
            return True
        except Exception as e:
            self.last_error = f"Error writing M{byte_offset}.{bit_offset}: {str(e)}"
            logger.error(self.last_error)
            return False

    # High-level methods for Dobot robot control

    def read_target_pose(self, db_number: int = 123) -> Dict[str, float]:
        """Read target X, Y, Z position from PLC (offset 0, 4, 8) in one operation"""
        if not snap7_available or self.client is None:
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}
        try:
            if not self.is_connected():
                return {'x': 0.0, 'y': 0.0, 'z': 0.0}

            # Read all 3 REAL values (12 bytes total) in one operation
            data = self.client.db_read(db_number, 0, 12)
            return {
                'x': get_real(data, 0),
                'y': get_real(data, 4),
                'z': get_real(data, 8)
            }
        except Exception as e:
            self.last_error = f"Error reading target pose from DB{db_number}: {str(e)}"
            logger.error(self.last_error)
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}

    def read_current_pose(self, db_number: int = 123) -> Dict[str, float]:
        """Read current X, Y, Z position from PLC (offset 12, 16, 20) in one operation"""
        if not snap7_available or self.client is None:
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}
        try:
            if not self.is_connected():
                return {'x': 0.0, 'y': 0.0, 'z': 0.0}

            # Read all 3 REAL values (12 bytes total) in one operation
            data = self.client.db_read(db_number, 12, 12)
            return {
                'x': get_real(data, 0),
                'y': get_real(data, 4),
                'z': get_real(data, 8)
            }
        except Exception as e:
            self.last_error = f"Error reading current pose from DB{db_number}: {str(e)}"
            logger.error(self.last_error)
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}

    def write_current_pose(self, pose: Dict[str, float], db_number: int = 123) -> bool:
        """Write current X, Y, Z position to PLC (offset 12, 16, 20) in one operation"""
        if not snap7_available or self.client is None:
            return False
        try:
            if not self.is_connected():
                return False

            # Write all 3 REAL values (12 bytes total) in one operation
            data = bytearray(12)
            set_real(data, 0, pose.get('x', 0.0))
            set_real(data, 4, pose.get('y', 0.0))
            set_real(data, 8, pose.get('z', 0.0))
            self.client.db_write(db_number, 12, data)
            return True
        except Exception as e:
            self.last_error = f"Error writing current pose to DB{db_number}: {str(e)}"
            logger.error(self.last_error)
            return False

    def read_control_bits(self) -> Dict[str, bool]:
        """Read all control bits from M0.0 - M0.7 in one operation"""
        if not snap7_available or self.client is None:
            return {
                'start': False, 'stop': False, 'home': False, 'estop': False,
                'suction': False, 'ready': False, 'busy': False, 'error': False
            }

        try:
            if not self.is_connected():
                return {
                    'start': False, 'stop': False, 'home': False, 'estop': False,
                    'suction': False, 'ready': False, 'busy': False, 'error': False
                }

            # Read entire byte M0 at once (contains all 8 bits)
            data = self.client.mb_read(0, 1)
            byte_value = data[0]

            # Extract individual bits from the byte
            return {
                'start': bool((byte_value >> 0) & 1),
                'stop': bool((byte_value >> 1) & 1),
                'home': bool((byte_value >> 2) & 1),
                'estop': bool((byte_value >> 3) & 1),
                'suction': bool((byte_value >> 4) & 1),
                'ready': bool((byte_value >> 5) & 1),
                'busy': bool((byte_value >> 6) & 1),
                'error': bool((byte_value >> 7) & 1)
            }
        except Exception as e:
            error_str = str(e)
            self.last_error = f"Error reading control bits: {error_str}"
            logger.debug(self.last_error)
            return {
                'start': False, 'stop': False, 'home': False, 'estop': False,
                'suction': False, 'ready': False, 'busy': False, 'error': False
            }

    def write_control_bit(self, bit_name: str, value: bool) -> bool:
        """Write a single control bit"""
        bit_map = {
            'start': (0, 0),
            'stop': (0, 1),
            'home': (0, 2),
            'estop': (0, 3),
            'suction': (0, 4),
            'ready': (0, 5),
            'busy': (0, 6),
            'error': (0, 7)
        }

        if bit_name not in bit_map:
            return False

        byte_offset, bit_offset = bit_map[bit_name]
        return self.write_m_bit(byte_offset, bit_offset, value)

    def get_status(self) -> Dict[str, Any]:
        """Get current PLC connection status"""
        try:
            # Use cached connection status to avoid blocking
            connected = self.connected
            # Only check actual connection if we think we're connected (quick check)
            if connected and self.client is not None:
                try:
                    # Quick non-blocking check with timeout
                    connected = self.client.get_connected()
                    self.connected = connected
                except Exception:
                    # If check fails, assume disconnected
                    self.connected = False
                    connected = False
        except Exception as e:
            logger.debug(f"Error checking connection status: {e}")
            connected = self.connected  # Fall back to cached value
        
        return {
            'connected': connected,
            'ip': self.ip,
            'rack': self.rack,
            'slot': self.slot,
            'last_error': self.last_error
        }

    # DB123 Vision System Tags Methods
    
    def read_db_int(self, db_number: int, offset: int) -> Optional[int]:
        """Read INT (16-bit signed integer) value from data block (thread-safe)"""
        if not snap7_available or self.client is None:
            return None
        try:
            if not self.is_connected():
                return None

            # Thread-safe: Only one Snap7 operation at a time
            # Use timeout to prevent deadlock with polling thread
            if not self.plc_lock.acquire(timeout=2.0):
                logger.warning(f"read_db_int: Failed to acquire PLC lock within 2 seconds for DB{db_number}.DBW{offset}")
                return None

            try:
                time.sleep(0.02)  # 20ms delay to avoid flooding
                data = self.client.db_read(db_number, offset, 2)
                return get_int(data, 0)
            finally:
                self.plc_lock.release()
        except Exception as e:
            self.last_error = f"Error reading DB{db_number}.DBW{offset}: {str(e)}"
            logger.error(self.last_error)
            return None

    def write_db_int(self, db_number: int, offset: int, value: int) -> bool:
        """Write INT (16-bit signed integer) value to data block"""
        if not snap7_available or self.client is None:
            return False
        try:
            if not self.is_connected():
                return False

            # Thread-safe: Only one Snap7 operation at a time
            with self.plc_lock:
                time.sleep(0.02)  # 20ms delay to avoid flooding
                data = bytearray(2)
                # snap7 uses set_int to write a 16-bit signed integer
                set_int(data, 0, value)
                self.client.db_write(db_number, offset, data)
                return True
        except Exception as e:
            self.last_error = f"Error writing DB{db_number}.DBW{offset}: {str(e)}"
            logger.error(self.last_error)
            return False

    def read_vision_tags(self, db_number: int = 123) -> Dict[str, Any]:
        """Read all vision system tags from DB123 (thread-safe)"""
        if not snap7_available or self.client is None:
            return {
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
        try:
            if not self.is_connected():
                return {
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

            # Thread-safe: Only one Snap7 operation at a time
            # Use timeout to prevent deadlock with polling thread
            if not self.plc_lock.acquire(timeout=2.0):
                logger.warning("read_vision_tags: Failed to acquire PLC lock within 2 seconds - returning default values")
                return {
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

            try:
                time.sleep(0.02)  # 20ms delay to avoid flooding
                # Read byte 40 (contains all bool flags)
                bool_data = self.client.db_read(db_number, 40, 1)
            finally:
                self.plc_lock.release()
            
            # Read INT values separately (outside lock to avoid holding it too long)
            # These use their own locks internally
            object_number = self.read_db_int(db_number, 42)
            defect_number = self.read_db_int(db_number, 44)
            
            return {
                'start': get_bool(bool_data, 0, 0),          # 40.0
                'connected': get_bool(bool_data, 0, 1),     # 40.1
                'busy': get_bool(bool_data, 0, 2),          # 40.2
                'completed': get_bool(bool_data, 0, 3),     # 40.3 (NEW)
                'object_detected': get_bool(bool_data, 0, 4),  # 40.4 (was 40.3)
                'object_ok': get_bool(bool_data, 0, 5),     # 40.5 (was 40.4)
                'defect_detected': get_bool(bool_data, 0, 6),  # 40.6 (was 40.5)
                'object_number': object_number if object_number is not None else 0,
                'defect_number': defect_number if defect_number is not None else 0
            }
        except Exception as e:
            self.last_error = f"Error reading vision tags from DB{db_number}: {str(e)}"
            logger.error(self.last_error)
            return {
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

    def write_vision_tags(self, tags: Dict[str, Any], db_number: int = 123) -> bool:
        """Write vision system tags to DB123 with retry logic for "Job pending" errors
        
        Args:
            tags: Dictionary with keys: connected, busy, completed, object_detected, 
                  object_ok, defect_detected, object_number, defect_number
                  NOTE: 'start' is READ-ONLY - only PLC can write to it (40.0)
            db_number: Data block number (default 123)
            
        Address mapping:
            - Start: 40.0 (READ-ONLY - PLC controlled)
            - Connected: 40.1
            - Busy: 40.2
            - Completed: 40.3
            - Object_Detected: 40.4
            - Object_OK: 40.5
            - Defect_Detected: 40.6
            - Object_Number: 42.0 (INT)
            - Defect_Number: 44.0 (INT)
        """
        if not snap7_available or self.client is None:
            return False

        # Thread-safe: Only one Snap7 operation at a time
        with self.plc_lock:
            try:
                if not self.is_connected():
                    return False

                # Small delay to avoid flooding PLC
                time.sleep(0.02)  # 20ms delay

                # Read current byte 40 to preserve other bits (especially Start bit)
                current_byte = bytearray(self.client.db_read(db_number, 40, 1))

                # Set individual bits (updated addresses with Completed at 40.3)
                # NOTE: 'start' bit (40.0) is READ-ONLY - only PLC can write to it
                # We preserve the existing Start bit value by not modifying it
                if 'connected' in tags:
                    set_bool(current_byte, 0, 1, bool(tags['connected']))  # 40.1
                if 'busy' in tags:
                    set_bool(current_byte, 0, 2, bool(tags['busy']))  # 40.2
                if 'completed' in tags:
                    set_bool(current_byte, 0, 3, bool(tags['completed']))  # 40.3 (NEW)
                if 'object_detected' in tags:
                    set_bool(current_byte, 0, 4, bool(tags['object_detected']))  # 40.4 (was 40.3)
                if 'object_ok' in tags:
                    set_bool(current_byte, 0, 5, bool(tags['object_ok']))  # 40.5 (was 40.4)
                if 'defect_detected' in tags:
                    set_bool(current_byte, 0, 6, bool(tags['defect_detected']))  # 40.6 (was 40.5)

                # Write byte 40 with all bool flags
                self.client.db_write(db_number, 40, current_byte)

                # Write INT values (these use internal locks, so no additional delay needed)
                if 'object_number' in tags:
                    self.write_db_int(db_number, 42, int(tags['object_number']))

                if 'defect_number' in tags:
                    self.write_db_int(db_number, 44, int(tags['defect_number']))

                return True

            except Exception as e:
                error_str = str(e)
                self.last_error = f"Error writing vision tags to DB{db_number}: {error_str}"
                logger.debug(self.last_error)
                return False

    # ==================================================
    # High-level Vision System Methods
    # ==================================================
    
    def write_vision_detection_results(self, object_count: int, defect_count: int, 
                                       object_ok: bool, defect_detected: bool, 
                                       busy: bool = False, completed: bool = False,
                                       db_number: int = 123) -> bool:
        """Write vision detection results to PLC DB123 tags
        
        This is a high-level method that combines all vision system data into one call.
        
        Args:
            object_count: Number of objects detected
            defect_count: Number of defects found
            object_ok: Whether objects are OK (no defects)
            defect_detected: Whether any defects were detected
            busy: Whether vision system is currently processing
            completed: Whether vision processing is completed
            db_number: Data block number (default 123)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        # Determine tag values
        connected = self.is_connected()
        object_detected = object_count > 0
        object_number = object_count
        defect_number = defect_count
        
        # Prepare tags dictionary
        tags = {
            'connected': connected,
            'busy': busy,
            'completed': completed,
            'object_detected': object_detected,
            'object_ok': object_ok,
            'defect_detected': defect_detected,
            'object_number': object_number,
            'defect_number': defect_number
        }
        
        # Add small delay before writing to avoid "Job pending" if polling just ran
        time.sleep(0.1)
        
        # Write using the unified write_vision_tags method
        success = self.write_vision_tags(tags, db_number)
        if success:
            logger.debug(f"Vision detection results written to DB{db_number}: {tags}")
        return success

    def read_vision_start_command(self, db_number: int = 123) -> bool:
        """Read Start command from PLC (DB123.DBX40.0) with basic stability filtering

        Simplified version with minimal retries to prevent "Job pending" errors.
        Uses simple debouncing: requires 2 consecutive matching reads to change state.

        Args:
            db_number: Data block number (default 123)

        Returns:
            True if Start command is active, False otherwise (filtered value)
        """
        if not self.is_connected():
            logger.debug("Cannot read Start command - PLC not connected")
            # Return last stable value if available, otherwise False
            return self.start_command_stable_value if self.start_command_stable_value is not None else False

        # Thread-safe: Only one Snap7 operation at a time
        with self.plc_lock:
            try:
                # Single read with small delay to avoid flooding
                time.sleep(0.02)  # 20ms delay
                bool_data = self.client.db_read(db_number, 40, 1)
                start_value = get_bool(bool_data, 0, 0)  # Bit 0 = Start

                # Add to history for stability check
                self.start_command_history.append(start_value)

                # Keep history size limited to 3 readings
                if len(self.start_command_history) > 3:
                    self.start_command_history.pop(0)

                # Initialize stable value on first read
                if self.start_command_stable_value is None:
                    self.start_command_stable_value = start_value
                    return start_value

                # Simple debouncing: require 2 consecutive matching reads to change state
                if len(self.start_command_history) >= 2:
                    last_two = self.start_command_history[-2:]
                    if last_two[0] == last_two[1] and last_two[1] != self.start_command_stable_value:
                        # State change confirmed by 2 consecutive reads
                        logger.info(f"Start command state changed: {self.start_command_stable_value} -> {last_two[1]}")
                        self.start_command_stable_value = last_two[1]

                return self.start_command_stable_value

            except Exception as e:
                error_str = str(e)
                logger.debug(f"Error reading Start command: {error_str}")
                # Return last stable value on error (prevents false triggers)
                return self.start_command_stable_value if self.start_command_stable_value is not None else False
    
    def write_vision_fault_bit(self, defects_found: bool, byte_offset: int = 1, bit_offset: int = 0) -> Dict[str, Any]:
        """Write vision fault status to PLC memory bit
        
        Args:
            defects_found: True if defects found, False if no defects
            byte_offset: M memory byte offset (default 1)
            bit_offset: M memory bit offset (default 0)
        
        Returns:
            Dictionary with write status and details
        """
        if not self.is_connected():
            return {'written': False, 'reason': 'plc_not_connected'}
        
        try:
            success = self.write_m_bit(byte_offset, bit_offset, defects_found)
            if success:
                logger.info(f"Vision fault bit M{byte_offset}.{bit_offset} set to {defects_found}")
                return {'written': True, 'address': f'M{byte_offset}.{bit_offset}', 'value': defects_found}
            else:
                logger.debug(f"Failed to write vision fault bit M{byte_offset}.{bit_offset}")
                return {'written': False, 'reason': 'write_failed', 'address': f'M{byte_offset}.{bit_offset}'}
        except Exception as e:
            logger.debug(f"Error writing vision fault bit: {e}")
            return {'written': False, 'reason': 'write_error', 'error': str(e)}