# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **PLC Handshaking System**: Implemented full handshaking protocol between Raspberry Pi and Siemens S7-1200 PLC
  - Vision system now waits for Start command (DB123.DBX40.0) from PLC before processing
  - Automatically processes vision detection when Start command is received
  - Sets Completed flag (DB123.DBX40.3) when processing is finished
  - Resets Completed flag when Start command is released
  - Processing runs in background thread to avoid blocking PLC polling

- **Unified PLC Communication Module**: Consolidated all S7 communication into `plc_client.py`
  - All PLC read/write operations now centralized in one module
  - Added `read_vision_start_command()` method to read Start command from PLC
  - Added `write_vision_detection_results()` high-level method for vision results
  - Added `write_vision_fault_bit()` method for fault status reporting
  - Improved code organization and maintainability

- **DB123 Vision Tags Support**: Full support for vision system tags in DB123
  - Added `read_vision_tags()` method to read all vision tags from DB123
  - Added `write_vision_tags()` method to write vision tags with retry logic
  - Support for all DB123 tags: Start, Connected, Busy, Completed, Object_Detected, Object_OK, Defect_Detected, Object_Number, Defect_Number

- **Retry Logic for PLC Communication**: Implemented automatic retry mechanism
  - Handles "Job pending" errors from S7-1200 PLC automatically
  - 3 retry attempts with 200ms delay between retries
  - Prevents communication failures due to transient PLC busy states
  - Applied to all read/write operations (BOOL, INT, REAL, M bits)

- **Completed Flag**: New tag at DB123.DBX40.3
  - Indicates when vision processing cycle is complete
  - Automatically set after object detection, image saving, and defect checking
  - Automatically reset when Start command goes low

### Changed
- **Address Mapping Updates**: Updated DB123 tag addresses to accommodate Completed flag
  - Completed: 40.3 (NEW)
  - Object_Detected: 40.4 (was 40.3)
  - Object_OK: 40.5 (was 40.4)
  - Defect_Detected: 40.6 (was 40.5)
  - Object_Number: 42.0 (unchanged)
  - Defect_Number: 44.0 (unchanged)

- **PLC Communication Architecture**: Refactored to use unified PLC client
  - All S7 communication now goes through `plc_client.py`
  - `app.py` wrapper functions call unified PLC client methods
  - Improved error handling and logging
  - Better separation of concerns

- **Vision Detection Flow**: Updated to support handshaking
  - Vision processing now triggered by PLC Start command
  - Automatic image saving and defect detection during handshake cycle
  - Results automatically written to PLC tags
  - Busy flag properly managed during processing

- **Default DB Number**: Changed from DB1 to DB123
  - All PLC operations now default to DB123
  - Configurable via `config.json`
  - Prevents "Address out of range" errors

- **Network Configuration**: Updated PLC IP address
  - Default PLC IP: 192.168.7.2
  - Raspberry Pi IP: 192.168.7.5
  - Network routing configured for PLC subnet access

### Fixed
- **"Job pending" Errors**: Fixed frequent PLC communication errors
  - Added retry logic with delays
  - Added small delays between consecutive writes
  - Prevents overwhelming S7-1200 PLC with rapid requests

- **"Address out of range" Errors**: Fixed DB1 access errors
  - Changed default DB number to DB123
  - Made DB number configurable
  - Suppressed errors when DB1 is not available

- **Duplicate Method Definitions**: Removed duplicate `write_vision_detection_results()` method
  - Consolidated into single implementation
  - Improved code clarity

- **Frontend JavaScript Errors**: Fixed undefined function errors
  - Fixed `handleStreamError` and `handleStreamLoad` event handlers
  - Fixed `plc_client` undefined error in frontend
  - Updated to use proper API endpoints

### Technical Details

#### PLC Communication Methods (`plc_client.py`)
- **Connection Management**:
  - `connect()` - Connect to PLC with retry logic
  - `disconnect()` - Disconnect from PLC
  - `is_connected()` - Check connection status
  - `get_status()` - Get connection status info

- **Low-Level Data Block Operations**:
  - `read_db_real()` - Read REAL (float) values
  - `write_db_real()` - Write REAL values
  - `read_db_bool()` - Read BOOL values
  - `write_db_bool()` - Write BOOL values
  - `read_db_int()` - Read INT values
  - `write_db_int()` - Write INT values

- **Memory (M) Operations**:
  - `read_m_bit()` - Read Merker bit
  - `write_m_bit()` - Write Merker bit

- **Robot Control Methods**:
  - `read_target_pose()` - Read target X, Y, Z position
  - `read_current_pose()` - Read current X, Y, Z position
  - `write_current_pose()` - Write current position
  - `read_control_bits()` - Read all control bits (M0.0-M0.7)
  - `write_control_bit()` - Write single control bit

- **Vision System Methods**:
  - `read_vision_tags()` - Read all DB123 vision tags
  - `write_vision_tags()` - Write DB123 vision tags (with retry logic)
  - `read_vision_start_command()` - Read Start command from PLC
  - `write_vision_detection_results()` - High-level method for vision results
  - `write_vision_fault_bit()` - Write vision fault bit to M memory

#### Handshaking Flow
1. PLC sets Start (DB123.DBX40.0) = True
2. Raspberry Pi detects rising edge in polling loop
3. Pi sets Busy (DB123.DBX40.2) = True
4. Pi processes vision:
   - Detects objects using YOLO
   - Saves counter images
   - Checks for defects
   - Writes results to PLC tags
5. Pi sets Completed (DB123.DBX40.3) = True, Busy = False
6. PLC reads Completed = True
7. PLC resets Start = False
8. Pi detects falling edge and resets Completed = False

#### Configuration (`config.json`)
```json
{
  "plc": {
    "ip": "192.168.7.2",
    "db_number": 123,
    "db123": {
      "enabled": true,
      "db_number": 123,
      "tags": {
        "start": {"byte": 40, "bit": 0},
        "connected": {"byte": 40, "bit": 1},
        "busy": {"byte": 40, "bit": 2},
        "completed": {"byte": 40, "bit": 3},
        "object_detected": {"byte": 40, "bit": 4},
        "object_ok": {"byte": 40, "bit": 5},
        "defect_detected": {"byte": 40, "bit": 6},
        "object_number": {"byte": 42},
        "defect_number": {"byte": 44}
      }
    }
  }
}
```

## Notes
- All S7 communication is now centralized in `plc_client.py` for better code organization
- Handshaking runs automatically in the background polling loop
- Vision processing is triggered by PLC Start command, ensuring synchronization
- Retry logic prevents communication failures due to transient PLC states
- Address mappings updated to accommodate new Completed flag

