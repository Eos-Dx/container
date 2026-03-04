"""Container Management - Locking, Archiving, and Status Control

This module provides functions for managing technical container lifecycle:
- Locking containers (HDF5 attribute + OS read-only)
- Archiving old containers when creating new ones
- Finding active containers by distance
- Marking measurements as primary/supplementary
"""

import logging
import os
import shutil
import stat
import time
import zipfile
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

import h5py

from . import schema

logger = logging.getLogger(__name__)


def _decode_attr(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe_archive_token(value: str, fallback: str = "unknown") -> str:
    token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (value or ""))
    token = token.strip("_")
    return token or fallback


def _resolve_archive_operator_token(container_path: Path) -> str:
    try:
        with h5py.File(container_path, "r") as f:
            operator = _decode_attr(f.attrs.get(schema.ATTR_OPERATOR_ID))
            if operator:
                return _safe_archive_token(operator)

            locked_by = _decode_attr(f.attrs.get("locked_by"))
            if locked_by:
                return _safe_archive_token(locked_by)

            runtime = f.get(schema.GROUP_RUNTIME)
            if runtime is not None:
                runtime_locked_by = _decode_attr(runtime.attrs.get(schema.ATTR_LOCKED_BY))
                if runtime_locked_by:
                    return _safe_archive_token(runtime_locked_by)
    except Exception:
        pass

    return "unknown"


def _set_container_lifecycle_state(
    container_file: Path,
    *,
    locked: Optional[bool] = None,
    lock_timestamp: Optional[str] = None,
    locked_by: Optional[str] = None,
    transfer_status: Optional[str] = None,
    transfer_timestamp: Optional[str] = None,
    clear_lock_metadata: bool = False,
    clear_transfer_timestamp: bool = False,
) -> None:
    """Mirror lifecycle state attrs at root and runtime level."""
    with h5py.File(container_file, "a") as f:
        runtime = f.get(schema.GROUP_RUNTIME)

        def _apply_attrs(target) -> None:
            if target is None:
                return

            if locked is not None:
                target.attrs[schema.ATTR_LOCKED] = bool(locked)
                target.attrs[schema.ATTR_LOCK_STATUS] = (
                    schema.LOCK_STATUS_LOCKED if locked else schema.LOCK_STATUS_UNLOCKED
                )
                if lock_timestamp is not None:
                    target.attrs[schema.ATTR_LOCKED_TIMESTAMP] = lock_timestamp
                    # Keep legacy root attrs for older callers/tests.
                    if target is f:
                        target.attrs["locked_timestamp"] = lock_timestamp
                if locked_by:
                    target.attrs[schema.ATTR_LOCKED_BY] = locked_by
                    if target is f:
                        target.attrs["locked_by"] = locked_by
                elif clear_lock_metadata:
                    for key in (schema.ATTR_LOCKED_BY, "locked_by"):
                        if key in target.attrs:
                            del target.attrs[key]

                if clear_lock_metadata:
                    for key in (schema.ATTR_LOCKED_TIMESTAMP, "locked_timestamp"):
                        if key in target.attrs:
                            del target.attrs[key]

            if transfer_status is not None:
                target.attrs[schema.ATTR_TRANSFER_STATUS] = str(transfer_status)
                if transfer_timestamp is not None:
                    target.attrs[schema.ATTR_TRANSFER_TIMESTAMP] = str(transfer_timestamp)
                elif clear_transfer_timestamp and schema.ATTR_TRANSFER_TIMESTAMP in target.attrs:
                    del target.attrs[schema.ATTR_TRANSFER_TIMESTAMP]

        _apply_attrs(f)
        _apply_attrs(runtime)


def _run_with_write_access(container_file: Path, callback) -> None:
    """Temporarily add user-write permission when updating a locked container."""
    current_perms = container_file.stat().st_mode
    restore_perms = None

    if not (current_perms & stat.S_IWUSR):
        restore_perms = current_perms
        os.chmod(container_file, current_perms | stat.S_IWUSR)

    try:
        callback()
    finally:
        if restore_perms is not None:
            os.chmod(container_file, restore_perms)


# ==================== Container Locking ====================

def is_container_locked(container_file: Path) -> bool:
    """Check if container is locked (closed).
    
    A locked container:
    - Has 'locked' attribute set to True in HDF5
    - Should be treated as read-only reference
    - Can be used by multiple session containers
    - Must be archived before creating new container at same distance
    
    Args:
        container_file: Path to container file
        
    Returns:
        True if locked, False otherwise
    """
    try:
        with h5py.File(container_file, 'r') as f:
            runtime = f.get(schema.GROUP_RUNTIME)
            if runtime is not None:
                return bool(runtime.attrs.get(schema.ATTR_LOCKED, f.attrs.get('locked', False)))
            return bool(f.attrs.get('locked', False))
    except Exception as e:
        logger.error(f"Could not check lock status for {container_file}: {e}")
        return False


def get_transfer_status(container_file: Path) -> str:
    """Return explicit transfer status for a container."""
    try:
        with h5py.File(container_file, "r") as f:
            runtime = f.get(schema.GROUP_RUNTIME)
            if runtime is not None and schema.ATTR_TRANSFER_STATUS in runtime.attrs:
                return _decode_attr(runtime.attrs.get(schema.ATTR_TRANSFER_STATUS)) or schema.TRANSFER_STATUS_UNSENT
            return _decode_attr(
                f.attrs.get(schema.ATTR_TRANSFER_STATUS, schema.TRANSFER_STATUS_UNSENT)
            ) or schema.TRANSFER_STATUS_UNSENT
    except Exception as e:
        logger.error(f"Could not read transfer status for {container_file}: {e}")
        return schema.TRANSFER_STATUS_UNSENT


def mark_container_transferred(
    container_file: Path,
    sent: bool,
    *,
    timestamp: Optional[str] = None,
) -> None:
    """Mark container as sent or unsent."""
    container_file = Path(container_file)
    if not container_file.exists():
        raise FileNotFoundError(f"Container not found: {container_file}")

    transfer_status = (
        schema.TRANSFER_STATUS_SENT if sent else schema.TRANSFER_STATUS_UNSENT
    )
    transfer_timestamp = timestamp or (time.strftime("%Y-%m-%d %H:%M:%S") if sent else None)
    _run_with_write_access(
        container_file,
        lambda: _set_container_lifecycle_state(
            container_file,
            transfer_status=transfer_status,
            transfer_timestamp=transfer_timestamp,
            clear_transfer_timestamp=not sent,
        ),
    )


def lock_container(container_file: Path, user_id: Optional[str] = None) -> None:
    """Lock container (make it read-only reference).
    
    Locking consists of two steps:
    1. Set 'locked' attribute in HDF5 file
    2. Set OS file permissions to read-only
    
    Once locked:
    - Container becomes read-only
    - Can be referenced by multiple sessions
    - Cannot be modified
    - Must be archived if new container created at same distance
    
    Args:
        container_file: Path to container file
        user_id: Optional user ID who locked it
        
    Raises:
        RuntimeError: If container already locked or cannot be locked
    """
    container_file = Path(container_file)
    
    if not container_file.exists():
        raise FileNotFoundError(f"Container not found: {container_file}")
    
    if is_container_locked(container_file):
        raise RuntimeError(f"Container already locked: {container_file}")
    
    # Step 1: Set HDF5 attribute
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        _set_container_lifecycle_state(
            container_file,
            locked=True,
            lock_timestamp=timestamp,
            locked_by=user_id,
        )
        
        logger.info(f"Set locked attribute for: {container_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to set locked attribute: {e}")
    
    # Step 2: Set OS read-only permissions
    try:
        # Get current permissions
        current_perms = container_file.stat().st_mode
        
        # Remove write permissions for user, group, others
        # Keep read and execute permissions
        read_only_perms = current_perms & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        
        # Apply new permissions
        os.chmod(container_file, read_only_perms)
        
        logger.info(f"Set OS read-only permissions for: {container_file}")
    except Exception as e:
        logger.warning(f"Could not set OS read-only permissions: {e}")
        # Don't fail if OS permissions can't be set (e.g., network drive)
    
    logger.info(f"✓ Locked container: {container_file}")


def unlock_container(container_file: Path) -> None:
    """Unlock container (for administrative purposes only).
    
    WARNING: This should only be used in special cases, not normal workflow.
    
    Args:
        container_file: Path to container file
    """
    container_file = Path(container_file)
    
    if not container_file.exists():
        raise FileNotFoundError(f"Container not found: {container_file}")
    
    # Step 1: Remove OS read-only
    try:
        current_perms = container_file.stat().st_mode
        writable_perms = current_perms | stat.S_IWUSR
        os.chmod(container_file, writable_perms)
        logger.info(f"Removed OS read-only for: {container_file}")
    except Exception as e:
        logger.warning(f"Could not remove OS read-only: {e}")
    
    # Step 2: Remove HDF5 locked attribute
    try:
        _set_container_lifecycle_state(
            container_file,
            locked=False,
            clear_lock_metadata=True,
        )
        
        logger.info(f"Removed locked attribute for: {container_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to remove locked attribute: {e}")
    
    logger.warning(f"⚠ Unlocked container (administrative): {container_file}")


# ==================== Archive Management ====================

def create_container_bundle(
    container_path: Path,
    source_folder: Optional[Path] = None,
    output_zip: Optional[Path] = None,
    include_patterns: Optional[list] = None,
    source_arcname: Optional[str] = None,
) -> Path:
    """Create ZIP bundle with container and optional operator folder structure.

    The container file is stored at ZIP root and source folder contents are
    preserved using relative paths under ``source_arcname``.

    Args:
        container_path: Path to .h5 container file
        source_folder: Optional folder containing operator-created structure/files
        output_zip: Optional output ZIP path (defaults to sibling ``<stem>.zip``)
        include_patterns: Optional glob patterns for source files
        source_arcname: Optional top-level folder name inside ZIP for source files

    Returns:
        Path to created ZIP bundle
    """
    container_path = Path(container_path)
    if not container_path.exists():
        raise FileNotFoundError(f"Container not found: {container_path}")

    if output_zip is None:
        output_zip = container_path.with_suffix(".zip")
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    selected_patterns = include_patterns or ["*"]
    source_root = Path(source_folder) if source_folder else None
    arc_root = source_arcname or (source_root.name if source_root else None)

    def _include_relative_path(relative_path: Path) -> bool:
        relative_str = relative_path.as_posix()
        base_name = relative_path.name
        for pattern in selected_patterns:
            normalized_pattern = str(pattern).replace("\\", "/")
            if fnmatch(base_name, normalized_pattern) or fnmatch(
                relative_str, normalized_pattern
            ):
                return True
        return False

    with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(container_path, arcname=container_path.name)

        if source_root and source_root.exists():
            for candidate in sorted(source_root.rglob("*")):
                if not candidate.is_file():
                    continue
                if candidate.resolve() == output_zip.resolve():
                    continue
                relative_path = candidate.relative_to(source_root)
                if not _include_relative_path(relative_path):
                    continue

                if arc_root:
                    arcname = (Path(arc_root) / relative_path).as_posix()
                else:
                    arcname = relative_path.as_posix()
                zf.write(candidate, arcname=arcname)

    logger.info("Created container bundle ZIP: %s", output_zip)
    return output_zip

def archive_technical_container(
    folder: Path,
    tech_file: Path,
    user_confirmed: bool = False
) -> Path:
    """Move locked technical container to archive.
    
    Args:
        folder: Base folder containing technical containers
        tech_file: Technical container to archive
        user_confirmed: Must be True to proceed (safety check)
        
    Returns:
        Path to archived file
        
    Raises:
        RuntimeError: If user confirmation not provided or container not locked
    """
    tech_file = Path(tech_file)
    folder = Path(folder)
    
    if not user_confirmed:
        raise RuntimeError(
            f"Cannot archive technical container without user confirmation.\n"
            f"Container: {tech_file}\n"
            f"Set user_confirmed=True to proceed."
        )
    
    if not tech_file.exists():
        raise FileNotFoundError(f"Container not found: {tech_file}")
    
    if not is_container_locked(tech_file):
        raise RuntimeError(
            f"Cannot archive unlocked container: {tech_file}\n"
            f"Lock it first using lock_container() before archiving."
        )
    
    # Create archive directory
    archive_dir = folder / 'archive'
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate archived filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    operator_token = _resolve_archive_operator_token(tech_file)
    archived_filename = (
        f"{tech_file.stem}_archived_{operator_token}_{timestamp}{tech_file.suffix}"
    )
    archived_path = archive_dir / archived_filename
    
    # Move file to archive
    try:
        shutil.move(str(tech_file), str(archived_path))
        logger.info(f"✓ Archived: {tech_file.name} → archive/{archived_filename}")
        return archived_path
    except Exception as e:
        raise RuntimeError(f"Failed to archive container: {e}")


def find_active_technical_container(
    folder: Path,
    distance_cm: float,
    tolerance_cm: float = 0.5
) -> Optional[Path]:
    """Find active (non-archived) technical container for given distance.
    
    Args:
        folder: Folder to search
        distance_cm: Required distance in cm
        tolerance_cm: Distance matching tolerance (default 0.5 cm)
        
    Returns:
        Path to active container, or None if not found
    """
    folder = Path(folder)
    
    if not folder.exists():
        return None
    
    # Search for technical containers (excluding archive subdirectory)
    for tech_file in folder.glob("technical_*.nxs.h5"):
        # Skip archived containers
        if 'archive' in str(tech_file.parent):
            continue
        
        # Read distance from container
        try:
            with h5py.File(tech_file, 'r') as f:
                file_distance = f.attrs.get('distance_cm', 0.0)
            
            # Check if distance matches (within tolerance)
            if abs(file_distance - distance_cm) <= tolerance_cm:
                logger.info(
                    f"Found active technical container: {tech_file.name} "
                    f"(distance: {file_distance:.1f} cm)"
                )
                return tech_file
        
        except Exception as e:
            logger.warning(f"Could not read {tech_file}: {e}")
            continue
    
    return None


# ==================== Primary/Supplementary Marking ====================

def set_measurement_primary_status(
    tech_file: Path,
    event_index: int,
    is_primary: bool,
    note: Optional[str] = None
) -> None:
    """Set primary/supplementary status for a technical event.
    
    Primary measurements are the main reference measurements for analysis.
    Supplementary measurements are additional measurements for verification.
    
    Args:
        tech_file: Technical container path
        event_index: Event index (1-based)
        is_primary: True for primary/major, False for supplementary
        note: Optional note for supplementary measurements
        
    Raises:
        ValueError: If event does not exist
        RuntimeError: If container is locked
    """
    tech_file = Path(tech_file)
    
    if not tech_file.exists():
        raise FileNotFoundError(f"Container not found: {tech_file}")
    
    if is_container_locked(tech_file):
        raise RuntimeError(
            f"Cannot modify locked container: {tech_file}\n"
            f"Container is locked and should not be modified."
        )
    
    event_id = schema.format_technical_event_id(event_index)
    event_path = f"{schema.GROUP_TECHNICAL}/{event_id}"
    
    try:
        with h5py.File(tech_file, 'a') as f:
            if event_path not in f:
                raise ValueError(
                    f"Event {event_path} does not exist in {tech_file}"
                )
            
            # Set primary status
            f[event_path].attrs['is_primary'] = is_primary
            
            # Add note for supplementary measurements
            if not is_primary and note:
                f[event_path].attrs['supplementary_note'] = note
            
            status_str = "PRIMARY" if is_primary else "SUPPLEMENTARY"
            logger.info(f"Set {event_path} to {status_str}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to set primary status: {e}")


def get_primary_measurements(tech_file: Path) -> dict:
    """Get list of primary measurements from technical container.
    
    Args:
        tech_file: Technical container path
        
    Returns:
        Dict mapping technical type to list of primary event indices
        Example: {'DARK': [1], 'EMPTY': [2], 'BACKGROUND': [3], 'AGBH': [4]}
    """
    tech_file = Path(tech_file)
    
    if not tech_file.exists():
        raise FileNotFoundError(f"Container not found: {tech_file}")
    
    primary_events = {}
    
    try:
        with h5py.File(tech_file, 'r') as f:
            if schema.GROUP_TECHNICAL not in f:
                return primary_events
            
            for event_id in f[schema.GROUP_TECHNICAL].keys():
                if not event_id.startswith('tech_evt_'):
                    continue
                
                event_path = f"{schema.GROUP_TECHNICAL}/{event_id}"
                is_primary = f[event_path].attrs.get('is_primary', True)
                
                if is_primary:
                    tech_type = f[event_path].attrs.get('type', 'UNKNOWN')
                    event_idx = int(event_id.split('_')[-1])
                    
                    if tech_type not in primary_events:
                        primary_events[tech_type] = []
                    primary_events[tech_type].append(event_idx)
        
        return primary_events
    
    except Exception as e:
        logger.error(f"Failed to get primary measurements: {e}")
        return {}


# ==================== New Locking Functions (with operator support) ====================

def archive_technical_data_files(
    container_path: Path,
    archive_folder: Path,
    file_patterns: Optional[list] = None
) -> int:
    """Archive data files associated with a technical container.
    
    Moves data files from the container directory to the archive folder.
    File patterns are detector-specific (e.g., Advacam uses .txt/.dsc/.npy,
    Bruker might use different formats).
    
    Args:
        container_path: Path to the technical container .h5 file
        archive_folder: Path to archive folder (will be created if needed)
        file_patterns: List of file patterns to archive (e.g., ['*.txt', '*.dsc', '*.npy', '*.poni', '*_state.json'])
                      If None, defaults to ['*.txt', '*.dsc', '*.npy', '*.poni', '*_state.json'] for Advacam detectors
    
    Returns:
        Number of files archived
    
    Example:
        # For Advacam detectors
        archive_technical_data_files(container, archive, ['*.txt', '*.dsc', '*.npy', '*.poni', '*_state.json'])
        
        # For Bruker detectors (hypothetical)
        archive_technical_data_files(container, archive, ['*.raw', '*.brml'])
    """
    container_path = Path(container_path)
    archive_folder = Path(archive_folder)
    container_dir = container_path.parent
    
    # Default patterns for Advacam detectors
    if file_patterns is None:
        file_patterns = ['*.txt', '*.dsc', '*.npy', '*.poni', '*_state.json']
    
    # Create archive folder
    archive_folder.mkdir(parents=True, exist_ok=True)
    
    archived_count = 0
    for pattern in file_patterns:
        for data_file in container_dir.glob(pattern):
            try:
                dest = archive_folder / data_file.name
                shutil.move(str(data_file), str(dest))
                archived_count += 1
                logger.debug(f"Archived: {data_file.name} -> {archive_folder.name}")
            except Exception as e:
                logger.warning(f"Failed to archive {data_file.name}: {e}")
    
    if archived_count > 0:
        logger.info(f"Archived {archived_count} data files to {archive_folder}")
    
    return archived_count


def lock_technical_container(
    tech_file: Path,
    locked_by: str,
    notes: Optional[str] = None
) -> None:
    """Lock technical container for production use.
    
    This locks the container and records who locked it and when.
    Locked containers are ready for session measurements.
    
    Note: This function only locks the container. Use archive_technical_data_files()
    separately to archive associated data files if needed.
    
    Args:
        tech_file: Path to technical container
        locked_by: Operator ID who is locking the container
        notes: Optional notes about locking
        
    Raises:
        RuntimeError: If container already locked
    """
    tech_file = Path(tech_file)
    
    if not tech_file.exists():
        raise FileNotFoundError(f"Container not found: {tech_file}")
    
    if is_container_locked(tech_file):
        raise RuntimeError(f"Container already locked: {tech_file}")
    
    # Step 1: Set all HDF5 attributes (including notes) BEFORE read-only
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        _set_container_lifecycle_state(
            tech_file,
            locked=True,
            lock_timestamp=timestamp,
            locked_by=locked_by,
        )
        with h5py.File(tech_file, 'a') as f:
            f.attrs['locked_notes'] = notes if notes else f.attrs.get('locked_notes', '')
            runtime = f.get(schema.GROUP_RUNTIME)
            if runtime is not None:
                if notes:
                    runtime.attrs['locked_notes'] = notes
                elif 'locked_notes' in runtime.attrs:
                    del runtime.attrs['locked_notes']
            if not notes and 'locked_notes' in f.attrs:
                del f.attrs['locked_notes']
        
        logger.info(f"Set locked attributes for: {tech_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to set locked attributes: {e}")
    
    # Step 2: Set OS read-only permissions
    try:
        current_perms = tech_file.stat().st_mode
        read_only_perms = current_perms & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        os.chmod(tech_file, read_only_perms)
        logger.info(f"Set OS read-only permissions for: {tech_file}")
    except Exception as e:
        logger.warning(f"Could not set OS read-only permissions: {e}")
    
    logger.info(f"Locked container for production: {tech_file} by {locked_by}")


def get_lock_info(tech_file: Path) -> dict:
    """Get lock information from container.
    
    Args:
        tech_file: Path to technical container
        
    Returns:
        Dict with keys: locked, locked_timestamp, locked_by, locked_notes
    """
    tech_file = Path(tech_file)
    
    info = {
        'locked': False,
        'locked_timestamp': None,
        'locked_by': None,
        'locked_notes': None
    }
    
    if not tech_file.exists():
        return info
    
    try:
        with h5py.File(tech_file, 'r') as f:
            runtime = f.get(schema.GROUP_RUNTIME)
            if runtime is not None:
                info['locked'] = bool(runtime.attrs.get(schema.ATTR_LOCKED, f.attrs.get('locked', False)))
            else:
                info['locked'] = bool(f.attrs.get('locked', False))
            
            if info['locked']:
                ts = (
                    runtime.attrs.get(schema.ATTR_LOCKED_TIMESTAMP)
                    if runtime is not None and schema.ATTR_LOCKED_TIMESTAMP in runtime.attrs
                    else f.attrs.get('locked_timestamp')
                )
                info['locked_timestamp'] = ts.decode('utf-8') if isinstance(ts, bytes) else ts
                
                by = (
                    runtime.attrs.get(schema.ATTR_LOCKED_BY)
                    if runtime is not None and schema.ATTR_LOCKED_BY in runtime.attrs
                    else f.attrs.get('locked_by')
                )
                info['locked_by'] = by.decode('utf-8') if isinstance(by, bytes) else by
                
                notes = (
                    runtime.attrs.get('locked_notes')
                    if runtime is not None and 'locked_notes' in runtime.attrs
                    else f.attrs.get('locked_notes')
                )
                info['locked_notes'] = notes.decode('utf-8') if isinstance(notes, bytes) else notes
    except Exception as e:
        logger.error(f"Failed to get lock info: {e}")
    
    return info


# ==================== Validation Helpers ====================

def validate_technical_container_format(tech_file: Path) -> bool:
    """Validate technical container filename format.
    
    Expected format:
    - technical_<id>_<yyyymmdd>.nxs.h5 (legacy)
    - technical_<id>_<distance>cm_<yyyymmdd>.nxs.h5 (current)
    
    Args:
        tech_file: Path to technical container
        
    Returns:
        True if format is valid
    """
    name = tech_file.name
    
    # Check pattern: technical_<16hexchars>[_<distance>cm]_<yyyymmdd>.nxs.h5
    import re
    pattern = r'^technical_[0-9a-f]{16}(?:_[0-9mp]+cm)?_\\d{8}\\.nxs\\.h5$'
    
    return bool(re.match(pattern, name))
