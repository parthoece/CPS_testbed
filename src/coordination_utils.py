import json
import fcntl
import logging

COORDINATION_FILE = "/shared/fault_state.json"

def read_coordination_file():
    """
    Safely read the coordination file with a file lock.
    """
    try:
        with open(COORDINATION_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for reading
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock
        return data
    except Exception as e:
        logging.error(f"Error reading coordination file: {e}")
        return {}

def write_coordination_file(data):
    """
    Safely write to the coordination file with a file lock.
    """
    try:
        with open(COORDINATION_FILE, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock for writing
            f.seek(0)  # Move to the beginning of the file
            json.dump(data, f, indent=4)
            f.truncate()  # Ensure the file is overwritten
            fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock
    except Exception as e:
        logging.error(f"Error writing coordination file: {e}")
