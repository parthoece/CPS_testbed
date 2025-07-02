import logging
from logging.handlers import SysLogHandler
import time
import os
import json
import random
from coordination_utils import read_coordination_file, write_coordination_file

from ics_sim.Device import PLC, SensorConnector, ActuatorConnector
from Configs import TAG, Connection, Controllers

# Function to determine simulation mode
def get_simulation_mode():
    mode_file = "./mode.conf"
    if os.path.exists(mode_file):
        with open(mode_file, "r") as f:
            for line in f:
                if line.startswith("mode="):
                    return line.strip().split("=")[1]
    return "normal"  # Default to "normal" if mode.conf is missing or invalid


# Configure logger
logger = logging.getLogger("PLC2")
simulation_mode = get_simulation_mode()

# Set logging level dynamically based on the simulation mode
if simulation_mode == "normal":
    logger.setLevel(logging.INFO)  # Log INFO and higher
elif simulation_mode == "faults":
    logger.setLevel(logging.DEBUG)  # Log DEBUG and higher for detailed fault insights

# Configure SysLogHandler for log-server
SYSLOG_SERVER = "log-server"  # Use the service name of your log-server
SYSLOG_PORT = 514             # Syslog port (UDP or TCP)
syslog_handler = SysLogHandler(address=(SYSLOG_SERVER, SYSLOG_PORT))
syslog_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
syslog_handler.setFormatter(syslog_formatter)
logger.addHandler(syslog_handler)


class PLC2(PLC):
    def __init__(self):
        # Initialize connectors
        sensor_connector = SensorConnector(Connection.CONNECTION)
        actuator_connector = ActuatorConnector(Connection.CONNECTION)

        # Configure sensors and faults
        sensor_connector.add_sensor(TAG.TAG_BOTTLE_LEVEL_VALUE, 0.01)
        sensor_connector.add_sensor_drift(TAG.TAG_BOTTLE_LEVEL_VALUE, 0.03)

        super().__init__(2, sensor_connector, actuator_connector, TAG.TAG_LIST, Controllers.PLCs)
        self.faults = [
            self._apply_sensor_drift,
            self._apply_conveyor_belt_sticking,
        ]
        self.current_fault_index = 0  # Start with the first fault
        self.simulation_mode = get_simulation_mode()

    def _logic(self):
        try:
            # Ensure coordination with PLC1
            self._wait_for_plc1_completion()

            # Start time for logic execution
            t1 = time.time()
            logger.debug("Starting logic execution at: %s", time.strftime("%Y-%m-%d %H:%M:%S"))

            # Retrieve sensor and actuator data
            belt_position = self._get(TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE)
            bottle_level = self._get(TAG.TAG_BOTTLE_LEVEL_VALUE)

            # Normal operation logic
            if belt_position > 1 or bottle_level > self._get(TAG.TAG_BOTTLE_LEVEL_MAX):
                self._set(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS, 1)
                logger.info("Conveyor belt started. Belt position: %.2f, Bottle level: %.2f", belt_position, bottle_level)
            else:
                self._set(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS, 0)
                logger.info("Conveyor belt stopped. Belt position: %.2f, Bottle level: %.2f", belt_position, bottle_level)

            # Apply one fault at a time if in "faults" mode
            if self.simulation_mode == "faults":
                self._apply_next_fault()

            # End time and execution duration
            t2 = time.time()
            execution_duration = t2 - t1
            logger.debug("Logic execution duration: %.4f seconds", execution_duration)

            # Update coordination status
            self._update_coordination_status()

        except Exception as e:
            logger.error("Error in PLC2 logic: %s", str(e), exc_info=True)

    def _apply_next_fault(self):
        if self.current_fault_index < len(self.faults):
            fault_function = self.faults[self.current_fault_index]
            fault_function()
            self.current_fault_index += 1  # Move to the next fault
        else:
            logger.info("All faults have been applied sequentially.")

    def _apply_sensor_drift(self):
        drift = random.uniform(-0.03, 0.03)
        bottle_level = self._get(TAG.TAG_BOTTLE_LEVEL_VALUE)
        self._set(TAG.TAG_BOTTLE_LEVEL_VALUE, bottle_level + drift)
        logger.warning("Sensor Drift fault applied. Drifted bottle level: %.2f", bottle_level + drift)

    def _apply_conveyor_belt_sticking(self):
        self._set(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS, 0)
        logger.error("Conveyor Belt Sticking fault applied. Belt stopped.")

    def _wait_for_plc1_completion(self):
        while True:
            coordination_data = read_coordination_file()
            if coordination_data["current_plc"] == "PLC1" and coordination_data["plc1_completed"]:
                logger.info("PLC1 has completed. Proceeding with PLC2 operations.")
                break
            else:
                logger.debug("Waiting for PLC1 to complete...")
                time.sleep(1)  # Poll every 1 second

    def _update_coordination_status(self):
        coordination_data = read_coordination_file()
        coordination_data["current_plc"] = "PLC2"
        coordination_data["plc2_completed"] = True
        write_coordination_file(coordination_data)


if __name__ == "__main__":
    plc2 = PLC2()
    plc2.set_record_variables(True)  # Record state changes for debugging or analysis
    logger.info("Starting PLC2 system in '%s' mode.", plc2.simulation_mode)
    plc2.start()