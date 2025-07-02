import logging
from logging.handlers import SysLogHandler
import time
import os
import random
import json

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
        sensor_connector.add_sensor(TAG.TAG_BOTTLE_LEVEL_VALUE, 0.05)  # Example fault level

         # Add actuators
        actuator_connector.add_actuator(TAG.TAG_BOTTLE_LEVEL_VALUE)  # Allow _set on TAG_BOTTLE_LEVEL_VALUE

        super().__init__(2, sensor_connector, actuator_connector, TAG.TAG_LIST, Controllers.PLCs)
        self.faults = [
            self._apply_sensor_drift,
            self._apply_conveyor_belt_sticking,
            self._apply_memory_corruption,
            self._apply_overheating,
        ]
        self.current_fault_index = 0  # Start with the first fault
        self.simulation_mode = self._get_simulation_mode()

    def _logic(self):
        try:
            if self.simulation_mode == "faults":
                logger.info("Executing system faults logic.")
                self._apply_next_fault()  # Continuously apply faults in a loop
            else:
                logger.info("Executing normal operation logic.")
                self._simulate_normal_operation()  # Perform normal operations
        except Exception as e:
            logger.error("Error in PLC2 logic: %s", str(e), exc_info=True)

    def _simulate_normal_operation(self):
        try:
            # Start time for logic execution
            t1 = time.time()
            logger.debug("Starting logic execution at: %s", time.strftime("%Y-%m-%d %H:%M:%S"))

            # Retrieve sensor and actuator data
            flow = self._get(TAG.TAG_TANK_OUTPUT_FLOW_VALUE)
            belt_position = self._get(TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE)
            bottle_level = self._get(TAG.TAG_BOTTLE_LEVEL_VALUE)

            # Normal operation logic
            if (belt_position > 1) or (flow == 0 and bottle_level > self._get(TAG.TAG_BOTTLE_LEVEL_MAX)):
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

        except Exception as e:
            logger.error("Error in PLC2 logic: %s", str(e), exc_info=True)

    
    def _apply_next_fault(self):
        """
        Applies faults sequentially and loops over the fault list in fault mode.
        """
        if self.simulation_mode == "faults":
         # Loop through the fault list repeatedly
            fault_function = self.faults[self.current_fault_index]
            fault_function()  # Apply the current fault
            self.current_fault_index = (self.current_fault_index + 1) % len(self.faults)  # Move to the next fault
        else:
            logger.info("No faults are applied in normal mode.")

    def _apply_sensor_drift(self):
        """
        Simulates sensor drift by dynamically modifying the sensor value for calculations.
        """
        try:
            # Read the drifted value directly using SensorConnector
            drifted_level = self._sensor_connector.read(TAG.TAG_BOTTLE_LEVEL_VALUE, apply_fault=True)
            self._set(TAG.TAG_BOTTLE_LEVEL_VALUE, drifted_level)
            # Log the drifted value for debugging
            logger.warning(f"Sensor drift applied. Drifted bottle level: {drifted_level:.2f}")

            # Use the drifted level for calculations
            # You could implement logic here using `drifted_level` directly

        except Exception as e:
            logger.error(f"Error applying sensor drift: {e}", exc_info=True)

    def _apply_conveyor_belt_sticking(self):
        self._set(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS, 0)
        logger.error("Conveyor Belt Sticking fault applied. Belt stopped.")


    def _apply_memory_corruption(self):
        corrupted_value = random.randint(0, 100)  # Random invalid value
        self._set(TAG.TAG_TANK_OUTPUT_FLOW_VALUE, corrupted_value)
        logger.critical("Memory Corruption fault applied. Tank output flow set to %d", corrupted_value)

    def _apply_overheating(self):
        time.sleep(2)  # Simulate processing delay
        logger.critical("Overheating fault applied. Processing delayed by 2 seconds.")

    def _get_simulation_mode(self):
        """
        Reads the simulation mode from the configuration file.
        """
        try:
            with open("/shared/mode.conf", "r") as f:
                for line in f:
                    if line.startswith("mode="):
                        return line.strip().split("=")[1]
        except Exception as e:
            logger.error(f"Error reading simulation mode: {e}")
        return "normal"

    def _get_fault_type(self):
        try:
            with open("/shared/fault_state.json", "r") as f:
                data = json.load(f)
                return data.get("current_fault", None)
        except Exception as e:
            logger.error(f"Error reading fault state: {e}")
        return None


if __name__ == "__main__":
    plc2 = PLC2()
    plc2.set_record_variables(True)  # Record state changes for debugging or analysis
    logger.info("Starting PLC2 system in '%s' mode.", plc2.simulation_mode)
    plc2.start()
