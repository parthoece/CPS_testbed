from ics_sim.Device import PLC, SensorConnector, ActuatorConnector
from Configs import TAG, Controllers, Connection
import logging
from logging.handlers import SysLogHandler
import random
import time
import json

# Configure Syslog
SYSLOG_SERVER = "192.168.0.10"  # Replace with the log-server's IP
SYSLOG_PORT = 514

# Set up a SysLogHandler
syslog_handler = SysLogHandler(address=(SYSLOG_SERVER, SYSLOG_PORT))
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
syslog_handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[syslog_handler])
logger = logging.getLogger("PLC1") 


class PLC1(PLC):
    def __init__(self):
        # Initialize connectors
        sensor_connector = SensorConnector(Connection.CONNECTION)
        actuator_connector = ActuatorConnector(Connection.CONNECTION)

        # Add sensors
        sensor_connector.add_sensor(TAG.TAG_TANK_LEVEL_VALUE, 0.05)
        #sensor_connector.add_sensor(TAG.TAG_BOTTLE_LEVEL_VALUE, 0.02)

         # Add actuators
        actuator_connector.add_actuator(TAG.TAG_TANK_LEVEL_VALUE)  # Allow _set on TAG_TANK_LEVEL_VALUE


        super().__init__(1, sensor_connector, actuator_connector, TAG.TAG_LIST, Controllers.PLCs)

        # Define faults
        self.faults = [
            self._apply_sensor_drift,
            self._apply_tank_leak,
            self._apply_overheating,
            self._apply_valve_sticking
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
            logger.error("Error in PLC1 logic: %s", str(e), exc_info=True)

    def _simulate_normal_operation(self):

        # Tank input valve logic
        if not self._check_manual_input(TAG.TAG_TANK_INPUT_VALVE_MODE, TAG.TAG_TANK_INPUT_VALVE_STATUS):
            tank_level = self._get(TAG.TAG_TANK_LEVEL_VALUE)
            if tank_level > self._get(TAG.TAG_TANK_LEVEL_MAX):
                self._set(TAG.TAG_TANK_INPUT_VALVE_STATUS, 0)
                logger.info(f"Tank input valve closed due to high tank level: {tank_level:.2f}")
            elif tank_level < self._get(TAG.TAG_TANK_LEVEL_MIN):
                self._set(TAG.TAG_TANK_INPUT_VALVE_STATUS, 1)
                logger.info(f"Tank input valve opened due to low tank level: {tank_level:.2f}")

        # Tank output valve logic
        if not self._check_manual_input(TAG.TAG_TANK_OUTPUT_VALVE_MODE, TAG.TAG_TANK_OUTPUT_VALVE_STATUS):
            bottle_level = self._get(TAG.TAG_BOTTLE_LEVEL_VALUE)
            belt_position = self._get(TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE)
            if bottle_level > self._get(TAG.TAG_BOTTLE_LEVEL_MAX) or belt_position > 1.0:
                self._set(TAG.TAG_TANK_OUTPUT_VALVE_STATUS, 0)
                logger.info(f"Tank output valve closed. Bottle level: {bottle_level:.2f}, Belt position: {belt_position:.2f}")
            else:
                self._set(TAG.TAG_TANK_OUTPUT_VALVE_STATUS, 1)
                logger.info(f"Tank output valve opened. Bottle level: {bottle_level:.2f}, Belt position: {belt_position:.2f}")

        

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
            drifted_level = self._sensor_connector.read(TAG.TAG_TANK_LEVEL_VALUE, apply_fault=True)
            self._set(TAG.TAG_TANK_LEVEL_VALUE, drifted_level)
            # Log the drifted value for debugging
            logger.warning(f"Sensor drift applied. Drifted tank level: {drifted_level:.2f}")

            # Use the drifted level for calculations
            # You could implement logic here using `drifted_level` directly

        except Exception as e:
            logger.error(f"Error applying sensor drift: {e}", exc_info=True)
    def _apply_tank_leak(self):
        """
        Simulates a tank leak by reducing the tank level.
        """
        leak_rate = random.uniform(0.01, 0.05)
        current_level = self._sensor_connector.read(TAG.TAG_TANK_LEVEL_VALUE, apply_fault=True)
        new_level = max(0, current_level - leak_rate)
        self._set(TAG.TAG_TANK_LEVEL_VALUE, new_level)
        logger.warning(f"Tank Leak fault applied. Tank level reduced to: {new_level:.2f}")

    def _apply_overheating(self):
        """
        Simulates overheating by introducing a delay in processing.
        """
        delay_time = random.uniform(1, 3)
        time.sleep(delay_time)
        logger.critical(f"Overheating fault applied. Processing delayed by {delay_time:.2f} seconds.")

    def _apply_valve_sticking(self):
        valve_state = random.choice([0, 1])  # Randomly set to open (1) or closed (0)
        self._set(TAG.TAG_TANK_INPUT_VALVE_STATUS, valve_state)
        self._set(TAG.TAG_TANK_OUTPUT_VALVE_STATUS, valve_state)
        logger.error("Valve Sticking fault applied. Input/Output valve set to %d", valve_state)


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

    def _post_logic_update(self):
        super()._post_logic_update()
        # Log the alive time and loop latency for performance monitoring
        alive_time = self.get_alive_time() / 1000  # Convert to seconds
        loop_latency = self.get_loop_latency() / 1000  # Convert to seconds
        logger.debug(f"Alive time: {alive_time:.2f} sec, Loop latency: {loop_latency:.4f} sec")


if __name__ == "__main__":
    plc1 = PLC1()
    plc1.set_record_variables(True)
    plc1.start()
