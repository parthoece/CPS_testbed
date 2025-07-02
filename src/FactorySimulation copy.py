import logging
import os
import random
from ics_sim.Device import HIL
from Configs import TAG, PHYSICS, Connection
from logging.handlers import SysLogHandler
import time

# Configure Syslog
SYSLOG_SERVER = "192.168.0.10"  # Replace with the log-server's IP
SYSLOG_PORT = 514

# Set up a SysLogHandler
syslog_handler = SysLogHandler(address=(SYSLOG_SERVER, SYSLOG_PORT))
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
syslog_handler.setFormatter(formatter)

# Configure the root logger
logging.basicConfig(level=logging.INFO, handlers=[syslog_handler])
logger = logging.getLogger("FactorySimulation")

# Function to get simulation mode
def get_simulation_mode():
    mode_file = "/shared/mode.conf"  # The file path inside the container
    try:
        with open(mode_file, "r") as f:
            for line in f:
                if line.startswith("mode="):
                    return line.split("=")[1].strip()
    except Exception as e:
        print(f"Error reading mode file: {e}")
        return "normal"  # Default to normal mode if file is not found or error occurs


class FactorySimulation(HIL):
    def __init__(self):
        super().__init__('Factory', Connection.CONNECTION, 100)
        self.init()
        self.simulation_mode = get_simulation_mode()
        self.faults = [
            self._apply_sensor_drift,
            self._apply_tank_leak,
            self._apply_conveyor_belt_sticking,
            self._apply_valve_sticking,
            self._apply_memory_corruption,
            self._apply_overheating,
        ]
        self.current_fault_index = 0  # Start with the first fault
        logger.info(f"Simulation mode: {self.simulation_mode}")

    def _logic(self):
        elapsed_time = self._current_loop_time - self._last_loop_time

        # Simulate normal or faults based on mode
        if self.simulation_mode == "faults":
            logger.info("Executing system faults logic.")
            self._simulate_faults(elapsed_time)
        else:
            logger.info("Executing normal operation logic.")
            self._simulate_normal_operation(elapsed_time)

    def _simulate_normal_operation(self, elapsed_time):
        # Simulate normal operation logic (existing logic moved here)
        tank_water_amount = self._get(TAG.TAG_TANK_LEVEL_VALUE) * PHYSICS.TANK_LEVEL_CAPACITY
        if self._get(TAG.TAG_TANK_INPUT_VALVE_STATUS):
            tank_water_amount += PHYSICS.TANK_INPUT_FLOW_RATE * elapsed_time

        if self._get(TAG.TAG_TANK_OUTPUT_VALVE_STATUS):
            tank_water_amount -= PHYSICS.TANK_OUTPUT_FLOW_RATE * elapsed_time

        tank_water_level = tank_water_amount / PHYSICS.TANK_LEVEL_CAPACITY

        if tank_water_level > PHYSICS.TANK_MAX_LEVEL:
            tank_water_level = PHYSICS.TANK_MAX_LEVEL
            self.report('Tank water overflowed', logging.WARNING)
        elif tank_water_level <= 0:
            tank_water_level = 0
            self.report('Tank water is empty', logging.WARNING)

        tank_water_flow = 0
        if self._get(TAG.TAG_TANK_OUTPUT_VALVE_STATUS) and tank_water_amount > 0:
            tank_water_flow = PHYSICS.TANK_OUTPUT_FLOW_RATE

        if self._get(TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE) > 1:
            bottle_water_amount = 0
            if self._get(TAG.TAG_TANK_OUTPUT_FLOW_VALUE):
                self.report('Water is wasting', logging.WARNING)
        else:
            bottle_water_amount = self._get(TAG.TAG_BOTTLE_LEVEL_VALUE) * PHYSICS.BOTTLE_LEVEL_CAPACITY
            bottle_water_amount += self._get(TAG.TAG_TANK_OUTPUT_FLOW_VALUE) * elapsed_time

        bottle_water_level = bottle_water_amount / PHYSICS.BOTTLE_LEVEL_CAPACITY

        if bottle_water_level > PHYSICS.BOTTLE_MAX_LEVEL:
            bottle_water_level = PHYSICS.BOTTLE_MAX_LEVEL
            self.report('Bottle water overflowed', logging.WARNING)

        bottle_distance_to_filler = self._get(TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE)
        if self._get(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS):
            bottle_distance_to_filler -= elapsed_time * PHYSICS.CONVEYOR_BELT_SPEED
            bottle_distance_to_filler %= PHYSICS.BOTTLE_DISTANCE

        self._set(TAG.TAG_TANK_LEVEL_VALUE, tank_water_level)
        self._set(TAG.TAG_TANK_OUTPUT_FLOW_VALUE, tank_water_flow)
        self._set(TAG.TAG_BOTTLE_LEVEL_VALUE, bottle_water_level)
        self._set(TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE, bottle_distance_to_filler)

    def _simulate_faults(self, elapsed_time):
        # Apply one fault at a time
        if self.current_fault_index < len(self.faults):
            fault_function = self.faults[self.current_fault_index]
            fault_function(elapsed_time)
            self.current_fault_index += 1
        else:
            logger.info("All faults have been applied.")

    # Fault simulation methods
    def _apply_sensor_drift(self, elapsed_time):
        drift = random.uniform(-0.05, 0.05)
        current_level = self._get(TAG.TAG_TANK_LEVEL_VALUE)
        self._set(TAG.TAG_TANK_LEVEL_VALUE, current_level + drift)
        logger.warning("Sensor Drift fault applied. Tank level drifted to: %.2f", current_level + drift)

    def _apply_tank_leak(self, elapsed_time):
        leak_rate = random.uniform(0.01, 0.02)
        current_level = self._get(TAG.TAG_TANK_LEVEL_VALUE)
        self._set(TAG.TAG_TANK_LEVEL_VALUE, max(0, current_level - leak_rate))
        logger.warning("Tank Leak fault applied. Tank level reduced to: %.2f", max(0, current_level - leak_rate))

    def _apply_conveyor_belt_sticking(self, elapsed_time):
        self._set(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS, 0)
        logger.error("Conveyor Belt Sticking fault applied. Conveyor belt stopped.")

    def _apply_valve_sticking(self, elapsed_time):
        valve_state = random.choice([0, 1])
        self._set(TAG.TAG_TANK_INPUT_VALVE_STATUS, valve_state)
        self._set(TAG.TAG_TANK_OUTPUT_VALVE_STATUS, valve_state)
        logger.error("Valve Sticking fault applied. Valve state set to %d.", valve_state)

    def _apply_memory_corruption(self, elapsed_time):
        corrupted_value = random.randint(0, 100)
        self._set(TAG.TAG_TANK_OUTPUT_FLOW_VALUE, corrupted_value)
        logger.critical("Memory Corruption fault applied. Output flow corrupted to %d.", corrupted_value)

    def _apply_overheating(self, elapsed_time):
        time.sleep(2)  # Simulate processing delay
        logger.critical("Overheating fault applied. Processing delayed by 2 seconds.")

    def init(self):
        initial_list = []
        for tag in TAG.TAG_LIST:
            initial_list.append((tag, TAG.TAG_LIST[tag]['default']))
        self._connector.initialize(initial_list)

    def report(self, message, level=logging.INFO, context=None):
        clean_message = message.strip()
        if context:
            clean_message += f" | Context: {context}"
        print(message)
        logger.log(level, clean_message)

    @staticmethod
    def recreate_connection():
        return True


if __name__ == '__main__':
    factory = FactorySimulation()
    factory.start()
