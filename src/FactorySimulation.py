import logging
import os
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
        self.simulation_mode = get_simulation_mode()
        super().__init__('Factory', Connection.CONNECTION, 100)
        self.init()
        

        # Set logging level based on simulation mode
        if self.simulation_mode == "normal":
            logger.setLevel(logging.INFO)
        elif self.simulation_mode == "faults":
            logger.setLevel(logging.WARNING)

        logger.info(f"Factory Simulation initialized in '{self.simulation_mode}' mode.")

    def _logic(self):
        elapsed_time = self._current_loop_time - self._last_loop_time

        # Update the physical properties of the system
        self._simulate_physical_system(elapsed_time)

    def _simulate_physical_system(self, elapsed_time):
        """
        Simulates the physical behavior of the system without fault injection.
        """
        # Update tank water level
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

        # Update tank water flow
        tank_water_flow = 0
        if self._get(TAG.TAG_TANK_OUTPUT_VALVE_STATUS) and tank_water_amount > 0:
            tank_water_flow = PHYSICS.TANK_OUTPUT_FLOW_RATE

        # Update bottle water
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

        # Update bottle position
        bottle_distance_to_filler = self._get(TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE)
        if self._get(TAG.TAG_CONVEYOR_BELT_ENGINE_STATUS):
            bottle_distance_to_filler -= elapsed_time * PHYSICS.CONVEYOR_BELT_SPEED
            bottle_distance_to_filler %= PHYSICS.BOTTLE_DISTANCE

        # Update physical properties
        self._set(TAG.TAG_TANK_LEVEL_VALUE, tank_water_level)
        self._set(TAG.TAG_TANK_OUTPUT_FLOW_VALUE, tank_water_flow)
        self._set(TAG.TAG_BOTTLE_LEVEL_VALUE, bottle_water_level)
        self._set(TAG.TAG_BOTTLE_DISTANCE_TO_FILLER_VALUE, bottle_distance_to_filler)

    def init(self):
        """
        Initializes the factory simulation by setting default values for all tags.
        """
        initial_list = []
        for tag in TAG.TAG_LIST:
            initial_list.append((tag, TAG.TAG_LIST[tag]['default']))
        self._connector.initialize(initial_list)
    """
    def report(self, message, level=logging.INFO, context=None):
        
        clean_message = message.strip()
        if context:
            clean_message += f" | Context: {context}"
        print(message)
        logger.log(level, clean_message)
    """
    def report(self, message, level=logging.INFO, context=None):
        """
        Logs a message with the specified logging level.
        """
        # Safeguard: Handle cases where simulation_mode is not defined yet
        simulation_mode = getattr(self, 'simulation_mode', 'unknown')
        if simulation_mode == "faults":
            level = logging.ERROR if level == logging.WARNING else level
        clean_message = message.strip()
        if context:
            clean_message += f" | Context: {context}"
        print(clean_message)
        logger.log(level, clean_message)

    @staticmethod
    def recreate_connection():
        """
        Stub method for recreating a connection, if needed.
        """
        return True


if __name__ == '__main__':
    factory = FactorySimulation()
    factory.start()
