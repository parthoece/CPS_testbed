import logging
import os
import sys
import time
import random

from ics_sim.Device import HMI
from Configs import TAG, Controllers
from logging.handlers import SysLogHandler

# Configure Syslog
SYSLOG_SERVER = "192.168.0.10"  # Replace with the log-server's IP
SYSLOG_PORT = 514

# Set up a SysLogHandler
syslog_handler = SysLogHandler(address=(SYSLOG_SERVER, SYSLOG_PORT))
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
syslog_handler.setFormatter(formatter)

# Configure the root logger
logging.basicConfig(level=logging.INFO, handlers=[syslog_handler])
logger = logging.getLogger("HMI3")

class HMI3(HMI):
    def __init__(self):
        super().__init__('HMI3', TAG.TAG_LIST, Controllers.PLCs)

    def _before_start(self):
        HMI._before_start(self)

        while True:
            response = input("Do you want to start auto manipulation of factory setting? \n")
            response = response.lower()
            if response in ['y', 'yes']:
                self._set_clear_scr(False)
                self.random_values = [
                    ["TANK LEVEL MIN", 1, 4.5],
                    ["TANK LEVEL MAX", 5.5, 9],
                    ["BOTTLE LEVEL MAX", 1, 1.9]
                ]
                logger.info("Auto manipulation mode started.")
                break
            else:
                logger.warning("Invalid response for auto manipulation mode.")
                continue

    def _display(self):
        try:
            n = random.randint(5, 20)
            logger.info("Sleeping for {} seconds".format(n))
            print("Sleep for {} seconds \n".format(n))
            time.sleep(n)
        except Exception as e:
            logger.error("Error in display operation: {}".format(str(e)), exc_info=True)

    def _operate(self):
        try:
            choice = self.__get_choice()
            input1, input2 = choice
            if input1 == 1:
                self._send(TAG.TAG_TANK_LEVEL_MIN, input2)
            elif input1 == 2:
                self._send(TAG.TAG_TANK_LEVEL_MAX, input2)
            elif input1 == 3:
                self._send(TAG.TAG_BOTTLE_LEVEL_MAX, input2)

            logger.info(
                "Automatically set {} to {:.2f}".format(self.random_values[input1 - 1][0], input2)
            )
            print("Set {} to the {} automatically".format(self.random_values[input1 - 1][0], input2))

        except ValueError as e:
            logger.warning("ValueError encountered: {}".format(str(e)))
            self.report(str(e))
        except Exception as e:
            logger.error("Invalid input: {}".format(str(e)), exc_info=True)
            self.report("The input is invalid: " + str(e))

    def __get_choice(self):
        try:
            input1 = random.randint(1, len(self.random_values))
            input2 = random.uniform(self.random_values[input1 - 1][1], self.random_values[input1 - 1][2])
            logger.debug(
                "Generated random choice: {} with value {:.2f}".format(self.random_values[input1 - 1][0], input2)
            )
            return input1, input2
        except Exception as e:
            logger.error("Error generating random choice: {}".format(str(e)), exc_info=True)
            raise


if __name__ == '__main__':
    hmi3 = HMI3()
    logger.info("Starting HMI3 system.")
    hmi3.start()
