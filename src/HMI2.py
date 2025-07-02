import logging
import os
import sys

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
logger = logging.getLogger("HMI2")
#logger.setLevel(logging.DEBUG)


class HMI2(HMI):
    def __init__(self):
        super().__init__('HMI2', TAG.TAG_LIST, Controllers.PLCs)

    def _display(self):
        try:
            menu_line = '{}) To change the {} press {} \n'
            menu = '\n'

            menu += self.__get_menu_line(1, 'empty level of tank')
            menu += self.__get_menu_line(2, 'full level of tank')
            menu += self.__get_menu_line(3, 'full level of bottle')
            menu += self.__get_menu_line(4, 'status of tank Input valve')
            menu += self.__get_menu_line(5, 'status of tank output valve')
            menu += self.__get_menu_line(6, 'status of conveyor belt engine')

            logger.info("Displaying menu:\n%s", menu)
            self.report(menu)
        except Exception as e:
            logger.error("Error while displaying menu: %s", str(e), exc_info=True)

    def __get_menu_line(self, number, text):
        return '{} To change the {} press {} \n'.format(
            self._make_text(str(number) + ')', self.COLOR_BLUE),
            self._make_text(text, self.COLOR_GREEN),
            self._make_text(str(number), self.COLOR_BLUE)
        )

    def _operate(self):
        try:
            choice = self.__get_choice()
            input1, input2 = choice

            if input1 == 1:
                self._send(TAG.TAG_TANK_LEVEL_MIN, input2)
                logger.info("Set empty level of tank to %.2f", input2)
            elif input1 == 2:
                self._send(TAG.TAG_TANK_LEVEL_MAX, input2)
                logger.info("Set full level of tank to %.2f", input2)
            elif input1 == 3:
                self._send(TAG.TAG_BOTTLE_LEVEL_MAX, input2)
                logger.info("Set full level of bottle to %.2f", input2)
            elif input1 == 4:
                self._send(TAG.TAG_TANK_INPUT_VALVE_MODE, input2)
                logger.info("Changed status of tank input valve to mode %d", input2)
            elif input1 == 5:
                self._send(TAG.TAG_TANK_OUTPUT_VALVE_MODE, input2)
                logger.info("Changed status of tank output valve to mode %d", input2)
            elif input1 == 6:
                self._send(TAG.TAG_CONVEYOR_BELT_ENGINE_MODE, input2)
                logger.info("Changed status of conveyor belt engine to mode %d", input2)

            logger.info("Operation completed successfully for choice %d", input1)
        except ValueError as e:
            logger.warning("ValueError encountered: %s", str(e))
            self.report(str(e))
        except Exception as e:
            logger.error("Invalid input: %s", str(e), exc_info=True)
            self.report("The input is invalid: " + str(e))

        input('Press Enter to continue ...')

    def __get_choice(self):
        try:
            input1 = int(input('Your choice (1 to 6): '))
            if input1 < 1 or input1 > 6:
                raise ValueError('Only integer values between 1 and 6 are acceptable')

            if input1 <= 3:
                input2 = float(input('Specify set point (positive real value): '))
                if input2 < 0:
                    raise ValueError('Negative numbers are not acceptable.')
            else:
                sub_menu = '\n'
                sub_menu += "1) Send command for manually off\n"
                sub_menu += "2) Send command for manually on\n"
                sub_menu += "3) Send command for auto operation\n"
                self.report(sub_menu)
                input2 = int(input('Command (1 to 3): '))
                if input2 < 1 or input2 > 3:
                    raise ValueError('Only 1, 2, and 3 are acceptable for command')

            logger.debug("User choice: %d, Input value: %s", input1, input2)
            return input1, input2
        except Exception as e:
            logger.error("Error while getting user choice: %s", str(e), exc_info=True)
            raise


if __name__ == '__main__':
    hmi2 = HMI2()
    logger.info("Starting HMI2 system.")
    hmi2.start()
