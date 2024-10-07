#src/exception.py

import sys
from src.logger import logging

class CustomError(Exception):
      def __init__(self, message, error_details:sys):
            super().__init__(message)
            self.error_details=error_details
            self.error_message=self.get_error_details_message()

      def get_error_details_message(self):
            _,_,exe_tb=self.error_details.exc_info()
            error_file = exe_tb.tb_frame.f_code.co_filename
            error_line = exe_tb.tb_lineno
            return f"Error file :{error_file}, Error line no: {error_line}, Error message : {self.args[0]}"
      
      def display_error(self):
            logging.info(self.error_message)
            print(self.error_message)

# Main function to test the exception
if __name__ == "__main__":
    logging.info("Testing exception.py")