import sys
from attention_yolo.logger import logging

import traceback

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in script: [{0}] at line number: [{1}] with error message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    full_traceback = traceback.format_exc()  # ✅ Captures full error traceback
    return f"{error_message}\nTraceback: {full_traceback}"

class CustomException(Exception):
    def __init__(self, error_message, error_details: sys, error_code=None):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)
        self.error_code = error_code  # ✅ Stores error code if provided
        logging.error(f"{self.error_message} | Error Code: {self.error_code}")

    def __str__(self):
        return f"{self.error_message} | Error Code: {self.error_code if self.error_code else 'N/A'}"
    
    