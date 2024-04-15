import sys
from src.logger import logging

def error_message_details(error_message, error_detail):
    file_name = error_detail[2].tb_frame.f_code.co_filename
    line_number = error_detail[2].tb_lineno
    return f"Error occurred in Python script '{file_name}' at line {line_number}: {error_message}"

class Custom_exception(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)
    
    def __str__(self):
        return self.error_message
    
        

# if __name__ == "__main__":
#     try:
#         a = 1/0
#     except Exception as e:
#         logging.info("divide by zero error occurred")
#         raise Custom_exception(e,sys.exc_info())