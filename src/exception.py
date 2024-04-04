import sys
import logging

def error_message_details(error_detail: tuple):
    exc_type, exc_obj, exc_tb = error_detail
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in Python script name [{0}] line number [{1}] error message [{2}].".format(
        file_name, exc_tb.tb_lineno, str(exc_obj))
    return error_message                                 

class Custom_exception(Exception):
    def __init__(self, error_detail: tuple):
        super().__init__(error_message_details(error_detail))
        