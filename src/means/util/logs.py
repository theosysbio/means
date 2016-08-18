from __future__ import absolute_import, print_function

import logging

def get_logger(name):
    logging.basicConfig()
    logger = logging.getLogger(name)
    return logger
