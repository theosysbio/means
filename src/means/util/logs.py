import logging

def get_logger(name):
    logging.basicConfig()
    logger = logging.getLogger(name)
    return logger