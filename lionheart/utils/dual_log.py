
"""Duallog

Based on https://github.com/acschaefer/duallog (MIT license)

This module contains a function "setup()" that sets up dual logging. 
All subsequent log messages are sent both to the console and to a logfile. 
Log messages are generated via the "logging" package.

Example:
    >>> import duallog
    >>> import logging
    >>> duallog.setup('mylogs')
    >>> logging.info('Test message')

"""


# Import required standard packages.
import datetime
import logging
import logging.handlers
import os
import sys

from utipy import random_alphanumeric

# Define default logfile format.
file_name_format = '{prefix}{year:04d}{month:02d}{day:02d}-'\
    '{hour:02d}{minute:02d}{second:02d}-{rand}.log'

# Define the default logging message formats.
file_msg_format = '%(asctime)s %(levelname)-8s: %(message)s'
console_msg_format = '%(levelname)s: %(message)s'

# Define the log rotation criteria.
max_bytes = 1024**2
backup_count = 100


def setup_logging(dir='logs', fname_prefix="", minLevel=logging.DEBUG, stream=sys.stdout):
    """ Set up dual logging to console and to logfile.

    When this function is called, it first creates the given logging output directory. 
    It then creates a logfile and passes all log messages to come to it. 
    The name of the logfile encodes the date and time when it was created, for example "20181115-153559.log". 
    All messages with a certain minimum log level are also forwarded to the console.

    Parameters
    ----------
    dir: str
        Path of the directory where to store the log files. Both a
        relative or an absolute path may be specified. If a relative path is
        specified, it is interpreted relative to the working directory.
        Defaults to "log".
    fname_prefix : str
        Prefix for the log filename.
    minLevel: logging level
        Defines the minimum level of the messages that will be shown on the console. Defaults to DEBUG. 
    """
    assert isinstance(fname_prefix, str)
    assert isinstance(dir, str)

    # Create the root logger.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Validate the given directory.
    dir = os.path.normpath(dir)

    # Create a folder for the logfiles.
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except FileExistsError:
            # Directory was likely made in-between
            # check and creation
            pass

    # Construct the name of the logfile.
    t = datetime.datetime.now()
    file_name = file_name_format.format(
        prefix=fname_prefix, year=t.year, month=t.month, day=t.day,
        hour=t.hour, minute=t.minute, second=t.second,
        rand=random_alphanumeric(10)
    )
    file_name = os.path.join(dir, file_name)

    # Set up logging to the logfile.
    file_handler = logging.handlers.RotatingFileHandler(
        filename=file_name,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(file_msg_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Set up logging to the console.
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setLevel(minLevel)
    stream_formatter = logging.Formatter(console_msg_format)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
