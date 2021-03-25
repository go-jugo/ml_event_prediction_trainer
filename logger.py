__all__ = ("get_logger", "init_logger")


import logging


logging_levels = {
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
    'debug': logging.DEBUG
}


color_code = {
    'CRITICAL': '\033[41m',
    'ERROR': '\033[31m',
    'WARNING': '\033[33m',
    'DEBUG': '\033[94m',
}


class LoggerError(Exception):
    pass


class ColorFormatter(logging.Formatter):
    def format(self, record):
        s = super().format(record)
        if record.levelname in color_code:
            return '{}{}{}'.format(color_code[record.levelname], s, '\033[0m')
        return s


msg_fmt = '%(asctime)s - %(levelname)s: [%(name)s] %(message)s'
date_fmt = '%m.%d.%Y %I:%M:%S %p'
standard_formatter = logging.Formatter(fmt=msg_fmt, datefmt=date_fmt)
color_formatter = ColorFormatter(fmt=msg_fmt, datefmt=date_fmt)

handler = logging.StreamHandler()
handler.setFormatter(standard_formatter)

logger = logging.getLogger("event-prediction")
logger.propagate = False
logger.addHandler(handler)


framework_loggers = [logging.getLogger(name) for name in ("dask", "fsspec", "sklearn", "concurrent")]
for framework_logger in framework_loggers:
    framework_logger.addHandler(handler)


def init_logger(level, color=False):
    if level not in logging_levels.keys():
        err = "unknown log level '{}'".format(level)
        raise LoggerError(err)
    if color:
        handler.setFormatter(color_formatter)
    logger.setLevel(logging_levels[level])


def get_logger(name: str) -> logging.Logger:
    return logger.getChild(name)
