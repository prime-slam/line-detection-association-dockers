import logging
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL

from cnn import config

colors = {
    NOTSET: 36,
    DEBUG: 33,
    INFO: 34,
    WARNING: .35,
    ERROR: 31,
    CRITICAL: "5;31"
}

active_logger = {}


class Logger:
    def __getattr__(self, attr):
        return getattr(self.logger, attr)

    def __setattr__(self, attr, val):
        if "_Logger__initialized" not in self.__dict__:
            self.__dict__[attr] = val
        else:
            setattr(self.logger, attr, val)

    def __init__(self, logger):
        self.logger = logger
        # super().__init__(name, lvl)

        self.explicit_lvl = False

        self.h = logging.StreamHandler()
        self.h.setFormatter(
            logging.Formatter("\r{bcolor}%(asctime)s "
                              "[\033[1;33m%(name)s{bcolor}] "
                              "%(module)s:%(lineno)d"
                              " - "

                              "[\033[1;%(color)sm%(levelname)s{bcolor}] "
                              "\033[0m%(spaces)s"

                              "\033[0m%(message)s"
                              .format(bcolor="\033[1;36m")))
        self.h.addFilter(self.rec_filter)
        self.logger.handlers = []
        self.addHandler(self.h)

        self.setLevel_(logging.DEBUG)
        # if config.has_option("", "default_logging_lvl"):
        #     self.setLevel_(config.default_logging_lvl)
        # self.setLevel_(config["default_logging_lvl", logging.INFO])
        self.__initialized = True

    def setLevel(self, val):
        self.explicit_lvl = True
        self.logger.setLevel(val)

    def setLevel_(self, val):
        self.logger.setLevel(val)

    def rec_filter(self, record):
        record.color = colors[record.levelno]
        record.spaces = " " * (8 - len(record.levelname))
        return True


def setGlobalLevel(lvl):
    config.default_logging_lvl = lvl
    for lg in active_logger.values():
        if not lg.explicit_lvl:
            lg.setLevel_(lvl)


def getLogger(name):
    if name not in active_logger:
        active_logger[name] = Logger(logging.getLogger(name))
    return active_logger[name]
