from configparser import ConfigParser, ExtendedInterpolation
from io import StringIO
import sys
default_name = "config"

__all__ = ["Conf"]


def tryeval(val):
    frm = sys._getframe()
    caller = frm.f_back.f_back.f_back
    glbl = caller.f_globals
    lcl = caller.f_locals
    try:
        import ast
        return ast.literal_eval(val)
    except: # noqa
        pass
    try:
        return eval(val, glbl, lcl)
    except: # noqa
        return val


class CleanSetAttrMeta(type):
    """Metaclass for overriding __setattr__"""
    def __call__(cls, *args, **kwargs):
        real_setattr = cls.__setattr__
        cls.__setattr__ = object.__setattr__
        self = super(CleanSetAttrMeta, cls).__call__(*args, **kwargs)
        cls.__setattr__ = real_setattr
        return self


class MyDict(dict):
    """Dictionary subclass for d.attribute and d["attribute"] query"""

    def __init__(self, **kwargs):
        for key in kwargs:
            self[key] = kwargs[key]

    def __contains__(self, item):
        return super().__contains__(item.lower())

    def __getattr__(self, attr):
        """For accessing dict values via `<obj>.<key>`"""
        if attr in self:
            return self[attr]
        return super().__getattr__(attr)

    def __setattr__(self, attr, val):
        if type(val) == dict:
            val = MyDict(**val)
        self[attr] = val

    def __getitem__(self, attr):
        """Access via `<obj>["<key>"]`.
        Also has the ability to set and return a default value
        via `<obj>["<key>", default_val]`"""

        key = attr
        if type(attr) == tuple:
            key = attr[0]
            self.setdefault(key, attr[1])

        return super().__getitem__(key.lower())

    def __setitem__(self, attr, val):
        if type(val) == dict:
            val = MyDict(**val)
        super().__setitem__(attr.lower(), val)


class Conf(metaclass=CleanSetAttrMeta):
    from contextlib import contextmanager

    __metaclass__ = CleanSetAttrMeta

    def __init__(self, **kwargs):
        self.__config = ConfigParser(default_section="default",
                                     interpolation=ExtendedInterpolation())
        self.__d = MyDict(**{self.__config.default_section: {}})
        self.set(**kwargs)

    @contextmanager
    def change(self, **kwargs):
        original = {k: self[k] for k in kwargs if k in self.keys()}
        self.set(**kwargs)
        yield
        self.set(**original)

    @contextmanager
    def rollback(self):
        import copy
        original = copy.deepcopy(self.__d)
        yield
        super().__setattr__("_Conf__d", original)

    def keys(self):
        return self.__d[self.__config.default_section].keys()

    def items(self):
        return self.__d[self.__config.default_section].items()

    def update_dict(self):
        for sec in [self.__config.default_section] + self.__config.sections():
            self.__d.setdefault(sec, MyDict())
            for k, v in self.__config[sec].items():
                self.__d[sec][k] = tryeval(v)

    def add_section(self, sec):
        self.__d.setdefault(sec, MyDict())
        return self.__config.add_section(sec)

    def has_section(self, sec):
        return sec in self.__d.keys()

    def options(self, sec):
        return self.__d.get(sec, None)

    def has_option(self, section, option):
        return section in self.__d and option in self.__d[section]

    def read(self, filenames, encoding):
        ret = self.__config.read(filenames, encoding)
        self.update_dict()
        return ret

    def read_file(self, *args, **kwargs):
        ret = self.__config.read_file(*args, **kwargs)
        self.update_dict()
        return ret

    def read_string(self, *args, **kwargs):
        ret = self.__config.read_string(*args, **kwargs)
        self.update_dict()
        return ret

    def read_dict(self, dictionary):
        for k, v in dictionary.items():
            self.__d.setdefault(k, {})
            self.__d[k].update(v)

    def set(self, **kwargs):
        self.read_dict({self.__config.default_section: kwargs})

    def sections(self):
        return list(self.__d.keys())

    def write(self, *args, **kwargs):
        tmp_d = {}
        for sec, v in self.__d.items():
            tmp_d.setdefault(sec, {})
            for key, value in v.items():
                if value is None:
                    value = "None"
                tmp_d[sec][key] = value
        self.__config.read_dict(tmp_d)
        return self.__config.write(*args, **kwargs)

    def __getattr__(self, attr):
        if hasattr(super(), attr):
            return getattr(super(), attr)

        the_dict = self.__d[self.__config.default_section]
        if attr in self.sections():
            the_dict = self.__d[attr]

        if attr not in the_dict:
            raise AttributeError("{} not found in Config object.".format(attr))
        else:
            return the_dict[attr]
    __getitem__ = __getattr__

    def __setattr__(self, attr, val):
        # if hasattr(super(), attr):
        #     super().__setattr__(attr, val)
        #     return
        if type(val) == dict:
            self.read_dict({attr: val})
        else:
            self.__d[self.__config.default_section][attr] = val


if __name__ == "__main__":
    def __inspect(obj):
        print(type(obj), end=" " * 5)
        print(obj)

    d = dict(
        default=dict(
            mode="preact",
            batch_size=128,
            min_batch_size="${batch_size} // 3"
        )
    )
    d2 = dict(
        default=dict(
            c_width=40,
        )
    )
    s = """
[default]

c_width = 40
c_height = 100

# resnet config
depth = 50
mode = preact

batch_size = 128
channels = 3

max_epoch = 100
steps_per_epoch = 100

min_batch_size = ${batch_size} // 3
default_logging_lvl = getattr(logging, "DEBUG")
    """
    s2 = """
[default]

c_width = 80

# resnet config
depth = 18
mode = preact
    """
    config = Conf(
        # mode = "preact",
        # batch_size = 128,
        # min_batch_size = "${batch_size} // 3"
    )
    testd = {'default':
             {'gpu': '0',
              'theta': 1.0,
              'depth': 10,
              'default_logging_lvl': 'getattr(logging, "DEBUG")',
              'dense_net_BC': False,
              'min_len': 15,
              'load': None,
              'mode': 'preact',
              'channels': 3,
              'use_dense_net': False,
              'log_devices': False,
              'log_ask': False,
              'debug': True,
              'growth_rate': 12,
              'steps_per_epoch': 10000,
              'c_height': 100,
              'max_epoch': 30,
              'cmd': 'train',
              'out': '',
              'c_width': 40,
              'debug_cutout': False,
              'ip': '0.0.0.0',
              'batch_size': 128,
              'port': 9645,
              'min_batch_size': 42,
              'conf': None}}

    config.read_dict(testd)
    # config.default = d2["default"]
    # config.read_string(s)
    # print(list(config.items()))
    # config.read_string(s2)
    # print(list(config.items()))
    # config.test = 5
    # print(list(config.items()))
    # config.set(**dict(a=1, b=2))
    # print(list(config.items()))
    # print(type(config._Conf__d))
    # print(type(config._Conf__d.default))
    # __inspect(config.min_batch_size)
    # import logging
    # __inspect(config.default_logging_lvl)
    # import logging
    # __inspect(config.default_logging_lvl)
    # __inspect(config.c_width)
    # __inspect(config.default.c_width)
    # config.c_width = 8
    # __inspect(config.default.c_width)
    # __inspect(config.min_batch_size)
    # d = dict(test = 5, bla = 7)
    # config.set(hallo=d)

    __inspect(config.load)
    s = StringIO()
    config.write(s)
    s.seek(0)
    # print(s.getvalue())

    __inspect(config.load)
    # __inspect(config.hallo.test)
    # config.hallo.test = "new test"
    # __inspect(config.hallo.test)
    # __inspect(config.sections())
