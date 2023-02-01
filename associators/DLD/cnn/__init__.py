import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# for line_profiler
import builtins
if "profile" not in builtins.__dict__:
    from functools import wraps

    def wrpr(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper

    profile = wrpr

from cnn.modules.config import Conf # noqa
config = Conf()
config.read(os.path.join(dir_path, "default.conf"), "utf-8")
