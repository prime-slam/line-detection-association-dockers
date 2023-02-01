# import threading
import socket
from struct import pack, unpack
# from functools import wraps
import ctypes

from misc.logger import getLogger, DEBUG

_L = getLogger("client")
_L.setLevel(DEBUG)


class ClientCls:
    class Answer:
        def __init__(self, data=b"", error=None):
            self.data = data
            self.error = error

    magic = ctypes.c_uint32(0x9E2B83C1).value
    pack_magic = pack("!I", magic)

    def __init__(self, ip="0.0.0.0", port=9645):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.is_connected = False

        self.msg_count = 0

    def __pack_msg(self, fmt, val):
        return pack("!{}".format(fmt), val)

    def __unpack_one(self, fmt, b):
        return unpack("!{}".format(fmt), b)[0]

    def connect(self, timeout=5):
        if self.is_connected:
            return

        try:
            self.socket.connect((self.ip, self.port))

            answer = self.read_answer()

            if answer.error:
                _L.error("Client Error whilst connecting: "
                         "{}".format(answer.error))
                self.socket.close()
                self.is_connected = False
            else:
                self.is_connected = True
                _L.info("Client connected")

            return self.is_connected, answer

        except Exception as e:
            _L.error("Error while connecting: {}".format(e))
            return False, ClientCls.Answer(error=str(e))

    def send(self, command, message=None, wait=True):
        # b_msg: the message will be sent in the order in which
        # it is given! (Independent from the magic packet.)
        if message is None:
            message = ""

        if type(message) == bytes:
            b_msg = bytes("{}:{}".format(self.msg_count, command),
                          "utf-8")
            b_msg += message
        else:
            b_msg = bytes("{}:{}{}".format(self.msg_count, command, message),
                          "utf-8")
        payload_size = len(b_msg)
        pack_ps = self.__pack_msg("I", payload_size)

        with self.socket.makefile("wb", len(b_msg) + 8) as wfile:
            wfile.write(ClientCls.pack_magic)
            wfile.write(pack_ps)
            wfile.write(b_msg)
            wfile.flush()
        self.msg_count += 1

        answer = None
        if wait:
            answer = self.read_answer()
            if answer.error:
                print("Sending Error: {}".format(answer.error))
        return answer

    def read_answer(self):
        ret = ClientCls.Answer()

        with self.socket.makefile("rb", 0) as rfile:
            _magic = rfile.read(4)
            if not _magic:
                ret.error = "remote disconnected"
                return ret

            # unpack magic
            umagic_i = unpack("!I", _magic)[0]
            if umagic_i != ClientCls.magic:
                ret.error = ("no matching magic number: "
                             "!{}".format(hex(umagic_i)))
                return ret

            # read payload_size
            _payload_size = rfile.read(4)
            payload_size = unpack("!I", _payload_size)[0]
            data_left = payload_size

            while data_left:
                ret.data += rfile.read(data_left)
                data_left = payload_size - len(ret.data)

        return ret


Client = ClientCls
# def memo(f):
#     clts = {}

#     @wraps(f)
#     def wrapper(*args, **kwargs):
#         key = str(args) + str(kwargs)
#         if key not in clts:
#             clts[key] = f(*args, **kwargs)
#         return clts[key]
#     return wrapper


# @memo
# def Client(ip, port):
#     return ClientCls(ip, port)
