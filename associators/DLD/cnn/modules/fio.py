from io import BytesIO

"""
Format | C Type             | Python Type       | size
x      | pad byte           | no value          |
c      | char               | bytes of length 1 | 1
b      | signed char        | integer           | 1
B      | unsigned char      | integer           | 1
?      | _Bool              | bool              | 1
h      | short              | integer           | 2
H      | unsigned short     | integer           | 2
i      | int                | integer           | 4
I      | unsigned int       | integer           | 4
l      | long               | integer           | 4
L      | unsigned long      | integer           | 4
q      | long long          | integer           | 8
Q      | unsigned long long | integer           | 8
n      | ssize_t            | integer           |
N      | size_t             | integer           |
f      | float              | float             | 4
d      | double             | float             | 8
s      | char[]             | bytes             |
p      | char[]             | bytes             |
P      | void *             | integer           |
"""


class FBytesIO(BytesIO):
    def unpack(self, fmt):
        """Unpack wrapper for automatic extraction of given format

        Arguments:
        fmt - Format of bytes to read (see struct.unpack)"""
        from struct import calcsize, unpack as s_unpack

        fmt_size = calcsize(fmt)
        data = self.read(fmt_size)
        if len(data) != fmt_size:
            exception = "Format mismatch: Tried to unpack {} bytes but got {}"
            raise IOError(exception.format(fmt_size, len(data)))

        ret = s_unpack(fmt, data)
        if len(ret) == 1:
            return ret[0]
        return ret

    def extract(self, fmt):
        """Unpack data of given format and yield a Generator object
        This generator keeps unpacking the last format.

        Arguments:
        fmt - Format of bytes to read (see struct.unpack)"""
        i = 0
        modifier = ""
        if (fmt[0] in "!<>"):
            i = 1
            modifier = fmt[0]

        try:
            while True:
                f = fmt[i]
                if i < (len(fmt) - 1):
                    i += 1
                try:
                    yield self.unpack(modifier + f)
                except Exception:
                    yield None
        finally:
            pass  # on GeneratorExit
