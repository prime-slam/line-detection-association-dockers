# -*- coding: utf-8 -*-

def hide_cursor(): # noqa
    import sys
    sys.stdout.write("[?25l")
    sys.stdout.flush()
def show_cursor(): # noqa
    import sys
    sys.stdout.write("[?25h")
    sys.stdout.flush()


def getch():
    """Query for one Key input.
    This function returns after getting exactly one keypress"""
    import termios
    import tty
    import sys
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
