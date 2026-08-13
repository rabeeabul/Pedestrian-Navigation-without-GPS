import threading


def start_enter_to_stop():
    # The PyCharm run console is not a real terminal, so single-key libraries such
    # as msvcrt do not see the keystroke. Reading a line does work, so Enter is the
    # stop key. Returns a function that becomes True once Enter has been pressed.
    state = {"stop": False}

    def _wait():
        try:
            input()
        except (EOFError, OSError, RuntimeError):
            return
        state["stop"] = True

    threading.Thread(target=_wait, daemon=True).start()
    return lambda: state["stop"]
