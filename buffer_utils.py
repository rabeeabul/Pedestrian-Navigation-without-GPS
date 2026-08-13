import pandas as pd


def trim_deque_by_time(dq, newest_time, max_age):
    while dq and dq[0][0] < newest_time - max_age:
        dq.popleft()


def make_acc_dataframe(acc_buffer):
    return pd.DataFrame(list(acc_buffer), columns=["seconds_elapsed", "x", "y", "z"])


def make_gyro_dataframe(gyro_buffer):
    return pd.DataFrame(list(gyro_buffer), columns=["seconds_elapsed", "y"])
