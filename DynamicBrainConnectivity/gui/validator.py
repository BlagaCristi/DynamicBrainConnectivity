import os


def check_if_path_valid(path):
    if os.path.exists(path):
        return True
    return False


def check_if_T_F(message):
    if message != 'T' and message != 'F':
        return False
    return True


def check_if_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def check_if_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def check_if_trial_or_window(message):
    if message != 'Trial' and message != 'Window':
        return False
    return True
