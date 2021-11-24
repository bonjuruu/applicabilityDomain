import time


def time2str(time_elapsed, formatstr='%Hh%Mm%Ss'):
    return time.strftime(formatstr, time.gmtime(time_elapsed))
