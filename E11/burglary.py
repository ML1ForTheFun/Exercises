'''This is the burglary example from Exercise 10.3'''
from bayesian.bbn import build_bbn

def f_burglary(burglary):
    if burglary is True:
        return 0.01
    return 0.99


def f_earthquake(earthquake):
    if earthquake is True:
        return 0.000001
    return 0.999999

def f_radioBroadcast(earthquake, radioBroadcast):
    table = dict()
    table['tt'] = 1.
    table['tf'] = 0.
    table['ft'] = 0.
    table['ff'] = 1.
    key = ''
    key = key + 't' if earthquake else key + 'f'
    key = key + 't' if radioBroadcast else key + 'f'
    return table[key]


def f_alarm(burglary, earthquake, alarm):
    table = dict()
    table['fft'] = 0.001
    table['fff'] = 0.999
    table['ftt'] = 0.41
    table['ftf'] = 0.59
    table['tft'] = 0.95
    table['tff'] = 0.05
    table['ttt'] = 0.98
    table['ttf'] = 0.02
    key = ''
    key = key + 't' if burglary else key + 'f'
    key = key + 't' if earthquake else key + 'f'
    key = key + 't' if alarm else key + 'f'
    return table[key]

if __name__ == '__main__':
    g = build_bbn(
        f_burglary,
        f_earthquake,
        f_radioBroadcast,
        f_alarm)
    g.q()
