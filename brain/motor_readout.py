"""Fixed descending-neuron → leg interface, shared by manual and maze modes.

This replaces maze steering rules, not the missing VNC. Rates are population
means smoothed over 100 ms by ConnectomeBrain. The gains are uncalibrated.
"""
import numpy as np

DESCENDING_GROUPS = tuple(f'{cell}_{side}' for cell in ('DNp09', 'DNa02', 'MDN')
                          for side in ('left', 'right'))


def descending_motor(rates):
    forward = (rates['DNp09_left'] + rates['DNp09_right']) / 200
    reverse = (rates['MDN_left'] + rates['MDN_right']) / 200
    turn = (rates['DNa02_left'] - rates['DNa02_right']) / 100
    gains = np.clip([forward-reverse-.6*turn, forward-reverse+.6*turn], -1.2, 1.2)
    return {'forward': float(forward), 'reverse': float(reverse),
            'turn': float(turn), 'gains': gains.tolist()}
