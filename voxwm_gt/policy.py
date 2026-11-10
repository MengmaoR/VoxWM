"""Real actions only; no per-frame camera or player-position assignments."""
import numpy as np

class RotatePolicy:
    def __init__(self, info, turns=2.5):
        self.yaw = float(info['player_yaw'])
        self.rotation = 0.
        self.turns = turns
        self.done = False

    def action(self, wrapper, info):
        yaw = float(info['player_yaw'])
        self.rotation += abs((yaw-self.yaw+180.) % 360.-180.)
        self.yaw = yaw
        self.done = self.rotation >= self.turns*360.
        action = np.zeros(4,dtype=np.int64)
        if not self.done: action[2] = wrapper.actions[2].index('mouse x+0.1')+1
        return action
