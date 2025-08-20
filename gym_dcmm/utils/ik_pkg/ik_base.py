import os, sys
sys.path.append(os.path.abspath('../../'))
import configs.env.DcmmCfg as DcmmCfg
import math
import numpy as np

def Damper(value, min_val, max_val):
    # 将输入值限制在 min_val 和 max_val 之间
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value

def IKBase(vx, vy):
    """
    Calculate the inverse kinematics for a slider mechanism.
    计算滑块的逆运动学

    Inputs:
        vx: Linear velocity of the slider along the x-axis (m/s)
        vy: Linear velocity of the slider along the y-axis (m/s)

    Outputs:
        vel_cmd: Velocity commands for the slider
    """
    # Disregard very small velocities to avoid noise
    if math.fabs(vy) < 0.01: vy = 0.0
    if math.fabs(vx) < 0.01: vx = 0.0
    
    # If both velocities are near zero, return zero velocity commands
    if math.fabs(vx) < 0.01 and math.fabs(vy) < 0.01:
        return np.array([0.0, 0.0])

    # Compute the total linear velocity (magnitude)
    vel_magnitude = math.hypot(vx, vy)

    # Command velocity is directly proportional to the linear velocity in x and y
    # You can further adjust scaling if needed based on system configuration
    vel_cmd = np.array([vx, vy])
    return vel_cmd