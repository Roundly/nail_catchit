import os
import numpy as np
from pathlib import Path

## Define the model path
path = os.path.realpath(__file__)
root = str(Path(path).parent)
ASSET_PATH = os.path.join(root, "../../assets")
# print("ASSET_PATH: ", ASSET_PATH)
# Use Leap Hand
XML_DCMM_LEAP_OBJECT_PATH = "urdf/nailrobot.xml"
XML_DCMM_LEAP_UNSEEN_OBJECT_PATH = "urdf/nailrobot_unseen.xml"
XML_ARM_PATH = "urdf/piper_right.xml"
## Weight Saved Path
WEIGHT_PATH = os.path.join(ASSET_PATH, "weights")

## The distance threshold to change the stage from 'tracking' to 'grasping'
distance_thresh = 0.25

## Define the initial joint positions of the arm and the hand
# 这个初始位置是PiPER机械臂的初始关节位置，防止机械臂被支架卡住
arm_joints = np.array([
   0.0, 1.57, -1.57, 0, 0, 0
])

hand_joints = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
])

## Define the reward weights
reward_weights = {
    "r_base_pos": 0.1,       # 基础位置奖励权重
    "r_ee_pos": 10.0,        # 末端执行器位置奖励权重
    "r_precision": 10.0,     # 精度奖励权重
    "r_orient": 1.0,         # 姿态奖励权重
    "r_touch": {            # 接触奖励权重
        'Tracking': 5,      # 跟踪阶段奖励权重
        'Catching': 0.1     # 捕捉阶段奖励权重
    },
    "r_constraint": 1.0,     # 约束奖励权重
    "r_stability": 20.0,     # 稳定性奖励权重
    "r_ctrl": {             # 控制奖励权重
        'base': 0.2,        # 底盘控制奖励权重
        'arm': 1.0,         # 机械臂控制奖励权重
        'hand': 0.2,        # 机械手控制奖励权重
    },
    "r_collision": -10.0,
}

## Define the camera params for the MujocoRenderer.
cam_config = {
    "name": "top",
    "width": 640,
    "height": 480,
}

## Define the params of the Double Ackerman model.
# 不使用双向Ackermann模型
RangerMiniV2Params = { 
  'wheel_radius': 0.1,                  # in meter //ranger-mini 0.1
  'steer_track': 0.364,                 # in meter (left & right wheel distance) //ranger-mini 0.364
  'wheel_base': 0.494,                   # in meter (front & rear wheel distance) //ranger-mini 0.494
  'max_linear_speed': 1.5,              # in m/s
  'max_angular_speed': 4.8,             # in rad/s
  'max_speed_cmd': 10.0,                # in rad/s
  'max_steer_angle_ackermann': 0.6981,  # 40 degree
  'max_steer_angle_parallel': 1.570,    # 180 degree
  'max_round_angle': 0.935671,
  'min_turn_radius': 0.47644,
}

## Define IK
ik_config = {
    "solver_type": "QP",   # 求解器类型：二次规划 (QP)
    "ps": 0.001,           # 步长参数
    "λΣ": 12.5,            # 正则化参数
    "ilimit": 100,         # 迭代次数上限
    "ee_tol": 1e-4         # 末端执行器容差
}

# Define the Randomization Params
## Wheel Drive
k_drive = np.array([0.75, 1.25])

## Arm Joints
k_arm = np.array([0.75, 1.25])
## Hand Joints
k_hand = np.array([0.75, 1.25])
## Object Shape and Size
object_shape = ["box", "cylinder", "sphere", "ellipsoid", "capsule"]
object_mesh = ["bottle_mesh", "bread_mesh", "bowl_mesh", "cup_mesh", "winnercup_mesh"]
object_size = {
    "sphere": np.array([[0.035, 0.045]]),   # 球体尺寸范围
    "capsule": np.array([[0.025, 0.035], [0.025, 0.04]]),  # 胶囊尺寸范围
    "cylinder": np.array([[0.025, 0.035], [0.025, 0.035]]),  # 圆柱尺寸范围
    "box": np.array([[0.025, 0.035], [0.025, 0.035], [0.025, 0.035]]),  # 盒子尺寸范围
    "ellipsoid": np.array([[0.03, 0.03], [0.045, 0.045], [0.045, 0.045]]),  # 椭球尺寸范围
}
object_mass = np.array([0.035, 0.075])      # 物体质量范围
object_damping = np.array([5e-3, 2e-2])      # 物体阻尼范围
object_static = np.array([0.5, 0.75])        # 物体静态摩擦系数范围
## Observation Noise
## 观测噪声参数
k_obs_base = 0.01
k_obs_arm = 0.001
k_obs_object = 0.01
k_obs_hand = 0.01
## Actions Noise
## 动作噪声参数
k_act = 0.025
## Action Delay
## 动作延迟参数
act_delay = {
    'base': [1,],       # 底盘动作延迟
    'arm': [1,],        # 机械臂动作延迟
    'hand': [1,],       # 机械手动作延迟
}

## Define PID params for wheel drive and steering. 
# driving
# 驾驶部分的 PID 参数
Kp_drive = 20          # 比例系数
Ki_drive = 1e-3       # 积分系数
Kd_drive = 1e-1       # 微分系数
llim_drive = -200     # 下限
ulim_drive = 200      # 上限

## Define PID params for the arm and hand. 
## 定义机械臂与机械手的 PID 参数PIPER用
# Kp_arm = np.array([15.0, 20.0, 20.0, 2.5, 10.0, 1.0])   # 机械臂比例系数 (显著降低)
# Ki_arm = np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.01])        # 机械臂积分系数 (适当增大)
# Kd_arm = np.array([20.0, 20.0, 20.0, 2.0, 5.0, 0.5])             # 机械臂微分系数 (可能需要后续微调)
# llim_arm = np.array([-300.0, -300.0, -300.0, -50.0, -50.0, -20.0])# 机械臂下限
# ulim_arm = np.array([300.0, 300.0, 300.0, 50.0, 50.0, 20.0])       # 机械臂上限

## 机械臂的 PID 参数（from catch it）
Kp_arm = np.array([300.0, 400.0, 400.0, 50.0, 200.0, 20.0])
Ki_arm = np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3])
Kd_arm = np.array([40.0, 40.0, 40.0, 5.0, 10.0, 1])
llim_arm = np.array([-300.0, -300.0, -300.0, -50.0, -50.0, -20.0])
ulim_arm = np.array([300.0, 300.0, 300.0, 50.0, 50.0, 20.0])

Kp_hand = np.array([4e-1, 1e-2, 2e-1, 2e-1,
                      4e-1, 1e-2, 2e-1, 2e-1,
                      4e-1, 1e-2, 2e-1, 2e-1,
                      1e-1, 1e-1, 1e-1, 1e-2,])
Ki_hand = 1e-2
Kd_hand = np.array([3e-2, 1e-3, 2e-3, 1e-3,
                      3e-2, 1e-3, 2e-3, 1e-3,
                      3e-2, 1e-3, 2e-3, 1e-3,
                      1e-2, 1e-2, 2e-2, 1e-3,])
llim_hand = -5.0
ulim_hand = 5.0
hand_mask = np.array([1, 0, 1, 1,
                      1, 0, 1, 1,
                      1, 0, 1, 1,
                      0, 1, 1, 1])