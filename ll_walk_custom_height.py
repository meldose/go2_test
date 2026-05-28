import argparse
import curses
import glob
import json
import math
import os
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LidarState_, LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

import unitree_legged_const as go2


TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_ODOM = "rt/odom"
TOPIC_LIDAR_STATE = "rt/utlidar/map_state"
TOPIC_LIDAR_CLOUD = "rt/utlidar/cloud"

CONTROL_DT = 0.002
POSTURE_BLEND_DT = 0.02
IDLE_POSE_BLEND_TIME = 0.30
EXIT_SIT_BLEND_TIME = 1.8
NORMAL_HEIGHT_M = 0.0
HEIGHT_STEP_M = 0.01
LOWERED_OFFSET_M = -0.05
MIN_HEIGHT_OFFSET_M = -0.10
MAX_HEIGHT_OFFSET_M = 0.06
COMMAND_RAMP_RATE = 0.45
GAIT_ENGAGE_RATE = 0.20
MAX_TILT_FOR_WALK_RAD = 0.30
MOVE_STEP = 0.20
TURN_STEP = 0.20
STEP_HEIGHT_STEP = 0.02
STEP_HEIGHT_SCALE_MIN = 1.0
STEP_HEIGHT_SCALE_MAX = 2.2
DEFAULT_LIFT_FOOT_Z = 0.10
MIN_SWING_CLEARANCE = 0.075
MAX_BODY_SHIFT_X = 0.025
MAX_BODY_SHIFT_Y = 0.025
BODY_SHIFT_GAIN = 0.9
IMU_PITCH_POS_GAIN = 0.035
IMU_ROLL_POS_GAIN = 0.03
IMU_GYRO_GAIN = 0.003
MAX_IMU_FOOT_Z = 0.025
LATERAL_WALK_SCALE = 0.20
YAW_WALK_SCALE = 0.15
ZMP_ROLL_P = 0.10
ZMP_ROLL_D = 0.02
ZMP_PITCH_P = 0.12
ZMP_PITCH_D = 0.02
ZMP_FORCE_GAIN = 0.0015
ZMP_EXT_CLAMP = 0.16
ZMP_THIGH_GAIN = 0.30
ZMP_CALF_GAIN = -0.45
REMOTE_DEADBAND = 0.12
Crawl_CONTACT_FORCE_MIN = 20.0
CRAWL_SHIFT_HOLD_SEC = 0.18
CRAWL_LIFT_SEC = 0.22
CRAWL_SWING_SEC = 0.30
CRAWL_LOWER_TIMEOUT_SEC = 0.40
CRAWL_SETTLE_SEC = 0.22
CRAWL_STEP_X = 0.09
CRAWL_SHIFT_X = 0.020
CRAWL_SHIFT_Y = 0.018
CRAWL_TILT_OK_RAD = 0.12
LOWLEVEL_STOP_WAIT = 3.0
SERVICE_TOGGLE_DELAY = 0.5
SERVICE_RESTART_WAIT = 2.0
SERVICES_TO_ENABLE = ("mcf", "sport_mode")
MODE_ALIASES = ("normal", "mcf")
IDLE_KP = 42.0
IDLE_KD = 3.5

LEG_INDEX = {
    "FR": (0, 1, 2),
    "FL": (3, 4, 5),
    "RR": (6, 7, 8),
    "RL": (9, 10, 11),
}
LEG_ORDER = ["FR", "FL", "RR", "RL"]
CRAWL_ORDER = ["RL", "FL", "RR", "FR"]
LEG_SIGNS = {
    "FL": {"left": 1.0, "front": 1.0},
    "FR": {"left": -1.0, "front": 1.0},
    "RL": {"left": 1.0, "front": -1.0},
    "RR": {"left": -1.0, "front": -1.0},
}

# Geometry extracted from the Go2 URDF and used for analytic leg IK.
HIP_ORIGIN = {
    "FL": (0.1934, 0.0465, 0.0),
    "FR": (0.1934, -0.0465, 0.0),
    "RL": (-0.1934, 0.0465, 0.0),
    "RR": (-0.1934, -0.0465, 0.0),
}
HIP_LATERAL_OFFSET = 0.0955
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213

GAITS = {
    "Walk": {
        "description": "Four-beat lateral walk: RL -> FL -> RR -> FR",
        "phase_offsets": {"RL": 0.00, "FL": 0.25, "RR": 0.50, "FR": 0.75},
        "duty": 0.72,
        "cycle_sec": 2.10,
        "step_height": 0.06,
        "step_length": 0.08,
        "body_roll": 0.015,
    },
    "Trot": {
        "description": "Diagonal trot",
        "phase_offsets": {"FR": 0.00, "RL": 0.00, "FL": 0.50, "RR": 0.50},
        "duty": 0.54,
        "cycle_sec": 1.05,
        "step_height": 0.28,
        "step_length": 0.28,
        "body_roll": 0.03,
    },
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp_pose(src, dst, alpha: float):
    return [(1.0 - alpha) * a + alpha * b for a, b in zip(src, dst)]


def normalize_axis(value: float, deadband: float) -> float:
    if abs(value) <= deadband:
        return 0.0
    scaled = (abs(value) - deadband) / max(1e-6, (1.0 - deadband))
    return math.copysign(clamp(scaled, 0.0, 1.0), value)


def smoothstep(alpha: float) -> float:
    alpha = clamp(alpha, 0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def vec3(values):
    return f"{values[0]: .2f} {values[1]: .2f} {values[2]: .2f}"


def quat_to_yaw(quat) -> float:
    x, y, z, w = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def extract_point_count(msg: Optional[PointCloud2_]) -> int:
    if msg is None:
        return 0
    return int(msg.width) * int(msg.height)


def clamp_abs(value: float, limit: float) -> float:
    return clamp(value, -limit, limit)


def leg_forward_kinematics(leg_name: str, joint_triplet):
    q_hip, q_thigh, q_calf = joint_triplet
    side_sign = LEG_SIGNS[leg_name]["left"]
    px = -THIGH_LENGTH * math.sin(q_thigh) - CALF_LENGTH * math.sin(q_thigh + q_calf)
    pzr = -THIGH_LENGTH * math.cos(q_thigh) - CALF_LENGTH * math.cos(q_thigh + q_calf)
    py = side_sign * HIP_LATERAL_OFFSET * math.cos(q_hip) - pzr * math.sin(q_hip)
    pz = side_sign * HIP_LATERAL_OFFSET * math.sin(q_hip) + pzr * math.cos(q_hip)
    return [px, py, pz]


def leg_inverse_kinematics(leg_name: str, foot_pos_hip):
    x, y, z = foot_pos_hip
    side_sign = LEG_SIGNS[leg_name]["left"]
    radial_sq = y * y + z * z - HIP_LATERAL_OFFSET * HIP_LATERAL_OFFSET
    radial = math.sqrt(max(radial_sq, 1e-9))
    q_hip = math.atan2(y, -z) - math.atan2(side_sign * HIP_LATERAL_OFFSET, radial)

    knee_cos = (
        x * x + radial_sq - THIGH_LENGTH * THIGH_LENGTH - CALF_LENGTH * CALF_LENGTH
    ) / (2.0 * THIGH_LENGTH * CALF_LENGTH)
    knee_cos = clamp(knee_cos, -1.0, 1.0)
    q_calf = -math.acos(knee_cos)
    q_thigh = math.atan2(-x, radial) - math.atan2(
        CALF_LENGTH * math.sin(q_calf),
        THIGH_LENGTH + CALF_LENGTH * math.cos(q_calf),
    )
    return [q_hip, q_thigh, q_calf]


def age_text(age: Optional[float]) -> str:
    if age is None:
        return "--"
    return f"{age:4.1f}s"


def status_text(age: Optional[float], timeout: float = 1.5) -> str:
    if age is None:
        return "waiting"
    if age <= timeout:
        return "live"
    return "stale"


def find_latest_recording():
    candidates = sorted(glob.glob("record_gait_*.jsonl"))
    if not candidates:
        return None
    return candidates[-1]


def decode_buttons(data):
    data1 = int(data[2])
    data2 = int(data[3])
    return {
        "R1": (data1 >> 0) & 1,
        "L1": (data1 >> 1) & 1,
        "Start": (data1 >> 2) & 1,
        "Select": (data1 >> 3) & 1,
        "R2": (data1 >> 4) & 1,
        "L2": (data1 >> 5) & 1,
        "F1": (data1 >> 6) & 1,
        "F3": (data1 >> 7) & 1,
        "A": (data2 >> 0) & 1,
        "B": (data2 >> 1) & 1,
        "X": (data2 >> 2) & 1,
        "Y": (data2 >> 3) & 1,
        "Up": (data2 >> 4) & 1,
        "Right": (data2 >> 5) & 1,
        "Down": (data2 >> 6) & 1,
        "Left": (data2 >> 7) & 1,
    }


def decode_remote(data):
    raw = bytes(data)
    return {
        "lx": struct.unpack("<f", raw[4:8])[0],
        "rx": struct.unpack("<f", raw[8:12])[0],
        "ry": struct.unpack("<f", raw[12:16])[0],
        "ly": struct.unpack("<f", raw[20:24])[0],
        "buttons": decode_buttons(raw),
        "raw_hex": raw.hex(),
    }


class RecordedGait:
    def __init__(self, path: str):
        self.path = path
        self.frames = []
        self.reference_pose = [0.0] * 12
        self.duration = 0.0
        self.valid = False
        self.error = ""
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw_frames = [json.loads(line) for line in f if line.strip()]
        except Exception as exc:
            self.error = str(exc)
            return

        if len(raw_frames) < 2:
            self.error = "recording has fewer than 2 frames"
            return

        t0 = float(raw_frames[0]["wall_time"])
        q_sum = [0.0] * 12
        parsed = []
        for frame in raw_frames:
            joints = frame.get("joints", [])
            if len(joints) < 12:
                continue
            q = [float(joints[i]["q"]) for i in range(12)]
            dq = [float(joints[i].get("dq", 0.0)) for i in range(12)]
            t = max(0.0, float(frame["wall_time"]) - t0)
            parsed.append({"t": t, "q": q, "dq": dq})
            for i in range(12):
                q_sum[i] += q[i]

        if len(parsed) < 2:
            self.error = "recording has too few valid joint frames"
            return

        count = float(len(parsed))
        self.reference_pose = [v / count for v in q_sum]
        self.frames = parsed
        self.duration = max(parsed[-1]["t"], CONTROL_DT)
        self.valid = True

    def sample(self, phase_time: float):
        if not self.valid:
            return None

        t = phase_time % self.duration
        frames = self.frames
        lo = frames[0]
        hi = frames[-1]
        for idx in range(1, len(frames)):
            if frames[idx]["t"] >= t:
                lo = frames[idx - 1]
                hi = frames[idx]
                break

        span = max(hi["t"] - lo["t"], 1e-6)
        alpha = clamp((t - lo["t"]) / span, 0.0, 1.0)
        q = [(1.0 - alpha) * lo["q"][i] + alpha * hi["q"][i] for i in range(12)]
        dq = [(1.0 - alpha) * lo["dq"][i] + alpha * hi["dq"][i] for i in range(12)]
        delta = [q[i] - self.reference_pose[i] for i in range(12)]
        return {"q": q, "dq": dq, "delta": delta}


@dataclass
class TopicSnapshot:
    low_state: Optional[LowState_] = None
    odom: Optional[Odometry_] = None
    lidar_state: Optional[LidarState_] = None
    lidar_cloud: Optional[PointCloud2_] = None
    low_state_time: float = 0.0
    odom_time: float = 0.0
    lidar_state_time: float = 0.0
    lidar_cloud_time: float = 0.0
    motion_mode: str = "unknown"
    motion_code: int = 0
    motion_time: float = 0.0
    motion_error: str = ""
    service_status: str = "hl active"
    service_time: float = 0.0
    last_button_event: str = ""
    last_button_time: float = 0.0
    remote_lx: float = 0.0
    remote_ly: float = 0.0
    remote_rx: float = 0.0


class LowLevelWalkController:
    def __init__(self, recorded_gait: Optional[RecordedGait]):
        self.Kp = 60.0
        self.Kd = 5.0
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()
        self.lock = threading.Lock()
        self.snapshots = TopicSnapshot()
        self.first_run = True

        self.stand_pose = [
            0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
            0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
        ]
        self.sit_pose = [
            0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
            -0.2, 1.36, -2.65, 0.2, 1.36, -2.65,
        ]
        self.height_pose = [
            -0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
            -0.5, 1.36, -2.65, 0.5, 1.36, -2.65,
        ]
        self.stand_foot_targets = self._build_nominal_foot_targets(self.stand_pose)
        self.lowered_foot_targets = self._build_nominal_foot_targets(self.height_pose)
        self.start_pose = [0.0] * 12
        self.current_pose = list(self.sit_pose)
        self.target_pose = list(self.sit_pose)
        self.height_offset_m = NORMAL_HEIGHT_M

        self.low_level_mode_active = False
        self.handoff_in_progress = False
        self.walk_enabled = False
        self.gait_name = "Walk"
        self.step_height_scale = 1.0
        self.move_x = 0.0
        self.move_y = 0.0
        self.move_yaw = 0.0
        self.command_move_x = 0.0
        self.command_move_y = 0.0
        self.command_move_yaw = 0.0
        self.gait_engage = 0.0
        self.phase = 0.0
        self.recorded_gait = recorded_gait
        self.recorded_phase_time = 0.0
        self.recorded_gait_speed = 1.0
        self.roll = 0.0
        self.pitch = 0.0
        self.gx = 0.0
        self.gy = 0.0
        self.foot_force = [0.0, 0.0, 0.0, 0.0]
        self.remote = None
        self.swing_legs = set()
        self.prev_toggle_combo = False
        self.prev_buttons = {}
        self.prev_height_up = False
        self.prev_height_down = False
        self.crawl_state = "idle"
        self.crawl_leg_index = 0
        self.crawl_state_started = 0.0
        self.crawl_support_started = 0.0
        self.button_log_path = os.path.abspath("ll_walk_custom_height_buttons.log")

        self.sequence = []
        self.sequence_done = True
        self.sequence_index = 0
        self.sequence_hold_started = 0.0
        self.manual_override = True

        self.transition_start_pose = list(self.sit_pose)
        self.transition_target_pose = list(self.sit_pose)
        self.transition_started = 0.0
        self.transition_duration = 1.0
        self.transition_active = False

        self.lowCmdWriteThreadPtr = None
        self.modePollThreadPtr = None

    def init(self, odom_topic: str, lidar_state_topic: str, lidar_cloud_topic: str):
        self._init_low_cmd()

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        self.odom_subscriber = ChannelSubscriber(odom_topic, Odometry_)
        self.odom_subscriber.Init(self._odom_handler, 10)

        self.lidar_state_subscriber = ChannelSubscriber(lidar_state_topic, LidarState_)
        self.lidar_state_subscriber.Init(self._lidar_state_handler, 10)

        self.lidar_cloud_subscriber = ChannelSubscriber(lidar_cloud_topic, PointCloud2_)
        self.lidar_cloud_subscriber.Init(self._lidar_cloud_handler, 10)

        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        self.robot_state = RobotStateClient()
        self.robot_state.SetTimeout(5.0)
        self.robot_state.Init()

    def start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=CONTROL_DT, target=self._low_cmd_write, name="ll_walk_custom_height_lowcmd"
        )
        self.lowCmdWriteThreadPtr.Start()
        self.modePollThreadPtr = RecurrentThread(
            interval=0.5, target=self._poll_motion_mode, name="ll_walk_custom_height_mode_poll"
        )
        self.modePollThreadPtr.Start()

    def stop(self):
        if self.lowCmdWriteThreadPtr is not None:
            self.lowCmdWriteThreadPtr.Wait(1.0)
        if self.modePollThreadPtr is not None:
            self.modePollThreadPtr.Wait(1.0)
        try:
            self.sc.StandDown()
        except Exception:
            pass

    def increase_height(self):
        with self.lock:
            self._set_height_offset_locked(self.height_offset_m + HEIGHT_STEP_M, 0.35)

    def decrease_height(self):
        with self.lock:
            self._set_height_offset_locked(self.height_offset_m - HEIGHT_STEP_M, 0.35)

    def increase_step_height(self):
        with self.lock:
            self.step_height_scale = clamp(
                self.step_height_scale + STEP_HEIGHT_STEP, STEP_HEIGHT_SCALE_MIN, STEP_HEIGHT_SCALE_MAX
            )

    def decrease_step_height(self):
        with self.lock:
            self.step_height_scale = clamp(
                self.step_height_scale - STEP_HEIGHT_STEP, STEP_HEIGHT_SCALE_MIN, STEP_HEIGHT_SCALE_MAX
            )

    def toggle_walk(self):
        with self.lock:
            self._cancel_sequence_locked()
            self.walk_enabled = not self.walk_enabled
            if not self.walk_enabled:
                self.move_x = 0.0
                self.move_y = 0.0
                self.move_yaw = 0.0
                self.command_move_x = 0.0
                self.command_move_y = 0.0
                self.command_move_yaw = 0.0
                self.phase = 0.0
                self.recorded_phase_time = 0.0
                self.gait_engage = 0.0
                self.crawl_leg_index = 0
                self._begin_crawl_state_locked("idle")
                self.target_pose = self._compose_base_pose_locked()

    def cycle_gait(self):
        with self.lock:
            gait_names = list(GAITS.keys())
            if self.recorded_gait is not None and self.recorded_gait.valid:
                gait_names.append("Recorded")
            idx = gait_names.index(self.gait_name)
            self.gait_name = gait_names[(idx + 1) % len(gait_names)]
            self.phase = 0.0
            self.recorded_phase_time = 0.0
            self.gait_engage = 0.0
            self.crawl_leg_index = 0
            self._begin_crawl_state_locked("idle")

    def adjust_move_x(self, delta: float):
        with self.lock:
            self.move_x = clamp(self.move_x + delta, -1.0, 1.0)
            if abs(self.move_x) < 0.05:
                self.move_x = 0.0

    def adjust_move_y(self, delta: float):
        with self.lock:
            self.move_y = clamp(self.move_y + delta, -1.0, 1.0)
            if abs(self.move_y) < 0.05:
                self.move_y = 0.0

    def adjust_move_yaw(self, delta: float):
        with self.lock:
            self.move_yaw = clamp(self.move_yaw + delta, -1.0, 1.0)
            if abs(self.move_yaw) < 0.05:
                self.move_yaw = 0.0

    def zero_commands(self):
        with self.lock:
            self.move_x = 0.0
            self.move_y = 0.0
            self.move_yaw = 0.0

    def get_snapshot(self):
        with self.lock:
            return {
                "low_state": self.snapshots.low_state,
                "odom": self.snapshots.odom,
                "lidar_state": self.snapshots.lidar_state,
                "lidar_cloud": self.snapshots.lidar_cloud,
                "low_state_age": self._age(self.snapshots.low_state_time),
                "odom_age": self._age(self.snapshots.odom_time),
                "lidar_state_age": self._age(self.snapshots.lidar_state_time),
                "lidar_cloud_age": self._age(self.snapshots.lidar_cloud_time),
                "motion_mode": self.snapshots.motion_mode,
                "motion_code": self.snapshots.motion_code,
                "motion_age": self._age(self.snapshots.motion_time),
                "motion_error": self.snapshots.motion_error,
                "service_status": self.snapshots.service_status,
                "service_age": self._age(self.snapshots.service_time),
                "last_button_event": self.snapshots.last_button_event,
                "last_button_age": self._age(self.snapshots.last_button_time),
                "height_offset_m": self.height_offset_m,
                "step_height_scale": self.step_height_scale,
                "low_level_mode_active": self.low_level_mode_active,
                "handoff_in_progress": self.handoff_in_progress,
                "walk_enabled": self.walk_enabled,
                "gait_name": self.gait_name,
                "gait_description": self._gait_description(),
                "move_x": self.move_x,
                "move_y": self.move_y,
                "move_yaw": self.move_yaw,
                "crawl_state": self.crawl_state,
                "crawl_leg": CRAWL_ORDER[self.crawl_leg_index % len(CRAWL_ORDER)],
                "sequence_done": self.sequence_done,
                "recorded_gait_loaded": self.recorded_gait is not None and self.recorded_gait.valid,
                "recorded_gait_path": self.recorded_gait.path if self.recorded_gait is not None else "",
                "recorded_gait_error": "" if self.recorded_gait is None else self.recorded_gait.error,
                "roll": self.roll,
                "pitch": self.pitch,
                "foot_force": list(self.foot_force),
                "remote_lx": self.snapshots.remote_lx,
                "remote_ly": self.snapshots.remote_ly,
                "remote_rx": self.snapshots.remote_rx,
            }

    def _age(self, ts: float) -> Optional[float]:
        if ts <= 0.0:
            return None
        return max(0.0, time.time() - ts)

    def _blend_height(self, offset_m: float):
        alpha = clamp(
            (offset_m - MIN_HEIGHT_OFFSET_M) / (MAX_HEIGHT_OFFSET_M - MIN_HEIGHT_OFFSET_M), 0.0, 1.0
        )
        return lerp_pose(self.height_pose, self.stand_pose, alpha)

    def _build_nominal_foot_targets(self, pose):
        foot_targets = {}
        for leg_name, indices in LEG_INDEX.items():
            joints = pose[indices[0]: indices[2] + 1]
            hip_origin = HIP_ORIGIN[leg_name]
            foot_hip = leg_forward_kinematics(leg_name, joints)
            foot_targets[leg_name] = [
                hip_origin[0] + foot_hip[0],
                hip_origin[1] + foot_hip[1],
                hip_origin[2] + foot_hip[2],
            ]
        return foot_targets

    def _height_foot_targets(self, offset_m: float):
        alpha = clamp(
            (offset_m - MIN_HEIGHT_OFFSET_M) / (MAX_HEIGHT_OFFSET_M - MIN_HEIGHT_OFFSET_M), 0.0, 1.0
        )
        targets = {}
        for leg_name in LEG_ORDER:
            stand_target = self.stand_foot_targets[leg_name]
            lowered_target = self.lowered_foot_targets[leg_name]
            targets[leg_name] = [
                (1.0 - alpha) * lowered_target[i] + alpha * stand_target[i] for i in range(3)
            ]
        return targets

    def _imu_balance_offsets(self):
        if self.low_state is None:
            return {leg: 0.0 for leg in LEG_ORDER}
        pitch_term = clamp_abs(
            -(IMU_PITCH_POS_GAIN * self.pitch + IMU_GYRO_GAIN * self.gy),
            MAX_IMU_FOOT_Z,
        )
        roll_term = clamp_abs(
            -(IMU_ROLL_POS_GAIN * self.roll + IMU_GYRO_GAIN * self.gx),
            MAX_IMU_FOOT_Z,
        )
        offsets = {}
        for leg_name, signs in LEG_SIGNS.items():
            offsets[leg_name] = signs["front"] * pitch_term + signs["left"] * roll_term
        return offsets

    def _support_body_shift(self, support_legs, foot_targets):
        if not support_legs:
            return (0.0, 0.0)
        centroid_x = sum(foot_targets[leg][0] for leg in support_legs) / len(support_legs)
        centroid_y = sum(foot_targets[leg][1] for leg in support_legs) / len(support_legs)
        shift_x = clamp_abs(BODY_SHIFT_GAIN * centroid_x, MAX_BODY_SHIFT_X)
        shift_y = clamp_abs(BODY_SHIFT_GAIN * centroid_y, MAX_BODY_SHIFT_Y)
        return (shift_x, shift_y)

    def _foot_targets_to_pose(self, foot_targets):
        pose = [0.0] * 12
        for leg_name, indices in LEG_INDEX.items():
            hip_origin = HIP_ORIGIN[leg_name]
            target = foot_targets[leg_name]
            foot_pos_hip = [
                target[0] - hip_origin[0],
                target[1] - hip_origin[1],
                target[2] - hip_origin[2],
            ]
            q_hip, q_thigh, q_calf = leg_inverse_kinematics(leg_name, foot_pos_hip)
            pose[indices[0]] = q_hip
            pose[indices[1]] = q_thigh
            pose[indices[2]] = q_calf
        return pose

    def _compose_base_pose_locked(self):
        foot_targets = self._height_foot_targets(self.height_offset_m)
        imu_balance = self._imu_balance_offsets()
        for leg_name in LEG_ORDER:
            foot_targets[leg_name][2] += imu_balance[leg_name]
        return self._foot_targets_to_pose(foot_targets)

    def _gait_description(self):
        if self.gait_name == "Recorded":
            if self.recorded_gait is None:
                return "Recorded gait unavailable"
            if not self.recorded_gait.valid:
                return f"Recorded gait error: {self.recorded_gait.error}"
            return f"Playback from {os.path.basename(self.recorded_gait.path)}"
        return GAITS[self.gait_name]["description"]

    def _begin_transition_locked(self, pose, duration: float):
        self.transition_start_pose = list(self.current_pose)
        self.transition_target_pose = list(pose)
        self.transition_started = time.time()
        self.transition_duration = max(duration, POSTURE_BLEND_DT)
        self.transition_active = True
        self.target_pose = list(pose)

    def _set_height_offset_locked(self, new_offset: float, duration: float = 0.35):
        self._cancel_sequence_locked()
        self.height_offset_m = clamp(new_offset, MIN_HEIGHT_OFFSET_M, MAX_HEIGHT_OFFSET_M)
        self._begin_transition_locked(self._compose_base_pose_locked(), duration)

    def _cancel_sequence_locked(self):
        self.manual_override = True
        self.sequence_done = True
        self.sequence_index = len(self.sequence)
        self.sequence_hold_started = 0.0

    def _advance_sequence_locked(self):
        if self.manual_override or self.sequence_done or not self.sequence:
            return
        if self.sequence_index >= len(self.sequence):
            self.sequence_done = True
            self.height_offset_m = NORMAL_HEIGHT_M
            self.target_pose = list(self.stand_pose)
            return
        if not self.transition_active and self.sequence_hold_started == 0.0:
            step = self.sequence[self.sequence_index]
            self._begin_transition_locked(step["pose"], step["duration"])
            return
        if self.transition_active:
            return
        if self.sequence_hold_started == 0.0:
            self.sequence_hold_started = time.time()
            return
        step = self.sequence[self.sequence_index]
        if time.time() - self.sequence_hold_started >= step["hold"]:
            self.sequence_index += 1
            self.sequence_hold_started = 0.0

    def _apply_transition_locked(self):
        if not self.transition_active:
            return
        elapsed = time.time() - self.transition_started
        alpha = smoothstep(elapsed / self.transition_duration)
        self.current_pose = lerp_pose(self.transition_start_pose, self.transition_target_pose, alpha)
        if alpha >= 1.0:
            self.current_pose = list(self.transition_target_pose)
            self.transition_active = False
            self.target_pose = list(self.current_pose)

    def _ramp_value(self, current: float, target: float, rate: float) -> float:
        step = rate * CONTROL_DT
        if abs(target - current) <= step:
            return target
        return current + step if target > current else current - step

    def _begin_crawl_state_locked(self, state: str):
        self.crawl_state = state
        self.crawl_state_started = time.time()
        if state == "shift":
            self.crawl_support_started = 0.0

    def _crawl_active_leg(self):
        return CRAWL_ORDER[self.crawl_leg_index % len(CRAWL_ORDER)]

    def _leg_force(self, leg_name: str) -> float:
        index_map = {"FR": 0, "FL": 1, "RR": 2, "RL": 3}
        return self.foot_force[index_map[leg_name]]

    def _crawl_height_scale(self) -> float:
        # Higher stance means less stable crawl. Reduce aggressiveness automatically.
        return clamp(1.0 - max(0.0, self.height_offset_m) * 4.0, 0.45, 1.0)

    def _crawl_targets_locked(self):
        base_targets = self._height_foot_targets(self.height_offset_m)
        base_pose = self._foot_targets_to_pose(base_targets)
        tilt_mag = max(abs(self.roll), abs(self.pitch))
        self.command_move_x = self._ramp_value(self.command_move_x, self.move_x, COMMAND_RAMP_RATE)

        if not self.walk_enabled or abs(self.command_move_x) < 0.03:
            self.crawl_state = "idle"
            self.crawl_support_started = 0.0
            self.swing_legs = set()
            self.target_pose = list(base_pose)
            return base_pose

        if tilt_mag > MAX_TILT_FOR_WALK_RAD:
            self.walk_enabled = False
            self.command_move_x = 0.0
            self.crawl_state = "idle"
            self.swing_legs = set()
            self.target_pose = list(base_pose)
            self.snapshots.service_status = "crawl halted: tilt limit"
            self.snapshots.service_time = time.time()
            return base_pose

        if self.crawl_state == "idle":
            self._begin_crawl_state_locked("shift")

        active_leg = self._crawl_active_leg()
        support_legs = [leg for leg in CRAWL_ORDER if leg != active_leg]
        shift_targets = {leg: list(target) for leg, target in base_targets.items()}
        direction = 1.0 if self.command_move_x >= 0.0 else -1.0
        height_scale = self._crawl_height_scale()
        shift_x = CRAWL_SHIFT_X * direction * height_scale
        shift_y = 0.0
        if active_leg in ("FL", "RL"):
            shift_y = -CRAWL_SHIFT_Y * height_scale
        else:
            shift_y = CRAWL_SHIFT_Y * height_scale
        for leg in support_legs:
            shift_targets[leg][0] -= shift_x
            shift_targets[leg][1] -= shift_y
        shift_targets[active_leg][0] -= shift_x * 0.35
        shift_targets[active_leg][1] -= shift_y * 0.35

        elapsed = time.time() - self.crawl_state_started
        tilt_ok = tilt_mag <= CRAWL_TILT_OK_RAD
        force_ok = min(self._leg_force(leg) for leg in support_legs) >= CRAWL_CONTACT_FORCE_MIN

        if self.crawl_state == "shift":
            if tilt_ok and force_ok:
                if self.crawl_support_started == 0.0:
                    self.crawl_support_started = time.time()
                elif time.time() - self.crawl_support_started >= CRAWL_SHIFT_HOLD_SEC:
                    self._begin_crawl_state_locked("lift")
            else:
                self.crawl_support_started = 0.0
            self.swing_legs = set()
            self.target_pose = self._foot_targets_to_pose(shift_targets)
            return self.target_pose

        self.swing_legs = {active_leg}
        swing_targets = {leg: list(target) for leg, target in shift_targets.items()}
        progress = 0.0
        if self.crawl_state == "lift":
            progress = clamp(elapsed / CRAWL_LIFT_SEC, 0.0, 1.0)
            swing_targets[active_leg][2] += DEFAULT_LIFT_FOOT_Z * height_scale * smoothstep(progress)
            if progress >= 1.0:
                self._begin_crawl_state_locked("swing")
        elif self.crawl_state == "swing":
            progress = clamp(elapsed / CRAWL_SWING_SEC, 0.0, 1.0)
            swing_targets[active_leg][2] += DEFAULT_LIFT_FOOT_Z * height_scale
            swing_targets[active_leg][0] += CRAWL_STEP_X * direction * height_scale * smoothstep(progress)
            if progress >= 1.0:
                self._begin_crawl_state_locked("lower")
        elif self.crawl_state == "lower":
            progress = clamp(elapsed / CRAWL_LOWER_TIMEOUT_SEC, 0.0, 1.0)
            swing_targets[active_leg][2] += DEFAULT_LIFT_FOOT_Z * height_scale * (1.0 - smoothstep(progress))
            swing_targets[active_leg][0] += CRAWL_STEP_X * direction * height_scale
            if self._leg_force(active_leg) >= CRAWL_CONTACT_FORCE_MIN or progress >= 1.0:
                self._begin_crawl_state_locked("settle")
                self.crawl_support_started = 0.0
        elif self.crawl_state == "settle":
            swing_targets[active_leg][0] += CRAWL_STEP_X * direction * height_scale
            if tilt_ok and force_ok and self._leg_force(active_leg) >= CRAWL_CONTACT_FORCE_MIN:
                if self.crawl_support_started == 0.0:
                    self.crawl_support_started = time.time()
                elif time.time() - self.crawl_support_started >= CRAWL_SETTLE_SEC:
                    self.crawl_leg_index = (self.crawl_leg_index + 1) % len(CRAWL_ORDER)
                    self._begin_crawl_state_locked("shift")
            else:
                self.crawl_support_started = 0.0

        imu_balance = self._imu_balance_offsets()
        for leg in support_legs:
            swing_targets[leg][2] += imu_balance[leg]
        pose = self._foot_targets_to_pose(swing_targets)
        self.target_pose = list(pose)
        self.snapshots.service_status = f"crawl {self.crawl_state}:{active_leg}"
        self.snapshots.service_time = time.time()
        return pose

    def _stance_swing_value(self, leg_phase: float, duty: float):
        if leg_phase < duty:
            stance_phase = leg_phase / duty
            return 1.0 - 2.0 * stance_phase, 0.0
        swing_phase = (leg_phase - duty) / (1.0 - duty)
        sweep = -1.0 + 2.0 * swing_phase
        lift = math.sin(math.pi * swing_phase)
        return sweep, lift

    def _apply_remote_command_locked(self):
        if not self.low_level_mode_active or self.remote is None:
            return
        buttons = self.remote["buttons"]
        self.move_x = normalize_axis(float(self.remote["ly"]), REMOTE_DEADBAND)
        self.move_y = 0.0
        self.move_yaw = 0.0
        self.walk_enabled = abs(self.move_x) >= 0.05
        up_pressed = bool(buttons.get("Up"))
        down_pressed = bool(buttons.get("Down"))
        if up_pressed and not self.prev_height_up:
            self._set_height_offset_locked(self.height_offset_m + HEIGHT_STEP_M, 0.35)
        elif down_pressed and not self.prev_height_down:
            self._set_height_offset_locked(self.height_offset_m - HEIGHT_STEP_M, 0.35)
        self.prev_height_up = up_pressed
        self.prev_height_down = down_pressed

    def _check_remote_toggle_locked(self):
        if self.remote is None:
            return
        buttons = self.remote["buttons"]
        combo = bool(buttons.get("L2") and buttons.get("Y"))
        if combo and not self.prev_toggle_combo and not self.handoff_in_progress:
            self.handoff_in_progress = True
            if self.low_level_mode_active:
                thread = threading.Thread(target=self._exit_low_level_mode_worker, name="ll_walk_exit", daemon=True)
            else:
                thread = threading.Thread(target=self._enter_low_level_mode_worker, name="ll_walk_enter", daemon=True)
            thread.start()
        self.prev_toggle_combo = combo

    def _log_button_edges_locked(self):
        if self.remote is None:
            return
        buttons = self.remote["buttons"]
        pressed = []
        for name, value in buttons.items():
            if value and not self.prev_buttons.get(name, 0):
                pressed.append(name)
        self.prev_buttons = dict(buttons)
        if not pressed:
            return

        event = "+".join(pressed)
        ts = time.time()
        self.snapshots.last_button_event = event
        self.snapshots.last_button_time = ts
        try:
            with open(self.button_log_path, "a", encoding="utf-8") as f:
                f.write(f"{ts:.3f} {event}\n")
        except Exception as exc:
            self.snapshots.last_button_event = f"log failed: {exc}"
            self.snapshots.last_button_time = ts

    def _gait_target_locked(self):
        return self._crawl_targets_locked()

    def _recorded_gait_target_locked(self):
        base_targets = self._height_foot_targets(self.height_offset_m)
        base = self._foot_targets_to_pose(base_targets)
        if self.recorded_gait is None or not self.recorded_gait.valid:
            self.target_pose = list(base)
            return base

        self.command_move_x = self._ramp_value(self.command_move_x, self.move_x, COMMAND_RAMP_RATE)
        self.command_move_y = self._ramp_value(self.command_move_y, self.move_y, COMMAND_RAMP_RATE)
        self.command_move_yaw = self._ramp_value(
            self.command_move_yaw, self.move_yaw, COMMAND_RAMP_RATE * 1.4
        )
        drive = max(abs(self.command_move_x), abs(self.command_move_y), abs(self.command_move_yaw))
        tilt_mag = max(abs(self.roll), abs(self.pitch))
        if tilt_mag > MAX_TILT_FOR_WALK_RAD:
            self.walk_enabled = False
            self.gait_engage = 0.0
            self.command_move_x = 0.0
            self.command_move_y = 0.0
            self.command_move_yaw = 0.0
            self.target_pose = list(base)
            self.snapshots.service_status = "walk halted: tilt limit"
            self.snapshots.service_time = time.time()
            return base
        if not self.walk_enabled or drive < 0.03:
            self.recorded_phase_time = 0.0
            self.gait_engage = self._ramp_value(self.gait_engage, 0.0, GAIT_ENGAGE_RATE)
            self.target_pose = list(base)
            return base
        self.gait_engage = self._ramp_value(self.gait_engage, 1.0, GAIT_ENGAGE_RATE)
        drive_mix = max(drive, 0.35)

        self.recorded_phase_time += CONTROL_DT * max(0.35, abs(self.command_move_x)) * self.recorded_gait_speed
        sampled = self.recorded_gait.sample(self.recorded_phase_time)
        if sampled is None:
            self.target_pose = list(base)
            return base

        turn_mix = self.command_move_yaw
        foot_targets = {leg: list(target) for leg, target in base_targets.items()}
        imu_balance = self._imu_balance_offsets()
        for leg in LEG_ORDER:
            front_sign = 1.0 if leg in ("FR", "FL") else -1.0
            side_sign = 1.0 if leg in ("FL", "RL") else -1.0

            hip_idx, thigh_idx, calf_idx = LEG_INDEX[leg]
            foot_targets[leg][0] += sampled["delta"][thigh_idx] * 0.08 * drive_mix * self.gait_engage
            foot_targets[leg][1] += 0.03 * sampled["delta"][hip_idx] * side_sign * self.gait_engage
            foot_targets[leg][1] += 0.04 * turn_mix * side_sign * (0.7 if front_sign > 0 else 1.0) * self.gait_engage
            foot_targets[leg][2] += -sampled["delta"][calf_idx] * 0.05 * drive_mix * self.gait_engage
            foot_targets[leg][2] += imu_balance[leg] * (1.0 - self.gait_engage * 0.5)

        self.target_pose = list(base)
        return self._foot_targets_to_pose(foot_targets)

    def _apply_zmp_stabilizer_locked(self, pose, base):
        q = list(pose)

        front_force = self.foot_force[0] + self.foot_force[1]
        rear_force = self.foot_force[2] + self.foot_force[3]
        left_force = self.foot_force[1] + self.foot_force[3]
        right_force = self.foot_force[0] + self.foot_force[2]

        u_pitch = -(ZMP_PITCH_P * self.pitch + ZMP_PITCH_D * self.gy)
        u_roll = -(ZMP_ROLL_P * self.roll + ZMP_ROLL_D * self.gx)
        u_pitch += ZMP_FORCE_GAIN * (rear_force - front_force)
        u_roll += ZMP_FORCE_GAIN * (right_force - left_force)

        u_pitch = clamp(u_pitch, -ZMP_EXT_CLAMP, ZMP_EXT_CLAMP)
        u_roll = clamp(u_roll, -ZMP_EXT_CLAMP, ZMP_EXT_CLAMP)

        def apply_extension(leg, ext):
            if leg in self.swing_legs:
                ext *= 0.15
            _, thigh_idx, calf_idx = LEG_INDEX[leg]
            q[thigh_idx] += ZMP_THIGH_GAIN * ext
            q[calf_idx] += ZMP_CALF_GAIN * ext

        for leg in ("FR", "FL"):
            apply_extension(leg, +u_pitch)
        for leg in ("RR", "RL"):
            apply_extension(leg, -u_pitch)
        for leg in ("FL", "RL"):
            apply_extension(leg, +u_roll)
        for leg in ("FR", "RR"):
            apply_extension(leg, -u_roll)

        limited = []
        for i in range(12):
            limited.append(clamp(q[i], base[i] - 0.50, base[i] + 0.50))
        return limited

    def _init_low_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def _release_mode_for_recovery(self):
        self.msc.ReleaseMode()
        time.sleep(1.0)

    def _reacquire_motion_mode(self):
        code, data = self.msc.CheckMode()
        if code == 0 and data is not None and data.get("name"):
            return
        for alias in MODE_ALIASES:
            code, _ = self.msc.SelectMode(alias)
            if code != 0:
                continue
            time.sleep(1.0)
            check_code, check_data = self.msc.CheckMode()
            if check_code == 0 and check_data is not None and check_data.get("name"):
                return

    def _ensure_services_enabled(self):
        code, services = self.robot_state.ServiceList()
        if code != 0 or services is None:
            raise RuntimeError(f"ServiceList failed with code {code}")
        by_name = {service.name: service for service in services}
        for service_name in SERVICES_TO_ENABLE:
            if service_name not in by_name:
                raise RuntimeError(f"Service not found: {service_name}")

            off_code = self.robot_state.ServiceSwitch(service_name, False)
            if off_code != 0:
                raise RuntimeError(f"ServiceSwitch({service_name}, off) failed with code {off_code}")
            time.sleep(SERVICE_TOGGLE_DELAY)

            switch_code = self.robot_state.ServiceSwitch(service_name, True)
            if switch_code != 0:
                raise RuntimeError(f"ServiceSwitch({service_name}, on) failed with code {switch_code}")
            time.sleep(SERVICE_RESTART_WAIT)

    def _try_stop_lowlevel(self, duration_sec: float):
        pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        pub.Init()

        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        for i in range(20):
            cmd.motor_cmd[i].mode = 0x00
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0

        start = time.time()
        while time.time() - start < duration_sec:
            cmd.crc = self.crc.Crc(cmd)
            pub.Write(cmd)
            time.sleep(0.02)

    def _enter_low_level_mode_worker(self):
        try:
            with self.lock:
                self.snapshots.service_status = "sitting before ll mode"
                self.snapshots.service_time = time.time()
            try:
                self.sc.StandDown()
            except Exception:
                pass
            time.sleep(1.5)

            with self.lock:
                self.snapshots.service_status = "releasing mcf mode"
                self.snapshots.service_time = time.time()
            self._release_mode_for_recovery()

            with self.lock:
                self.low_level_mode_active = True
                self.first_run = True
                self.manual_override = True
                self.sequence_done = True
                self.sequence_index = 0
                self.sequence_hold_started = 0.0
                self.walk_enabled = False
                self.move_x = 0.0
                self.move_y = 0.0
                self.move_yaw = 0.0
                self.command_move_x = 0.0
                self.command_move_y = 0.0
                self.command_move_yaw = 0.0
                self.gait_engage = 0.0
                self.crawl_leg_index = 0
                self._begin_crawl_state_locked("idle")
                self.phase = 0.0
                self.recorded_phase_time = 0.0
                self.height_offset_m = NORMAL_HEIGHT_M
                if self.low_state is not None:
                    self.start_pose = [self.low_state.motor_state[i].q for i in range(12)]
                    self.current_pose = list(self.start_pose)
                    self.target_pose = list(self.start_pose)
                self.transition_active = False
                self._begin_transition_locked(self._compose_base_pose_locked(), 0.8)
                self.snapshots.service_status = "ll mode active"
                self.snapshots.service_time = time.time()
        except Exception as exc:
            with self.lock:
                self.snapshots.service_status = f"ll mode enter failed: {exc}"
                self.snapshots.service_time = time.time()
        finally:
            with self.lock:
                self.handoff_in_progress = False

    def _exit_low_level_mode_worker(self):
        try:
            with self.lock:
                self.snapshots.service_status = "lowering before hl handoff"
                self.snapshots.service_time = time.time()
                self.first_run = True
                self.height_offset_m = MIN_HEIGHT_OFFSET_M
                self.walk_enabled = False
                self.move_x = 0.0
                self.move_y = 0.0
                self.move_yaw = 0.0
                self.gait_engage = 0.0
                self.crawl_leg_index = 0
                self._begin_crawl_state_locked("idle")
                self._begin_transition_locked(self._compose_base_pose_locked(), EXIT_SIT_BLEND_TIME)
            time.sleep(EXIT_SIT_BLEND_TIME + 0.6)

            with self.lock:
                self.snapshots.service_status = "neutralizing low-level control"
                self.snapshots.service_time = time.time()
                self.low_level_mode_active = False
            self._try_stop_lowlevel(LOWLEVEL_STOP_WAIT)

            with self.lock:
                self.snapshots.service_status = "releasing motion mode"
                self.snapshots.service_time = time.time()
            self._release_mode_for_recovery()

            with self.lock:
                self.snapshots.service_status = "enabling mcf/sport_mode"
                self.snapshots.service_time = time.time()
            self._ensure_services_enabled()

            with self.lock:
                self.snapshots.service_status = "reacquiring motion mode"
                self.snapshots.service_time = time.time()
            self._reacquire_motion_mode()

            with self.lock:
                self.current_pose = list(self.sit_pose)
                self.target_pose = list(self.sit_pose)
                self.transition_active = False
                self.snapshots.service_status = "hl active"
                self.snapshots.service_time = time.time()
        except Exception as exc:
            with self.lock:
                self.snapshots.service_status = f"hl handoff failed: {exc}"
                self.snapshots.service_time = time.time()
        finally:
            with self.lock:
                self.handoff_in_progress = False

    def _low_state_handler(self, msg: LowState_):
        with self.lock:
            self.low_state = msg
            self.snapshots.low_state = msg
            self.snapshots.low_state_time = time.time()
            self.remote = decode_remote(msg.wireless_remote)
            self.snapshots.remote_lx = float(self.remote["lx"])
            self.snapshots.remote_ly = float(self.remote["ly"])
            self.snapshots.remote_rx = float(self.remote["rx"])
            imu = msg.imu_state
            self.roll = float(imu.rpy[0])
            self.pitch = float(imu.rpy[1])
            self.gx = float(imu.gyroscope[0])
            self.gy = float(imu.gyroscope[1])
            self.foot_force = [float(v) for v in msg.foot_force[:4]]
            self._log_button_edges_locked()
            self._check_remote_toggle_locked()
            self._apply_remote_command_locked()

    def _odom_handler(self, msg: Odometry_):
        with self.lock:
            self.snapshots.odom = msg
            self.snapshots.odom_time = time.time()

    def _lidar_state_handler(self, msg: LidarState_):
        with self.lock:
            self.snapshots.lidar_state = msg
            self.snapshots.lidar_state_time = time.time()

    def _lidar_cloud_handler(self, msg: PointCloud2_):
        with self.lock:
            self.snapshots.lidar_cloud = msg
            self.snapshots.lidar_cloud_time = time.time()

    def _poll_motion_mode(self):
        try:
            code, result = self.msc.CheckMode()
            with self.lock:
                self.snapshots.motion_code = code
                self.snapshots.motion_time = time.time()
                self.snapshots.motion_mode = (result or {}).get("name", "") or "released"
                self.snapshots.motion_error = ""
        except Exception as exc:
            with self.lock:
                self.snapshots.motion_time = time.time()
                self.snapshots.motion_error = str(exc)

    def _low_cmd_write(self):
        with self.lock:
            if not self.low_level_mode_active:
                return
            if self.low_state is None:
                return
            if self.first_run:
                for i in range(12):
                    self.start_pose[i] = self.low_state.motor_state[i].q
                self.current_pose = list(self.start_pose)
                self.transition_start_pose = list(self.start_pose)
                self.transition_target_pose = self._compose_base_pose_locked()
                self.first_run = False

            self._advance_sequence_locked()
            self._apply_transition_locked()
            if self.transition_active:
                pose = list(self.current_pose)
            elif self.walk_enabled:
                pose = self._gait_target_locked()
                pose = self._apply_zmp_stabilizer_locked(pose, self.target_pose)
                self.current_pose = list(pose)
            else:
                self.command_move_x = self._ramp_value(self.command_move_x, 0.0, COMMAND_RAMP_RATE)
                self.command_move_y = self._ramp_value(self.command_move_y, 0.0, COMMAND_RAMP_RATE)
                self.command_move_yaw = self._ramp_value(self.command_move_yaw, 0.0, COMMAND_RAMP_RATE * 1.4)
                stand_target = self._compose_base_pose_locked()
                alpha = clamp(CONTROL_DT / IDLE_POSE_BLEND_TIME, 0.0, 1.0)
                self.current_pose = lerp_pose(self.current_pose, stand_target, alpha)
                pose = list(self.current_pose)
                self.target_pose = list(stand_target)

            kp = self.Kp
            kd = self.Kd
            if not self.transition_active and not self.walk_enabled:
                kp = IDLE_KP
                kd = IDLE_KD

        for i in range(12):
            self.low_cmd.motor_cmd[i].q = pose[i]
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = kp
            self.low_cmd.motor_cmd[i].kd = kd
            self.low_cmd.motor_cmd[i].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


def draw_panel(stdscr, controller: LowLevelWalkController, odom_topic: str, lidar_state_topic: str, lidar_cloud_topic: str):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    snapshot = controller.get_snapshot()
    low_state = snapshot["low_state"]
    odom = snapshot["odom"]
    lidar_state = snapshot["lidar_state"]
    lidar_cloud = snapshot["lidar_cloud"]

    lines = [
        "Low-Level Walk Custom Height",
        "Remote: L2+Y toggles LL mode. Left stick Y: forward/back closed-loop crawl. D-pad Up/Down: height. q: quit",
        "",
        f"Control owner: {'low-level' if snapshot['low_level_mode_active'] else 'high-level MCF'}",
        f"Handoff: {snapshot['service_status']} (age {age_text(snapshot['service_age'])})",
        f"Last button: {snapshot['last_button_event'] or '--'} (age {age_text(snapshot['last_button_age'])})",
        f"Height offset: {snapshot['height_offset_m']:+.3f} m",
        f"Step height scale: {snapshot['step_height_scale']:.2f}",
        f"Walk: {'on' if snapshot['walk_enabled'] else 'off'}",
        f"Crawl: {snapshot['crawl_state']}  leg={snapshot['crawl_leg']}",
        f"Gait: {snapshot['gait_name']}  {snapshot['gait_description']}",
        f"Command x/y/yaw: {snapshot['move_x']:+.2f} / {snapshot['move_y']:+.2f} / {snapshot['move_yaw']:+.2f}",
        f"Remote lx/ly/rx: {snapshot['remote_lx']:+.2f} / {snapshot['remote_ly']:+.2f} / {snapshot['remote_rx']:+.2f}",
        f"IMU roll/pitch: {snapshot['roll']:+.3f} / {snapshot['pitch']:+.3f}",
        f"Foot force: {' '.join(str(int(v)) for v in snapshot['foot_force'])}",
        f"Startup sequence: {'done' if snapshot['sequence_done'] else 'running'}",
        "",
        f"LowState  {status_text(snapshot['low_state_age'])}  age={age_text(snapshot['low_state_age'])}  topic={TOPIC_LOWSTATE}",
    ]

    if low_state is not None:
        imu = low_state.imu_state
        lines.extend(
            [
                f"  Power V/A: {float(low_state.power_v):.2f} / {float(low_state.power_a):.2f}",
                f"  IMU rpy:   {vec3([float(v) for v in imu.rpy])}",
                f"  IMU gyro:  {vec3([float(v) for v in imu.gyroscope])}",
                f"  IMU acc:   {vec3([float(v) for v in imu.accelerometer])}",
                f"  Foot force:{' '.join(str(int(v)) for v in low_state.foot_force)}",
            ]
        )
    else:
        lines.append("  waiting for rt/lowstate")

    lines.extend(
        [
            "",
            f"Odometry  {status_text(snapshot['odom_age'])}  age={age_text(snapshot['odom_age'])}  topic={odom_topic}",
        ]
    )
    if odom is not None:
        pos = odom.pose.pose.position
        quat = odom.pose.pose.orientation
        lin = odom.twist.twist.linear
        ang = odom.twist.twist.angular
        yaw = quat_to_yaw([quat.x, quat.y, quat.z, quat.w])
        lines.extend(
            [
                f"  Pos xyz:  {pos.x: .2f} {pos.y: .2f} {pos.z: .2f}",
                f"  Yaw deg:  {yaw: .1f}",
                f"  Lin vel:  {lin.x: .2f} {lin.y: .2f} {lin.z: .2f}",
                f"  Ang vel:  {ang.x: .2f} {ang.y: .2f} {ang.z: .2f}",
            ]
        )
    else:
        lines.append("  waiting for rt/odom")

    lines.extend(
        [
            "",
            f"UTLiDAR state  {status_text(snapshot['lidar_state_age'])}  age={age_text(snapshot['lidar_state_age'])}  topic={lidar_state_topic}",
        ]
    )
    if lidar_state is not None:
        lines.extend(
            [
                f"  Cloud size: {int(lidar_state.cloud_size)}",
                f"  Cloud freq: {float(lidar_state.cloud_frequency):.2f} Hz",
                f"  Cloud loss: {float(lidar_state.cloud_packet_loss_rate):.3f}",
                f"  IMU rpy:    {vec3([float(v) for v in lidar_state.imu_rpy])}",
            ]
        )
    else:
        lines.append("  waiting for utlidar state")

    lines.extend(
        [
            "",
            f"UTLiDAR cloud  {status_text(snapshot['lidar_cloud_age'])}  age={age_text(snapshot['lidar_cloud_age'])}  topic={lidar_cloud_topic}",
            f"  Points: {extract_point_count(lidar_cloud)}",
            "",
            f"Motion switcher  age={age_text(snapshot['motion_age'])}",
            f"  mode={snapshot['motion_mode']}  code={snapshot['motion_code']}",
        ]
    )
    if snapshot["motion_error"]:
        lines.append(f"  error={snapshot['motion_error']}")
    if snapshot["recorded_gait_loaded"]:
        lines.append(f"Recorded gait: {os.path.basename(snapshot['recorded_gait_path'])}")
    elif snapshot["recorded_gait_error"]:
        lines.append(f"Recorded gait error: {snapshot['recorded_gait_error']}")

    for idx, line in enumerate(lines[: max(0, h - 1)]):
        stdscr.addnstr(idx, 0, line, max(0, w - 1))
    stdscr.refresh()


def tui_main(stdscr, controller: LowLevelWalkController, odom_topic: str, lidar_state_topic: str, lidar_cloud_topic: str):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    while True:
        draw_panel(stdscr, controller, odom_topic, lidar_state_topic, lidar_cloud_topic)
        key = stdscr.getch()
        if key == curses.KEY_UP:
            controller.increase_height()
        elif key == curses.KEY_DOWN:
            controller.decrease_height()
        elif key == curses.KEY_LEFT:
            controller.adjust_move_yaw(+TURN_STEP)
        elif key == curses.KEY_RIGHT:
            controller.adjust_move_yaw(-TURN_STEP)
        elif key in (ord("a"), ord("A")):
            controller.adjust_move_y(+MOVE_STEP)
        elif key in (ord("d"), ord("D")):
            controller.adjust_move_y(-MOVE_STEP)
        elif key in (ord("w"), ord("W")):
            controller.toggle_walk()
        elif key in (ord("g"), ord("G")):
            controller.cycle_gait()
        elif key in (ord("i"), ord("I")):
            controller.adjust_move_x(+MOVE_STEP)
        elif key in (ord("k"), ord("K")):
            controller.adjust_move_x(-MOVE_STEP)
        elif key in (ord("j"), ord("J")):
            controller.zero_commands()
        elif key == ord("["):
            controller.decrease_step_height()
        elif key == ord("]"):
            controller.increase_step_height()
        elif key in (ord("q"), ord("Q")):
            break


def parse_args():
    parser = argparse.ArgumentParser(
        description="Low-level Go2 walk controller with adjustable stand height."
    )
    parser.add_argument("iface", nargs="?", default=None, help="Robot network interface")
    parser.add_argument("--odom-topic", default=TOPIC_ODOM)
    parser.add_argument("--lidar-state-topic", default=TOPIC_LIDAR_STATE)
    parser.add_argument("--lidar-cloud-topic", default=TOPIC_LIDAR_CLOUD)
    parser.add_argument("--recording", default=find_latest_recording(), help="Optional recorded gait jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    print("WARNING: This script streams low-level joint commands for standing and walking.")
    print("Use only with clearance, a spotter, and a ready emergency stop.")
    input("Press Enter to continue...")

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    recorded = None
    if args.recording:
        recorded = RecordedGait(args.recording)

    controller = LowLevelWalkController(recorded)
    controller.init(args.odom_topic, args.lidar_state_topic, args.lidar_cloud_topic)
    controller.start()

    try:
        curses.wrapper(
            tui_main,
            controller,
            args.odom_topic,
            args.lidar_state_topic,
            args.lidar_cloud_topic,
        )
    finally:
        controller.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
