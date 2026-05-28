import argparse
import curses
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
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
NORMAL_HEIGHT_M = 0.0
HEIGHT_STEP_M = 0.01
MIN_HEIGHT_OFFSET_M = -0.10
MAX_HEIGHT_OFFSET_M = 0.06
DEFAULT_LIFT_FOOT_Z = 0.06
LIFT_STEP_M = 0.01
MIN_LIFT_FOOT_Z = 0.0
MAX_LIFT_FOOT_Z = 0.12
MAX_BODY_SHIFT_X = 0.055
MAX_BODY_SHIFT_Y = 0.045
BODY_SHIFT_GAIN = 0.9
IMU_PITCH_POS_GAIN = 0.035
IMU_ROLL_POS_GAIN = 0.03
IMU_GYRO_GAIN = 0.003
MAX_IMU_FOOT_Z = 0.025

LEG_INDEX = {
    "FR": (0, 1, 2),
    "FL": (3, 4, 5),
    "RR": (6, 7, 8),
    "RL": (9, 10, 11),
}

LEG_SIGNS = {
    "FL": {"left": 1.0, "front": 1.0},
    "FR": {"left": -1.0, "front": 1.0},
    "RL": {"left": 1.0, "front": -1.0},
    "RR": {"left": -1.0, "front": -1.0},
}

# Geometry extracted from ../sim/unitree_description-master/model/go2/go2.urdf
HIP_ORIGIN = {
    "FL": (0.1934, 0.0465, 0.0),
    "FR": (0.1934, -0.0465, 0.0),
    "RL": (-0.1934, 0.0465, 0.0),
    "RR": (-0.1934, -0.0465, 0.0),
}
HIP_LATERAL_OFFSET = 0.0955
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213

LIFT_MODES = {
    "none": (),
    "fr": ("FR",),
    "fl": ("FL",),
    "rr": ("RR",),
    "rl": ("RL",),
}

LIFT_LABELS = {
    "none": "all feet down",
    "fr": "lift FR",
    "fl": "lift FL",
    "rr": "lift RR",
    "rl": "lift RL",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp_pose(src, dst, alpha: float):
    return [(1.0 - alpha) * a + alpha * b for a, b in zip(src, dst)]


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


class Go2BalanceDiagController:
    def __init__(self):
        self.Kp = 60.0
        self.Kd = 5.0
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()
        self.lock = threading.Lock()
        self.snapshots = TopicSnapshot()
        self.running = True
        self.first_run = True

        self.stand_pose = [
            0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
            0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
        ]
        self.height_pose = [
            -0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
            -0.5, 1.36, -2.65, 0.5, 1.36, -2.65,
        ]
        self.sit_pose = [
            0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
            -0.2, 1.36, -2.65, 0.2, 1.36, -2.65,
        ]
        self.start_pose = [0.0] * 12
        self.current_pose = list(self.sit_pose)
        self.target_pose = list(self.sit_pose)
        self.height_offset_m = NORMAL_HEIGHT_M
        self.lift_foot_z = DEFAULT_LIFT_FOOT_Z
        self.lift_mode = "none"
        self.nominal_foot_targets = self._build_nominal_foot_targets()

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

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        self._release_motion_mode()

    def start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=CONTROL_DT, target=self._low_cmd_write, name="go2_balance_diag_lowcmd"
        )
        self.lowCmdWriteThreadPtr.Start()
        self.modePollThreadPtr = RecurrentThread(
            interval=0.5, target=self._poll_motion_mode, name="go2_balance_diag_mode_poll"
        )
        self.modePollThreadPtr.Start()

    def stop(self):
        self.running = False
        if self.lowCmdWriteThreadPtr is not None:
            self.lowCmdWriteThreadPtr.Wait(1.0)
        if self.modePollThreadPtr is not None:
            self.modePollThreadPtr.Wait(1.0)

    def increase_height(self):
        with self.lock:
            self.height_offset_m = clamp(
                self.height_offset_m + HEIGHT_STEP_M, MIN_HEIGHT_OFFSET_M, MAX_HEIGHT_OFFSET_M
            )
            self._begin_transition(self._compose_pose(), 0.35)

    def decrease_height(self):
        with self.lock:
            self.height_offset_m = clamp(
                self.height_offset_m - HEIGHT_STEP_M, MIN_HEIGHT_OFFSET_M, MAX_HEIGHT_OFFSET_M
            )
            self._begin_transition(self._compose_pose(), 0.35)

    def set_lift_mode(self, mode: str):
        if mode not in LIFT_MODES:
            return
        with self.lock:
            self.lift_mode = mode
            self._begin_transition(self._compose_pose(), 0.6)

    def increase_lift_height(self):
        with self.lock:
            self.lift_foot_z = clamp(
                self.lift_foot_z + LIFT_STEP_M, MIN_LIFT_FOOT_Z, MAX_LIFT_FOOT_Z
            )
            self._begin_transition(self._compose_pose(), 0.25)

    def decrease_lift_height(self):
        with self.lock:
            self.lift_foot_z = clamp(
                self.lift_foot_z - LIFT_STEP_M, MIN_LIFT_FOOT_Z, MAX_LIFT_FOOT_Z
            )
            self._begin_transition(self._compose_pose(), 0.25)

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
                "height_offset_m": self.height_offset_m,
                "lift_foot_z": self.lift_foot_z,
                "lift_mode": self.lift_mode,
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

    def _build_nominal_foot_targets(self):
        foot_targets = {}
        stand_triplets = {
            "FR": self.stand_pose[0:3],
            "FL": self.stand_pose[3:6],
            "RR": self.stand_pose[6:9],
            "RL": self.stand_pose[9:12],
        }
        for leg_name, joints in stand_triplets.items():
            hip_origin = HIP_ORIGIN[leg_name]
            foot_hip = leg_forward_kinematics(leg_name, joints)
            foot_targets[leg_name] = [
                hip_origin[0] + foot_hip[0],
                hip_origin[1] + foot_hip[1],
                hip_origin[2] + foot_hip[2],
            ]
        return foot_targets

    def _body_shift_for_mode(self):
        if self.lift_mode == "none":
            return (0.0, 0.0)
        lifted_legs = set(LIFT_MODES[self.lift_mode])
        support_legs = [leg for leg in LEG_INDEX if leg not in lifted_legs]
        if not support_legs:
            return (0.0, 0.0)
        centroid_x = sum(self.nominal_foot_targets[leg][0] for leg in support_legs) / len(support_legs)
        centroid_y = sum(self.nominal_foot_targets[leg][1] for leg in support_legs) / len(support_legs)
        shift_x = clamp_abs(BODY_SHIFT_GAIN * centroid_x, MAX_BODY_SHIFT_X)
        shift_y = clamp_abs(BODY_SHIFT_GAIN * centroid_y, MAX_BODY_SHIFT_Y)
        return (shift_x, shift_y)

    def _imu_balance_offsets(self):
        if self.low_state is None:
            return {leg: 0.0 for leg in LEG_INDEX}
        imu = self.low_state.imu_state
        roll = float(imu.rpy[0])
        pitch = float(imu.rpy[1])
        gx = float(imu.gyroscope[0])
        gy = float(imu.gyroscope[1])
        pitch_term = clamp_abs(
            -(IMU_PITCH_POS_GAIN * pitch + IMU_GYRO_GAIN * gy),
            MAX_IMU_FOOT_Z,
        )
        roll_term = clamp_abs(
            -(IMU_ROLL_POS_GAIN * roll + IMU_GYRO_GAIN * gx),
            MAX_IMU_FOOT_Z,
        )
        offsets = {}
        for leg_name, signs in LEG_SIGNS.items():
            offsets[leg_name] = signs["front"] * pitch_term + signs["left"] * roll_term
        return offsets

    def _compose_pose(self):
        alpha = clamp(
            (self.height_offset_m - MIN_HEIGHT_OFFSET_M) / (MAX_HEIGHT_OFFSET_M - MIN_HEIGHT_OFFSET_M), 0.0, 1.0
        )
        nominal_targets = {}
        for leg_name in self.nominal_foot_targets:
            stand_target = self.nominal_foot_targets[leg_name]
            lowered_joints = self.height_pose[LEG_INDEX[leg_name][0]: LEG_INDEX[leg_name][2] + 1]
            lowered_target_hip = leg_forward_kinematics(leg_name, lowered_joints)
            lowered_target = [
                HIP_ORIGIN[leg_name][0] + lowered_target_hip[0],
                HIP_ORIGIN[leg_name][1] + lowered_target_hip[1],
                HIP_ORIGIN[leg_name][2] + lowered_target_hip[2],
            ]
            nominal_targets[leg_name] = [
                (1.0 - alpha) * lowered_target[i] + alpha * stand_target[i] for i in range(3)
            ]

        body_shift_x, body_shift_y = self._body_shift_for_mode()
        imu_balance = self._imu_balance_offsets()
        lifted_legs = set(LIFT_MODES[self.lift_mode])
        pose = [0.0] * 12

        for leg_name, indices in LEG_INDEX.items():
            target_base = list(nominal_targets[leg_name])
            target_base[0] -= body_shift_x
            target_base[1] -= body_shift_y
            if leg_name in lifted_legs:
                target_base[2] += self.lift_foot_z
            else:
                target_base[2] += imu_balance[leg_name]

            hip_origin = HIP_ORIGIN[leg_name]
            foot_pos_hip = [
                target_base[0] - hip_origin[0],
                target_base[1] - hip_origin[1],
                target_base[2] - hip_origin[2],
            ]
            q_hip, q_thigh, q_calf = leg_inverse_kinematics(leg_name, foot_pos_hip)
            pose[indices[0]] = q_hip
            pose[indices[1]] = q_thigh
            pose[indices[2]] = q_calf

        return pose

    def _begin_transition(self, pose, duration: float):
        self.transition_start_pose = list(self.current_pose)
        self.transition_target_pose = list(pose)
        self.transition_started = time.time()
        self.transition_duration = max(duration, POSTURE_BLEND_DT)
        self.transition_active = True
        self.target_pose = list(pose)

    def _apply_transition(self):
        if not self.transition_active:
            return
        elapsed = time.time() - self.transition_started
        alpha = clamp(elapsed / self.transition_duration, 0.0, 1.0)
        self.current_pose = lerp_pose(self.transition_start_pose, self.transition_target_pose, alpha)
        if alpha >= 1.0:
            self.current_pose = list(self.transition_target_pose)
            self.transition_active = False

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

    def _release_motion_mode(self):
        while True:
            code, result = self.msc.CheckMode()
            name = (result or {}).get("name", "") if result is not None else ""
            if code == 0 and not name:
                return
            self.msc.ReleaseMode()
            time.sleep(1.0)

    def _low_state_handler(self, msg: LowState_):
        with self.lock:
            self.low_state = msg
            self.snapshots.low_state = msg
            self.snapshots.low_state_time = time.time()

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
            if self.low_state is None:
                return
            if self.first_run:
                for i in range(12):
                    self.start_pose[i] = self.low_state.motor_state[i].q
                self.current_pose = list(self.start_pose)
                self.transition_start_pose = list(self.start_pose)
                self.transition_target_pose = self._compose_pose()
                self.transition_started = time.time()
                self.transition_duration = 1.5
                self.transition_active = True
                self.first_run = False
            desired_pose = self._compose_pose()
            if self.transition_active:
                self.transition_target_pose = list(desired_pose)
                self.target_pose = list(desired_pose)
                self._apply_transition()
                pose = list(self.current_pose)
            else:
                self.current_pose = list(desired_pose)
                self.target_pose = list(desired_pose)
                pose = list(desired_pose)

        for i in range(12):
            self.low_cmd.motor_cmd[i].q = pose[i]
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = self.Kp
            self.low_cmd.motor_cmd[i].kd = self.Kd
            self.low_cmd.motor_cmd[i].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


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


def draw_panel(stdscr, controller: Go2BalanceDiagController, odom_topic: str, lidar_state_topic: str, lidar_cloud_topic: str):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    snapshot = controller.get_snapshot()
    low_state = snapshot["low_state"]
    odom = snapshot["odom"]
    lidar_state = snapshot["lidar_state"]
    lidar_cloud = snapshot["lidar_cloud"]

    lines = [
        "Go2 Balance Foot Lift TUI",
        "Up/Down: body height  Left/Right: lift height  1: FR  2: FL  3: RR  4: RL  0: down  q: quit",
        "",
        f"Height offset: {snapshot['height_offset_m']:+.3f} m",
        f"Lift height: {snapshot['lift_foot_z']:+.3f} m",
        f"Lift mode: {LIFT_LABELS[snapshot['lift_mode']]}",
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

    if h < 18 or w < 80:
        lines.append("")
        lines.append("Terminal is small; widen it for the full panel.")

    for idx, line in enumerate(lines[: max(0, h - 1)]):
        stdscr.addnstr(idx, 0, line, max(0, w - 1))
    stdscr.refresh()


def tui_main(stdscr, controller: Go2BalanceDiagController, odom_topic: str, lidar_state_topic: str, lidar_cloud_topic: str):
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
        elif key == curses.KEY_RIGHT:
            controller.increase_lift_height()
        elif key == curses.KEY_LEFT:
            controller.decrease_lift_height()
        elif key == ord("1"):
            controller.set_lift_mode("fr")
        elif key == ord("2"):
            controller.set_lift_mode("fl")
        elif key == ord("3"):
            controller.set_lift_mode("rr")
        elif key == ord("4"):
            controller.set_lift_mode("rl")
        elif key == ord("0"):
            controller.set_lift_mode("none")
        elif key in (ord("q"), ord("Q")):
            break


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Go2 single-foot lift diagnostic with fixed-height stand control."
    )
    parser.add_argument("iface", nargs="?", default=None, help="Robot network interface")
    parser.add_argument("--odom-topic", default=TOPIC_ODOM)
    parser.add_argument("--lidar-state-topic", default=TOPIC_LIDAR_STATE)
    parser.add_argument("--lidar-cloud-topic", default=TOPIC_LIDAR_CLOUD)
    return parser.parse_args()


def main():
    args = parse_args()
    print("WARNING: This script can intentionally reduce the support polygon by lifting one foot.")
    print("Use only with spotters and clear floor contact on the supporting legs.")
    input("Press Enter to continue...")

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    controller = Go2BalanceDiagController()
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
