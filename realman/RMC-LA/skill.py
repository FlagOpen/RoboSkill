import json
import os
import socket
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from groundingdino.util.inference import annotate, load_image, load_model, predict
from mcp.server.fastmcp import FastMCP
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

# chassis
chassis_host = "127.0.0.1"
chassis_port = 5000

# arm
arm_host = "127.0.0.1"
arm_port = 5000
speed = 20
radius = 0
connect = 0
block = 1


class RealmanChassis:
    """
    A client class for communicating with the Realman robot chassis over TCP.
    """

    def __init__(self, host: str, port: int):
        """
        Initialize the RealmanChassis client.

        Args:
            host (str): IP address or hostname of the chassis server.
            port (int): Port number to connect to.

        Raises:
            ConnectionError: If the socket connection fails.
        """
        self.host = host
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client.connect((self.host, self.port))
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}")

    def get_current_pose(self) -> Optional[dict]:
        """
        Query the current pose of the robot.

        Returns:
            dict or None: The current pose dictionary if successful, else None.
        """
        try:
            self.client.sendall(b"/api/robot_status")
            response = self.client.recv(2048).decode()
            data = json.loads(response)
            return data.get("results", {}).get("current_pose")
        except Exception as e:
            print(f"[ERROR] Failed to get current pose: {e}")
            return None

    def move_to_position(
        self, marker_name: str
    ) -> Tuple[Optional[dict], Optional[dict], str]:
        """
        Move the robot to a specified marker position.

        Args:
            marker_name (str): The name of the marker to move to.

        Returns:
            Tuple[Optional[dict], Optional[dict], str]:
                - start_pose: Pose before the move
                - target_pose: Pose after the move
                - status: Move status string (e.g., "success", "failed")
        """
        start_pose = self.get_current_pose()
        request_move = f"/api/move?marker={marker_name}"
        self.client.sendall(request_move.encode("utf-8"))
        response_move = self.client.recv(2048).decode()
        try:
            data_move = json.loads(response_move)
        except json.JSONDecodeError:
            return start_pose, None, "Failed to parse move response"

        # Wait for move to complete (code "01002" indicates completion)
        while True:
            try:
                response_status = self.client.recv(2048).decode()
                data_status = json.loads(response_status)
                if data_status.get("code") == "01002":
                    break
            except json.JSONDecodeError:
                print(f"[WARN] Ignoring invalid JSON chunk.")
                continue

        target_pose = self.get_current_pose()
        return start_pose, target_pose, data_move.get("status", "unknown")

    def cancel_current_move(self):
        """
        Cancel the current move command if in progress.
        """
        try:
            self.client.sendall(b"/api/move/cancel")
            response = self.client.recv(2048).decode()
            print(f"[INFO] Cancel move response: {response}")
        except Exception as e:
            print(f"[ERROR] Cancel move failed: {e}")

    def close_connection(self):
        """
        Close the socket connection to the chassis server.
        """
        self.client.close()
        print(f"[INFO] Connection closed")


class Camera:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Camera, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if Camera._initialized:
            print("[Warning] Camera already initialized. Skipping...")
            return

        try:
            self.connect_device = []
            self._init_camera()
            Camera._initialized = True
            self.first_call = True
        except Exception as e:
            Camera._initialized = False
            raise RuntimeError(f"[ERROR] Camera initialization failed: {e}")

    def _init_camera(self):
        ctx = rs.context()
        if not ctx.devices:
            raise RuntimeError("No RealSense device found.")

        for d in ctx.devices:
            dev_name = d.get_info(rs.camera_info.name)
            dev_sn = d.get_info(rs.camera_info.serial_number)
            print(f"[INFO] Found device: {dev_name} {dev_sn}")
            self.connect_device.append(dev_sn)

        if len(self.connect_device) != 1:
            raise RuntimeError(
                f"[ERROR] Expected 1 device, but found {len(self.connect_device)}"
            )

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.connect_device[0])
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.align = rs.align(rs.stream.color)
        self.profile = self.pipeline.start(self.config)
        self.depth_scale = (
            self.profile.get_device().first_depth_sensor().get_depth_scale()
        )
        print("[INFO] Camera initialized successfully")

    def record_wrist_frame(self, output_dir: str = "./output") -> Tuple[str, str]:
        os.makedirs(output_dir, exist_ok=True)

        if self.first_call:
            for _ in range(45):
                self.pipeline.wait_for_frames()
            self.first_call = False
            print("[INFO] Warm-up frames completed")

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to capture frames.")

        color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        depth_image = (
            np.asanyarray(depth_frame.get_data(), dtype=np.float32)
            * self.depth_scale
            * 1000
        )

        color_path = os.path.join(output_dir, "wrist_obs.png")
        depth_path = os.path.join(output_dir, "wrist_obs_depth.npy")

        cv2.imwrite(color_path, color_image)
        np.save(depth_path, depth_image)

        print(f"[INFO] Saved color image: {color_path}")
        print(f"[INFO] Saved depth data: {depth_path}")
        return color_path, depth_path

    def shutdown(self):
        if hasattr(self, "pipeline"):
            self.pipeline.stop()
        Camera._initialized = False
        print("[INFO] Camera shut down")

    def __del__(self):
        self.shutdown()


class Dino:
    def __init__(
        self,
        config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        weights_path="GroundingDINO/weights/groundingdino_swint_ogc.pth",
        device=None,
    ):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = load_model(config_path, weights_path)
        print(f"[INFO] DINO model loaded on {self.device}")

    def predict(self, image_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
        image_source, image = load_image(image_path)
        image = image.to(self.device)
        boxes, logits, phrases = predict(
            self.model,
            image,
            text_prompt,
            box_threshold,
            text_threshold,
            device=self.device,
        )
        h, w, _ = image_source.shape
        boxes_point = boxes * torch.tensor([w, h, w, h])
        self.save_annotated_image(
            image_source, boxes, logits, phrases, "./output/annotated_image.png"
        )
        return boxes_point, logits

    def save_annotated_image(self, image_source, boxes, logits, phrases, output_path):
        annotated = annotate(image_source, boxes, logits, phrases)
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Annotated image saved: {output_path}")

    @staticmethod
    def uv_to_xyz(uv, depth_file, extrinsics):
        depth = np.load(depth_file)
        z = depth[uv[1], uv[0]] * 0.001
        fx, fy, cx, cy = 600.42, 600.68, 328.08, 238.68
        x = (uv[0] - cx) * z / fx
        y = (uv[1] - cy) * z / fy
        cam_point = np.array([x, y, z, 1.0]).reshape(4, 1)
        base_point = extrinsics @ cam_point
        return base_point[:3].flatten()

    @staticmethod
    def transform_camera2base(gripper_pose):
        R = np.array(
            [[-0.034, -0.999, 0], [0.999, -0.034, -0.006], [0.006, 0, 1]],
            dtype=np.float32,
        )
        t = np.array([[0.09], [-0.03], [0.02]], dtype=np.float32)

        def euler_to_rot(rx, ry, rz):
            Rx = np.array(
                [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
            )
            Ry = np.array(
                [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
            )
            Rz = np.array(
                [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
            )
            return Rz @ Ry @ Rx

        T = np.eye(4)
        T[:3, :3] = euler_to_rot(*gripper_pose[3:])
        T[:3, 3] = gripper_pose[:3]
        T_gripper = T

        T_cam = np.eye(4)
        T_cam[:3, :3] = R
        T_cam[:3, 3] = t.flatten()

        return T_gripper @ T_cam


class RealmanArm:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, host, port, speed, radius, connect, block):
        if self._initialized:
            return
        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.robot.rm_create_robot_arm(host, port)
        self.v, self.r = speed, radius
        self.connect, self.block = connect, block
        print(f"[INFO] Robot handle: {self.handle.id}")
        self._initialized = True

    def observe(self):
        self.robot.rm_movej(
            [2, 84, -74, -12, -100, 4], self.v, self.r, self.connect, self.block
        )
        print("[INFO] Arm moved to observe pose")

    def grasp(self, pose):
        self.robot.rm_set_gripper_release(500, True, 10)
        self.robot.rm_movej_p(pose, self.v, self.r, self.connect, self.block)
        # pose[0] += 0.025
        self.robot.rm_movej_p(pose, self.v, self.r, self.connect, self.block)
        self.robot.rm_set_gripper_position(5, False, 5)
        time.sleep(1)
        pose[0] -= 0.10
        flag = self.robot.rm_movej_p(pose, self.v, self.r, self.connect, self.block)
        print("[INFO] Arm grasp sequence complete")
        return flag


# Initialize FastMCP server
mcp = FastMCP("robots", stateless_http=True, host="0.0.0.0", port=8000)


@mcp.tool()
def navigate_to_target(marker_name: str) -> dict:
    """
    Perform navigation of the robot to a specified marker position.

    Args:
        marker_name (str): The destination marker name.

    Returns:
        str: A message indicating whether navigation to the marker succeeded or failed.

    Raises:
        FileNotFoundError: If `config.yaml` does not exist.
        KeyError: If required keys (host, port) are missing in the config file.
        Exception: Any error during robot communication or movement execution.

    """

    robot = RealmanChassis(chassis_host, chassis_port)
    robot.cancel_current_move()
    _, _, status = robot.move_to_position(marker_name)
    robot.close_connection()

    return (
        f"Navigation to '{marker_name}' {'succeeded' if status == 'ok' else 'failed'}"
    )


dino = Dino()
camera = Camera()
arm = RealmanArm(arm_host, arm_port, speed, radius, connect, block)


@mcp.tool()
def grasp_object(object: str):
    """
    Perform object grasping using Realman robotic arm, wrist-mounted camera, and Grounding DINO model.

    Args:
        object (str): The name of the target object to grasp (e.g., "apple", "bottle").

    Returns:
        str: A message indicating whether the grasping operation succeeded or failed.
    """

    arm.observe()
    color_path, depth_path = camera.record_wrist_frame()
    boxes, _ = dino.predict(color_path, object)

    if boxes is None or boxes.numel() == 0:
        return f"{object} not found."

    uv = [int(x) for x in boxes[0][:2].numpy()]
    gripper_pose = arm.robot.rm_get_current_arm_state()[1]["pose"]
    extrinsics = dino.transform_camera2base(gripper_pose)
    xyz = dino.uv_to_xyz(uv, depth_path, extrinsics)

    grasp_pose = xyz.tolist() + [1.57, -1.5707, 1.5]
    grasp_pose[0] -= 0.09
    grasp_pose[1] += 0.015
    grasp_pose[2] -= 0.0

    success = arm.grasp(grasp_pose)
    return f"'{object}' has been {'successfully' if success == 'ok' else 'failed'}  grasped"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="streamable-http")
