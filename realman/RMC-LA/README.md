# 🤖 Realman Robot Skill Server

This project provides skill services for a Realman robot, including object grasping with a wrist-mounted camera and target navigation. It integrates:

- 🦾 Realman Arm SDK
- 🦿 Realman Chassis Control (TCP)
- 🎯 [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for grounding-based visual detection
- 📷 Intel RealSense Camera
- 🧠 FastMCP protocol server

---

## 📦 Prerequisites

- Python 3.10+
- RealSense D435i (or compatible)
- PyTorch 2.6.0
- Redis server
- Realman robot (arm + chassis)
- GroundingDINO (see below)

---

## ⚙️ Installation
You can run the project using either **Docker** (recommended) or directly from the **source code**.


### 🚀 Option 1: Using Docker (Recommended)

1. **Pull the prebuilt Docker image:**

```bash
docker pull flagrelease-registry.cn-beijing.cr.aliyuncs.com/flagrelease/flagrelease:realman-rmc-la-d345

```

2. **Launch the container:**
```bash
docker run -it \
  -e CHASSIS_HOST=127.0.0.1 \   
  -e CHASSIS_PORT=5000 \
  -e ARM_HOST=127.0.0.1 \
  -e ARM_PORT=8080 \
  -v /dev:/dev \
  --device-cgroup-rule "c 81:* rmw" \
  --device-cgroup-rule "c 189:* rmw" \
  -p 8000:8000 \
  realman-rmc-la-d345:latest
```


### 🧩 Option 2: Run from Source Code

1. **Clone the repository**

```bash
git clone https://github.com/FlagOpen/RoboSkill

cd RoboSkill/realman/RMC-LA

```

2. **(Optional) Setup conda environment**
It is recommended to use a fresh virtual environment (e.g., with Conda or venv). Then install dependencies:
```bash
pip install -r requirements.txt
```

3. **Install GroundingDINO**

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git

cd GroundingDINO

pip install -e .

```
4. **Download the weights:**
```bash
mkdir weights
wget https://huggingface.co/IDEA-Research/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth -P weights/
```

5. **🔧 Configuration**
Before running, you must configure IP and port settings in `skill.py`:

```bash
# Chassis configuration
chassis_host = "127.0.0.1"   # 🔁 Replace with your chassis IP address
chassis_port = 5000

# Robotic arm configuration
arm_host = "127.0.0.1"       # 🔁 Replace with your arm IP address
arm_port = 5000

```

5. 🚀 Running the Service
```bash
python skill.py
```
This will:

+ Start the MCP-compatible skill server on http://0.0.0.0:8000

+ Load the GroundingDINO model

+ Initialize the RealSense camera

+ Initialize the Realman robotic arm



## 📂 Output

+ `./output/wrist_obs.png`: Captured color image

+ `./output/annotated_image.png`: Visualized prediction with bounding boxes

+ `./output/wrist_obs_depth.npy`: Depth image



## 🛠️ Troubleshooting

+ ❌ No RealSense Device Found
Make sure the camera is plugged in and pyrealsense2 is installed.

+ ❌ Grasping Fails
Ensure good lighting and that the target object is in the field of view.

+ ❌ Cannot connect to robot
Double-check the IP/port values in skill.py.


## 📄 License



## ✨ Acknowledgments
+ [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
+ [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)