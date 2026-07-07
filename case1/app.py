import sys
sys.modules.setdefault("app", sys.modules[__name__])
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
import os
import cv2
import numpy as np
import base64
import time
import json
from datetime import datetime
import database
from ascend_inference import FaceSystem
from camera import VideoCamera

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局对象
face_system = None
video_camera = None
latest_temperature = None
latest_thermal_matrix = None
latest_thermal_time = 0.0
latest_thermal_info = {}

THERMAL_FACE_MIN = 29.2
THERMAL_FACE_MAX = 38.5
THERMAL_FACE_DELTA_MIN = 0.25
THERMAL_FACE_ROI_RADIUS = 6
THERMAL_FACE_MIN_HOT_POINTS = 1
THERMAL_FACE_STALE_SECONDS = 3.0





latest_thermal_info = {}
latest_liveness = False
latest_thermal_time = 0.0



latest_thermal_info = {}
latest_liveness = False
thermal_live_history = []
thermal_live_streak = 0
thermal_dead_streak = 0
# 严格活体检测参数


TEMP_MIN = 30.8
TEMP_MAX = 38.5

def get_face_system():
    global face_system
    if face_system is None:
        try:
            face_system = FaceSystem()
        except Exception as e:
            print(f"初始化人脸系统失败: {e}")
            face_system = None
    return face_system

def _load_thermal_homography():
    path = "thermal_calibration.json"
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        H = np.array(data.get("homography"), dtype=np.float32)
        if H.shape == (3, 3):
            return H
    except Exception as e:
        print(f"load thermal calibration failed: {e}")

    return None


def _map_camera_face_to_thermal(face_box, frame_shape):
    x1, y1, x2, y2 = face_box
    frame_h, frame_w = frame_shape[:2]

    # Use upper-middle face point: forehead / nose bridge / cheeks area
    cx = (x1 + x2) / 2.0
    cy = y1 + (y2 - y1) * 0.45

    H = _load_thermal_homography()

    if H is not None:
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)
        tx, ty = mapped[0][0]
        source = "homography"
    else:
        # Fallback when calibration is not finished.
        # For best accuracy, create thermal_calibration.json later.
        tx = cx / max(frame_w, 1) * 31.0
        ty = cy / max(frame_h, 1) * 23.0
        source = "rough_scale_no_calibration"

    tx = int(round(tx))
    ty = int(round(ty))

    tx = max(0, min(31, tx))
    ty = max(0, min(23, ty))

    return tx, ty, source


def get_live_temperature_for_camera(face_box=None, frame_shape=None):
    global latest_thermal_matrix, latest_thermal_time, latest_thermal_info

    if latest_thermal_matrix is None:
        print("Liveness failed: no thermal matrix")
        return {
            "roi_live": False,
            "temperature": 0.0,
            "thermal_center": None,
            "reason": "no thermal matrix"
        }

    age = time.time() - latest_thermal_time
    if age > THERMAL_FACE_STALE_SECONDS:
        print(f"Liveness failed: thermal frame stale, age={age:.2f}s")
        return {
            "roi_live": False,
            "temperature": 0.0,
            "thermal_center": None,
            "reason": f"thermal frame stale, age={age:.2f}s"
        }

    if face_box is None or frame_shape is None:
        print("Liveness failed: no face box for ROI check")
        return {
            "roi_live": False,
            "temperature": 0.0,
            "thermal_center": None,
            "reason": "no face box"
        }

    arr = latest_thermal_matrix
    ambient = float(np.median(arr))

    tx, ty, map_source = _map_camera_face_to_thermal(face_box, frame_shape)

    # Search near mapped face point because visible camera and MLX90640 are not perfectly aligned.
    search_radius = max(THERMAL_FACE_ROI_RADIUS, 4)

    sx1 = max(0, tx - search_radius)
    sx2 = min(32, tx + search_radius + 1)
    sy1 = max(0, ty - search_radius)
    sy2 = min(24, ty + search_radius + 1)

    search = arr[sy1:sy2, sx1:sx2]
    if search.size == 0:
        print("Liveness failed: empty search ROI")
        return {
            "roi_live": False,
            "temperature": 0.0,
            "thermal_center": None,
            "reason": "empty search ROI"
        }

    best = None
    local_radius = 1

    for yy in range(search.shape[0]):
        for xx in range(search.shape[1]):
            cx = sx1 + xx
            cy = sy1 + yy

            x1 = max(0, cx - local_radius)
            x2 = min(32, cx + local_radius + 1)
            y1 = max(0, cy - local_radius)
            y2 = min(24, cy + local_radius + 1)

            roi = arr[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_p90 = float(np.percentile(roi, 90))
            roi_p95 = float(np.percentile(roi, 95))
            roi_max = float(np.max(roi))
            roi_mean = float(np.mean(roi))
            roi_delta = roi_p90 - ambient

            hot_mask = (
                (roi >= THERMAL_FACE_MIN) &
                (roi <= THERMAL_FACE_MAX) &
                ((roi - ambient) >= THERMAL_FACE_DELTA_MIN)
            )
            hot_points = int(np.sum(hot_mask))

            score = roi_p90 + roi_delta + hot_points * 0.15

            item = {
                "score": score,
                "cx": int(cx),
                "cy": int(cy),
                "roi": [int(x1), int(y1), int(x2), int(y2)],
                "roi_mean": roi_mean,
                "roi_p90": roi_p90,
                "roi_p95": roi_p95,
                "roi_max": roi_max,
                "roi_delta": roi_delta,
                "hot_points": hot_points
            }

            if best is None or item["score"] > best["score"]:
                best = item

    if best is None:
        print("Liveness failed: no candidate ROI")
        return {
            "roi_live": False,
            "temperature": 0.0,
            "thermal_center": None,
            "reason": "no candidate ROI"
        }

    temp_ok = THERMAL_FACE_MIN <= best["roi_p90"] <= THERMAL_FACE_MAX
    delta_ok = best["roi_delta"] >= THERMAL_FACE_DELTA_MIN
    count_ok = best["hot_points"] >= THERMAL_FACE_MIN_HOT_POINTS
    too_hot = best["roi_max"] > 39.5

    roi_live = bool(temp_ok and delta_ok and count_ok and not too_hot)

    reason = "roi liveness passed"
    if not temp_ok:
        reason = "face ROI temperature invalid"
    elif not delta_ok:
        reason = "face ROI delta too low"
    elif not count_ok:
        reason = "not enough hot points in face ROI"
    elif too_hot:
        reason = "face ROI too hot"

    latest_thermal_info = {
        "mode": "face_roi_sync_ready",
        "ambient": ambient,
        "face_box": [int(v) for v in face_box],
        "mapped_point": [int(tx), int(ty)],
        "search_roi": [int(sx1), int(sy1), int(sx2), int(sy2)],
        "map_source": map_source,
        "best_roi": best,
        "thermal_center": [int(best["cx"]), int(best["cy"])],
        "temp_ok": temp_ok,
        "delta_ok": delta_ok,
        "count_ok": count_ok,
        "too_hot": too_hot,
        "roi_live": roi_live,
        "age_ms": int(age * 1000),
        "reason": reason
    }

    print(
        f"Face ROI thermal: roi_live={roi_live}, "
        f"map={map_source}, mapped=({tx},{ty}), "
        f"thermal_center=({best['cx']},{best['cy']}), "
        f"ambient={ambient:.2f}, p90={best['roi_p90']:.2f}, "
        f"delta={best['roi_delta']:.2f}, hot_points={best['hot_points']}, "
        f"age={age*1000:.0f}ms, reason={reason}"
    )

    return {
        "roi_live": roi_live,
        "temperature": float(best["roi_p90"]) if roi_live else 0.0,
        "thermal_center": (float(best["cx"]), float(best["cy"])),
        "mapped_point": (float(tx), float(ty)),
        "ambient": ambient,
        "roi_p90": float(best["roi_p90"]),
        "roi_delta": float(best["roi_delta"]),
        "hot_points": int(best["hot_points"]),
        "reason": reason,
        "map_source": map_source,
        "age_ms": int(age * 1000)
    }


def get_video_camera():
    global video_camera

    if video_camera is None:
        fs = get_face_system()
        if fs:
            try:
                video_camera = VideoCamera(
                    fs,
                    get_temperature_func=get_live_temperature_for_camera,
                    temp_min=TEMP_MIN,
                    temp_max=TEMP_MAX,
                    camera_indices=[2, 0, 1, 3]
                )
            except Exception as e:
                print(f"Init VideoCamera failed: {e}")
                video_camera = None

    return video_camera


@app.route('/api/temperature', methods=['POST'])
def receive_temperature():
    global latest_temperature, latest_thermal_matrix, latest_thermal_time, latest_thermal_info

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "success": False,
            "error": "missing json body"
        }), 400

    try:
        now_time = time.time()

        if "temperatures" in data:
            temps = data.get("temperatures", [])

            if not isinstance(temps, list):
                return jsonify({
                    "success": False,
                    "error": "temperatures must be list"
                }), 400

            temps = [float(x) for x in temps]

            if len(temps) < 768:
                latest_thermal_matrix = None
                latest_temperature = 0.0
                latest_thermal_time = now_time
                return jsonify({
                    "success": False,
                    "error": "temperature frame points not enough",
                    "points": len(temps),
                    "allow_face_recognition": False
                }), 400

            arr = np.array(temps[:768], dtype=np.float32).reshape(24, 32)
            flat = arr.reshape(-1)

            min_temp = float(np.min(flat))
            max_temp = float(np.max(flat))
            avg_temp = float(np.mean(flat))
            ambient = float(np.median(flat))
            p90_temp = float(np.percentile(flat, 90))
            p95_temp = float(np.percentile(flat, 95))

            latest_thermal_matrix = arr
            latest_thermal_time = now_time

            # 注意：这里不再用整帧判定活体。
            # 活体必须在人脸识别成功后，由 get_live_temperature_for_camera(face_box, frame_shape) 判断。
            latest_temperature = p90_temp
            latest_thermal_info = {
                "mode": "stored_frame_only",
                "points": 768,
                "min_temperature": min_temp,
                "max_temperature": max_temp,
                "avg_temperature": avg_temp,
                "ambient": ambient,
                "p90_temperature": p90_temp,
                "p95_temperature": p95_temp,
                "updated_at": latest_thermal_time,
                "reason": "frame stored; liveness waits for matched face ROI"
            }

            print(
                f"Thermal frame stored: "
                f"min={min_temp:.2f}, avg={avg_temp:.2f}, max={max_temp:.2f}, "
                f"ambient={ambient:.2f}, p90={p90_temp:.2f}, p95={p95_temp:.2f}"
            )

            return jsonify({
                "success": True,
                "temperature": p90_temp,
                "allow_face_recognition": False,
                "thermal": latest_thermal_info
            })

        if "temperature" in data:
            # 单点温度无法可靠区分人、热水壶、环境热源。
            # 为了安全，单点模式只保存，不允许直接活体通过。
            temperature = float(data["temperature"])
            latest_temperature = temperature
            latest_thermal_matrix = None
            latest_thermal_time = now_time
            latest_thermal_info = {
                "mode": "single_temperature_ignored_for_liveness",
                "raw_temperature": temperature,
                "updated_at": latest_thermal_time,
                "reason": "single temperature is not accepted as liveness"
            }

            print(f"Single temperature stored but not accepted for liveness: {temperature:.2f}")

            return jsonify({
                "success": True,
                "temperature": temperature,
                "allow_face_recognition": False,
                "thermal": latest_thermal_info
            })

        return jsonify({
            "success": False,
            "error": "missing temperature or temperatures",
            "allow_face_recognition": False
        }), 400

    except Exception as e:
        latest_thermal_matrix = None
        latest_temperature = 0.0
        latest_thermal_time = time.time()

        print(f"receive_temperature error: {e}")

        return jsonify({
            "success": False,
            "error": str(e),
            "allow_face_recognition": False
        }), 500



@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/users_page')
def users_page():
    return render_template('users.html')

@app.route('/attendance_page')
def attendance_page():
    return render_template('attendance.html')

@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        users = database.get_users()
        result = []

        for index, user in enumerate(users, start=1):
            u_dict = dict(user)

            # 前端列表只需要显示信息，不返回 bytes 类型的人脸特征
            clean_user = {}
            for key, value in u_dict.items():
                if isinstance(value, (bytes, bytearray, memoryview)):
                    continue
                clean_user[key] = value

            # seq 是页面显示序号，id 仍然保留给删除/修改使用
            clean_user["seq"] = index
            result.append(clean_user)

        return jsonify(result)
    except Exception as e:
        print(f"获取用户列表失败: {e}")
        return jsonify([])

@app.route('/api/camera/capture', methods=['POST'])
def capture_from_device():
    cam = get_video_camera()
    if cam is None:
        return jsonify({"error": "摄像头不可用"}), 503

    frame = cam.get_snapshot()
    if frame is None:
        return jsonify({"error": "抓取画面失败"}), 500

    # 保存到临时文件
    filename = f"capture_{int(time.time())}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, frame)

    return jsonify({"success": True, "temp_path": filename})

@app.route('/api/users', methods=['POST'])
def add_user():
    name = request.form.get('name')
    img = None

    # 检查是使用上传文件、抓拍的画面还是 base64 数据
    if 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        filename = f"{int(time.time())}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = cv2.imread(filepath)
    elif 'temp_path' in request.form:
        temp_filename = request.form.get('temp_path')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        if os.path.exists(filepath):
            img = cv2.imread(filepath)
        else:
            return jsonify({"error": "抓拍文件未找到"}), 400
    elif 'image_base64' in request.form:
        data = request.form['image_base64']
        if ',' in data:
            data = data.split(',')[1]
        img_data = base64.b64decode(data)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
         return jsonify({"error": "未提供图片"}), 400

    if img is None:
        return jsonify({"error": "无效的图片"}), 400

    fs = get_face_system()
    if fs is None:
        return jsonify({"error": "人脸系统未初始化"}), 500

    try:
        faces = fs.detect(img)
        if len(faces) > 0:
            best_face = max(faces, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            x1, y1, x2, y2 = map(int, best_face)
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_img = img[y1:y2, x1:x2]
        else:
            face_img = img

        embedding = fs.get_embedding(face_img)
        embedding_blob = embedding.tobytes()

        # 保存头像
        avatar_filename = f"avatar_{int(time.time())}.jpg"
        avatar_path = os.path.join(app.config['UPLOAD_FOLDER'], avatar_filename)
        cv2.imwrite(avatar_path, face_img)

        user_id = database.add_user(name, embedding_blob, avatar_filename)
        print(f"用户已添加: {name} (ID: {user_id})")
        return jsonify({"success": True, "user_id": user_id})
    except Exception as e:
        print(f"添加用户失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    database.delete_user(user_id)
    return jsonify({"success": True})

@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    name = request.json.get('name')
    if name:
        database.update_user_name(user_id, name)
        return jsonify({"success": True})
    return jsonify({"error": "姓名不能为空"}), 400

@app.route('/api/clockin', methods=['POST'])
def clockin():
    # 手动打卡（上传或客户端摄像头）
    img = None
    filepath = "unknown"

    if 'image' in request.files:
        file = request.files['image']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"clockin_{int(time.time())}.jpg")
        file.save(filepath)
        img = cv2.imread(filepath)
    elif 'image_base64' in request.form:
        data = request.form['image_base64']
        if ',' in data:
            data = data.split(',')[1]
        img_data = base64.b64decode(data)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        filepath = "client_camera"

    if img is None:
        return jsonify({"error": "无图片数据"}), 400
        
    if latest_temperature is None:
        return jsonify({
            "success": False,
            "error": "未收到温度数据，禁止人脸识别"
        }), 400

    if latest_temperature <= TEMP_MIN or latest_temperature >= TEMP_MAX:
        return jsonify({
            "success": True,
            "match": False,
            "temperature": latest_temperature,
            "allow_face_recognition": False,
            "message": "体温异常，禁止人脸识别打卡"
        }), 200
        
    fs = get_face_system()
    
    faces = fs.detect(img)
    if len(faces) > 0:
        best_face = max(faces, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        x1, y1, x2, y2 = map(int, best_face)
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_img = img[y1:y2, x1:x2]
    else:
        face_img = img
    
    target_embedding = fs.get_embedding(face_img)
    
    users = database.get_users()
    max_sim = -1.0
    best_match = None
    threshold = 0.39
    
    for u in users:
        db_emb = np.frombuffer(u['embedding'], dtype=np.float32)
        sim = np.dot(target_embedding, db_emb) / (np.linalg.norm(target_embedding) * np.linalg.norm(db_emb) + 1e-6)
        if sim > max_sim:
            max_sim = sim
            best_match = u

    if best_match and max_sim > threshold:
        database.add_attendance(best_match['id'], 'manual', filepath, latest_temperature)
        return jsonify({
            "success": True, 
            "match": True, 
            "user": best_match['name'], 
            "similarity": float(max_sim)
        })
    else:
        return jsonify({
            "success": True, 
            "match": False,
            "similarity": float(max_sim)
        })


@app.route('/api/attendance/<int:attendance_id>', methods=['DELETE'])
def delete_attendance_record(attendance_id):
    global video_camera

    try:
        result = database.delete_attendance(attendance_id)

        if isinstance(result, tuple):
            deleted, user_id = result
        else:
            deleted = bool(result)
            user_id = None

        if not deleted:
            return jsonify({
                "success": False,
                "error": "考勤记录不存在"
            }), 404

        # 删除考勤后，清除该用户的自动考勤冷却状态，方便马上再次测试打卡
        if user_id is not None and video_camera is not None:
            try:
                if hasattr(video_camera, "last_attendance"):
                    video_camera.last_attendance.pop(int(user_id), None)
                    print(f"已清除用户 {user_id} 的考勤冷却状态")
            except Exception as e:
                print(f"清除考勤冷却状态失败: {e}")

        return jsonify({
            "success": True,
            "deleted_id": attendance_id,
            "user_id": user_id
        })

    except Exception as e:
        print(f"删除考勤记录失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/attendance', methods=['GET'])
def list_attendance():
    records = database.get_attendance()
    return jsonify([dict(r) for r in records])

def gen(camera):
    # 限制网页视频流帧率，避免 JPEG 编码占满 CPU
    stream_interval = 0.10  # 10 FPS

    while True:
        start_time = time.time()

        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.05)
            continue

        elapsed = time.time() - start_time
        sleep_time = stream_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

@app.route('/video_feed')
def video_feed():
    cam = get_video_camera()
    if cam is None:
        return "摄像头不可用", 503
    return Response(gen(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



# ================= RECTANGLE_PHONE_THERMAL_PATCH_V1 =================
# 作用：
# 在人脸 ROI 温度通过后，继续判断红外热区形状。
# 如果热源呈现明显矩形、面积大、填充率高、边缘规整、温度分布过于均匀，
# 则认为可能是手机屏幕/平板屏幕/矩形热源，直接阻止进入眨眼检测。

import time as _rect_phone_time
import math as _rect_phone_math

RECT_PHONE_ENABLE = True

# 手机/屏幕型热源判定阈值
RECT_PHONE_MIN_HOT_POINTS = 10       # 热点太少不判断为手机
RECT_PHONE_MIN_FILL_RATIO = 0.62     # 热区在外接矩形内填充越满，越像矩形屏幕
RECT_PHONE_MIN_AREA_RATIO = 0.18     # 热区外接矩形占局部搜索区域比例
RECT_PHONE_MIN_RECT_SCORE = 0.70     # 综合矩形分数阈值
RECT_PHONE_MAX_TEMP_STD = 1.60       # 温度过于均匀，越像屏幕/热板
RECT_PHONE_MIN_ASPECT = 0.55         # 矩形宽高比下限
RECT_PHONE_MAX_ASPECT = 2.20         # 矩形宽高比上限

def _rect_phone_to_matrix(matrix):
    if matrix is None:
        return None

    try:
        if hasattr(matrix, "tolist"):
            matrix = matrix.tolist()

        flat = []

        if isinstance(matrix, list) and len(matrix) == 24 and isinstance(matrix[0], list):
            return [[float(v) for v in row[:32]] for row in matrix[:24]]

        if isinstance(matrix, list):
            for item in matrix:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)

        if len(flat) < 768:
            return None

        flat = [float(v) for v in flat[:768]]
        return [flat[i * 32:(i + 1) * 32] for i in range(24)]

    except Exception:
        return None


def _rect_phone_percentile(values, q):
    if not values:
        return 0.0

    values = sorted(values)
    if len(values) == 1:
        return float(values[0])

    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _rect_phone_std(values):
    if not values:
        return 0.0

    avg = sum(values) / len(values)
    var = sum((v - avg) * (v - avg) for v in values) / len(values)
    return _rect_phone_math.sqrt(var)


def _rect_phone_get_latest_matrix():
    """
    兼容不同版本里的全局温度缓存变量名。
    """
    g = globals()

    for name in [
        "latest_thermal_matrix",
        "thermal_matrix",
        "latest_temperature_matrix",
        "latest_temperatures",
        "temperature_matrix",
    ]:
        if name in g and g[name] is not None:
            m = _rect_phone_to_matrix(g[name])
            if m is not None:
                return m

    # 有些版本会存在 latest_thermal_info / latest_temperature_frame 字典
    for name in [
        "latest_thermal_info",
        "latest_temperature_info",
        "latest_temperature_frame",
        "latest_thermal_frame",
    ]:
        info = g.get(name)
        if isinstance(info, dict):
            for key in ["matrix", "temperatures", "data", "thermal", "frame"]:
                if key in info:
                    m = _rect_phone_to_matrix(info.get(key))
                    if m is not None:
                        return m

    return None


def _rect_phone_hot_shape(matrix, center=None, radius=8):
    """
    从 32x24 红外矩阵中判断热区是否像矩形屏幕。
    返回：
    {
        phone_like_rect: bool,
        rect_score: float,
        reason: str,
        ...
    }
    """
    m = _rect_phone_to_matrix(matrix)
    if m is None:
        return {
            "phone_like_rect": False,
            "rect_score": 0.0,
            "rect_reason": "no thermal matrix"
        }

    h = 24
    w = 32

    all_values = [m[y][x] for y in range(h) for x in range(w)]
    ambient = _rect_phone_percentile(all_values, 0.30)

    if center is None:
        # 如果拿不到人脸映射中心，就围绕全局最高温点做形状判断
        max_y, max_x = 0, 0
        max_v = -999.0
        for yy in range(h):
            for xx in range(w):
                if m[yy][xx] > max_v:
                    max_v = m[yy][xx]
                    max_y, max_x = yy, xx
        cx, cy = max_x, max_y
    else:
        try:
            cx, cy = center
            cx, cy = int(round(cx)), int(round(cy))
        except Exception:
            cx, cy = 16, 12

    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    r = max(4, min(10, int(radius)))

    x1 = max(0, cx - r)
    x2 = min(w - 1, cx + r)
    y1 = max(0, cy - r)
    y2 = min(h - 1, cy + r)

    roi_points = []
    for yy in range(y1, y2 + 1):
        for xx in range(x1, x2 + 1):
            roi_points.append((xx, yy, m[yy][xx]))

    if not roi_points:
        return {
            "phone_like_rect": False,
            "rect_score": 0.0,
            "rect_reason": "empty roi"
        }

    roi_values = [p[2] for p in roi_points]
    roi_p80 = _rect_phone_percentile(roi_values, 0.80)
    roi_p90 = _rect_phone_percentile(roi_values, 0.90)
    roi_max = max(roi_values)

    # 热点阈值：必须比环境高，也要接近 ROI 高温区域
    hot_threshold = max(
        ambient + 1.2,
        roi_p80,
        roi_max - 2.2
    )

    hot = []
    for xx, yy, temp in roi_points:
        if temp >= hot_threshold:
            hot.append((xx, yy, temp))

    if len(hot) < RECT_PHONE_MIN_HOT_POINTS:
        return {
            "phone_like_rect": False,
            "rect_score": 0.0,
            "rect_reason": "not enough hot points",
            "hot_points": len(hot),
            "ambient": round(ambient, 2),
            "hot_threshold": round(hot_threshold, 2),
            "roi_p90": round(roi_p90, 2),
        }

    xs = [p[0] for p in hot]
    ys = [p[1] for p in hot]
    temps = [p[2] for p in hot]

    bx1, bx2 = min(xs), max(xs)
    by1, by2 = min(ys), max(ys)

    bw = bx2 - bx1 + 1
    bh = by2 - by1 + 1
    bbox_area = max(1, bw * bh)
    roi_area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))

    fill_ratio = len(hot) / bbox_area
    area_ratio = bbox_area / roi_area
    aspect = bw / max(1, bh)
    temp_std = _rect_phone_std(temps)

    # 边缘规整性：矩形热源通常外接框四边都会有热点贴边
    top_count = sum(1 for xx, yy, tt in hot if yy == by1)
    bottom_count = sum(1 for xx, yy, tt in hot if yy == by2)
    left_count = sum(1 for xx, yy, tt in hot if xx == bx1)
    right_count = sum(1 for xx, yy, tt in hot if xx == bx2)

    edge_score = 0.0
    if bw >= 3:
        edge_score += min(top_count / bw, 1.0) * 0.25
        edge_score += min(bottom_count / bw, 1.0) * 0.25
    if bh >= 3:
        edge_score += min(left_count / bh, 1.0) * 0.25
        edge_score += min(right_count / bh, 1.0) * 0.25

    fill_score = min(fill_ratio / RECT_PHONE_MIN_FILL_RATIO, 1.0)
    area_score = min(area_ratio / RECT_PHONE_MIN_AREA_RATIO, 1.0)
    uniform_score = max(0.0, min((RECT_PHONE_MAX_TEMP_STD - temp_std) / RECT_PHONE_MAX_TEMP_STD, 1.0))

    aspect_ok = RECT_PHONE_MIN_ASPECT <= aspect <= RECT_PHONE_MAX_ASPECT

    rect_score = (
        fill_score * 0.38 +
        area_score * 0.22 +
        edge_score * 0.25 +
        uniform_score * 0.15
    )

    phone_like = (
        aspect_ok
        and fill_ratio >= RECT_PHONE_MIN_FILL_RATIO
        and area_ratio >= RECT_PHONE_MIN_AREA_RATIO
        and rect_score >= RECT_PHONE_MIN_RECT_SCORE
        and temp_std <= RECT_PHONE_MAX_TEMP_STD
    )

    if phone_like:
        reason = (
            "blocked rectangular thermal source, possible phone screen: "
            f"score={rect_score:.2f}, fill={fill_ratio:.2f}, area={area_ratio:.2f}, "
            f"aspect={aspect:.2f}, std={temp_std:.2f}"
        )
    else:
        reason = (
            "thermal shape not rectangular phone-like: "
            f"score={rect_score:.2f}, fill={fill_ratio:.2f}, area={area_ratio:.2f}, "
            f"aspect={aspect:.2f}, std={temp_std:.2f}"
        )

    return {
        "phone_like_rect": bool(phone_like),
        "rect_score": round(rect_score, 3),
        "rect_reason": reason,
        "rect_fill_ratio": round(fill_ratio, 3),
        "rect_area_ratio": round(area_ratio, 3),
        "rect_aspect": round(aspect, 3),
        "rect_temp_std": round(temp_std, 3),
        "rect_edge_score": round(edge_score, 3),
        "hot_points": len(hot),
        "hot_bbox": [int(bx1), int(by1), int(bx2), int(by2)],
        "ambient": round(ambient, 2),
        "hot_threshold": round(hot_threshold, 2),
        "roi_p90": round(roi_p90, 2),
    }


def _rect_phone_center_from_temp_info(temp_info):
    if not isinstance(temp_info, dict):
        return None

    for key in ["mapped_point", "thermal_center"]:
        pt = temp_info.get(key)
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                return int(round(float(pt[0]))), int(round(float(pt[1])))
            except Exception:
                pass

    return None


# 包装原来的 get_live_temperature_for_camera：
# 原函数继续负责 ROI 温度判断；
# 新包装函数只在 ROI 温度通过后增加“矩形热源/手机屏幕”拦截。
if RECT_PHONE_ENABLE and "get_live_temperature_for_camera" in globals():
    if "_original_get_live_temperature_for_camera_rect_phone" not in globals():
        _original_get_live_temperature_for_camera_rect_phone = get_live_temperature_for_camera

        def get_live_temperature_for_camera(face_box=None, frame_shape=None):
            temp_info = _original_get_live_temperature_for_camera_rect_phone(
                face_box=face_box,
                frame_shape=frame_shape
            )

            if not isinstance(temp_info, dict):
                return temp_info

            # 温度本身没通过，就不再做形状判断
            if not temp_info.get("roi_live", False):
                temp_info["phone_like_rect"] = False
                return temp_info

            matrix = _rect_phone_get_latest_matrix()
            center = _rect_phone_center_from_temp_info(temp_info)

            shape_info = _rect_phone_hot_shape(
                matrix,
                center=center,
                radius=8
            )

            temp_info.update(shape_info)

            if shape_info.get("phone_like_rect", False):
                temp_info["roi_live"] = False
                temp_info["reason"] = shape_info.get(
                    "rect_reason",
                    "blocked rectangular thermal source"
                )
                print(
                    "Phone-like thermal source blocked: "
                    f"score={shape_info.get('rect_score')}, "
                    f"fill={shape_info.get('rect_fill_ratio')}, "
                    f"area={shape_info.get('rect_area_ratio')}, "
                    f"aspect={shape_info.get('rect_aspect')}, "
                    f"std={shape_info.get('rect_temp_std')}"
                )
            else:
                print(
                    "Thermal shape passed: "
                    f"score={shape_info.get('rect_score')}, "
                    f"fill={shape_info.get('rect_fill_ratio')}, "
                    f"area={shape_info.get('rect_area_ratio')}, "
                    f"aspect={shape_info.get('rect_aspect')}, "
                    f"std={shape_info.get('rect_temp_std')}"
                )

            return temp_info

        print("RECTANGLE_PHONE_THERMAL_PATCH_V1 loaded")

# ================= END RECTANGLE_PHONE_THERMAL_PATCH_V1 =================




# ================= PHONE_SCREEN_THERMAL_BLOCK_V2 =================
# 目的：
# 手机播放照片/视频时，即使人脸识别通过，也要在眨眼检测前尽量拦截。
# 思路：
# 1. 不只看人脸 ROI，而是在人脸映射点周围扩大区域搜索热源形状；
# 2. 如果热源呈现“大面积、较均匀、填充率高、宽高比像屏幕/手机”的块状区域，则判定疑似手机；
# 3. 疑似手机时直接把 roi_live 改成 False，原有流程就不会进入眨眼检测。

import math as _phone_v2_math

PHONE_SCREEN_BLOCK_ENABLE = True

# 这些阈值是偏保守拦截手机的版本：
# 如果真人也被误拦，再把 PHONE_V2_SCORE_BLOCK 从 0.58 调到 0.68。
PHONE_V2_SEARCH_RADIUS = 10
PHONE_V2_MIN_HOT_POINTS = 8
PHONE_V2_MIN_FILL_RATIO = 0.48
PHONE_V2_MIN_AREA_RATIO = 0.12
PHONE_V2_SCORE_BLOCK = 0.58
PHONE_V2_MAX_STD_FOR_SCREEN = 2.20
PHONE_V2_MIN_ASPECT = 0.45
PHONE_V2_MAX_ASPECT = 2.80

def _phone_v2_matrix(matrix):
    if matrix is None:
        return None

    try:
        if hasattr(matrix, "tolist"):
            matrix = matrix.tolist()

        if isinstance(matrix, list) and len(matrix) == 24 and isinstance(matrix[0], list):
            out = []
            for row in matrix[:24]:
                if len(row) < 32:
                    return None
                out.append([float(v) for v in row[:32]])
            return out

        flat = []
        if isinstance(matrix, list):
            for item in matrix:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)

        if len(flat) < 768:
            return None

        flat = [float(v) for v in flat[:768]]
        return [flat[i * 32:(i + 1) * 32] for i in range(24)]

    except Exception:
        return None


def _phone_v2_percentile(values, q):
    if not values:
        return 0.0

    values = sorted(float(v) for v in values)
    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _phone_v2_std(values):
    if not values:
        return 0.0

    avg = sum(values) / len(values)
    return _phone_v2_math.sqrt(sum((v - avg) * (v - avg) for v in values) / len(values))


def _phone_v2_latest_matrix():
    g = globals()

    # 常见全局变量名
    for name in [
        "latest_thermal_matrix",
        "thermal_matrix",
        "latest_temperature_matrix",
        "latest_temperatures",
        "temperature_matrix",
    ]:
        if name in g and g[name] is not None:
            m = _phone_v2_matrix(g[name])
            if m is not None:
                return m

    # 常见字典缓存名
    for name in [
        "latest_thermal_info",
        "latest_temperature_info",
        "latest_temperature_frame",
        "latest_thermal_frame",
    ]:
        info = g.get(name)
        if isinstance(info, dict):
            for key in ["matrix", "temperatures", "data", "thermal", "frame"]:
                if key in info:
                    m = _phone_v2_matrix(info.get(key))
                    if m is not None:
                        return m

    return None


def _phone_v2_center(temp_info):
    if not isinstance(temp_info, dict):
        return None

    for key in ["mapped_point", "thermal_center"]:
        pt = temp_info.get(key)
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                return int(round(float(pt[0]))), int(round(float(pt[1])))
            except Exception:
                pass

    return None


def _phone_v2_biggest_component(mask):
    h = len(mask)
    w = len(mask[0]) if h else 0
    seen = [[False for _ in range(w)] for _ in range(h)]
    best = []

    for y in range(h):
        for x in range(w):
            if seen[y][x] or not mask[y][x]:
                continue

            stack = [(x, y)]
            seen[y][x] = True
            comp = []

            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))

                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and mask[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((nx, ny))

            if len(comp) > len(best):
                best = comp

    return best


def _phone_v2_detect_screen_like(matrix, center=None):
    m = _phone_v2_matrix(matrix)

    if m is None:
        return {
            "phone_like_rect": False,
            "phone_screen_score": 0.0,
            "phone_screen_reason": "no thermal matrix"
        }

    H = 24
    W = 32
    all_values = [m[y][x] for y in range(H) for x in range(W)]
    ambient = _phone_v2_percentile(all_values, 0.30)

    if center is None:
        # 没有人脸映射点时，围绕全局最高温搜索
        max_x, max_y, max_v = 16, 12, -999.0
        for yy in range(H):
            for xx in range(W):
                if m[yy][xx] > max_v:
                    max_v = m[yy][xx]
                    max_x, max_y = xx, yy
        cx, cy = max_x, max_y
    else:
        cx, cy = center

    cx = max(0, min(W - 1, int(cx)))
    cy = max(0, min(H - 1, int(cy)))

    r = PHONE_V2_SEARCH_RADIUS
    x1 = max(0, cx - r)
    x2 = min(W - 1, cx + r)
    y1 = max(0, cy - r)
    y2 = min(H - 1, cy + r)

    roi_values = []
    for yy in range(y1, y2 + 1):
        for xx in range(x1, x2 + 1):
            roi_values.append(m[yy][xx])

    if not roi_values:
        return {
            "phone_like_rect": False,
            "phone_screen_score": 0.0,
            "phone_screen_reason": "empty roi"
        }

    roi_p70 = _phone_v2_percentile(roi_values, 0.70)
    roi_p80 = _phone_v2_percentile(roi_values, 0.80)
    roi_p90 = _phone_v2_percentile(roi_values, 0.90)
    roi_max = max(roi_values)

    # 手机屏幕/机身热源有时温升不一定特别高，所以阈值不能设太高。
    # 但仍要求明显高于背景。
    hot_threshold = max(
        ambient + 0.9,
        roi_p70,
        roi_max - 2.8
    )

    local_w = x2 - x1 + 1
    local_h = y2 - y1 + 1

    mask = []
    for yy in range(y1, y2 + 1):
        row = []
        for xx in range(x1, x2 + 1):
            row.append(m[yy][xx] >= hot_threshold)
        mask.append(row)

    comp = _phone_v2_biggest_component(mask)

    if len(comp) < PHONE_V2_MIN_HOT_POINTS:
        return {
            "phone_like_rect": False,
            "phone_screen_score": 0.0,
            "phone_screen_reason": "not enough hot component points",
            "phone_hot_points": len(comp),
            "ambient": round(ambient, 2),
            "hot_threshold": round(hot_threshold, 2),
            "roi_p90": round(roi_p90, 2),
        }

    xs = [p[0] for p in comp]
    ys = [p[1] for p in comp]

    bx1, bx2 = min(xs), max(xs)
    by1, by2 = min(ys), max(ys)

    bw = bx2 - bx1 + 1
    bh = by2 - by1 + 1
    bbox_area = max(1, bw * bh)
    roi_area = max(1, local_w * local_h)

    fill_ratio = len(comp) / bbox_area
    area_ratio = bbox_area / roi_area
    aspect = bw / max(1, bh)

    comp_temps = []
    for lx, ly in comp:
        gx = x1 + lx
        gy = y1 + ly
        comp_temps.append(m[gy][gx])

    temp_std = _phone_v2_std(comp_temps)

    # 判断四边是否比较“贴边规整”
    top = sum(1 for lx, ly in comp if ly == by1)
    bottom = sum(1 for lx, ly in comp if ly == by2)
    left = sum(1 for lx, ly in comp if lx == bx1)
    right = sum(1 for lx, ly in comp if lx == bx2)

    edge_score = 0.0
    if bw >= 3:
        edge_score += min(top / bw, 1.0) * 0.25
        edge_score += min(bottom / bw, 1.0) * 0.25
    if bh >= 3:
        edge_score += min(left / bh, 1.0) * 0.25
        edge_score += min(right / bh, 1.0) * 0.25

    fill_score = min(fill_ratio / PHONE_V2_MIN_FILL_RATIO, 1.0)
    area_score = min(area_ratio / PHONE_V2_MIN_AREA_RATIO, 1.0)
    uniform_score = max(0.0, min((PHONE_V2_MAX_STD_FOR_SCREEN - temp_std) / PHONE_V2_MAX_STD_FOR_SCREEN, 1.0))

    aspect_ok = PHONE_V2_MIN_ASPECT <= aspect <= PHONE_V2_MAX_ASPECT

    # 屏幕/手机型热源综合分数
    screen_score = (
        fill_score * 0.34 +
        area_score * 0.26 +
        edge_score * 0.20 +
        uniform_score * 0.20
    )

    # 额外规则：
    # 如果热区面积比较大、填充率高、温度又比较均匀，即使边缘不完美，也拦截。
    large_uniform_block = (
        area_ratio >= 0.18
        and fill_ratio >= 0.42
        and temp_std <= 2.4
        and aspect_ok
    )

    phone_like = (
        aspect_ok
        and fill_ratio >= PHONE_V2_MIN_FILL_RATIO
        and area_ratio >= PHONE_V2_MIN_AREA_RATIO
        and screen_score >= PHONE_V2_SCORE_BLOCK
        and temp_std <= PHONE_V2_MAX_STD_FOR_SCREEN
    ) or large_uniform_block

    reason = (
        f"score={screen_score:.2f}, fill={fill_ratio:.2f}, area={area_ratio:.2f}, "
        f"aspect={aspect:.2f}, std={temp_std:.2f}, hot={len(comp)}, "
        f"bbox=({x1 + bx1},{y1 + by1})-({x1 + bx2},{y1 + by2})"
    )

    return {
        "phone_like_rect": bool(phone_like),
        "phone_screen_score": round(screen_score, 3),
        "phone_screen_reason": reason,
        "phone_fill_ratio": round(fill_ratio, 3),
        "phone_area_ratio": round(area_ratio, 3),
        "phone_aspect": round(aspect, 3),
        "phone_temp_std": round(temp_std, 3),
        "phone_edge_score": round(edge_score, 3),
        "phone_hot_points": len(comp),
        "phone_hot_bbox": [int(x1 + bx1), int(y1 + by1), int(x1 + bx2), int(y1 + by2)],
        "ambient": round(ambient, 2),
        "hot_threshold": round(hot_threshold, 2),
        "roi_p90": round(roi_p90, 2),
    }


if PHONE_SCREEN_BLOCK_ENABLE and "get_live_temperature_for_camera" in globals():
    if "_original_get_live_temperature_for_camera_phone_v2" not in globals():
        _original_get_live_temperature_for_camera_phone_v2 = get_live_temperature_for_camera

        def get_live_temperature_for_camera(face_box=None, frame_shape=None):
            temp_info = _original_get_live_temperature_for_camera_phone_v2(
                face_box=face_box,
                frame_shape=frame_shape
            )

            if not isinstance(temp_info, dict):
                return temp_info

            # 原本温度没通过，直接返回
            if not temp_info.get("roi_live", False):
                temp_info.setdefault("phone_like_rect", False)
                return temp_info

            matrix = _phone_v2_latest_matrix()
            center = _phone_v2_center(temp_info)

            shape = _phone_v2_detect_screen_like(matrix, center=center)
            temp_info.update(shape)

            if shape.get("phone_like_rect", False):
                temp_info["roi_live"] = False
                temp_info["reason"] = "blocked phone screen thermal source: " + shape.get("phone_screen_reason", "")
                print("Phone screen thermal blocked: " + shape.get("phone_screen_reason", ""))
            else:
                print("Phone screen check passed: " + shape.get("phone_screen_reason", ""))

            return temp_info

        print("PHONE_SCREEN_THERMAL_BLOCK_V2 loaded")

# ================= END PHONE_SCREEN_THERMAL_BLOCK_V2 =================








# ================= THERMAL_FRAME_SANITY_ROUTE_GUARD_V2 =================
# 作用：
# 拦截异常 MLX90640 温度帧，防止串口错位/脏数据进入缓存。
# 这个版本不修改 /api/temperature 原函数内部，而是在 Flask 路由层包装它，
# 所以不会出现 return outside function。

THERMAL_VALID_MIN = -20.0
THERMAL_VALID_MAX = 80.0
THERMAL_REASONABLE_AVG_MIN = 5.0
THERMAL_REASONABLE_AVG_MAX = 45.0
THERMAL_MIN_VALID_POINTS = 700

def _thermal_route_guard_flatten(values):
    flat = []

    def walk(x):
        if isinstance(x, (list, tuple)):
            for item in x:
                walk(item)
        else:
            try:
                flat.append(float(x))
            except Exception:
                pass

    walk(values)
    return flat


def _thermal_route_guard_pick_payload(data):
    if not isinstance(data, dict):
        return None

    # 常见字段名
    for key in [
        "temperatures",
        "matrix",
        "thermal_matrix",
        "data",
        "frame",
        "thermal",
    ]:
        if key in data:
            return data.get(key)

    # 有些脚本可能嵌套在 payload 里
    payload = data.get("payload")
    if isinstance(payload, dict):
        for key in [
            "temperatures",
            "matrix",
            "thermal_matrix",
            "data",
            "frame",
            "thermal",
        ]:
            if key in payload:
                return payload.get(key)

    return None


def _thermal_route_guard_check(values):
    flat = _thermal_route_guard_flatten(values)

    if len(flat) < THERMAL_MIN_VALID_POINTS:
        return False, f"not enough valid points: {len(flat)}"

    flat = flat[:768]

    bad = [
        v for v in flat
        if v < THERMAL_VALID_MIN or v > THERMAL_VALID_MAX
    ]

    if bad:
        return (
            False,
            f"temperature out of range: bad_count={len(bad)}, "
            f"min={min(flat):.2f}, max={max(flat):.2f}"
        )

    avg = sum(flat) / len(flat)

    if avg < THERMAL_REASONABLE_AVG_MIN or avg > THERMAL_REASONABLE_AVG_MAX:
        return False, f"avg out of range: avg={avg:.2f}"

    return True, f"ok: min={min(flat):.2f}, avg={avg:.2f}, max={max(flat):.2f}"


def _install_thermal_route_guard_v2():
    try:
        target_endpoint = None

        for rule in app.url_map.iter_rules():
            if rule.rule == "/api/temperature":
                target_endpoint = rule.endpoint
                break

        if not target_endpoint:
            print("THERMAL_FRAME_SANITY_ROUTE_GUARD_V2: /api/temperature route not found")
            return

        original_view = app.view_functions.get(target_endpoint)

        if original_view is None:
            print("THERMAL_FRAME_SANITY_ROUTE_GUARD_V2: endpoint view not found")
            return

        if getattr(original_view, "_thermal_guard_v2_wrapped", False):
            print("THERMAL_FRAME_SANITY_ROUTE_GUARD_V2 already installed")
            return

        def guarded_temperature_view(*args, **kwargs):
            data = request.get_json(silent=True) or {}
            values = _thermal_route_guard_pick_payload(data)

            ok, reason = _thermal_route_guard_check(values)

            if not ok:
                print("thermal sanity rejected: " + reason)
                return jsonify({
                    "success": False,
                    "stored": False,
                    "live": False,
                    "reason": "thermal sanity rejected: " + reason
                }), 400

            return original_view(*args, **kwargs)

        guarded_temperature_view.__name__ = getattr(original_view, "__name__", "guarded_temperature_view")
        guarded_temperature_view._thermal_guard_v2_wrapped = True

        app.view_functions[target_endpoint] = guarded_temperature_view
        print("THERMAL_FRAME_SANITY_ROUTE_GUARD_V2 loaded for endpoint: " + target_endpoint)

    except Exception as e:
        print("THERMAL_FRAME_SANITY_ROUTE_GUARD_V2 install failed: " + str(e))


_install_thermal_route_guard_v2()

# ================= END THERMAL_FRAME_SANITY_ROUTE_GUARD_V2 =================


if __name__ == '__main__':
    database.init_db()
    get_face_system()
    # 启动时初始化摄像头（如果可用）
    get_video_camera()

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)





