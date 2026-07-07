import cv2
import threading
import time
import random
import numpy as np
import database
import os


class VideoCamera(object):
    def __init__(self, face_system, get_temperature_func=None, temp_min=30.8, temp_max=38.5, camera_indices=None):
        self.get_temperature_func = get_temperature_func
        self.temp_min = temp_min
        self.temp_max = temp_max

        self.lock = threading.Lock()
        self.infer_lock = threading.Lock()

        self.face_system = face_system
        self.last_frame = None
        self.last_faces = []
        self.last_match_text = ""

        self.running = True
        self.video = None
        self.camera_index = None
        self.camera_indices = camera_indices or [2, 0, 1, 3]

        self.capture_sleep = 0.02
        self.check_interval = 0.45

        self.last_attendance = {}
        self.attendance_cooldown = 24 * 60 * 60

        # 测试阶段阈值，最终使用建议调回 0.42~0.48
        self.face_threshold = 0.39

        # 随机动作活体：blink / mouth
        self.challenge = None
        self.challenge_timeout = 8.0
        self.challenge_actions = ["blink"]

        # OpenCV 自带眼睛检测器
        self.eye_cascade = None
        try:
            eye_xml = cv2.data.haarcascades + "haarcascade_eye.xml"
            self.eye_cascade = cv2.CascadeClassifier(eye_xml)
            if self.eye_cascade.empty():
                self.eye_cascade = None
                print("Warning: eye cascade load failed")
        except Exception as e:
            self.eye_cascade = None
            print(f"Warning: eye cascade unavailable: {e}")

        self.open_camera()

        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.attendance_thread = threading.Thread(target=self.attendance_loop, daemon=True)

        self.capture_thread.start()
        self.attendance_thread.start()

    def open_camera(self):
        if self.video is not None:
            try:
                self.video.release()
            except Exception:
                pass
            self.video = None

        for idx in self.camera_indices:
            print(f"Trying camera /dev/video{idx} ...")
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)

            if not cap.isOpened():
                print(f"/dev/video{idx} open failed")
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok = False
            first_frame = None

            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    ok = True
                    first_frame = frame
                    break
                time.sleep(0.05)

            if ok:
                self.video = cap
                self.camera_index = idx
                with self.lock:
                    self.last_frame = first_frame.copy()
                    self.last_faces = []
                    self.last_match_text = "Camera ready"
                print(f"Camera opened successfully: /dev/video{idx}")
                return True

            print(f"/dev/video{idx} opened but cannot read frame")
            cap.release()

        print("Error: no usable camera found. Tried:", self.camera_indices)
        return False

    def is_camera_ready(self):
        return self.video is not None and self.video.isOpened()

    def release(self):
        self.running = False
        if self.video is not None:
            try:
                self.video.release()
            except Exception:
                pass
            self.video = None

    def __del__(self):
        self.release()

    def capture_loop(self):
        while self.running:
            try:
                if not self.is_camera_ready():
                    print("Camera is not ready, retrying open...")
                    self.open_camera()
                    time.sleep(0.5)
                    continue

                success, frame = self.video.read()

                if not success or frame is None or frame.size == 0:
                    print("Camera read failed, reopening camera...")
                    self.open_camera()
                    time.sleep(0.3)
                    continue

                with self.lock:
                    self.last_frame = frame.copy()

                time.sleep(self.capture_sleep)

            except Exception as e:
                print(f"capture_loop error: {e}")
                time.sleep(0.3)

    def attendance_loop(self):
        while self.running:
            try:
                with self.lock:
                    frame = self.last_frame.copy() if self.last_frame is not None else None

                if frame is not None:
                    self.process_attendance(frame)

                time.sleep(self.check_interval)

            except Exception as e:
                print(f"attendance_loop error: {e}")
                time.sleep(self.check_interval)

    def get_frame(self):
        with self.lock:
            if self.last_frame is None:
                return None

            frame = self.last_frame.copy()
            faces = list(self.last_faces)
            text = self.last_match_text

        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

        if text:
            cv2.putText(
                frame,
                text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 255),
                2
            )

        ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not ret:
            return None

        return jpeg.tobytes()

    def get_snapshot(self):
        with self.lock:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()

    def crop_face_with_margin(self, frame, box, margin_ratio=0.18):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]

        bw = x2 - x1
        bh = y2 - y1

        mx = int(bw * margin_ratio)
        my = int(bh * margin_ratio)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        return frame[y1:y2, x1:x2]

    def match_registered_user(self, face_img):
        emb = self.face_system.get_embedding(face_img)

        users = database.get_users()
        max_sim = -1.0
        best_match = None

        for u in users:
            if "embedding" not in u.keys():
                continue

            if u["embedding"] is None:
                continue

            db_emb = np.frombuffer(u["embedding"], dtype=np.float32)

            if db_emb.size != emb.size:
                print(
                    f"Skip user {u['id']}: embedding size mismatch, "
                    f"camera={emb.size}, db={db_emb.size}"
                )
                continue

            sim = np.dot(emb, db_emb) / (
                np.linalg.norm(emb) * np.linalg.norm(db_emb) + 1e-6
            )

            if sim > max_sim:
                max_sim = sim
                best_match = u

        return best_match, max_sim

    def get_thermal_result(self, face_box=None, frame_shape=None):
        if not self.get_temperature_func:
            return {
                "roi_live": False,
                "temperature": 0.0,
                "thermal_center": None,
                "reason": "no temperature callback"
            }

        try:
            result = self.get_temperature_func(face_box=face_box, frame_shape=frame_shape)
        except TypeError:
            result = self.get_temperature_func()
        except Exception as e:
            print(f"get thermal result error: {e}")
            return {
                "roi_live": False,
                "temperature": 0.0,
                "thermal_center": None,
                "reason": str(e)
            }

        if isinstance(result, dict):
            return result

        try:
            value = float(result)
        except Exception:
            value = 0.0

        return {
            "roi_live": value > 0.0,
            "temperature": value,
            "thermal_center": None,
            "reason": "legacy temperature callback"
        }

    def reset_challenge(self):
        self.challenge = None

    def ensure_challenge(self, user_id):
        now = time.time()

        if (
            self.challenge is None or
            self.challenge.get("user_id") != user_id or
            now - self.challenge.get("start_time", 0) > self.challenge_timeout
        ):
            action = random.choice(self.challenge_actions)

            if action == "blink":
                phase = "need_open"
                prompt = "Blink: open-close-open"
            else:
                phase = "collect_closed"
                prompt = "Mouth: close then open"

            self.challenge = {
                "user_id": user_id,
                "action": action,
                "phase": phase,
                "start_time": now,
                "prompt": prompt,
                "mouth_min_score": 999.0,
                "mouth_max_score": 0.0,
                "mouth_baseline": None,
                "mouth_collect_count": 0
            }

            print(f"New action challenge for user {user_id}: {prompt}")

        return self.challenge


    def detect_eyes_open(self, face_img):
        if self.eye_cascade is None or face_img is None or face_img.size == 0:
            return False

        h, w = face_img.shape[:2]
        upper = face_img[0:int(h * 0.55), :]

        if upper.size == 0:
            return False

        gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(18, 12)
        )

        # 检测到至少一只眼睛就认为当前为睁眼状态
        return len(eyes) >= 1

    def detect_mouth_score(self, face_img):
        if face_img is None or face_img.size == 0:
            return 0.0, {
                "dark_ratio": 0.0,
                "area_ratio": 0.0,
                "height_ratio": 0.0,
                "score": 0.0
            }

        h, w = face_img.shape[:2]

        # 嘴部区域：脸下半部分中央，范围放宽一点，适配不同脸型和角度
        y1 = int(h * 0.50)
        y2 = int(h * 0.92)
        x1 = int(w * 0.12)
        x2 = int(w * 0.88)

        mouth = face_img[y1:y2, x1:x2]
        if mouth.size == 0:
            return 0.0, {
                "dark_ratio": 0.0,
                "area_ratio": 0.0,
                "height_ratio": 0.0,
                "score": 0.0
            }

        gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 用 Otsu 自适应找嘴巴内部暗区，不再依赖固定阈值
        _, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 去掉小噪点
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        dark_ratio = float(np.sum(mask > 0)) / float(mask.size)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi_h, roi_w = gray.shape[:2]
        largest_area = 0.0
        largest_w = 0
        largest_h = 0

        for c in contours:
            area = float(cv2.contourArea(c))
            x, y, cw, ch = cv2.boundingRect(c)

            # 嘴巴张开一般是横向暗区，过滤太小或太窄的噪点
            if cw < roi_w * 0.12 or ch < roi_h * 0.08:
                continue

            if area > largest_area:
                largest_area = area
                largest_w = cw
                largest_h = ch

        area_ratio = largest_area / float(max(1, roi_w * roi_h))
        height_ratio = float(largest_h) / float(max(1, roi_h))
        width_ratio = float(largest_w) / float(max(1, roi_w))

        # 综合分数：张嘴时 dark_ratio、面积、高度都会明显增加
        score = (
            dark_ratio * 0.45 +
            area_ratio * 0.35 +
            height_ratio * 0.15 +
            width_ratio * 0.05
        )

        info = {
            "dark_ratio": dark_ratio,
            "area_ratio": area_ratio,
            "height_ratio": height_ratio,
            "width_ratio": width_ratio,
            "score": score
        }

        return score, info

    def detect_mouth_open(self, face_img):
        score, info = self.detect_mouth_score(face_img)

        # 绝对判定保留，但阈值放宽；真正挑战里还会使用动态变化判定
        open_by_score = score >= 0.105
        open_by_shape = (
            info["dark_ratio"] >= 0.075 and
            info["height_ratio"] >= 0.16 and
            info["area_ratio"] >= 0.018
        )

        return bool(open_by_score or open_by_shape)


    def check_action_challenge(self, user_id, face_img):
        challenge = self.ensure_challenge(user_id)
        action = challenge["action"]
        phase = challenge["phase"]

        eyes_open = self.detect_eyes_open(face_img)
        mouth_score, mouth_info = self.detect_mouth_score(face_img)
        mouth_open = self.detect_mouth_open(face_img)

        if action == "blink":
            if phase == "need_open":
                if eyes_open:
                    challenge["phase"] = "need_closed"
                    return False, "blink: now close eyes"
                return False, "blink: show open eyes first"

            if phase == "need_closed":
                if not eyes_open:
                    challenge["phase"] = "need_reopen"
                    return False, "blink: now open eyes"
                return False, "blink: waiting for closed eyes"

            if phase == "need_reopen":
                if eyes_open:
                    return True, "blink passed"
                return False, "blink: waiting for eyes reopen"

        if action == "mouth":
            # 第一步：采集闭嘴基准。闭嘴时 mouth_score 通常较低。
            if phase == "collect_closed":
                challenge["mouth_collect_count"] += 1
                challenge["mouth_min_score"] = min(challenge["mouth_min_score"], mouth_score)
                challenge["mouth_max_score"] = max(challenge["mouth_max_score"], mouth_score)

                # 采集 2 帧作为基准，避免单帧误差
                if challenge["mouth_collect_count"] >= 2:
                    challenge["mouth_baseline"] = challenge["mouth_min_score"]
                    challenge["phase"] = "need_open"
                    return False, (
                        f"mouth: now open mouth, "
                        f"baseline={challenge['mouth_baseline']:.3f}, "
                        f"score={mouth_score:.3f}"
                    )

                return False, f"mouth: keep mouth closed, score={mouth_score:.3f}"

            # 第二步：张嘴后分数相对闭嘴基准明显升高即可通过
            if phase == "need_open":
                baseline = challenge.get("mouth_baseline")
                if baseline is None:
                    baseline = challenge.get("mouth_min_score", mouth_score)

                diff = mouth_score - baseline

                # 动态变化阈值：嘴部暗区分数比闭嘴基准高 0.030 以上
                # 或绝对张嘴检测通过
                dynamic_ok = diff >= 0.030 and mouth_score >= 0.075
                absolute_ok = mouth_open

                if dynamic_ok or absolute_ok:
                    return True, (
                        f"mouth open passed, score={mouth_score:.3f}, "
                        f"baseline={baseline:.3f}, diff={diff:.3f}, "
                        f"dark={mouth_info['dark_ratio']:.3f}, "
                        f"area={mouth_info['area_ratio']:.3f}, "
                        f"height={mouth_info['height_ratio']:.3f}"
                    )

                return False, (
                    f"mouth: waiting for open, score={mouth_score:.3f}, "
                    f"baseline={baseline:.3f}, diff={diff:.3f}, "
                    f"dark={mouth_info['dark_ratio']:.3f}, "
                    f"area={mouth_info['area_ratio']:.3f}, "
                    f"height={mouth_info['height_ratio']:.3f}"
                )

        return False, "unknown action challenge"


    def process_attendance(self, frame):
        if self.face_system is None:
            with self.lock:
                self.last_match_text = "FaceSystem not ready"
            return

        with self.infer_lock:
            faces = self.face_system.detect(frame)

            if len(faces) == 0:
                self.reset_challenge()
                with self.lock:
                    self.last_faces = []
                    self.last_match_text = "No face"
                return

            best_face = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, best_face)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_w = x2 - x1
            face_h = y2 - y1

            with self.lock:
                self.last_faces = [(x1, y1, x2, y2)]

            if face_w < 90 or face_h < 90:
                self.reset_challenge()
                with self.lock:
                    self.last_match_text = "Face too small"
                print(f"Face too small: {face_w}x{face_h}, move closer")
                return

            face_img = self.crop_face_with_margin(frame, (x1, y1, x2, y2))

            if face_img.size == 0:
                self.reset_challenge()
                with self.lock:
                    self.last_match_text = "Empty face"
                print("Detected face crop is empty")
                return

            best_match, max_sim = self.match_registered_user(face_img)

        if not best_match or max_sim <= self.face_threshold:
            self.reset_challenge()
            with self.lock:
                self.last_match_text = f"No match {max_sim:.2f}"
            print(f"Face detected but no registered user matched, best similarity={max_sim:.2f}")
            return

        user_id = int(best_match["id"])
        user_name = best_match["name"]

        print(f"Face matched: user={user_name}, id={user_id}, similarity={max_sim:.2f}")

        now = time.time()
        last_time = self.last_attendance.get(user_id, 0)

        if now - last_time < self.attendance_cooldown:
            with self.lock:
                self.last_match_text = f"{user_name}: already checked"
            print(f"User {user_name} already checked in recently, skipped")
            return

        thermal_result = self.get_thermal_result(
            face_box=(x1, y1, x2, y2),
            frame_shape=frame.shape
        )

        if not thermal_result.get("roi_live"):
            self.reset_challenge()
            with self.lock:
                self.last_match_text = f"{user_name}: thermal fail"
            print(f"User {user_name} matched, but thermal ROI liveness failed: {thermal_result.get('reason')}")
            return

        action_ok, action_reason = self.check_action_challenge(user_id, face_img)

        if not action_ok:
            prompt = self.challenge.get("prompt", "action") if self.challenge else "action"
            with self.lock:
                self.last_match_text = f"{user_name}: {prompt}"
            print(f"User {user_name} matched, action challenge not passed: {action_reason}")
            return

        print(f"Action liveness passed: {action_reason}")

        temperature = float(thermal_result.get("temperature", 0.0))

        if temperature <= 0.0:
            print(f"User {user_name} matched, but temperature invalid after action")
            return

        if temperature <= self.temp_min or temperature >= self.temp_max:
            print(
                f"User {user_name} matched, but temperature rejected: "
                f"temperature={temperature:.2f}, allowed=({self.temp_min}, {self.temp_max})"
            )
            return

        os.makedirs("uploads", exist_ok=True)

        filename = f"attendance_{user_id}_{int(time.time())}.jpg"
        filepath = os.path.join("uploads", filename)
        cv2.imwrite(filepath, face_img)

        try:
            database.add_attendance(user_id, "camera_auto", filename, temperature)
        except TypeError:
            database.add_attendance(user_id, "camera_auto", filename)

        self.last_attendance[user_id] = now
        self.reset_challenge()

        with self.lock:
            self.last_match_text = f"{user_name}: success"

        print(
            f"Attendance success: user={user_name}, "
            f"similarity={max_sim:.2f}, temperature={temperature:.2f}, action={action_reason}"
        )








# ================= CLEAN_BLINK_LIVENESS_PATCH =================
# 说明：
# - 移动图片/移动手机不会直接通过活体
# - 必须同时满足：人脸 ROI 温度通过 + 眨眼变化通过
# - 这里采用较宽松的“眼睛状态变化”判断，避免一直无法考勤

import time as _clean_blink_time

def _clean_find_local(local_vars, *names):
    for name in names:
        if name in local_vars and local_vars[name] is not None:
            return local_vars[name]
    return None


def _clean_crop_face(frame, face_box):
    if frame is None or face_box is None:
        return None

    try:
        h, w = frame.shape[:2]
        vals = list(face_box)

        if len(vals) < 4:
            return None

        x1, y1, a, b = [int(v) for v in vals[:4]]

        if a > x1 and b > y1:
            x2, y2 = a, b
        else:
            x2, y2 = x1 + a, y1 + b

        pad_x = max(10, int((x2 - x1) * 0.20))
        pad_y = max(10, int((y2 - y1) * 0.20))

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    except Exception:
        return None


def _clean_eye_score(face_img):
    try:
        import cv2

        if face_img is None or getattr(face_img, "size", 0) == 0:
            return None

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        if h < 45 or w < 45:
            return None

        eye_roi = gray[0:int(h * 0.62), :]
        eye_roi = cv2.equalizeHist(eye_roi)

        cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        eye_cascade = cv2.CascadeClassifier(cascade_path)

        if eye_cascade.empty():
            print("活体检测：找不到 haarcascade_eye.xml")
            return None

        eyes = eye_cascade.detectMultiScale(
            eye_roi,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(10, 8)
        )

        eye_count_score = min(len(eyes), 2) / 2.0

        edges = cv2.Canny(eye_roi, 50, 120)
        edge_score = float((edges > 0).sum()) / float(edges.size)

        return eye_count_score * 0.75 + min(edge_score * 8.0, 1.0) * 0.25

    except Exception as e:
        print(f"活体检测：眼睛检测异常: {e}")
        return None


def _clean_blink_liveness_gate(self, user_id, local_vars):
    now = _clean_blink_time.time()

    frame = _clean_find_local(
        local_vars,
        "frame",
        "current_frame",
        "img",
        "image"
    )

    face_box = _clean_find_local(
        local_vars,
        "face_box",
        "box",
        "bbox",
        "rect"
    )

    face_img = _clean_find_local(
        local_vars,
        "face_img",
        "face_crop",
        "crop",
        "face_image"
    )

    if face_img is None:
        face_img = _clean_crop_face(frame, face_box)

    # 1. 温度 ROI 必须先通过
    try:
        try:
            from __main__ import get_live_temperature_for_camera
        except Exception:
            from app import get_live_temperature_for_camera
        frame_shape = frame.shape if frame is not None else None

        temp_info = get_live_temperature_for_camera(
            face_box=face_box,
            frame_shape=frame_shape
        )

        if not isinstance(temp_info, dict) or not temp_info.get("roi_live", False):
            print(f"活体失败：温度未通过 temp_info={temp_info}")
            return False

    except Exception as e:
        print(f"活体失败：温度检测异常: {e}")
        return False

    # 2. 眨眼变化检测
    score = _clean_eye_score(face_img)

    if not hasattr(self, "clean_blink_states"):
        self.clean_blink_states = {}

    st = self.clean_blink_states.get(user_id)

    if st is None or now - st.get("start", now) > 4.0:
        st = {
            "start": now,
            "min_score": 999.0,
            "max_score": -999.0,
            "last_print": 0,
        }
        self.clean_blink_states[user_id] = st
        print("活体挑战：请眨一下眼")
        return False

    if score is not None:
        st["min_score"] = min(st["min_score"], score)
        st["max_score"] = max(st["max_score"], score)

    diff = st["max_score"] - st["min_score"]

    # 这个阈值已经放低，避免一直过不了
    if diff >= 0.08:
        print(f"活体通过：检测到眨眼变化 diff={diff:.3f}")
        self.clean_blink_states.pop(user_id, None)
        return True

    if now - st.get("last_print", 0) > 0.8:
        print(
            f"活体等待眨眼：score={score}, "
            f"min={st['min_score']:.3f}, "
            f"max={st['max_score']:.3f}, "
            f"diff={diff:.3f}"
        )
        st["last_print"] = now

    return False


try:
    VideoCamera.blink_fix_liveness_gate = _clean_blink_liveness_gate
    print("CLEAN_BLINK_LIVENESS_PATCH loaded")
except Exception as e:
    print(f"CLEAN_BLINK_LIVENESS_PATCH load failed: {e}")

# ================= END CLEAN_BLINK_LIVENESS_PATCH =================





