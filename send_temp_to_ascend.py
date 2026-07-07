import serial
import requests
import time
import re
import threading
from serial.tools import list_ports

PREFERRED_SERIAL_PORT = "COM5"
BAUD_RATE = 115200

ASCEND_URL = "http://10.148.111.242:5000/api/temperature"

COLS = 32
ROWS = 24
POINTS_PER_FRAME = COLS * ROWS

SEND_INTERVAL = 0.25
HTTP_TIMEOUT = 0.3

latest_frame = None
latest_frame_id = 0
latest_frame_time = 0.0
frame_lock = threading.Lock()


def find_serial_port():
    ports = list(list_ports.comports())

    print("当前检测到的串口：")
    if not ports:
        print("没有检测到任何串口")
        return None

    for p in ports:
        print(f"  {p.device} - {p.description}")

    available = [p.device for p in ports]

    if PREFERRED_SERIAL_PORT in available:
        print(f"使用指定串口: {PREFERRED_SERIAL_PORT}")
        return PREFERRED_SERIAL_PORT

    selected = available[0]
    print(f"指定串口 {PREFERRED_SERIAL_PORT} 不存在，自动使用: {selected}")
    return selected


def extract_numbers(line):
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", line)]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, p):
    if not values:
        return 0.0

    data = sorted(values)
    if len(data) == 1:
        return data[0]

    k = (len(data) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(data) - 1)

    if f == c:
        return data[f]

    return data[f] + (data[c] - data[f]) * (k - f)


def median(values):
    return percentile(values, 50)


def serial_reader(ser):
    global latest_frame, latest_frame_id, latest_frame_time

    frame_buffer = []

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                continue

            nums = extract_numbers(line)

            if len(nums) < COLS:
                continue

            frame_buffer.extend(nums[:COLS])

            # 如果缓存过多，说明旧数据积压，直接只保留最后一帧
            if len(frame_buffer) > POINTS_PER_FRAME * 2:
                frame_buffer = frame_buffer[-POINTS_PER_FRAME:]

            if len(frame_buffer) >= POINTS_PER_FRAME:
                frame = frame_buffer[:POINTS_PER_FRAME]
                frame_buffer = frame_buffer[POINTS_PER_FRAME:]

                with frame_lock:
                    latest_frame = frame
                    latest_frame_id += 1
                    latest_frame_time = time.time()

                print(
                    f"更新最新温度帧 #{latest_frame_id}: "
                    f"min={min(frame):.2f}, "
                    f"avg={mean(frame):.2f}, "
                    f"max={max(frame):.2f}, "
                    f"p90={percentile(frame, 90):.2f}"
                )

        except Exception as e:
            print("串口读取错误:", e)
            time.sleep(0.1)


def send_frame(frame, frame_id, frame_time, ser):
    now = time.time()

    payload = {
        "rows": ROWS,
        "cols": COLS,
        "points": len(frame),
        "frame_id": frame_id,
        "client_timestamp": frame_time,
        "client_age_ms": int((now - frame_time) * 1000),
        "temperatures": [round(float(x), 2) for x in frame],
        "min_temperature": round(min(frame), 2),
        "max_temperature": round(max(frame), 2),
        "avg_temperature": round(mean(frame), 2),
        "ambient": round(median(frame), 2),
        "p90_temperature": round(percentile(frame, 90), 2),
        "p95_temperature": round(percentile(frame, 95), 2),
    }

    try:
        r = requests.post(
            ASCEND_URL,
            json=payload,
            timeout=HTTP_TIMEOUT
        )

        if r.ok:
            data = r.json()
            thermal = data.get("thermal", {})
            print(
                f"发送最新帧 #{frame_id} 成功: "
                f"age={payload['client_age_ms']}ms, "
                f"live={data.get('allow_face_recognition')}, "
                f"reason={thermal.get('reason')}"
            )
        else:
            print(f"发送失败: HTTP {r.status_code}, {r.text}")

    except requests.exceptions.Timeout:
        print("发送超时，丢弃本帧，继续发送最新帧")

    except requests.exceptions.ConnectionError as e:
        print("连接昇腾板子失败，请检查 IP、网络、Flask 是否启动")
        print("错误:", e)

    except Exception as e:
        print("发送异常:", e)

    # 发送期间串口可能堆积旧帧，这里清空旧数据，保证下一帧尽量新
    try:
        ser.reset_input_buffer()
    except Exception:
        pass


def sender_loop(ser):
    last_sent_id = 0

    while True:
        try:
            with frame_lock:
                frame = latest_frame[:] if latest_frame is not None else None
                frame_id = latest_frame_id
                frame_time = latest_frame_time

            if frame is not None and frame_id != last_sent_id:
                last_sent_id = frame_id
                send_frame(frame, frame_id, frame_time, ser)

            time.sleep(SEND_INTERVAL)

        except Exception as e:
            print("发送线程错误:", e)
            time.sleep(0.2)


def main():
    serial_port = find_serial_port()

    if serial_port is None:
        print("没有可用串口，退出")
        return

    ser = serial.Serial(serial_port, BAUD_RATE, timeout=0.2)

    ser.reset_input_buffer()

    print("开始读取串口:", serial_port, BAUD_RATE)
    print("发送目标:", ASCEND_URL)
    print("模式: 只发送最新完整温度帧，自动丢弃旧帧")

    reader_thread = threading.Thread(target=serial_reader, args=(ser,), daemon=True)
    sender_thread = threading.Thread(target=sender_loop, args=(ser,), daemon=True)

    reader_thread.start()
    sender_thread.start()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()