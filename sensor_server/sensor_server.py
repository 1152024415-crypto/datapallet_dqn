import json
import os
import threading
import time
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timedelta
from enum import Enum
from .mcp_client import MCPClientSession
import shutil
import queue
from .wgs84_to_gcj02 import wgs84_to_gcj02
import numpy as np
import pandas as pd
import librosa
from .bitmap_to_png import bitmap_to_png
import base64
from typing import List, Dict, Any, Optional, Tuple
import pathlib
# from models import SCENEPHOTO_DIR, PERSON_PICTURE_DIR
# from scene_summary import SceneDivisionProcessor
# from llm_client import LLMClient
from pathlib import Path
import struct
import urllib.parse
import glob
from datapallet.enums import LightIntensity, SoundIntensity, ActivityMode

summary_queue = queue.Queue()
# testbed = TestBed()
class ArEventType(Enum):
    EVENT_NONE = 0
    EVENT_ENTER = 1
    EVENT_EXIT = 2
    EVENT_BOTH = 3

SUMMARY_OUTPUT =   'summary_output.jsonl'               # 活动日志记录
SUMMARY_TRANSFER = 'summary_transfer.jsonl'             # 内嵌了活动日志的prompt messages，用于LLM 活动摘要生成
SUMMARY_OUTPUT_LLM = 'active_log_summaries.jsonl'       # LLM 活动摘要生成记录
ORI_SENSOR_DATA_FILE = 'origin_sensor_datas.jsonl'      # sensorhub原始传感器数据
BACKUP_DIR = 'backup_files'
DB_FILE = 'Kuzu_db.db'

THRESHOLD = 1.1

DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
AR_MAPPING = {
    "AR_ACTIVITY_VEHICLE": ActivityMode.CAR,
    "AR_ACTIVITY_RIDING": ActivityMode.RIDING,
    "AR_ACTIVITY_WALK_SLOW": ActivityMode.STROLLING,
    "AR_ACTIVITY_RUN_FAST": ActivityMode.RUNNING,
    "AR_ACTIVITY_STATIONARY": ActivityMode.STANDING,
    "AR_ACTIVITY_TILT": ActivityMode.TILT,
    "AR_ACTIVITY_END": ActivityMode.NULL,

    "AR_VE_BUS": ActivityMode.BUS,
    "AR_VE_CAR": ActivityMode.CAR,
    "AR_VE_METRO": ActivityMode.SUBWAY,
    "AR_VE_HIGH_SPEED_RAIL": ActivityMode.HIGH_SPEED_TRAIN,
    "AR_VE_AUTO": ActivityMode.CAR,
    "AR_VE_RAIL": ActivityMode.TRAIN,
    "AR_CLIMBING_MOUNT": ActivityMode.HIKING,
    "AR_FAST_WALK": ActivityMode.BRISK_WALKING,

    "AR_ACTIVITY_ELEVATOR": ActivityMode.ELEVATOR,
}

def delayed_call(delay_seconds):
    """装饰器：延迟指定秒数后调用函数"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer = threading.Timer(delay_seconds, func, args=args, kwargs=kwargs)
            timer.start()
            return timer  # 返回Timer对象以便后续控制
        return wrapper
    return decorator

# class SummaryUpdater:
#     """
#     负责把 http 收到的图片时间戳回填到 summary_output.jsonl 的最后一条记录里。
#     """

#     # ==================== 对外接口 ====================
#     @delayed_call(5)
#     def update_summary_output_record(self, image_ts_str: str, image_file_name: str, image_full_path: str, photoid: str) -> None:
#         """
#         主入口：
#         image_ts  : str 从 http header 解析出的时间戳
#         image_file_name  : str               仅文件名，写回 file_name 字段
#         image_full_path  : str               本地磁盘路径，用于读取 png 内容
#         """
#         if not os.path.exists(SUMMARY_OUTPUT):
#             return

#         if not os.path.exists(image_full_path):
#             print(f'{image_full_path}文件不存在（可能已被删除）, 跳过')
#             return

#         # 若为scenephoto则才更新到summary_output.jsonl活动日志文件中，person不需要
#         pid = query_person_by_photoid(photoid)
#         if pid is not None:
#             print(f"Enter update_summary_output_record image field : 为person id {pid}的人脸照片，跳过, {photoid}")
#             return

#         print(f"Enter update_summary_output_record image field, 场景照片 {photoid} {pid}")

#         # 1. 读文件，拿到所有记录
#         records = self._load_records()
#         if not records:
#             return

#         # 2. 只关心最后一条
#         last = records[-1]
#         last_dt = self._parse_dt(last.get("Date_time", ""))
#         if last_dt is None:
#             return  # 无法解析时间 → 跳过
        
#         image_ts = self._parse_dt(image_ts_str)

#         # 3. 时间差判断
#         delta = abs((image_ts - last_dt).total_seconds())
#         print(f"cur_image_ts - last_record_datetime delta : {delta}")
#         if delta <= 5 * 60:
#             # 4. 更新 image 字段并写回
#             # last["image"] = image_file_name
#             # 读 png 并编码
#             b64_data = self._png_to_base64(image_full_path)
#             last["image"] = {
#                 "file_name": image_file_name,
#                 "data": b64_data
#             }
#             self._save_records(records)
#             # 创建 ScenePhoto 节点
#             # 图片摘要处理 LLM ----- TODO
#             llm = LLMClient(model="qwen2.5vl:7b")

#             # 构造多模态消息
#             msgs = [
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "image_url",
#                             "image_url": {"url": f"data:image/png;base64,{b64_data}"},
#                         },
#                         {
#                             "type": "text",
#                             "text": (
#                                 "请用中文一句话总结这张图片："
#                                 "说明画面中的主要物体、场景、氛围等，不要输出任何多余解释或标点。"
#                             ),
#                         },
#                     ],
#                 }
#             ]
#             description = llm.chat(msgs, temperature=0.3).strip()
#             # description = description.removeprefix("<think>\n\n</think>\n\n")
#             create_scene_photo_node(image_file_name, description, image_full_path)

#     def update_summary_output_personid_record(self, new_ts_str: str, new_person_id: int) -> None:
#         """
#         主入口：
#         image_ts  : str 从 http header 解析出的时间戳
#         image_file_name  : str               仅文件名，写回 file_name 字段
#         image_full_path  : str               本地磁盘路径，用于读取 png 内容
#         """
#         if not os.path.exists(SUMMARY_OUTPUT):
#             return

#         print("Enter update_summary_output_personid_record field ")

#         # 1. 读文件，拿到所有记录
#         records = self._load_records()
#         if not records:
#             return

#         # 2. 只关心最后一条
#         last = records[-1]
#         last_dt = self._parse_dt(last.get("Date_time", ""))
#         if last_dt is None:
#             return  # 无法解析时间 → 跳过
        
#         new_ts = self._parse_dt(new_ts_str)

#         # 3. 时间差判断
#         delta = abs((new_ts - last_dt).total_seconds())
#         print(f"new_ts - last_record_datetime delta : {delta}")
#         if delta <= 5 * 60:
#             # 4. 更新 personID 列表
#             person_list: list = last.setdefault("personID", [])
#             if new_person_id not in person_list:
#                 person_list.append(new_person_id) 
#             self._save_records(records)

#     # ==================== 内部工具 ====================
#     @staticmethod
#     def _parse_dt(dt_str: str):
#         """
#         把 "1970-2-15 5:7:23" 之类字符串转成 datetime.datetime。
#         解析失败返回 None。
#         """
#         try:
#             return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
#         except ValueError:
#             return None

#     def _load_records(self) -> List[Dict[str, Any]]:
#         """
#         读取整个 jsonl，返回 list[dict]。
#         若文件为空或格式异常，返回空列表。
#         """
#         try:
#             with open(SUMMARY_OUTPUT, "r", encoding="utf-8") as f:
#                 return [json.loads(line) for line in f if line.strip()]
#         except (FileNotFoundError, json.JSONDecodeError):
#             return []

#     def _save_records(self, records: List[Dict[str, Any]]) -> None:
#         """
#         把修改后的 records 整体写回 jsonl。
#         """
#         tmp_path = SUMMARY_OUTPUT + ".tmp"
#         with open(tmp_path, "w", encoding="utf-8") as f:
#             for rec in records:
#                 f.write(json.dumps(rec, ensure_ascii=False) + "\n")
#         # 原子替换
#         os.replace(tmp_path, SUMMARY_OUTPUT)

#     @staticmethod
#     def _png_to_base64(png_path: str) -> str:
#         """
#         读取 png 文件并以 base64 编码返回字符串。
#         文件不存在返回空字符串。
#         """
#         if not os.path.exists(png_path):
#             return ""
#         try:
#             with open(png_path, "rb") as f:
#                 return base64.b64encode(f.read()).decode("utf-8")
#         except Exception:
#             return ""

def _rotate(filename, output_dir='backup_files'):
    """文件达到阈值时的滚动动作：重命名并移到 backup_files 目录"""
    # 1. 确保 output 目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 2. 构造备份文件名：原文件名_2025-08-01_15-42-03.jsonl
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base, ext = os.path.splitext(os.path.basename(filename))  # 只保留文件名
    backup_name = f"{base}_{ts}{ext}"

    # 3. 计算目标路径
    dest = os.path.join(output_dir, backup_name)

    # 4. 移动（重命名）文件
    shutil.move(filename, dest)          # 推荐，跨分区也能用
    # 或者 os.rename(filename, dest)    # 仅在同一文件系统下可用

def append_to_jsonl(filename, data):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def ambient_light_label(lux: float) -> str:
    """
    根据光照强度(Lux)返回语义标签
    参数:
        lux (float): 光照强度值(单位:Lux)
    返回:
        str: 语义标签
    """
    if lux < 5:
        return LightIntensity.EXTREMELY_DARK
    elif 5 <= lux < 50:
        return LightIntensity.DIM
    elif 50 <= lux < 300:
        return LightIntensity.MODERATE_BRIGHTNESS
    elif 300 <= lux < 1000:
        return LightIntensity.BRIGHT
    else:  # lux >= 1000
        return LightIntensity.HARSH_LIGHT

def ambient_sound_label_from_level(level: int) -> str:
    """
    根据声音强度(dBFS)返回语义标签
    参数:
        level: 噪声强度划分等级
    返回:
        str: 语义标签
    """
    if level < 2:
        return SoundIntensity.VERY_QUIET
    elif 2 <= level < 4:
        return SoundIntensity.SOFT_SOUND
    elif 4 <= level < 6:
        return SoundIntensity.NORMAL_SOUND
    elif 6 <= level < 8:
        return SoundIntensity.NOISY
    else:  # dba >= 80
        return SoundIntensity.VERY_NOISY

# ================= Person ================
# ---------- 内存缓存 ----------
CACHE_FILE = "cache.jsonl"          # 每行一条 JSON  [personid, photoid, vec_list]
cached: List[Tuple[int, str, np.ndarray]] = []

# ---------- 启动加载 ----------
def load_cache():
    global cached
    cached = []
    if not os.path.exists(CACHE_FILE):
        return
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            personid, photoid, vec_list = json.loads(line)
            cached.append((personid, photoid, np.array(vec_list, dtype=np.float16)))

load_cache()

# ---------- 追加单条 ----------
def append_cache_line(personid: int, photoid: str, vec: np.ndarray):
    # 1. 内存
    cached.append((personid, photoid, vec))
    # 2. 追加一行
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        json.dump((personid, photoid, vec.tolist()), f, ensure_ascii=False)
        f.write("\n")          # 换行分隔
    print(f"cache 插入新纪录: person {personid} photo {photoid} ")

# ---------- 工具 ----------
def u32_to_fp16_arr(u32_list: List[int]) -> np.ndarray:
    if len(u32_list) != 128:
        raise ValueError("faceid data length != 128")
    # print(f"u32_list : {u32_list}")
    fp16_bytes = b"".join(struct.pack("<i", u32) for u32 in u32_list)
    # print(f"fp16_bytes : {fp16_bytes}")
    arr = np.frombuffer(fp16_bytes, dtype=np.float16).astype(np.float32)
    # print(f"arr : {arr}")
    if arr.shape != (256,):
        raise ValueError("parsed vec dim != 256")
    return arr

def sq_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum((a - b) ** 2))

def find_match(vec: np.ndarray) -> Optional[Tuple[int, str]]:
    for personid, photoid, v in cached:
        print(f"cached: personid {personid}")
        if sq_dist(vec, v) < THRESHOLD:
            print(f"cache 匹配成功  personid {personid}")
            return personid, photoid
    print(f"cache 匹配失败")
    return None

def query_person(pid: int) -> Optional[dict]:
    conn = get_processor().conn          # 拿到 kuzu.Connection
    result = conn.execute(
        "MATCH (p:Person {id: $pid}) RETURN p.name AS n, p.description AS d",
        {"pid": pid},
    )
    if result.has_next():
        n, d = result.get_next()
        print(f"query_person 结果: pid: {pid} name: {n} desc: {d}")
        return {"name": n or "", "description": d or ""}
    return None

def query_person_by_photoid(photoid: str):
    conn = get_processor().conn          # 拿到 kuzu.Connection
    result = conn.execute(
        "MATCH (p:Person {photoid: $photoid}) RETURN p.id AS id",
        {"photoid": photoid},
    )
    if result.has_next():
        id = result.get_next()
        print(f"query_person_by_photoid 结果: photoid: {photoid} personid: {id} ")
        return id or None
    return None

def create_person_node(photoid: str, vec: np.ndarray) -> int:
    try:
        conn = get_processor().conn          # 拿到 kuzu.Connection
        print(f"创建Person节点: Connection {conn} ")
        result = conn.execute(
            """
            CREATE (p:Person {
                photoid: $photoid,
                photo: $photo,
                faceid: $vec
            })
            RETURN p.id
            """,
            {"photoid": photoid, "photo": f"{photoid}.png", "vec": vec.tolist()},
        )
        # SA 传输photoid IMG_20251128_173907.jpg.png   ts和sh的ts 格式对齐 近似匹配（时间差小于1s），
        # 获取到的png name 替换这里的photoid，最终体现到db中的photoid和photo，以及cache和summary output log
        # SA 图片上传 需要先于sensorhub faceid上报
        personid = result.get_next()[0]
    except Exception as e:
        print("创建 Person 节点失败")
        raise RuntimeError("DB insert failed") from e

    append_cache_line(personid, photoid, vec)   # 只追加一行
    print(f"创建Person节点成功: {personid} {photoid} ")
    
    return personid

# photoid 近似匹配
# SCAN_WINDOW = timedelta(seconds=5)   # 只扫最近 5 秒区间，可改

def parse_png_basename(ts_str: str):
    """把 png 文件名里的时间戳转成 datetime 对象"""
    date_part, time_part = ts_str.split('_')[1:3]          # ['20251128', '173907']
    return datetime.strptime(date_part + time_part, '%Y%m%d%H%M%S')

def scan_recent_files(center_ts: datetime, delta: timedelta):
    """
    扫描 pictures 目录下【文件名合法 & 文件修改时间落在区间】的 png
    返回 list[(datetime, basename)]，按时间升序
    """
    low, high = center_ts - delta, center_ts + delta
    results = []
    # 如需递归子目录，把 glob 改成 os.walk 或 glob.glob('**/*.png', recursive=True)
    for png in glob.glob(os.path.join(PERSON_PICTURE_DIR, 'IMG_*.png')):
        try:
            base = os.path.basename(png)[:-4]
            ts = parse_png_basename(base)

            if low <= ts <= high:
                results.append((ts, base))
        except Exception:
            continue
    results.sort(key=lambda x: x[0])
    return results

def find_closest_basename(client_ts: datetime, delta=timedelta(seconds=1)):
    """
    动态扫描，返回与 client_ts 相差 ≤delta 的 basename（无后缀）
    有多个取最早一个，没有返回 None
    """
    candidates = scan_recent_files(client_ts, delta)
    print(f'查找{PERSON_PICTURE_DIR}下匹配的png文件：{candidates}')
    return candidates[0][1] if candidates else None

# ---------- 创建 ScenePhoto 节点 ----------
def create_scene_photo_node(png_name: str, description: str, image_full_path: str) -> None:
    """
    为单张场景 PNG 创建/更新 ScenePhoto 节点。
    参数
    ----
    png_name : str
        形如 "xxx.png" 的文件名（仅文件名，不含路径）。
    """

    print(f"ScenePhoto：{png_name} {description}")

    conn = get_processor().conn          # 拿到 kuzu.Connection

    if not os.path.exists(image_full_path):
        print(f"{image_full_path} 不存在，跳过 ScenePhoto 节点创建")
        return

    try:
        # MERGE 保证幂等：节点存在就更新，不存在就创建
        conn.execute(
            "MERGE (n:ScenePhoto {name: $name}) "
            "SET n.description = $description",
            {"name": png_name, "description": description}
        )
    except Exception as e:
        print(f"创建 ScenePhoto 节点失败：{png_name}，{e}")
        return

    print(f"ScenePhoto 节点创建/更新完成：{png_name} {description}")

def parse_person_picture_data_full_png(self):
    # 2. 读取图片原始数据
    try:
        content_length = int(self.headers.get('Content-Length', 0))
        image_data = self.rfile.read(content_length)
        # print(content_length, image_data)
    except Exception as e:
        self.send_error(400, f'Failed to read image data: {e}')
        return

    # 3. 解析时间戳
    ts_header = self.headers.get('x-timestamp')
    if ts_header is None:
        self.send_error(400, 'Missing X-Timestamp header')
        return
    
    save_dir = PERSON_PICTURE_DIR          
    os.makedirs(save_dir, exist_ok=True)

    photoid = self.headers.get('photoid')
    # if photoid == -1:
    #     self.send_error(400, f'Invalid photoid')
    #     return

    filename = f'{photoid}.png'
    filepath = os.path.join(save_dir, filename)

    # 存储图片信息
    with open(filepath, 'wb') as f:
        f.write(image_data)


    updater = SummaryUpdater()
    updater.update_summary_output_record(ts_header, filename, filepath, photoid)

    # 5. 返回成功
    self.send_response(200)
    self.end_headers()
    self.wfile.write(b'Upload OK')

def parse_picture_data(self):
    # 获取type
    type = self.headers.get('type')
    if type == 'person':
        parse_person_picture_data_full_png(self)

def parse_noise_level_data(data):
    """
    从传入的传感器数据中解析环境光数据
    参数:
        data: 传感器数据
    返回:
        noise_level_str: 噪声强度语义标签
    """
    noise_level_str = "Unkown"
    for item in data:
        if "audio" not in item:
            continue
        if item["audio"] != "noise":
            continue
        noise_level = item["data"]
        if noise_level >= 0:
            noise_level_str = ambient_sound_label_from_level(noise_level)
        break
    return noise_level_str

def parse_als_data(data):
    """
    从传入的传感器数据中解析环境光数据
    参数:
        data: 传感器数据
    返回:
        als_level: 光照等级
    """
    als_level = "Unkown"
    for item in data:
        if "sensor" not in item:
            continue
        if item["sensor"] != "als":
            continue
        als_data = item["data"]
        als_flat = [x[0] for x in als_data]
        # als_mean = np.mean(als_flat) # 均值
        als_median = np.median(als_flat)  # 中位数
        als_level = ambient_light_label(als_median)
        break
    return als_level

def parse_activity_data(data):
    """
    从传入的传感器数据中解析活动类别数据
    参数:
        data: 传感器数据
    返回:
        dict类型数据, 包含date time和识别到活动类别数据
    """
    latest_time = None
    latest_ar = None
    latest_ar_data = None
    for sensor_data in data:
        if "ar" not in sensor_data:
            continue
        if sensor_data["data"] == 0:
            continue
        if latest_time is None:
            latest_time = datetime.strptime(sensor_data["timestamp"].strip(), DATE_TIME_FORMAT) # 预计传感器时间格式为%Y-%m-%d %H:%M:%S
            latest_ar = sensor_data["ar"].strip()
            latest_ar_data = sensor_data["data"]
            continue
        current_time = datetime.strptime(sensor_data["timestamp"].strip(), DATE_TIME_FORMAT)
        if current_time < latest_time:
            continue
        if current_time == latest_time:
            if sensor_data["data"] == ArEventType.EVENT_ENTER.value and latest_ar_data == ArEventType.EVENT_EXIT.value:
                latest_ar = sensor_data["ar"].strip()
                latest_ar_data = sensor_data["data"]
        else:
            latest_time = current_time
            latest_ar = sensor_data["ar"].strip()
            latest_ar_data = sensor_data["data"]
    if latest_ar not in AR_MAPPING:
        print(f"no suitable activity data in {data}")
        return {}
    return {
        "Date time": latest_time.strftime(DATE_TIME_FORMAT),
        "Activity type": AR_MAPPING[latest_ar]
    }

async def get_address_info(longitude, latitude):
    mcp_client_session = MCPClientSession("D:\myArxiv\datapallet\datapallet\sensor_server\servers_config.json")
    await mcp_client_session.start()

    all_tools = await mcp_client_session.get_all_tools()
    all_tools_desc = "".join([tool.format_for_llm() for tool in all_tools])
    # print(f"#### all_tools: {all_tools_desc}\n")

    location = longitude + "," + latitude
    amap_json_str = """{
    "tool": "maps_regeo",
    "arguments": {
        "location": "%s",
        "radius": "500"
    }
}""" % (location)
    status, result = await mcp_client_session.process_tool_call(amap_json_str)
    # print(f"####status= {status}, json_str= {result}")

    await mcp_client_session.stop()
    if status:
        return result
    return None

_last_result = {
    "Specific address": "",
    "Location type": "",
    "Longitude": 0.0,
    "Latitude": 0.0
}

def parse_gnss_data(data):
    """
    从传入的传感器数据中解析位置信息
    参数:
        data: 传感器数据
    返回:
        dict类型数据, 包含date time和识别到的位置数据
    """
    global _last_result

    longitude = None
    latitude = None
    for sensor_data in data:
        if "gnss" not in sensor_data:
            continue
        cur_gnss_data = sensor_data["data"]
        if len(cur_gnss_data) < 4:
            print(" GNSS data is None\n")
            print("GNSS data is None")
            return _last_result
        latitude_h = cur_gnss_data[0]
        latitude_l = cur_gnss_data[1]
        longitude_h = cur_gnss_data[2]
        longitude_l = cur_gnss_data[3]
        wgs_longitude = longitude_h * (2 ** -23) + longitude_l * (2 ** -37)
        wgs_latitude = latitude_h * (2 ** -23) + latitude_l * (2 ** -37)
        
        longitude, latitude = wgs84_to_gcj02(wgs_longitude, wgs_latitude)
        print(f"wgs_lon: {wgs_longitude}, wgs_lat: {wgs_latitude} gcj_lon: {longitude}, gcj_lat: {latitude}")
        rec = f"[{longitude},{latitude}],"
        append_to_jsonl("out_lg_gcj.txt", rec)
    if longitude is None or latitude is None:
        print("longitude or latitude is None")
        return _last_result
    results = None
    results = asyncio.run(get_address_info(str(longitude), str(latitude)))
    if results is None:
        print(f"Call gaode api error with longitude={longitude}, latitude={latitude}")
        return _last_result

    result = json.loads(results)
    data = result["results"][0]  # 通常只有一条记录     

    # 1. 构造 Specific address
    parts = [
        data.get("country", ""),
        data.get("province", ""),
        data.get("city", ""),
        data.get("district", ""),
        data.get("township", "")
    ]

    poi_name = data.get("poi_name", "")
    parts.append(f" {poi_name}")

    specific_address = "".join(filter(None, parts))

    # 2. 其他字段
    location_type = data.get("poi_type", "")
    location_type_last = location_type.split(";")[-1] if location_type else ""
    
    _last_result = { # 含义上可能不太对应，后续可以改下
        "Specific address": specific_address,
        "Location type": location_type_last,
        "Longitude": longitude,
        "Latitude": latitude
    }
    return _last_result

def parse_ts(ts: str) -> datetime:
    """把字符串时间转成 datetime 对象；1900-0-0 这种非法时间返回最小值"""
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return dt
    except ValueError:
        # 处理“1900-0-0 0:0:0”之类无法解析的字符串
        return datetime.min

def get_latest_time(data):
    latest = max(data, key=lambda d: parse_ts(d["timestamp"]))
    latest_ts = parse_ts(latest["timestamp"]) + timedelta(hours=8)
    return latest_ts.strftime("%Y-%m-%d %H:%M:%S")

def process_sensor_data(data):
    """
    根据不同传感器类型，解析对应数据
    """
    latest_ts = get_latest_time(data)  # 获取最新的时间戳

    # 初始化结果字典
    # result = {
    #     "Date time": latest_ts,
    #     "Activity type": None,
    #     "Specific address": None,
    #     "Location type": None,
    #     "Longitude": None,
    #     "Latitude": None,
    #     "ALS level": None,
    #     "Sound level": None
    # }
    result = {
        "timestamp": latest_ts,
        "data_id": None,
        "value": None,
    }

    # 遍历数据，根据字段类型进行处理
    for item in data:
        if "ar" in item:
            # 处理活动类型数据
            activity_data = parse_activity_data([item])
            if activity_data:
                result["data_id"] = "activity_mode"
                result["value"] = activity_data["Activity type"]
                return result
                # result["Activity type"] = activity_data["Activity type"]
        elif "gnss" in item:
            result["data_id"] = "Location"
            result["value"] = 1
            return result
            # 处理位置信息数据
            # gnss_data = parse_gnss_data([item])
            # gnss_data = {}
            # if gnss_data:
            #     result.update({
            #         "Specific address": 1,
            #         "Location type": 1,
            #         "Longitude": 1,
            #         "Latitude": 1
            #         # "Specific address": gnss_data["Specific address"],
            #         # "Location type": gnss_data["Location type"],
            #         # "Longitude": gnss_data["Longitude"],
            #         # "Latitude": gnss_data["Latitude"]
            #     })
        elif "audio" in item:
            # 处理声音数据
            sound_level = parse_noise_level_data([item])
            result["data_id"] = "Sound_Intensity"
            result["value"] = sound_level
            return result
            # result["Sound level"] = sound_level
        elif "sensor" in item:
            # 处理环境光数据
            if item["sensor"] == "als":
                als_level = parse_als_data([item])
                result["data_id"] = "Light_Intensity"
                result["value"] = als_level
                return result
                # result["ALS level"] = als_level
    

    # 检查是否所有关键字段都已填充
    if not any(result.values()):
        raise ValueError("No valid sensor data found")

    return result
# ====================== 记录活动日志，包含活动日志去重 ===================
def key(rec: dict) -> tuple:
    loc_info = rec.get("location information", {})
    # 构造一个新的 location 信息字典，排除 Longitude 和 Latitude
    filtered_loc = {k: v for k, v in loc_info.items() if k not in {"Longitude", "Latitude"}}
    return (
        filtered_loc,
        rec.get("Activity type"),
        rec.get('enviromental_parameter')
    )

begin_rec = False
first_rec_time = "2025-09-03 12:40:09"
def update_if_same(new_record: dict) -> bool:
    """
    如果新纪录与 summary_output.jsonl 中最后一条记录的
    location information 与 Activity type 完全一致，
    则把最后一条记录的 Date_time 更新为 new_record 中的时间，
    同时文件行数不变。
    按以下规则更新：
    1. 第一行永不变。
    2. 关键字段一旦变化，该变化区间的第一条记录也不更新。
    3. 只有同一 key 的连续区间 >=2 条时，才更新区间最后一条的时间戳。
    返回 True 表示已更新，False 表示未更新（内容不一致或文件为空）。
    """
    global begin_rec
    global first_rec_time

    new_key = key(new_record)

    # 1. 打开文件，读到最后一条记录
    try:
        with open(SUMMARY_OUTPUT, "r+b") as fp:
            # 找到最后一个换行符的位置
            fp.seek(0, os.SEEK_END)
            end = fp.tell()
            if end == 0:          # 空文件
                first_rec_time = new_record["Date_time"]
                return False

            # 从后往前找倒数第一个 '\n'（若文件只有一行，可能找不到）
            pos = end - 1
            while pos >= 0 and fp.read(1) != b'\n':
                pos -= 1
                if pos < 0 : # 文件中只有一行数据，不更新，始终保留首行数据
                    # break
                    return False
                fp.seek(pos)
            last_line_start = pos + 1

            # 读取最后一行并解析
            fp.seek(last_line_start)
            last_line = fp.readline()
            last_rec = json.loads(last_line.decode("utf-8"))

            # 2. 比对 key
            if key(last_rec) != new_key:
                begin_rec = True
                first_rec_time = new_record["Date_time"]
                return False
            
            # 若相隔超过10min，则无需更新原有记录，新增条目即可 --- TBD：有问题 需要调整  记录一个阶段内首条记录时间 而非last
            last_time = datetime.strptime(first_rec_time, "%Y-%m-%d %H:%M:%S")
            new_time = datetime.strptime(new_record["Date_time"], "%Y-%m-%d %H:%M:%S")
            print(f"update_if_same : last_time {last_time} new_time {new_time} first_rec_time {first_rec_time}")
            if new_time - last_time > timedelta(minutes=10):
                begin_rec = True
                first_rec_time = new_record["Date_time"]
                print(f"not update")
                print(f"距离该阶段首条记录时间超过10min, 不作更新，restart")
                return False

            if begin_rec:
                begin_rec = False
                return False

            # 3. 就地更新 Date_time
            last_rec["Date_time"] = new_record["Date_time"]
            last_rec["location information"]["Longitude"] = new_record["location information"]["Longitude"]
            last_rec["location information"]["Latitude"] = new_record["location information"]["Latitude"]
            new_line = json.dumps(last_rec, ensure_ascii=False) + "\n"

            # 如果新行比旧行长，直接追加会覆盖掉下一行，因此需要重写
            # 这里简单处理：把文件截断到 last_line_start，然后写入新行
            fp.seek(last_line_start)
            fp.truncate()
            fp.write(new_line.encode("utf-8"))
            return True

    except FileNotFoundError:
        # 文件不存在，无法更新
        return False

def save_sensor_data(sensor_data):
    """
    当前只有AR和Date time数据
    """
    record = {
        "Date_time": sensor_data["Date time"],
        "location information": {
            "Specific address": sensor_data["Specific address"],
            "Location type": sensor_data["Location type"],
            "Longitude":sensor_data["Longitude"],
            "Latitude":sensor_data["Latitude"]
        },
        "Activity type": sensor_data["Activity type"],
        "enviromental_parameter": {
            "light_intensity": sensor_data["ALS level"],
            "noise_level": sensor_data["Sound level"]
        },
        "image": {
            "file_name": "",
            "data": ""
        },
        "personID": []
    }
    # print("save_sensor_data 111")
    # updated = update_if_same(record)
    # if not updated:
    #     append_to_jsonl(SUMMARY_OUTPUT, record)
    # print("save_sensor_data 222")

def delete_photo(photoid: str) -> None:
    """删除指定 photoid 的 PNG 文件"""
    file_path = os.path.join(PERSON_PICTURE_DIR, f'{photoid}.png')

    if not os.path.exists(file_path):
        print(f'文件不存在：{file_path}')
        return

    try:
        # file_path.unlink()
        os.remove(file_path)
        print(f'已删除：{file_path}')
    except OSError as e:
        print(f'删除失败：{e}')

def process_person_data_from_sensorhub(self, data):
    latest_ts = get_latest_time(data)
    swings = {item.get("swing") for item in data}
    if swings != {"faceid", "photoid"}:
        self.send_error(400, "must contain both faceid and photoid")
        return
    face_item = next(i for i in data if i["swing"] == "faceid")
    photo_item = next(i for i in data if i["swing"] == "photoid")
    # photoid = photo_item["data"]
    client_dt = datetime.strptime(latest_ts, '%Y-%m-%d %H:%M:%S')
    photoid = find_closest_basename(client_dt)
    try:
        vec = u32_to_fp16_arr(face_item["data"])
    except Exception as e:
        self.send_error(400, f"faceid parse error: {e}")
        return

    # 比对
    person_id = None
    match = find_match(vec) # personid,photoid
    if match:
        matched_personid, matched_photoid = match
    else:
        matched_personid, matched_photoid = None, None
    if matched_personid is not None:
        print(f"匹配成功：matched_personid {matched_personid}")
        # 删除本次照片
        if photoid is not None:
            delete_photo(photoid)
        info = query_person(matched_personid) or {"name": "", "description": ""}
        resp = {
            "matched": True,
            "personid": matched_personid,
            "photoid": matched_photoid,
            "name": info["name"],
            "description": info["description"],
        }
        person_id = matched_personid
    else:
        print(f"匹配失败：photoid {photoid}, 新建Person节点")
        try:
            # photoid近似真值获取，具体原因在create_person_node内部描述
            personid = create_person_node(photoid, vec) 
        except RuntimeError as e:
            self.send_error(400, f"db insert failed: {e}")
            return

        resp = {
            "matched": False,
            "personid": personid,
            "photoid": photoid,
            "name": "",
            "description": "",
        }
        person_id = personid

    # 更新到summary_output.jsonl活动日志文件最后一条记录中
    updater = SummaryUpdater()
    updater.update_summary_output_personid_record(latest_ts, person_id)

    # 返回 JSON
    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.end_headers()
    self.wfile.write(json.dumps(resp).encode())


class SensorHTTPRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, code=200):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        
        # 解析headers 若为image则独立处理
        print(self.headers)
        print(f"HTTP headers : {self.headers}")
        content_type = self.headers.get('Content-Type', '')
        # if content_type.startswith('image/'):
        if content_type == 'application/jpeg':
            parse_picture_data(self)
            return
        
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data)
            # print(data) # 调整成save to file
            print(f"Sensor data : {data}")
            # append_to_jsonl(ORI_SENSOR_DATA_FILE,data)

            if isinstance(data, list):
                swings = {item.get("swing") for item in data}
                if not swings or swings <= {None, ""}:
                    processed = process_sensor_data(data)
                    # save_sensor_data(processed)

                    # # todo:处理sensor数据
                    # # 将处理后的数据上报给TestBed
                    # for item in data:
                    #     data_id = item.get("data_id")
                    #     value = item.get("value")
                    #     timestamp_str = item.get("timestamp")
                    #     if data_id and value and timestamp_str:
                    #         timestamp = datetime.strptime(timestamp_str, DATE_TIME_FORMAT)
                    #         testbed.receive_and_transmit_data(data_id, value, timestamp)

                    response = {'status': 'received', 'processed': 'ok'}
                    self._set_headers(200)
                # else:
                #     # 处理swing：faceid & photoid; 并更新到活动日志
                #     process_person_data_from_sensorhub(self, data)
                #     return
            else:
                response = {'status': 'received', 'processed': 'not json list'}
                self._set_headers(400)
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=SensorHTTPRequestHandler, port=8100):

    def run_http_server():
        server_address = ('', port)
        print(f"Server address: {server_address}")
        try:
            httpd = server_class(server_address, handler_class)
            print(f'Starting http server on port {port}...')
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt received, shutting down server...")
            finally:
                httpd.shutdown()
                httpd.server_close()
                print("Http server closed\n")
        except Exception as e:
            print(f"Error starting server: {e}")

    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    try:
        http_thread.join()  # 确保主线程等待线程完成
    except KeyboardInterrupt:
        print("\nMain thread KeyboardInterrupt received, shutting down...")
        http_thread.join()

if __name__ == '__main__':
    run()
