# logger.py 全局 logger 工厂，供所有模块统一导入使用
import logging
logging.basicConfig(level=logging.DEBUG, handlers=[])

import os
import sys
# from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "kuzu_demo.log"
MAX_BYTES = 10 * 1024 * 1024   # 10 MB
BACKUP_COUNT = 5

def get_logger(name: str) -> logging.Logger:
    try:
        LOG_FILE.touch(exist_ok=True)
    except Exception as e:
        sys.stderr.write(f"无法创建日志文件 {LOG_FILE} : {e}\n")

    logger = logging.getLogger(name)
    logger.propagate = False   # <-- 关键：不向上找 root ！！！！！！！！！ 解决部分级别日记不输出、不写入log文件的问题
    # print(f"[logger.py] get_logger called: {name}, handlers={logger.handlers}")
    # print(f" logger : {logger}, name : {name} ")
    if logger.hasHandlers():          # 避免重复挂载
        # print(f"[logger.py] 已有handler")
        return logger

    # print(f"[logger.py] 导入: {name}")

    logger.setLevel(logging.DEBUG)
    

    # 1. 控制台带颜色
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    # 关键变更：增加 %(lineno)d 显示行号
    fmt_c = "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    try:
        from colorlog import ColoredFormatter
        formatter_c = ColoredFormatter(fmt_c, log_colors={
            "DEBUG": "cyan", "INFO": "green", "WARNING": "yellow",
            "ERROR": "red", "CRITICAL": "bold_red"})
    except ModuleNotFoundError:       # 没装 colorlog 就退化为普通格式
        formatter_c = logging.Formatter(fmt_c.replace("%(log_color)s", ""))
    console.setFormatter(formatter_c)

    # 2. 文件按天滚动
    # file_handler = TimedRotatingFileHandler(
    #     LOG_FILE, when="midnight", backupCount=7, encoding="utf8")
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, encoding="utf-8", maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    # 文件日志也带行号
    fmt_f = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    file_handler.setFormatter(logging.Formatter(fmt_f))
    # 关键：把内部 stream 换成行缓冲
    # file_handler.stream = open(file_handler.baseFilename, mode='a', encoding='utf-8', buffering=1)

    # print(f"[logger.py] 导入file_handler: {file_handler}")

    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    logger.critical("日志系统初始化完成，日志将写入 %s", LOG_FILE.resolve())
    return logger