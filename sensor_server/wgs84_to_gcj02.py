"""
功能：将WGS-84坐标系（GPS标准）转换为GCJ-02坐标系（中国火星坐标系）
"""

import math
import json
from typing import List, Tuple
 
class CoordinatorTransformer:
    """
    坐标转换类，实现WGS-84到GCJ-02的转换
    """
    
    # 定义常量
    EARTH_RADIUS = 6378245.0  # 地球长轴半径
    ECCENTRICITY_SQ = 0.00669342162296594323  # 偏心率平方
    PI = 3.1415926535897932384626  # 圆周率
    
    # 中国区域边界（粗略定义）
    CHINA_RECT = {
        'min_lng': 72.004,
        'max_lng': 137.8347,
        'min_lat': 0.8293,
        'max_lat': 55.8271
    }
    
    def __init__(self):
        """初始化坐标转换器"""
        pass
    
    def is_in_china(self, lng: float, lat: float) -> bool:
        """
        判断坐标是否在中国范围内
        
        Args:
            lng: 经度
            lat: 纬度
            
        Returns:
            bool: 是否在中国范围内
        """
        return (self.CHINA_RECT['min_lng'] <= lng <= self.CHINA_RECT['max_lng'] and
                self.CHINA_RECT['min_lat'] <= lat <= self.CHINA_RECT['max_lat'])
    
    def transform_lat(self, x: float, y: float) -> float:
        """
        纬度转换公式
        
        Args:
            x: 经度偏移量
            y: 纬度偏移量
            
        Returns:
            float: 纬度偏移值
        """
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * self.PI) + 20.0 * math.sin(2.0 * x * self.PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * self.PI) + 40.0 * math.sin(y / 3.0 * self.PI)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * self.PI) + 320 * math.sin(y * self.PI / 30.0)) * 2.0 / 3.0
        return ret
    
    def transform_lng(self, x: float, y: float) -> float:
        """
        经度转换公式
        
        Args:
            x: 经度偏移量
            y: 纬度偏移量
            
        Returns:
            float: 经度偏移值
        """
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * self.PI) + 20.0 * math.sin(2.0 * x * self.PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * self.PI) + 40.0 * math.sin(x / 3.0 * self.PI)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * self.PI) + 300.0 * math.sin(x / 30.0 * self.PI)) * 2.0 / 3.0
        return ret
    
    def wgs84_to_gcj02(self, lng: float, lat: float) -> Tuple[float, float]:
        """
        WGS-84坐标转换为GCJ-02坐标
        
        Args:
            lng: WGS-84经度
            lat: WGS-84纬度
            
        Returns:
            Tuple[float, float]: GCJ-02经纬度
        """
        # 如果不在中国范围内，则不进行转换
        if not self.is_in_china(lng, lat):
            return lng, lat
        
        # 计算偏移量
        d_lat = self.transform_lat(lng - 105.0, lat - 35.0)
        d_lng = self.transform_lng(lng - 105.0, lat - 35.0)
        
        # 计算弧度纬度
        rad_lat = lat / 180.0 * self.PI
        magic = math.sin(rad_lat)
        magic = 1 - self.ECCENTRICITY_SQ * magic * magic
        sqrt_magic = math.sqrt(magic)
        
        # 计算最终偏移量
        d_lat = (d_lat * 180.0) / ((self.EARTH_RADIUS * (1 - self.ECCENTRICITY_SQ)) / (magic * sqrt_magic) * self.PI)
        d_lng = (d_lng * 180.0) / (self.EARTH_RADIUS / sqrt_magic * math.cos(rad_lat) * self.PI)
        
        # 应用偏移量
        gcj_lat = lat + d_lat
        gcj_lng = lng + d_lng
        
        return gcj_lng, gcj_lat
        
    def get_accuracy_estimate(self, lng: float, lat: float) -> float:
        """
        获取精度估计（米）
        
        注意：这只是基于经验的粗略估计
        实际精度可能因地区而异
        
        Args:
            lng: 经度
            lat: 纬度
            
        Returns:
            float: 估计误差（米）
        """
        # 基于经验的精度估计模型
        # 沿海地区精度较高，内陆地区精度较低
        if not self.is_in_china(lng, lat):
            return 0.0
        
        # 简单的地理位置精度估计
        base_accuracy = 5.0  # 基础误差5米
        
        # 东部沿海地区精度较高
        if lng > 115.0 and lat > 25.0 and lat < 35.0:
            return base_accuracy + 2.0
        
        # 西部内陆地区精度较低
        if lng < 100.0:
            return base_accuracy + 15.0
        
        return base_accuracy + 8.0
    def estimate_offset_distance(self, lng: float, lat: float) -> float:
        """
        估计WGS-84到GCJ-02的大致偏移距离（米）
        
        Args:
            lng: 经度
            lat: 纬度
            
        Returns:
            float: 估计的偏移距离（米）
        """
        if not self.is_in_china(lng, lat):
            return 0.0
        
        # 计算偏移量
        d_lat = self.transform_lat(lng - 105.0, lat - 35.0)
        d_lng = self.transform_lng(lng - 105.0, lat - 35.0)
        
        rad_lat = lat / 180.0 * self.PI
        magic = math.sin(rad_lat)
        magic = 1 - self.ECCENTRICITY_SQ * magic * magic
        sqrt_magic = math.sqrt(magic)
        
        d_lat = (d_lat * 180.0) / ((self.EARTH_RADIUS * (1 - self.ECCENTRICITY_SQ)) / 
                                  (magic * sqrt_magic) * self.PI)
        d_lng = (d_lng * 180.0) / (self.EARTH_RADIUS / sqrt_magic * math.cos(rad_lat) * self.PI)
        
        # 将经纬度偏移转换为米距离（近似）
        # 1度纬度约111公里，1度经度约111×cos(纬度)公里
        lat_distance = abs(d_lat) * 111000  # 纬度方向距离（米）
        lng_distance = abs(d_lng) * 111000 * math.cos(rad_lat)  # 经度方向距离（米）
        
        # 欧几里得距离
        return math.sqrt(lat_distance**2 + lng_distance**2)

    def transform_batch(self, coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        批量转换坐标
        
        Args:
            coordinates: WGS-84坐标列表，格式为[(lng1, lat1), (lng2, lat2), ...]
            
        Returns:
            List[Tuple[float, float]]: GCJ-02坐标列表
        """
        return [self.wgs84_to_gcj02(lng, lat) for lng, lat in coordinates]
    
    def transform_file(self, input_file: str, output_file: str, input_format: str = 'json'):
        """
        转换文件中的坐标数据
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            input_format: 输入文件格式，支持 'json', 'csv'
        """
        if input_format == 'json':
            self._transform_json_file(input_file, output_file)
        elif input_format == 'csv':
            self._transform_csv_file(input_file, output_file)
        else:
            raise ValueError(f"不支持的格式: {input_format}")
    
    def _transform_json_file(self, input_file: str, output_file: str):
        """转换JSON文件中的坐标"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 假设JSON结构为 {"coordinates": [[lng, lat], ...]}
        if 'coordinates' in data:
            data['coordinates'] = self.transform_batch(data['coordinates'])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _transform_csv_file(self, input_file: str, output_file: str):
        """转换CSV文件中的坐标"""
        import csv
        
        with open(input_file, 'r', encoding='utf-8') as f_in:
            reader = csv.reader(f_in)
            rows = list(reader)
        
        # 假设CSV格式为: lng,lat,其他列...
        for i, row in enumerate(rows):
            if i == 0:  # 跳过标题行
                continue
            if len(row) >= 2:
                try:
                    lng = float(row[0])
                    lat = float(row[1])
                    gcj_lng, gcj_lat = self.wgs84_to_gcj02(lng, lat)
                    row[0] = str(gcj_lng)
                    row[1] = str(gcj_lat)
                except ValueError:
                    # 如果无法转换为浮点数，跳过该行
                    continue
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerows(rows)
 
 
# 提供简单易用的函数接口
def wgs84_to_gcj02(lng: float, lat: float) -> Tuple[float, float]:
    """
    WGS-84坐标转换为GCJ-02坐标（简便函数）
    
    Args:
        lng: WGS-84经度
        lat: WGS-84纬度
        
    Returns:
        Tuple[float, float]: GCJ-02经纬度
    """
    transformer = CoordinatorTransformer()
    return transformer.wgs84_to_gcj02(lng, lat)
 
 
def transform_coordinates(coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    批量转换坐标（简便函数）
    
    Args:
        coordinates: WGS-84坐标列表，格式为[(lng1, lat1), (lng2, lat2), ...]
        
    Returns:
        List[Tuple[float, float]]: GCJ-02坐标列表
    """
    transformer = CoordinatorTransformer()
    return transformer.transform_batch(coordinates)
 
 
# 示例使用
if __name__ == "__main__": 
    # 单个坐标转换示例
    wgs_lng, wgs_lat = 120.97059875173, 31.073238071985543
    wgs84_to_gcj02(wgs_lng, wgs_lat)
