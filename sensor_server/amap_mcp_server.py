import argparse
import inspect
from typing import Any, Callable, Dict, Optional
import requests
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools.base import Tool

class AmapTool(Tool):
    @classmethod
    def from_function(cls, fn: Callable[..., Any], **kwargs) -> Tool:
        tool = super().from_function(fn, **kwargs)

        # 从函数docstring解析描述和参数
        docstring = inspect.getdoc(fn) or ""
        func_description = []
        param_descriptions = {}

        for line in docstring.split('\n'):
            line = line.strip()
            if line.startswith(':'):
                # 参数描述行
                parts = line.split(':')
                if len(parts) >= 3 and parts[1].strip() in tool.parameters['properties']:
                    param_name = parts[1].strip()
                    param_descriptions[param_name] = parts[2].strip()
            elif line and not line.startswith(':'):
                # 函数描述行
                func_description.append(line)

        # 设置清理后的函数描述
        tool.description = '\n'.join(func_description)

        # 简化参数schema
        simplified_props = {}
        for name, prop in tool.parameters['properties'].items():
            # 处理类型定义
            prop_type = 'string'
            if 'anyOf' in prop:
                prop_type = prop['anyOf'][0]['type']
            elif 'type' in prop:
                prop_type = prop['type']

            # 获取参数描述
            description = param_descriptions.get(name, '')
            if not description and name in tool.fn_metadata.arg_model.model_fields:
                description = tool.fn_metadata.arg_model.model_fields[name].description or ''

            simplified_props[name] = {
                'type': prop_type,
                'description': description
            }

        tool.parameters = {
            'type': 'object',
            'properties': simplified_props,
            'required': tool.parameters['required']
        }
        return tool

mcp = FastMCP("amap-maps")

def amap_tool(*args, **kwargs):
    def decorator(fn):
        tool = AmapTool.from_function(fn, *args, **kwargs)
        if hasattr(tool.fn_metadata, 'output_schema'):
            tool.fn_metadata.output_schema = None
        mcp._tool_manager._tools[tool.name] = tool
        return fn
    return decorator

def get_api_key() -> str:
    """Get the Amap Maps API key from environment variables"""
    api_key = "f958afc6d5a1d92710122ce9b0a2ddcd"
    return api_key

AMAP_MAPS_API_KEY = get_api_key()

@amap_tool()
def maps_geo(address: str, city: Optional[str] = None) -> Dict[str, Any]:
    """将详细的结构化地址转换为经纬度坐标。支持对地标性名胜景区、建筑物名称解析为经纬度坐标
    :address: 待解析的结构化地址信息
    :city: 指定查询的城市
    """
    try:
        params = {
            "key": AMAP_MAPS_API_KEY,
            "address": address,
            "output": "json"
        }
        if city:
            params["city"] = city

        response = requests.get(
            "https://restapi.amap.com/v3/geocode/geo",
            params=params,
            verify=False
        )

        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"results": []}

        if not data.get("geocodes"):
            return {"results": []}

        results = []
        for geo in data["geocodes"]:
            result = {
                "country": geo.get("country", "中国"),
                "province": geo.get("province", ""),
                "city": geo.get("city", ""),
                "citycode": geo.get("citycode", ""),
                "district": geo.get("district", ""),
                "street": geo.get("street", []),
                "number": geo.get("number", []),
                "adcode": geo.get("adcode", ""),
                "location": geo.get("location", ""),
                "level": geo.get("level", "公交地铁站点")
            }
            results.append(result)

        return {"results": results}
    except requests.exceptions.RequestException as e:
        return {"results": []}

@amap_tool()
def maps_regeo(location: str, radius: str = "1000") -> Dict[str, Any]:
    """将经纬度信息转换为地理位置
    :location: 经纬度坐标，经度在前，纬度在后，经度和纬度用","分割
    :radius: 搜索半径(米)
    """
    try:
        params = {
            "key": AMAP_MAPS_API_KEY,
            "location": location,
            "radius": radius,
            "extensions": "all",
            "output": "json"
        }

        response = requests.get(
            "https://restapi.amap.com/v3/geocode/regeo",
            params=params,
            verify=False
        )

        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"results": []}

        if not data.get("regeocode"):
            return {"results": []}

        regeocode = data["regeocode"]
        address_component = regeocode.get("addressComponent", {})

        # 获取第一个AOI的address信息
        aois = regeocode.get("aois", [])
        aoi_name = ""
        aoi_distance = ""
        aoi_type = ""
        aoi_loc = ""
        if aois and len(aois) > 0:
            aoi_name = aois[0].get("name", "")
            aoi_distance = aois[0].get("distance", "")
            aoi_type = aois[0].get("type", "")
            aoi_loc = aois[0].get("location", "")

        # 获取第一个POI的address信息
        pois = regeocode.get("pois", [])
        poi_address = ""
        poi_name = ""
        poi_distance = ""
        poi_type = ""
        poi_loc = ""
        if pois and len(pois) > 0:
            poi_address = pois[0].get("address", "")
            poi_name = pois[0].get("name", "")
            poi_distance = pois[0].get("distance", "")
            poi_type = pois[0].get("type", "")
            poi_loc = pois[0].get("location", "")

        result = {
            "country": address_component.get("country", "中国"),
            "province": address_component.get("province", ""),
            "city": address_component.get("city", ""),
            # "citycode": address_component.get("citycode", ""),
            "district": address_component.get("district", ""),
            # "adcode": address_component.get("adcode", ""),
            "township": address_component.get("township", ""),
            # "towncode": address_component.get("towncode", ""),
            "neighborhood_name": address_component.get("neighborhood", {}).get("name", ""),
            "neighborhood_type": address_component.get("neighborhood", {}).get("type", ""),
            "building_name": address_component.get("building", {}).get("name", ""),
            "building_type": address_component.get("building", {}).get("type", ""),
            "street": address_component.get("streetNumber", {}).get("street", ""),
            "number": address_component.get("streetNumber", {}).get("number", ""),
            "distance": address_component.get("streetNumber", {}).get("distance", ""),
            "location": address_component.get("streetNumber", {}).get("location", ""),
            "ori_location": location,
            "poi_address": poi_address,
            "poi_name":poi_name,
            "poi_distance":poi_distance,
            "poi_type":poi_type,
            "poi_loc":poi_loc,
			"aoi_name": aoi_name,
            "aoi_distance":aoi_distance,
            "aoi_type":aoi_type,
            "aoi_loc":aoi_loc
        }

        return {"results": [result]}
    except requests.exceptions.RequestException as e:
        return {"results": []}


@amap_tool()
def maps_around_search(location: str, radius: str = "1000", keywords: str = "") -> Dict[str, Any]:
    """周边搜，根据用户传入关键词以及坐标location，搜索出radius半径范围的POI
    :location: 中心点经度纬度，经度在前，纬度在后，经度和纬度用","分割
    :radius: 搜索半径(米)
    :keywords: 搜索关键词
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/place/around",
            params={
                "key": AMAP_MAPS_API_KEY,
                "location": location,
                "radius": radius,
                "keywords": keywords,
                "sortrule": "weight"
            },
            verify=False
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"pois": []}

        pois = []
        for poi in data.get("pois", []):
            biz_ext = poi.get("biz_ext", {})
            photos = poi.get("photos", [])

            # 处理营业时间 - 优先opentime2，其次open_time
            open_time = biz_ext.get("opentime2", biz_ext.get("open_time", ""))

            pois.append({
                "id": poi.get("id", ""),
                "name": poi.get("name", ""),
                "address": poi.get("address", ""),
                "typecode": poi.get("typecode", ""),
                "photo": photos[0]["url"] if photos else "",
                "phone": poi.get("tel", ""),
                "cost": biz_ext.get("cost", ""),
                "rating": biz_ext.get("rating", ""),
                "open_time": open_time,
                "distance": poi.get("distance", "")
            })

        return {"pois": pois}
    except requests.exceptions.RequestException as e:
        return {"pois": []}

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Amap MCP Server")
    parser.add_argument('transport', nargs='?', default='sse', choices=['stdio', 'sse', 'streamable-http'],
                        help='Transport type (stdio, sse, or streamable-http)')
    args = parser.parse_args()

    # Run the MCP server with the specified transport
    mcp.run(transport=args.transport)
