import requests
import os
from datapallet.enums import SceneType

SCENE_TYPE_MAPPING = {
    "conference_room": SceneType.MEETINGROOM,
    "conference_center": SceneType.MEETINGROOM,
    "lecture_room": SceneType.MEETINGROOM,
    # 工位办公 (适配细粒度分类结果：对应 labels.txt 中的 office, office_cubicles, computer_room)
    "office": SceneType.WORKSPACE,
    "office_cubicles": SceneType.WORKSPACE,
    "computer_room": SceneType.WORKSPACE,
    "cubicle/office": SceneType.WORKSPACE,
    "home_office": SceneType.WORKSPACE,

    # 餐厅就餐 (适配细粒度分类结果：对应 labels.txt 中的 cafeteria, food_court, dining_hall)
    "cafeteria": SceneType.DINING,
    "food_court": SceneType.DINING,
    "dining_hall": SceneType.DINING,
    "restaurant": SceneType.DINING,
    "dining_room": SceneType.DINING,
    "kitchen": SceneType.DINING,
    "pantry": SceneType.DINING,

    # 室外园区散步 (适配细粒度分类结果：对应 labels.txt 中的 park, campus, garden)
    "park": SceneType.OUTDOOR_PARK,
    "campus": SceneType.OUTDOOR_PARK,
    "botanical_garden": SceneType.OUTDOOR_PARK,
    "garden": SceneType.OUTDOOR_PARK,
    "promenade": SceneType.OUTDOOR_PARK,
    "street": SceneType.OUTDOOR_PARK,
    "crosswalk": SceneType.OUTDOOR_PARK,

    # 地铁站 (适配细粒度分类结果：对应 labels.txt 中的 subway_station/platform)
    "platform": SceneType.SUBWAY_STATION,
}
def predict_scene(image_path, server_url="http://127.0.0.1:8200/predict/"):
    """
    向场景分类 API 发送图片并获取预测结果。

    :param image_path: 要进行分类的图片路径。
    :param server_url: FastAPI 服务器的 URL。
    """
    if not os.path.exists(image_path):
        print(f"错误：找不到文件 {image_path}")
        return

    print(f"正在向服务器发送图片: {image_path} ...")

    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/png')}

        try:
            response = requests.post(server_url, files=files)

            # 检查响应状态码
            if response.status_code == 200:
                print("成功接收到服务器响应！")
                print("-" * 30)
                result = response.json()
                print("预测消息:", result.get("message"))
                print("服务器保存路径:", result.get("result_image_path"))
                print("\nTop 1 预测详情:")
                if result.get("predictions"):
                    label = result["predictions"][0]['label']
                    print(label)
                    mapped_label = SCENE_TYPE_MAPPING.get(label, SceneType.NULL)
                    return mapped_label
            else:
                print(f"请求失败，状态码: {response.status_code}")
                print("服务器返回信息:", response.text)
                return None

        except requests.exceptions.RequestException as e:
            print(f"请求过程中发生错误: {e}")


if __name__ == "__main__":
    # 使用相对路径定位本目录下的png文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_path = os.path.join(script_dir, "scene_images", "meeting_room.png")
    
    # 检查文件是否存在
    if not os.path.exists(test_image_path):
        print(f"错误：找不到测试图片文件 {test_image_path}")
        print("请确保 scene_images/meeting_room.png 文件存在")
        exit(1)
    
    print(f"使用测试图片: {test_image_path}")
    scene_type = predict_scene(test_image_path)