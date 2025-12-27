import requests
import os


def predict_scene(image_path, server_url="http://127.0.0.1:8000/predict/"):
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
                print("\nTop 5 预测详情:")
                if result.get("predictions"):
                    for pred in result["predictions"]:
                        print(f" - 标签: {pred['label']}, 置信度: {pred['probability']}")
                print("-" * 30)
            else:
                print(f"请求失败，状态码: {response.status_code}")
                print("服务器返回信息:", response.text)

        except requests.exceptions.RequestException as e:
            print(f"请求过程中发生错误: {e}")


if __name__ == "__main__":
    test_image_path = "/Users/cannkit/PycharmProjects/AO/sceneclassify/scene_images/meeting_room.png"
    predict_scene(test_image_path)