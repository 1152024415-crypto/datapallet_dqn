import requests
import json

# 服务器地址和端口
server_url = "http://127.0.0.1:8100"

# 第一组数据
data1 = [
    {'ar': 'AR_ACTIVITY_VEHICLE', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0},
    {'ar': 'AR_ACTIVITY_RIDING', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0},
    {'ar': 'AR_ACTIVITY_WALK_SLOW', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0},
    {'ar': 'AR_ACTIVITY_RUN_FAST', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0},
    {'ar': 'AR_ACTIVITY_STATIONARY', 'duration': 0, 'timestamp': '2070-2-17 16:4:10', 'data': 1},
    {'ar': 'AR_ACTIVITY_TILT', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0},
    {'ar': 'AR_ACTIVITY_END', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0},
    {'ar': 'AR_VE_BUS', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0},
    {'ar': 'AR_VE_CAR', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0},
    {'ar': 'AR_VE_METRO', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0},
    {'ar': 'AR_VE_HIGH_SPEED_RAIL', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0},
    {'ar': 'AR_VE_AUTO', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0},
    {'ar': 'AR_VE_RAIL', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0},
    {'ar': 'AR_CLIMBING_MOUNT', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0},
    {'ar': 'AR_FAST_WALK', 'duration': 0, 'timestamp': '2000-0-0 0:0:0', 'data': 0}
]

# 第二组数据
data2 = [
    {'gnss': 'gps', 'duration': 1, 'timestamp': '2000-0-0 0:0:0', 'data': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    {'audio': 'noise', 'duration': 1, 'timestamp': '2000-0-0 0:0:0', 'data': 0}
]

# 第三组数据
data3 = [
    {'sensor': 'acc', 'duration': 1000, 'timestamp': '2025-12-24 6:17:19', 'data': [[63841, 55252, 1015768], [66120, 55436, 1013754], [65100, 54802, 1012622], [64944, 55008, 1013206], [64300, 54997, 1013496]]},
    {'sensor': 'gyro', 'duration': 1000, 'timestamp': '2025-12-24 6:17:20', 'data': [[65, 81, 55], [19, -41, -66], [-1, -6, -39], [65, 74, -79], [-11, 1, 30]]},
    {'sensor': 'mag', 'duration': 1000, 'timestamp': '2025-12-24 6:17:19', 'data': [[1556, -372, -976, 0], [1568, -368, -976, 0], [1552, -380, -976, 0], [1568, -368, -976, 0], [1568, -368, -976, 0]]},
    {'sensor': 'airpress', 'duration': 1000, 'timestamp': '2025-12-24 6:17:20', 'data': [[2083307, -33234, 1153], [2083382, -33539, 1153], [2083355, -33429, 1147], [2083330, -33328, 1143], [2083337, -33356, 1138]]},
    {'sensor': 'als', 'duration': 4000, 'timestamp': '2025-12-24 6:17:17', 'data': [[1536], [447], [436], [458], [457]]},
]

# 发送数据
def send_data(data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(server_url, headers=headers, data=json.dumps(data))
    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.content.decode()}")

# 发送第一组数据
print("Sending data1...")
send_data(data1)

# 发送第二组数据
print("Sending data2...")
send_data(data2)

# 发送第三组数据
print("Sending data3...")
send_data(data3)