import json


class SmartSensingSystem:
    def __init__(self, config_path="system_config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.mappings = self.config['mappings']
        self.rules = self.config['rules']
        self.thresholds = self.config['thresholds']

        # prev_state holds the last valid values from the previous second
        self.prev_state = {
            "Sound": 0, "Light": 0, "Activity": 0,
            "Location": 0, "Scene": 0
        }

        # current_state: initialized in run_step
        self.state = {
            "Sound": 0, "Light": 0, "Activity": 0,
            "Location": 0, "Scene": 0
        }

        self.activity_duration = 0
        self.current_activity_label = "null"
        self.last_action = "Nothing"
        # 构建mapping的反向查找表 name -> id
        # "Activity": { "4": "strolling", "5": "running" } ---> "Activity": { "strolling": 4, "running": 5, ... },
        self.name_to_id = {
            cat: {v: int(k) for k, v in val.items()}
            for cat, val in self.mappings.items()
        }

    '''
    输入：类别（如 "Activity"）和 整数ID（如 4）。
    逻辑：从原始配置 self.mappings 中查找。注意它将输入的 value_id 转为了字符串 str(value_id)，因为 JSON 的键通常是字符串。
    输出：返回对应的标签（如 "strolling"）。如果找不到，默认返回 "null"。
    场景：当传感器上报了 ID 4，代码需要将其转为 "strolling"，以便放入 ctx 上下文中供 eval(rule['condition']) 评估。
    '''
    def _get_label(self, category, value_id):
        return self.mappings[category].get(str(value_id), "null")
    '''
    输入：类别（如 "Activity"）和 字符串标签（如 "strolling"）。
    逻辑：从第一步生成的 self.name_to_id 中查找。
    输出：返回对应的整数 ID（如 4）。如果找不到，默认返回 0（通常代表 null/未知）。
    场景：
    当外部输入直接给出字符串描述时，系统需要将其转为 ID 存入 self.state。
    初始化 prev_state 时，需要将配置中的字符串转为 ID。
    '''
    def _get_id(self, category, label):
        return self.name_to_id[category].get(label, 0)
    '''
    允许外部（通常是测试程序）手动初始化或注入“上一时刻的状态”，并自动处理数据类型的标准化。
    '''
    def set_previous_state(self, state_dict):
        # 遍历传入的状态字典
        for k, v in state_dict.items():
            # 1. 安全检查：只处理系统设计中已知的状态字段
            # (即 self.prev_state 中存在的键，如 "Activity", "Sound" 等)
            if k in self.prev_state:

                # 2. 智能转换：处理字符串输入
                if isinstance(v, str):
                    # 如果输入是 "strolling" (字符串)，调用辅助函数转为 ID (如 4)
                    # 这让测试用例写起来非常直观，不用去查 ID 表
                    self.prev_state[k] = self._get_id(k, v)

                # 3. 直接赋值：处理整数输入
                else:
                    # 如果输入已经是整数 ID (如 4)，直接存入
                    self.prev_state[k] = v

    def set_last_action(self, action_name):
        self.last_action = action_name


    '''
    分层感知：低功耗 -> 中功耗 -> 高功耗，层层递进触发。
    状态机思想：不仅看当前输入，还看 prev_state（变化量）和 duration（持续量）。
    '''
    def run_step(self, external_world):
        intermediate_actions = []

        # T1: Get Cache (Carry Forward Previous State)
        # Instead of resetting to 0, we start with what we knew last second.
        # 状态保持机制。如果这一帧没有触发 Location 更新，self.state["Location"] 依然保留上一帧的值，而不是变成 null。这保证了上下文的连续性。
        self.state = self.prev_state.copy()

        # --- T2: Low-Cost Sensors (Always Update) ---
        # "Activity/Light/Sound is always updated through request"
        # We overwrite the carried-forward value with the fresh reading.
        # --- T2: 低功耗传感器（总是更新） ---
        # 动作(Activity)、光线(Light)、声音(Sound) 是低功耗的，每次都重新请求并覆盖旧值。
        self.state["Sound"] = self._get_id("Sound", external_world.get("Sound", "null"))
        intermediate_actions.append("Request_Sound_Intensity")

        self.state["Light"] = self._get_id("Light", external_world.get("Light", "null"))
        intermediate_actions.append("Request_Light_Intensity")

        self.state["Activity"] = self._get_id("Activity", external_world.get("Activity", "null"))
        intermediate_actions.append("Request_Activity")

        # --- T3 / T4 / T5: Determine if Location Update is needed ---
        request_location = False

        # Check for changes against previous state
        # 检查变化：对比“当前刚刚获取的低功耗数据”与“上一时刻的旧数据”
        if self.state["Sound"] != self.prev_state["Sound"]: request_location = True
        if self.state["Light"] != self.prev_state["Light"]: request_location = True
        if self.state["Activity"] != self.prev_state["Activity"]: request_location = True

        # Action: Request Location
        # If triggered, we overwrite the carried-forward location with a fresh request.
        # If NOT triggered, self.state["Location"] remains as it was (Persistence).
        if request_location:
            # 只有触发了，才去 external_world 拿 Location，并覆盖缓存
            self.state["Location"] = self._get_id("Location", external_world.get("Location", "null"))
            intermediate_actions.append("Request_Location")
        # 如果没有触发，self.state["Location"] 保持的是 T1 阶段 copy 来的旧值。
        # --- T6 / T7: Determine if Scene Update is needed ---
        # --- T6 / T7: 决定是否需要更新场景
        request_scene = False

        # T6: Location Known (Not Null) AND Activity Changed
        # Note: self.state["Location"] is the valid location (either carried forward or just updated)
        # T6: 位置已知（非空） 且 动作发生了变化（比如从走到坐）
        if self.state["Location"] != 0 and (self.state["Activity"] != self.prev_state["Activity"]):
            request_scene = True

        # T7: Location Changed
        # We compare the current state (potentially updated above) vs previous state
        # T7: 位置发生了变化
        # 到了新地方，自然要重新看一眼环境
        if self.state["Location"] != self.prev_state["Location"]:
            request_scene = True

        # Action: Request Scene
        # 动作：请求场景
        # 触发拍照 - 场景分类 （拍照功耗高）
        if request_scene:
            self.state["Scene"] = self._get_id("Scene", external_world.get("Scene", "null"))
            intermediate_actions.append("Request_Scene")

        # --- UPDATE HISTORY ---
        # --- 更新历史 ---
        # 当前状态将在下一帧变成“前一状态”
        self.prev_state = self.state.copy()

        # --- DURATION TRACKING ---
        # --- 持续时间追踪 (Duration Tracking) ---
        '''
        假设外部每1秒调用一次 run_step，那么 activity_duration 就等于秒数。
        如果调用频率不稳定，这个时间就不准。
        '''
        current_activity = self._get_label("Activity", self.state["Activity"])
        if current_activity == self.current_activity_label:
            # 如果动作没变，计数器+1
            self.activity_duration += 1
        else:
            # 如果动作变了，重置为1，并更新当前动作标签
            self.activity_duration = 1
            self.current_activity_label = current_activity

        # --- RULE EVALUATION ---
        # --- 规则评估 ---
        # 1. 构建上下文 (Context)：把内部的整数 ID 转回人类可读的字符串
        ctx = {
            "sound_class": self._get_label("Sound", self.state["Sound"]),
            "light_class": self._get_label("Light", self.state["Light"]),
            "activity_class": self.current_activity_label,
            "location_class": self._get_label("Location", self.state["Location"]),
            "scene_class": self._get_label("Scene", self.state["Scene"]),
            "activity_duration": self.activity_duration,
            "last_action": self.last_action,
            "threshold_walk_duration": self.thresholds['walk_duration'],
            "threshold_relax_duration": self.thresholds['relax_duration']
        }

        terminal_action = "Nothing"
        for rule in self.rules:
            '''
            # 动态执行 DSL
            # rule['condition'] 是字符串，例如: "activity_duration >= 30 and ..."
            # eval 会在 ctx 字典的环境下计算这个布尔表达式 (Python eval 函数，让 JSON 配置文件中的字符串变成了可执行的代码逻辑。)
            '''
            try:
                if eval(rule['condition'], {}, ctx):
                    terminal_action = rule['action']
                    break
            except Exception as e:
                pass

        if terminal_action != "Nothing":
            self.last_action = terminal_action

        return intermediate_actions, terminal_action
