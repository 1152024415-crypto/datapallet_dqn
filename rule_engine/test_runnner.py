import json
from system_engine import SmartSensingSystem


def run_tests(test_file="test_cases.json"):
    try:
        with open(test_file, 'r') as f:
            scenarios = json.load(f)
    except FileNotFoundError:
        print(f"Error: {test_file} not found.")
        return

    system = SmartSensingSystem('system_config.json')

    print(f"RUNNING TESTS FROM: {test_file}")
    # Updated Header to show both Requests and Actions
    print(f"{'SCENARIO':<30} | {'DESC':<50} | {'ACTUAL REQS':<25} | {'ACTUAL ACTION':<25} | {'RESULT'}")
    print("-" * 145)

    for scenario in scenarios:
        # 系统重置：如果测试用例要求 reset，创建一个全新的系统实例
        # 这是为了防止上一个测试用例的状态污染当前用例
        if scenario.get("reset"):
            system = SmartSensingSystem('system_config.json')

        # Inject Previous State if defined
        # 注入初始状态
        # 允许测试用例指定一个“前置条件”，比如“用户之前是坐着的”
        if "previous_state" in scenario:
            system.set_previous_state(scenario["previous_state"])

        # Inject Last Action if defined
        if "previous_action" in scenario:
            system.set_last_action(scenario["previous_action"])

        for i, step in enumerate(scenario['steps']):
            inputs = step['input'] # 当前这一步的传感器输入
            wait_count = step.get('wait', 0) # 需要模拟等待的时间步数

            # Simulate wait
            # 模拟时间流逝（Wait）
            # 如果规则要求“持续30秒”，测试数据可能只给了一个 Start 和一个 End
            # 这里通过循环空跑 wait_count 次 run_step 来模拟中间的时间流逝
            if wait_count > 0:
                for _ in range(wait_count - 1):
                    system.run_step(inputs)

            # Execute Step
            # 执行关键步骤
            # 调用核心函数，获取实际输出：中间请求(inter) 和 最终动作(term)
            inter, term = system.run_step(inputs)

            # --- STRICT VALIDATION LOGIC ---

            # 1. Filter Actual Requests (High Cost Only) to compare with inter_check
            # 验证点 1：中间请求 (Request_...)
            # 过滤掉低功耗的请求（因为它们总是发生），只关注高成本的 Location/Scene 请求
            actual_high_cost = sorted([
                x for x in inter
                if x in ["Request_Location", "Request_Scene"]
            ])

            # 2. Get Expected Requests
            # 获取期望的请求列表
            expected_high_cost = sorted(step.get('inter_check', []))

            # 3. Check Exact Match for BOTH Requests and Action
            # 验证点 2：最终动作 (Action)
            pass_inter = (actual_high_cost == expected_high_cost)
            pass_term = (term == step.get('expect_term', "Nothing"))

            status = "PASS" if (pass_term and pass_inter) else "FAIL"

            # Format output strings
            inter_str = " | ".join([x.replace("Request_", "") for x in actual_high_cost])
            if not inter_str: inter_str = "-"

            # Print Table Row
            print(f"{scenario['id']:<30} | {step['desc']:<50} | {inter_str:<25} | {term:<25} | {status}")

            # Detailed Failure Logs
            if status == "FAIL":
                if not pass_inter:
                    print(f"   >>> REQ MISMATCH! Expected: {expected_high_cost} | Got: {actual_high_cost}")
                if not pass_term:
                    print(f"   >>> ACT MISMATCH! Expected: {step.get('expect_term')} | Got: {term}")


if __name__ == "__main__":
    # You can switch between the files here:
    run_tests("test_cases.json")
    print()
    print()
    run_tests("test_cases_difficult.json")