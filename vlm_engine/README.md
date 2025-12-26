# VLM
```
python test.py
```

1. VLM服务初始化
# 初始化并获取模块和回调
```python
python
from src import initialize_vlm_service

vlm_module, analysis_callback = initialize_vlm_service(
    image_dir="./images",
    metadata_list=metadata_list
)
```


2. 触发场景分析
```python
# 使用回调函数触发分析
success = analysis_callback(start_idx=0, batch_size=5)

# 或者通过TestEngine触发
test_engine.setup_callback(analysis_callback)
success = test_engine.trigger_analysis(start_idx=0, batch_size=5)
```


3. 获取分析结果
```python
# 检查是否正在处理
is_busy = vlm_module.is_busy()

# 获取最后的结果
result = vlm_module.get_last_result()

if result:
    print(f"批次ID: {result.batch_id}")
    print(f"总帧数: {result.total_frames}")
    print(f"场景数量: {result.total_scenes}")
    
    for scene in result.scenes:
        print(f"场景 {scene.scene_id}:")
        print(f"  帧范围: {scene.start_frame}-{scene.end_frame}")
        print(f"  主要活动: {scene.main_activity}")
        print(f"  描述: {scene.description}")
```