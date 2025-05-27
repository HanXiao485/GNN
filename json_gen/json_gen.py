# 重新导入所需库
import json
import numpy as np

# 参数配置（恢复之前上下文）
num_dicts = 256
num_keys_per_dict = 3
array_length = 4

# 重新生成原始数据 v3（包含4维）
json_data_v3 = []
for _ in range(num_dicts):
    entry = {}
    for key in range(num_keys_per_dict):
        matrix = []
        int_value = np.random.randint(-10, 10)
        matrix.append([int_value] * array_length)
        for _ in range(3):
            matrix.append(np.round(np.random.uniform(-5, 5, array_length), 2).tolist())
        entry[str(key)] = matrix
    entry[str(num_keys_per_dict)] = np.round(np.random.uniform(-5, 5, 3), 2).tolist()
    json_data_v3.append(entry)


# 保存到新文件
file_path_v4 = "json_gen/data_v4.json"
with open(file_path_v4, "w") as f:
    json.dump(json_data_v3, f)
