import json
import os
import torch


def append_to_json(file_path, data):
    """
    将 data 追加写入到指定的 JSON 文件。

    如果文件不存在，会新建文件并写入一个包含 data 的列表；
    如果文件存在，假定文件内容为 JSON 列表，读取列表后追加 data 并写回；
    支持原子写入以避免写入过程中文件损坏。

    参数：
        file_path (str): JSON 文件路径
        data (dict or list): 要追加的 JSON 对象或对象列表
    """
    # 确保 data 为列表，方便批量追加
    entries = data if isinstance(data, list) else [data]

    # 如果文件存在，先读取现有内容
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                raise ValueError(f"JSON 文件 {file_path} 内容不是列表，无法追加")
        except json.JSONDecodeError:
            # 如果文件为空或内容无效，重置为空列表
            existing = []
    else:
        existing = []

    # 合并列表
    combined = existing + entries

    # 写入临时文件，保证原子性
    tmp_file = file_path + '.tmp'
    with open(tmp_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=4)

    # 替换原文件
    os.replace(tmp_file, file_path)


# 示例用法
if __name__ == '__main__':
    log_file = 'data.json'
    new_record = {
        "user": "alice",
        "action": "login",
        "time": "2025-05-23T10:00:00"
    }
    append_to_json(log_file, new_record)
    # 也可以一次追加多个：
    # append_to_json(log_file, [record1, record2])
    new_record = {
        "user": "bob",
        "action": "login",
        "time": "2025-05-23T10:00:00"
    }
    append_to_json(log_file, new_record)
