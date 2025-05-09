import json
import argparse
from pathlib import Path


# 常量定义
SEPARATOR = "### Response:\n"
ID_KEY = "id"
LANG_KEY = "language"
INST_KEY = "instruction"
RESP_KEY = "response"
SOURCE_KEY = "source"


def load_json(file_path):
    """加载 JSON 或 JSONL 文件内容到列表"""
    try:
        data = json.load(open(file_path, "r"))
    except json.JSONDecodeError:
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    return data


def write_jsonl(data, file_path):
    """将数据写入 JSONL 文件"""
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def merge_files(input_dir, output_file):
    """
    合并指定目录下的所有 JSONL 文件，并按 ID_KEY 排序。

    Args:
        input_dir (str): 包含 JSONL 文件的目录。
        output_file (str): 合并后的 JSONL 文件路径。
    """
    input_dir = Path(input_dir)
    jsonl_files = list(input_dir.glob("*.jsonl"))

    # 检查文件数量
    if not jsonl_files:
        print(f"No JSONL files found in directory: {input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files in directory: {input_dir}")

    # 合并所有文件内容
    merged_data = []
    for file in jsonl_files:
        print(f"Loading data from: {file}")
        data = load_json(file)
        merged_data.extend(data)

    # 按 ID_KEY 的值从小到大排序
    print(f"Sorting data by {ID_KEY}...")
    sorted_data = sorted(merged_data, key=lambda x: x[ID_KEY])

    # 将合并后的数据写入输出文件
    print(f"Writing merged data to: {output_file}")
    write_jsonl(sorted_data, output_file)

    print(f"Merge completed! Total records: {len(sorted_data)}")


def main():
    """主函数：解析命令行参数并调用合并函数"""
    parser = argparse.ArgumentParser(description="Merge multiple JSONL files into one.")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing the JSONL files to merge."
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path to save the merged JSONL file."
    )
    args = parser.parse_args()

    # 调用合并函数
    merge_files(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()