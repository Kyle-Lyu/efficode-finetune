import json
import argparse


ID_KEY = "id"
SOURCE_KEY = "source"

def load_json(file_path):
    try:
        data = json.load(open(file_path, "r"))
    except json.JSONDecodeError:
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    return data


def write_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_args():
    parser = argparse.ArgumentParser(description="Calculate final scores and sort data.")
    parser.add_argument(
        "--complexity_file", type=str, required=True,
        help="Path to the complexity scores file (JSONL format)."
    )
    parser.add_argument(
        "--quality_file", type=str, required=True,
        help="Path to the quality scores file (JSONL format)."
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path to save the final sorted results (JSONL format)."
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # 加载复杂度分数文件
    complexity_data = load_json(args.complexity_file)
    # 加载质量分数文件
    quality_data = load_json(args.quality_file)

    # 将数据按 id 映射为字典，方便快速查找
    complexity_dict = {item[ID_KEY]: item for item in complexity_data}
    quality_dict = {item[ID_KEY]: item for item in quality_data}

    # 计算 final_score 并合并数据
    results = []
    for id in complexity_dict:
        if id in quality_dict:
            complexity_score = complexity_dict[id]["complexity_score"]
            quality_score = quality_dict[id]["quality_score"]
            final_score = complexity_score * quality_score

            result = {
                ID_KEY: id,
                SOURCE_KEY: complexity_dict[id][SOURCE_KEY],
                "complexity_score": complexity_score,
                "quality_score": quality_score,
                "final_score": final_score,
            }
            results.append(result)

    # 按 final_score 从大到小排序
    results.sort(key=lambda x: x["final_score"], reverse=True)

    # 保存结果到文件
    write_jsonl(results, args.output_file)

    print(f"Processing completed! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()