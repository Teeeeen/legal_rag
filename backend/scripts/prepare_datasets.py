"""数据准备脚本 — 下载并转换多种法律数据源

用法:
    cd backend
    python -m scripts.prepare_datasets

功能:
1. 转换 CAIL2018 刑事案例为系统兼容的 JSONL 格式
2. 转换 CrimeKgAssitant QA 数据为评估参考答案
3. 处理 LawRefBook Markdown 法律条文（若已下载）
4. 从 HuggingFace 下载 DISC-Law-SFT（可选）
"""

import json
import os
import re
import sys
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LAWS_DIR = DATA_DIR / "laws"
CASES_DIR = DATA_DIR / "cases"
QA_DIR = DATA_DIR / "qa"
REFERENCE_DIR = DATA_DIR / "reference"


def convert_cail2018(max_cases: int = 10000):
    """将 CAIL2018 刑事案例转换为系统兼容的 JSONL 格式

    CAIL2018 原始格式:
        {"fact": "...", "meta": {"accusation": [...], "relevant_articles": [...], ...}}

    转换为:
        {"fact": "...", "accusation": "...", "relevant_articles": "...",
         "criminals": "...", "imprisonment": ..., "source": "CAIL2018"}
    """
    output_dir = CASES_DIR / "CAIL2018"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cail2018_cases.jsonl"

    # 查找已解压的 CAIL2018 数据文件
    source_files = []
    for search_dir in [
        CASES_DIR / "CAIL2018_extracted",
        CASES_DIR / "CAIL_repo" / "CAIL2018",
    ]:
        if search_dir.exists():
            for root, _dirs, files in os.walk(search_dir):
                for f in files:
                    if f.endswith(".json") and f != "README.md":
                        source_files.append(os.path.join(root, f))

    # 如果没找到数据文件，尝试解压 zip
    if not source_files:
        zip_path = CASES_DIR / "CAIL2018_ALL_DATA.zip"
        if zip_path.exists():
            print(f"  解压 {zip_path.name}...")
            import zipfile
            extract_dir = CASES_DIR / "CAIL2018_extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            # 查找解压后的 JSON 文件
            for root, _dirs, files in os.walk(extract_dir):
                for f in files:
                    if f.endswith(".json"):
                        source_files.append(os.path.join(root, f))
        else:
            print("  ✗ 未找到 CAIL2018 数据文件或 ZIP 包")
            return 0

    if not source_files:
        print("  ✗ 未找到任何 CAIL2018 JSON 数据文件")
        return 0

    print(f"  找到 {len(source_files)} 个源文件")
    count = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for src in sorted(source_files):
            if count >= max_cases:
                break
            fname = os.path.basename(src)
            try:
                with open(src, "r", encoding="utf-8") as f:
                    for line in f:
                        if count >= max_cases:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        fact = obj.get("fact", "")
                        if not fact or len(fact) < 50:
                            continue

                        meta = obj.get("meta", {})
                        # 构建带结构化元数据的案例记录
                        record = {
                            "fact": fact,
                            "accusation": "；".join(meta.get("accusation", [])),
                            "relevant_articles": "；".join(
                                str(a) for a in meta.get("relevant_articles", [])
                            ),
                            "criminals": "；".join(meta.get("criminals", [])),
                            "source": "CAIL2018",
                        }
                        # 刑期信息
                        term = meta.get("term_of_imprisonment", {})
                        if term.get("death_penalty"):
                            record["sentence"] = "死刑"
                        elif term.get("life_imprisonment"):
                            record["sentence"] = "无期徒刑"
                        else:
                            months = term.get("imprisonment", 0)
                            if months > 0:
                                record["sentence"] = f"有期徒刑{months}个月"
                            else:
                                record["sentence"] = "免予刑事处罚"

                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1

                print(f"  处理 {fname}: 累计 {count} 条")
            except Exception as e:
                print(f"  ✗ {fname}: {e}")

    print(f"  ✓ CAIL2018 转换完成: {count} 条案例 → {output_file.name}")
    return count


def convert_crime_qa_to_reference(max_qa: int = 5000):
    """将 CrimeKgAssitant QA 数据转为评估参考答案

    输入格式(每行 JSON):
        {"question": "...", "answers": ["...", ...], "category": "..."}

    输出: reference_answers.json (dict: question → answer)
    """
    qa_file = QA_DIR / "CrimeKgAssitant" / "data" / "qa_corpus.json"
    if not qa_file.exists():
        print("  ✗ CrimeKgAssitant QA 语料不存在")
        return 0

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    output_file = REFERENCE_DIR / "crime_qa_reference.json"
    qa_for_eval = DATA_DIR / "reference_answers.json"

    references = {}
    all_qa = []
    count = 0

    with open(qa_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = obj.get("question", "").strip()
            answers = obj.get("answers", [])
            category = obj.get("category", "")

            if not question or not answers:
                continue

            # 取最长的答案作为参考答案（通常最详细）
            best_answer = max(answers, key=len)
            if len(best_answer) < 10:
                continue

            all_qa.append({
                "question": question,
                "answer": best_answer,
                "category": category,
            })
            count += 1

    # 按类别均匀采样
    categories = {}
    for qa in all_qa:
        cat = qa["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(qa)

    # 每个类别最多取 max_qa / num_categories 条
    sampled = []
    per_cat = max(max_qa // max(len(categories), 1), 50)
    for cat, items in categories.items():
        random.seed(42)
        sample = random.sample(items, min(len(items), per_cat))
        sampled.extend(sample)

    if len(sampled) > max_qa:
        random.seed(42)
        sampled = random.sample(sampled, max_qa)

    # 写入完整 QA 参考文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    # 写入精选评估参考答案 (question → answer 映射)
    eval_refs = {}
    # 先保留已有的参考答案
    if qa_for_eval.exists():
        try:
            with open(qa_for_eval, "r", encoding="utf-8") as f:
                eval_refs = json.load(f)
        except Exception:
            pass

    # 每个类别取 5 个高质量 QA 对作为评估基准
    for cat, items in categories.items():
        # 选择答案较长的（更详细）
        sorted_items = sorted(items, key=lambda x: len(x["answer"]), reverse=True)
        for item in sorted_items[:5]:
            eval_refs[item["question"]] = item["answer"]

    with open(qa_for_eval, "w", encoding="utf-8") as f:
        json.dump(eval_refs, f, ensure_ascii=False, indent=2)

    print(f"  ✓ QA 参考数据: {len(sampled)} 条 → {output_file.name}")
    print(f"  ✓ 评估参考答案: {len(eval_refs)} 条 → reference_answers.json")
    return len(sampled)


def convert_crime_kg():
    """将犯罪知识图谱转为可检索的法律知识文件"""
    kg_file = QA_DIR / "CrimeKgAssitant" / "data" / "kg_crime.json"
    if not kg_file.exists():
        print("  ✗ 犯罪知识图谱文件不存在")
        return 0

    output_dir = LAWS_DIR / "CrimeKG"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "犯罪知识图谱.txt"

    # JSONL 格式：每行一个 JSON 对象
    field_map = {
        "gainian": "概念与定义",
        "tezheng": "犯罪构成特征",
        "rending": "认定与区分",
        "chufa": "量刑处罚",
        "fatiao": "相关法条",
        "jieshi": "司法解释",
    }

    lines = []
    count = 0
    with open(kg_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            crime_big = obj.get("crime_big", "")
            crime_small = obj.get("crime_small", "")
            if not crime_small:
                continue

            parts = [f"【{crime_small}】（{crime_big}）"]

            for key, label in field_map.items():
                value = obj.get(key, [])
                if isinstance(value, list) and value:
                    # 合并列表条目为文本段落
                    text = "\n".join(v.strip() for v in value if v.strip())
                    if text:
                        parts.append(f"\n{label}：\n{text}")
                elif isinstance(value, str) and value.strip():
                    parts.append(f"\n{label}：\n{value.strip()}")

            if len(parts) > 1:
                lines.append("\n".join(parts))
                count += 1

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n" + "=" * 50 + "\n\n".join(lines))

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ 犯罪知识图谱: {count} 个罪名, {size_mb:.1f} MB → {output_file.name}")
    return count


def process_lawrefbook():
    """处理 LawRefBook Markdown 法律条文

    LawRefBook 的 Markdown 文件已经可以被系统直接读取(.md 格式)，
    但需要过滤掉非法律内容的文件（如 README、配置文件等）。
    """
    lawref_dir = LAWS_DIR / "LawRefBook"
    if not lawref_dir.exists():
        print("  ✗ LawRefBook 目录不存在（未下载或下载失败）")
        return 0

    # 统计可用的法律文件
    law_files = []
    skip_patterns = {"README", "LICENSE", "SUMMARY", ".git", "node_modules", "package"}
    for root, dirs, files in os.walk(lawref_dir):
        # 跳过 .git 目录
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.endswith((".md", ".txt")):
                if not any(skip in f for skip in skip_patterns):
                    law_files.append(os.path.join(root, f))

    print(f"  ✓ LawRefBook: 发现 {len(law_files)} 个法律文件（.md/.txt）")
    print(f"    目录位于: {lawref_dir}")
    print(f"    import_data.py 会自动递归扫描该目录")
    return len(law_files)


def download_disc_law_sft(max_samples: int = 5000):
    """尝试从 HuggingFace 下载 DISC-Law-SFT 数据"""
    output_dir = REFERENCE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "disc_law_sft.json"

    if output_file.exists():
        print(f"  ✓ DISC-Law-SFT 已存在: {output_file}")
        return 0

    try:
        from datasets import load_dataset
        print("  正在从 HuggingFace 下载 DISC-Law-SFT...")
        ds = load_dataset("ShengbinYue/DISC-Law-SFT", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            samples.append({
                "input": item.get("input", ""),
                "output": item.get("output", ""),
                "category": item.get("category", ""),
            })
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"  ✓ DISC-Law-SFT: {len(samples)} 条 → {output_file.name}")
        return len(samples)
    except ImportError:
        print("  ⚠ 需要安装 datasets 库: pip install datasets")
        print("    安装后重新运行即可自动下载")
        return 0
    except Exception as e:
        print(f"  ✗ DISC-Law-SFT 下载失败: {e}")
        return 0


def print_summary():
    """打印数据目录汇总统计"""
    print(f"\n{'=' * 55}")
    print("数据目录汇总")
    print(f"{'=' * 55}")

    for name, path in [
        ("法律条文 (laws/)", LAWS_DIR),
        ("案例 (cases/)", CASES_DIR),
        ("QA 数据 (qa/)", QA_DIR),
        ("参考答案 (reference/)", REFERENCE_DIR),
    ]:
        if not path.exists():
            print(f"  {name}: 不存在")
            continue
        total_size = 0
        file_count = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if not f.startswith(".") and not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
                    file_count += 1
        size_mb = total_size / (1024 * 1024)
        print(f"  {name}: {file_count} 个文件, {size_mb:.1f} MB")


def main():
    print("=" * 55)
    print("法律 RAG 系统 — 数据准备")
    print("=" * 55)

    # 1. 转换 CAIL2018
    print("\n[1/5] 转换 CAIL2018 刑事案例")
    convert_cail2018(max_cases=10000)

    # 2. 转换 CrimeKgAssitant QA
    print("\n[2/5] 转换 CrimeKgAssitant QA 为参考答案")
    convert_crime_qa_to_reference(max_qa=5000)

    # 3. 转换犯罪知识图谱
    print("\n[3/5] 转换犯罪知识图谱为可检索知识")
    convert_crime_kg()

    # 4. 处理 LawRefBook
    print("\n[4/5] 检查 LawRefBook 法律条文")
    process_lawrefbook()

    # 5. DISC-Law-SFT
    print("\n[5/5] 下载 DISC-Law-SFT (可选)")
    download_disc_law_sft()

    # 汇总
    print_summary()


if __name__ == "__main__":
    main()
