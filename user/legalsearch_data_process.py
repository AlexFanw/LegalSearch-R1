"""Utility script to convert LegalSearch KQA data into parquet for the deep research agent."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from time import gmtime, strftime
import pandas as pd

import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CURRENT_DATE = strftime("%Y-%m-%d", gmtime())

SYSTEM_PROMPT = f"""
## 背景信息

今天日期是 {CURRENT_DATE}。
你是一名专业的中国法律研究助手，但你记忆中的法律知识并不完整或最新。
为了确保你提供的信息准确且有依据，你必须使用可用的工具来检索真实的法律条文、司法解释或权威信息。

## 你的任务

用户会提出中国法律相关的问题，禁止在任何情况下使用非搜索工具得到的信息来推理和回答问题。
你必须先制定系统化的搜索计划，然后调用工具检索真实的法律条文、司法解释或权威信息，最后基于检索结果给出严谨且可溯源的法律分析。

## 强制要求

1.在第一轮，你必须在<think>和<plan>标签内输出你的思考过程和Step by step的研究计划，每一步计划应该简明且以行动为导向，如工具调用、回答等。
2.每一轮都必须以<think>开头，禁止任何例外。
3.最后一轮必须在<think>和<answer>标签内输出你的思考过程和最终答案。

## 工具使用原则

你必须根据问题内容判断应使用的工具：

1. 如果问题涉及具体法律条文（如包含“第×条”“刑法”“民法典”“条款”等词），使用rag_retrieve。
2. 如果问题涉及法律理论、法治思想、制度建设、案例分析或司法实践等非条文性内容，应使用web_search。

## 输出格式

每一轮可以从下面三种输出格式中选择，但是不能混合在一起

1.第一轮必须进行思考和计划，禁止在这一轮进行tool_call：

<think> 
你的思考过程
</think> 
<plan>
Step by step的研究计划。每一轮的计划都应简明且以行动为导向。如工具调用、回答。
</plan>

2.后续轮次进行思考和工具调用：

<think> 
你的思考过程
</think> 
<tool_call> 
符合格式要求的工具调用
</tool_call>

3.最终轮次进行思考和回答：

<think>
你的思考过程 
</think>
<answer> 
符合用户格式要求的答案
</answer>

"""

@dataclass
class ProcessConfig:
    jsonl_path: Path
    output_path: Path
    data_source: str


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(record: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
    """Construct system and user messages following the tool agent format."""
    # system_prompt = (record.get("system_prompt_cot") or record.get("system_prompt") or "").strip()
    
    user_parts: List[str] = []
    for key in ("deepresearch_task_prompt", "question"):
        value = record.get(key)
        if value:
            value = value.strip()
            if value:
                user_parts.append(value)

    user_content = "\n".join(user_parts)

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    return user_content, prompt


def build_reward_model(user_content: str, answer: str) -> Dict[str, Any]:
    return {
        "style": "rule",
        "ground_truth": answer,
        "question": user_content,
    }


def build_extra_info(record: Dict[str, Any], user_content: str, answer: str, data_source: str) -> Dict[str, Any]:
    tools_kwargs = {
        "web_search": {"create_kwargs": {"question": user_content}},
        "browse_webpage": {"create_kwargs": {"question": user_content}},
        "rag_retrieve": {"create_kwargs": {"query": user_content}},
    }

    extra_info = {
        "index": record.get("id"),
        "type": record.get("type"),
        "split": "train" if data_source == "legalsearch_training" else "test",
        "question": user_content,
        "answer": answer,
        "system_prompt": SYSTEM_PROMPT,
        "deepresearch_task_prompt": record.get("deepresearch_task_prompt"),
        "need_tools_kwargs": True,
        "tools_kwargs": tools_kwargs,
        "interaction_kwargs": {
            "query": user_content,
            "ground_truth": answer,
        },
    }

    return extra_info

def save_sample_to_parquet(
    df: pd.DataFrame,
    output_path: Path,
    group_col: str,
    rows_per_group: int,
) -> None:
    """Save a deterministic sample with a fixed number of rows per group."""
    sampled_df = df.groupby(group_col, group_keys=False).head(rows_per_group)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_parquet(output_path, index=False)

def process_dataset(config: ProcessConfig) -> pd.DataFrame:
    records = load_jsonl(config.jsonl_path)

    processed_rows: List[Dict[str, Any]] = []

    for record in records:
        answer = (record.get("answer") or "").strip()
        user_content, prompt = build_prompt(record)

        processed_rows.append(
            {
                "data_source": record.get("type"),
                "agent_name": "tool_agent",
                "prompt": prompt,
                "ability": "deepresearch",
                "reward_model": build_reward_model(user_content, answer),
                "extra_info": build_extra_info(record, user_content, answer, config.data_source),
            }
        )

    df = pd.DataFrame(processed_rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.output_path, index=False)
    if config.data_source == "legalsearch_training":
        save_sample_to_json(df, output_path=DATA_DIR / "train_sample_10.json", num_rows=10)
    elif config.data_source == "legalsearch_evaluation":
        save_sample_to_json(df, output_path=DATA_DIR / "test_sample_10.json", num_rows=10)
        save_sample_to_parquet(
            df,
            output_path=DATA_DIR / "legalsearch_rag_test_sample16.parquet",
            group_col="data_source",
            rows_per_group=16,
        )
        print("\n样本保存成功！")
    print(f"处理完成: {config.jsonl_path}")
    print(f"输出文件: {config.output_path}")
    print(f"总记录数: {df.shape[0]}")
    return df


def default_configs() -> List[ProcessConfig]:
    return [
        ProcessConfig(
            jsonl_path=DATA_DIR / "legalsearch_training.jsonl",
            output_path=DATA_DIR / "legalsearch_rag_train.parquet",
            data_source="legalsearch_training",
        ),
        ProcessConfig(
            jsonl_path=DATA_DIR / "legalsearch_evaluation.jsonl",
            output_path=DATA_DIR / "legalsearch_rag_test.parquet",
            data_source="legalsearch_evaluation",
        ),
        ProcessConfig(
            jsonl_path=DATA_DIR / "legalsearch_ood_evaluation.jsonl",
            output_path=DATA_DIR / "legalsearch_rag_ood_test.parquet",
            data_source="legalsearch_ood_evaluation",
        ),
    ]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process LegalSearch JSONL files into parquet")
    parser.add_argument(
        "--config",
        nargs="*",
        metavar="JSONL:PARQUET:SOURCE",
        help="Optional custom mappings; each entry formatted as input_jsonl:output_parquet:data_source",
    )
    return parser.parse_args()

def save_sample_to_json(df, output_path=DATA_DIR / "train_sample_10.json", num_rows=10):
    """
    将train数据的前num_rows行保存为JSON文件
    """
    try:
        # 取前num_rows行数据
        sample_data = df.head(num_rows)
        
        # 转换为字典格式
        sample_dict = sample_data.to_dict('records')
        
        # 保存为JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_dict, f, ensure_ascii=False, indent=2)
        
        print(f"\n成功保存数据前{num_rows}行到: {output_path}")
        print(f"文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"\n保存JSON文件时出错: {e}")

def main() -> None:
    args = parse_args()

    if args.config:
        configs: List[ProcessConfig] = []
        for item in args.config:
            try:
                jsonl_str, parquet_str, source = item.split(":", 2)
            except ValueError as exc:
                raise ValueError(
                    f"配置 '{item}' 无法解析，需使用 input_jsonl:output_parquet:data_source 格式"
                ) from exc

            configs.append(
                ProcessConfig(
                    jsonl_path=Path(jsonl_str).expanduser().resolve(),
                    output_path=Path(parquet_str).expanduser().resolve(),
                    data_source=source,
                )
            )
    else:
        configs = default_configs()

    for config in configs:
        if not config.jsonl_path.exists():
            print(f"跳过: 输入文件不存在 -> {config.jsonl_path}")
            continue

        process_dataset(config)


if __name__ == "__main__":
    main()
