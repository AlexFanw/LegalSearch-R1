from typing import Optional

from openai import OpenAI
import re
import difflib
import string
import json
import argparse
import os
import random
import time
from tqdm import tqdm
import pandas as pd
import sys
import math
from collections import Counter

from openai.types import Completion as OpenAICompletion
from openai import RateLimitError as OpenAIRateLimitError
from openai import APIError as OpenAIAPIError
from openai import Timeout as OpenAITimeout

from user.tools.utils import get_response_from_llm
import yaml
import os


# Load API keys from config
def _load_config():
    config_path = "./user/tools/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

_config = _load_config()
LLM_API_KEY = _config.get("openai_api_key", "")
LLM_BASE_URL = _config.get("openai_api_base", "https://openrouter.ai/api/v1")


_openai_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not LLM_API_KEY:
            raise RuntimeError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY or OPENAI_API_KEY."
            )
        _openai_client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        )
    return _openai_client

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)

def check_tags_balance_simple(solution_str: str) -> bool:
    """Check if tags are correctly matched

    Args:
        solution_str: The string to check

    Returns:
        bool: Whether all tags are correctly matched
    """
    # Tags to check
    tags_to_check = ['tool_call', 'think', 'answer', 'plan', 'tool_response']
    
    for tag in tags_to_check:
        # Count the number of start and end tags
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)
        
        # Return False if counts are not equal
        if start_count != end_count:
            print(f"Unmatched tags: {start_tag} and {end_tag}")
            return False
            
        # Check nesting order (ensure end tag does not come before start tag)
        last_pos = -1
        while True:
            start_pos = solution_str.find(start_tag, last_pos + 1)
            if start_pos == -1:
                break
                
            end_pos = solution_str.find(end_tag, start_pos)
            if end_pos == -1:
                # print(f"Unmatched start tag: {start_tag}")
                return False
                
            last_pos = end_pos
            
    return True

def check_tags_balance_enhanced(solution_str: str) -> bool:
    """
    Enhanced tag balance checking:
    1. All tags must appear at least once.
    2. Tag open/close counts must match.
    3. Count(think) == Count(plan) + Count(tool_call) + Count(answer)
    """

    tags = ['think', 'plan', 'tool_call', 'answer', 'tool_response']

    # --- Step 1: All tags must appear ---
    for tag in tags:
        if f"<{tag}>" not in solution_str or f"</{tag}>" not in solution_str:
            print(f"Tag <{tag}> missing.")
            return False

    # --- Step 2: Count matching open/close tags ---
    counts = {}
    for tag in tags:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)

        if start_count != end_count:
            print(f"Unmatched tags: {start_tag} ({start_count}) and {end_tag} ({end_count})")
            return False

        counts[tag] = start_count

    # --- Step 3: think count must equal plan + tool_call + answer ---
    if counts['think'] != counts['plan'] + counts['tool_call'] + counts['answer']:
        print(f"Tag count mismatch: think={counts['think']} but plan+tool_call+answer="
              f"{counts['plan'] + counts['tool_call'] + counts['answer']}")
        return False

    # --- Step 4: simple linear ordering check (keep your original logic) ---
    for tag in tags:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        last_pos = -1
        while True:
            start_pos = solution_str.find(start_tag, last_pos + 1)
            if start_pos == -1:
                break

            end_pos = solution_str.find(end_tag, start_pos)
            if end_pos == -1:
                print(f"Unmatched start tag: {start_tag}")
                return False

            last_pos = end_pos

    return True

def check_tags_balance(solution_str: str) -> bool:
    """
    Check that:
    1. Only allowed tags appear.
    2. All tags are matched and balanced.
    3. Tag blocks are not nested (linear structure only).
    """
    allowed_tags = ['think', 'plan', 'answer', 'tool_call']
    full_tag_pattern = re.compile(r'</?([a-zA-Z0-9_]+)>')
    valid_tag_pattern = re.compile(r'</?(' + '|'.join(allowed_tags) + r')>')

    # Step 1: Check for illegal tags
    for match in full_tag_pattern.finditer(solution_str):
        tag_name = match.group(1)
        if tag_name not in allowed_tags:
            return False  # Found an illegal tag

    # Step 2: Check for linear, non-nested structure
    tag_stack = []
    last_close_idx = -1

    for match in valid_tag_pattern.finditer(solution_str):
        tag = match.group(0)
        tag_name = match.group(1)
        is_closing = tag.startswith('</')

        if is_closing:
            if not tag_stack:
                return False  # Closing tag without opening
            last_tag, open_end_idx = tag_stack.pop()
            if last_tag != tag_name:
                return False  # Mismatched tag

            # Check if anything is nested between <tag>...</tag>
            between = solution_str[open_end_idx:match.start()]
            if valid_tag_pattern.search(between):
                return False  # Nested tag found

        else:
            tag_stack.append((tag_name, match.end()))

    return len(tag_stack) == 0

def extract_answer(solution_str: str) -> str:
    """Extract the answer content from the solution string."""

    if not solution_str:
        return ""

    match = ANSWER_PATTERN.search(solution_str)
    if not match:
        return ""
    return match.group(1).strip()


def normalize_for_metrics(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", text)


def tokenise(text: str) -> list[str]:
    return list(text)


def longest_common_subsequence(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0

    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, start=1):
            temp = dp[j]
            if token_a == token_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l_score(reference: list[str], hypothesis: list[str], beta: float = 1.2) -> float:
    if not reference or not hypothesis:
        return 0

    lcs = longest_common_subsequence(reference, hypothesis)
    recall = lcs / len(reference)
    precision = lcs / len(hypothesis)
    if recall == 0 or precision == 0:
        return 0

    beta_sq = beta * beta
    return (1 + beta_sq) * recall * precision / (recall + beta_sq * precision)


def f1_score(reference: list[str], hypothesis: list[str]) -> float:
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0

    reference_counts = Counter(reference)
    hypothesis_counts = Counter(hypothesis)

    overlap = sum(min(reference_counts[token], hypothesis_counts[token]) for token in reference_counts)
    if overlap == 0:
        return 0

    precision = overlap / len(hypothesis)
    recall = overlap / len(reference)
    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


def jaccard_score(reference: list[str], hypothesis: list[str]) -> float:
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0

    ref_set = set(reference)
    hyp_set = set(hypothesis)
    union = ref_set | hyp_set
    if not union:
        return 1.0

    intersection = ref_set & hyp_set
    return len(intersection) / len(union)


def compute_generation_metrics(ground_truth: str, prediction: str) -> dict[str, float]:
    reference_text = normalize_for_metrics(ground_truth)
    hypothesis_text = normalize_for_metrics(prediction)

    reference_tokens = tokenise(reference_text)
    hypothesis_tokens = tokenise(hypothesis_text)

    rouge_l = rouge_l_score(reference_tokens, hypothesis_tokens)
    f1 = f1_score(reference_tokens, hypothesis_tokens)
    jaccard = jaccard_score(reference_tokens, hypothesis_tokens)
    bleu = bleu_score(reference_tokens, hypothesis_tokens)

    return {
        "rouge_l": rouge_l,
        "f1": f1,
        "jaccard": jaccard,
        "bleu": bleu,
    }


def modified_precision(reference: list[str], hypothesis: list[str], n: int) -> tuple[int, int]:
    if len(hypothesis) < n:
        return 0, 0

    ref_ngrams = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
    hyp_ngrams = Counter(tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1))

    overlap = 0
    for ngram, count in hyp_ngrams.items():
        overlap += min(count, ref_ngrams.get(ngram, 0))

    return overlap, sum(hyp_ngrams.values())


def brevity_penalty(reference: list[str], hypothesis: list[str]) -> float:
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if hyp_len == 0:
        return 0
    if hyp_len > ref_len:
        return 1.0
    return math.exp(1 - ref_len / hyp_len)


def bleu_score(reference: list[str], hypothesis: list[str], max_order: int = 4, smooth: bool = True) -> float:
    if not hypothesis:
        return 0

    weights = [1 / max_order] * max_order
    precisions = []
    for n in range(1, max_order + 1):
        overlap, total = modified_precision(reference, hypothesis, n)
        if total == 0:
            precisions.append(0)
            continue
        if overlap == 0:
            precisions.append(1 / (total * (2 ** n)) if smooth else 0)
        else:
            precisions.append(overlap / total)

    if any(p == 0 for p in precisions):
        geo_mean = 0
    else:
        log_precision_sum = sum(w * math.log(p) for w, p in zip(weights, precisions))
        geo_mean = math.exp(log_precision_sum)

    bp = brevity_penalty(reference, hypothesis)
    return bp * geo_mean


def split_answers(answer: str) -> list[str]:
    parts = re.split(r"[;；]", answer)
    return [part.strip() for part in parts if part.strip()]


def answers_match_unordered(expected: str, predicted: str) -> bool:
    expected_parts = split_answers(expected)
    predicted_parts = split_answers(predicted)
    if not expected_parts and not predicted_parts:
        return True
    if len(expected_parts) != len(predicted_parts):
        return False
    for part in expected_parts:
        if part not in predicted_parts and part + "罪" not in predicted_parts:
            return False
    return True


def score_lar_answer(ground_truth: str, prediction: str) -> float:
    metrics = compute_generation_metrics(ground_truth, prediction)
    rouge_l = metrics.get("rouge_l", 0)
    if rouge_l >= 0.95:
        return 1.0
    else:
        return 0


def score_structured_answer(task_type: str, ground_truth: str, prediction: str) -> float:
    task_type = (task_type or "").lower()
    prediction = prediction.strip()

    if not ground_truth:
        return 0

    best_score = 0

    candidate = ground_truth.strip()

    if task_type in {"lar"}:
        score = score_lar_answer(candidate, prediction)
    elif task_type == "ccp":
        score = 1.0 if answers_match_unordered(candidate, prediction) else 0
    elif task_type in {"kqa", "ptp", "lap", "lca"}:
        score = 1.0 if prediction == candidate else 0
    else:
        score = 1.0 if prediction.lower() == candidate.lower() else 0

    best_score = max(best_score, score)

    return best_score


LCS_PROMPT = """
你是一名法律评审专家，任务是判断模型回答是否正确，且法律依据引用是否正确。

请严格输出 JSON，格式如下：
{
  "judgement_answer": "correct" 或 "incorrect",
  "judgement_basis": "correct" 或 "incorrect"
}

评分逻辑如下（务必严格遵守）：

1. judgement_answer（回答正确性）
   - 如果模型给出的最终答案与标准答案在逻辑和语义上完全一致，哪怕表述上有差别，也为 "correct"
   - 否则为 "incorrect"

2. judgement_basis（法律依据正确性）
   - 如果模型引用的法律条文、案例或其他法律依据与标准答案中的法律依据一致，哪怕表述上有差别，也为 "correct"
   - 如果条文错误、引用不存在的条文、条文内容错误、或引用条文无法支撑结论，则为 "incorrect"

务必只输出 JSON，不要输出其他内容。
"""
def get_json(raw: str) -> dict:
    json_pattern = re.compile(r"\{.*\}", re.DOTALL)
    match = json_pattern.search(raw)
    if not match:
        raise ValueError("No JSON object found in the response.")
    json_str = match.group(0)
    return json.loads(json_str)

def get_lcs_result(question, gt_answer, pred_answer):
    try_cnt = 0
    while try_cnt < 20:
        prompt = LCS_PROMPT + f"""
        【问题】
        {question}

        【标准答案】
        {gt_answer}

        【模型回答】
        {pred_answer}
        """
        try:
            raw = call_api(prompt)
            data = get_json(raw)

            if "judgement_answer" in data and "judgement_basis" in data:
                answer_correct = 0.5 if data["judgement_answer"] == "correct" else 0
                basis_correct = 0.5 if data["judgement_basis"] == "correct" else 0
                return answer_correct, basis_correct
            else:
                try_cnt += 1
                print(f"Missing keys in LCS API response. Retrying ({try_cnt})...")
        except Exception as e:
            try_cnt += 1
            print(f"Error in LCS API call or JSON parsing: {e}. Retrying ({try_cnt})...")
    return 0, 0


def call_api(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = get_response_from_llm(
            messages=messages,
            client=get_openai_client(),
            model="qwen/qwen3-235b-a22b-2507",
            stream=False,
            temperature=0.6,
            timeout=60,
        )
        return response.get("content", "")
    except Exception as exc:
        print(f"Error calling OpenAI client: {exc}")
        return ""

def compute_score(solution_str, ground_truth, val_type, question, data_source) -> dict:
    if not check_tags_balance_enhanced(solution_str):
        return {"score": 0, "ground_truth": ground_truth, "answer": extract_answer(solution_str)}

    lower_solution = solution_str.lower()
    if "<plan>" not in lower_solution or "</plan>" not in lower_solution:
        return {"score": 0, "ground_truth": ground_truth, "answer": extract_answer(solution_str)}

    # if "<tool_call>" not in lower_solution or "</tool_call>" not in lower_solution:
    #     format_score = 0
    # else:
    #     format_score = 0.5

    format_weight = 0
    answer_weight = 1.0 - format_weight

    format_score = 1

    answer_content = extract_answer(solution_str)
    if not answer_content:
        return {"score": 0, "ground_truth": ground_truth, "answer": ""}

    if data_source == "lcs":
        answer_correct, basis_correct = get_lcs_result(question, ground_truth, answer_content)
        answer_score = answer_correct + basis_correct
        if answer_score != 1.0:
            answer_score = 0
        total_score = format_score * format_weight + answer_score * answer_weight
        return {
            "score": total_score,
            "ground_truth": ground_truth,
            "answer": answer_content
        }

    answer_score = score_structured_answer(data_source, ground_truth, answer_content)
    total_score = format_score * format_weight + answer_score * answer_weight
    return {"score": total_score, "ground_truth": ground_truth, "answer": answer_content}

if __name__ == "__main__":
    # Example usage
    pass