"""Metrics calculation utility for Legalsearch evaluation outputs.

This module reads a JSONL file where each row contains the following keys:

* ``id``: unique identifier for the sample.
* ``type``: task type, one of ``kqa``, ``ccp``, ``ptp``, ``lar``, ``lap``, ``lca``, ``lcs``.
* ``prompt``: the prompt presented to the model (passed through unchanged).
* ``ground_truth``: the reference answer.
* ``prediction``: the model prediction which should include an ``<answer>...</answer>``
  span.

For ``kqa``, ``ccp``, ``ptp``, ``lap``, and ``lca`` tasks the prediction is scored with strict
accuracy after extracting the answer span. For the ``lar`` and ``lcs`` tasks, the prediction
is evaluated with ROUGE-L, token level F1, Jaccard similarity, and BLEU-4.

The script writes two artifacts:

1. A JSONL file mirroring the input rows and adding a ``score`` key containing
   either the numeric accuracy (for KQA/CCP/PTP) or a dictionary of metrics (for LAR).
2. A plain-text summary file including the per-task average scores and the overall
   average across the four tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
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

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
SUPPORTED_TASKS = {
	"kqa",
	"ccp",
	"ptp",
	"lar",
	"lap",
	"lca",
	"lcs",
	"cpa",
	"pae",
	"nje",
	"ungee",
	"lbk",
	"pfe",
}

OOD_ACCURACY_TASKS = ["cpa", "pae", "nje", "ungee", "lbk", "pfe"]


def _initialise_task_scores(is_ood: bool) -> Dict[str, object]:
	if is_ood:
		return {task: [] for task in OOD_ACCURACY_TASKS}

	return {
		"kqa": [],
		"ccp": [],
		"ptp": [],
		"lap": [],
		"lca": [],
		"lar": {"rouge_l": [], "f1": [], "jaccard": [], "bleu": []},
		"lcs": [],
	}


def read_jsonl(path: Path) -> List[Dict[str, object]]:
	with path.open("r", encoding="utf-8") as file:
		return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
	with path.open("w", encoding="utf-8") as file:
		for row in rows:
			file.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_answer(text: str | None) -> str:
	if not text:
		return ""
	match = ANSWER_PATTERN.search(text)
	if not match:
		return text.strip()
	return match.group(1).strip()


def normalize_for_metrics(text: str | None) -> str:
	if not text:
		return ""
	return re.sub(r"\s+", "", text)


def tokenise(text: str) -> List[str]:
	"""Tokenise text at the character level for Chinese-friendly metrics."""

	return list(text)


def longest_common_subsequence(a: List[str], b: List[str]) -> int:
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


def rouge_l_score(reference: List[str], hypothesis: List[str], beta: float = 1.2) -> float:
	if not reference or not hypothesis:
		return 0.0

	lcs = longest_common_subsequence(reference, hypothesis)
	recall = lcs / len(reference)
	precision = lcs / len(hypothesis)
	if recall == 0 or precision == 0:
		return 0.0

	beta_sq = beta * beta
	return (1 + beta_sq) * recall * precision / (recall + beta_sq * precision)


def f1_score(reference: List[str], hypothesis: List[str]) -> float:
	if not reference and not hypothesis:
		return 1.0
	if not reference or not hypothesis:
		return 0.0

	reference_counts = Counter(reference)
	hypothesis_counts = Counter(hypothesis)

	overlap = sum(min(reference_counts[token], hypothesis_counts[token]) for token in reference_counts)
	if overlap == 0:
		return 0.0

	precision = overlap / len(hypothesis)
	recall = overlap / len(reference)
	if precision + recall == 0:
		return 0.0

	return 2 * precision * recall / (precision + recall)


def jaccard_score(reference: List[str], hypothesis: List[str]) -> float:
	if not reference and not hypothesis:
		return 1.0
	if not reference or not hypothesis:
		return 0.0

	ref_set = set(reference)
	hyp_set = set(hypothesis)
	union = ref_set | hyp_set
	if not union:
		return 1.0

	intersection = ref_set & hyp_set
	return len(intersection) / len(union)


def modified_precision(reference: List[str], hypothesis: List[str], n: int) -> Tuple[int, int]:
	if len(hypothesis) < n:
		return 0, 0

	ref_ngrams = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
	hyp_ngrams = Counter(tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1))

	overlap = 0
	for ngram, count in hyp_ngrams.items():
		overlap += min(count, ref_ngrams.get(ngram, 0))

	return overlap, sum(hyp_ngrams.values())


def brevity_penalty(reference: List[str], hypothesis: List[str]) -> float:
	ref_len = len(reference)
	hyp_len = len(hypothesis)
	if hyp_len == 0:
		return 0.0
	if hyp_len > ref_len:
		return 1.0
	return math.exp(1 - ref_len / hyp_len)


def bleu_score(reference: List[str], hypothesis: List[str], max_order: int = 4, smooth: bool = True) -> float:
	if not hypothesis:
		return 0.0

	weights = [1 / max_order] * max_order
	precisions = []
	for n in range(1, max_order + 1):
		overlap, total = modified_precision(reference, hypothesis, n)
		if total == 0:
			precisions.append(0.0)
			continue
		if overlap == 0:
			precisions.append(1 / (total * (2 ** n)) if smooth else 0.0)
		else:
			precisions.append(overlap / total)

	if any(p == 0 for p in precisions):
		geo_mean = 0.0
	else:
		log_precision_sum = sum(w * math.log(p) for w, p in zip(weights, precisions))
		geo_mean = math.exp(log_precision_sum)

	bp = brevity_penalty(reference, hypothesis)
	return bp * geo_mean


def split_answers(answer: str) -> List[str]:
	parts = re.split(r"[;；]", answer)
	return [part.strip() for part in parts if part.strip()]


def answers_match_unordered(expected: str, predicted: str) -> bool:
	expected_parts = split_answers(expected)
	predicted_parts = split_answers(predicted)
	if not expected_parts and not predicted_parts:
		return True
	if len(expected_parts) != len(predicted_parts):
		return False
	count = 0
	for part in expected_parts:
		if part in predicted_parts or part+"罪" in predicted_parts:
			count += 1
	if count == len(expected_parts):
		return True
	return False


def score_structured_task(task_type: str, ground_truth: str, prediction: str) -> float:
	extracted = extract_answer(prediction)
	normalized_ground = ground_truth.strip()

	if task_type == "ccp":
		# print("Scoring CCP task")
		if answers_match_unordered(normalized_ground, extracted):
			return 1.0
		return 0.0

	return 1.0 if extracted == normalized_ground else 0.0


def score_lar_task(ground_truth: str, prediction: str) -> Dict[str, float]:
	reference_text = normalize_for_metrics(ground_truth)
	hypothesis_text = normalize_for_metrics(extract_answer(prediction))
	# print(reference_text)
	reference_tokens = tokenise(reference_text)
	hypothesis_tokens = tokenise(hypothesis_text)
	# print(hypothesis_text)
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


def summarise_scores(task_scores: Dict[str, object], *, is_ood: bool = False) -> Tuple[str, float]:
	lines: List[str] = []
	aggregated: List[float] = []

	def append_line(label: str, value: float) -> None:
		lines.append(f"{label}: {value*100:.2f}")
		aggregated.append(value)

	if is_ood:
		accuracy_tasks = [(task, task.upper()) for task in OOD_ACCURACY_TASKS]
		metric_tasks: List[Tuple[str, str]] = []
	else:
		accuracy_tasks = [
			("kqa", "KQA"),
			("ccp", "CCP"),
			("ptp", "PTP"),
			("lap", "LAP"),
			("lca", "LCA"),
			("lcs", "LCS"),
		]
		metric_tasks = [
			("lar", "LAR"),
		]

	for task_key, label in accuracy_tasks:
		scores = task_scores.get(task_key, [])
		avg = mean(scores) if scores else 0.0
		if label == "LCS":
			print(f"-------------\n{label} average mbe", avg)
			append_line(f"-------------\n{label} average mbe", avg)
		else:
			print(f"-------------\n{label} average accuracy", avg)
			append_line(f"-------------\n{label} average accuracy", avg)

	metrics_section_started = False

	for task_key, label in metric_tasks:
		metrics = task_scores.get(task_key, {})
		block_lines: List[str] = []

		if isinstance(metrics, dict) and metrics:
			# print(f"{label} metrics sums:", {k: sum(v) for k, v in metrics.items()})
			for metric_name, values in metrics.items():
				avg_value = mean(values) if values else 0.0
				if metric_name == "rouge_l":
					append_line(f"-------------\n{label} average rouge_l", avg_value)
				print(f"{label} average {metric_name.upper()}: {avg_value*100:.2f}")
				block_lines.append(f"{label} average {metric_name.upper()}: {avg_value*100:.2f}")
			composite = mean([
				mean(values) if values else 0.0 for values in metrics.values()
			]) if metrics else 0.0
		else:
			block_lines.append(f"{label} average metrics unavailable: 0.0000")
			composite = 0.0
			
		if block_lines:
			if not metrics_section_started:
				lines.append("\n-------------\n")
				metrics_section_started = True
			else:
				lines.append("")
			lines.extend(block_lines)

		# append_line(f"{label} composite average", composite)

	overall = mean(aggregated) if aggregated else 0.0
	lines.append(f"\n-------------\nOverall average score: {overall*100:.2f}")

	return "\n".join(lines), overall

def print_latex_scores(task_scores: Dict[str, object], *, is_ood: bool = False) -> Tuple[str, float]:
	print("\n================= SUMMARY =================")
	if is_ood:
		acc_order = OOD_ACCURACY_TASKS
		acc_values = [mean(task_scores.get(key, [])) if task_scores.get(key) else 0.0 for key in acc_order]
		header = "  &  ".join([t.upper() for t in acc_order]) + " Accuracy  &  AVG"
		values = "  &  ".join([f"{v*100:.2f}" for v in acc_values]) if acc_values else ""
		overall = mean(acc_values) if acc_values else 0.0
		augmented_values = (values + f"  &  {overall*100:.2f}") if values else f"{overall*100:.2f}"
		text = header + ("\n & " + augmented_values + "\n")
		print(text)
		print("===========================================\n")
		return text, overall

	acc_order = ["kqa", "lap", "ccp", "ptp", "lca"]
	acc_values = [mean(task_scores.get(key, [])) if task_scores.get(key) else 0.0 for key in acc_order]

	lcs_vals = task_scores.get("lcs", [])
	lcs_avg = mean(lcs_vals) if lcs_vals else 0.0

	lar_vals = task_scores.get("lar", {}).get("rouge_l", [])
	lar_avg = mean(lar_vals) if lar_vals else 0.0

	header = "LAR Rouge-L  &  LCS MBE  &  " + " & ".join([t.upper() for t in acc_order]) + " Accuracy  &  AVG"

	values = (
		f"{lar_avg*100:.2f}  &  {lcs_avg*100:.2f}  &  "
		+ " & ".join([f"{v*100:.2f}" for v in acc_values])
		+ f"  &  {mean([lar_avg, lcs_avg] + acc_values) * 100:.2f}"
	)

	text = header + "\n & " + values + "\n"
	print(text)
	print("===========================================\n")

	overall = mean(acc_values + [lar_avg, lcs_avg])

	return text, overall


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


def call_api(prompt):
	api_key = _config.get("openai_api_key", "")
	base_url = _config.get("openai_api_base", "https://openrouter.ai/api/v1")
	url = f"{base_url}/chat/completions"
	headers = {
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json",
	}
	data = {
		"model": "qwen/qwen3-235b-a22b-2507",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt}
		],
		"temperature": 0.7
	}

	response = requests.post(url, headers=headers, json=data, timeout=60)

	if response.status_code == 200:
		result = response.json()
		return result["choices"][0]["message"]["content"]
	else:
		return f"Error {response.status_code}: {response.text}"


def evaluate_multithread(
	input_path: Path,
	output_path: Path,
	summary_path: Path,
	*,
	is_ood: bool = False,
) -> None:
	rows = read_jsonl(input_path)

	output_rows: List[Dict[str, object]] = []
	task_scores: Dict[str, object] = _initialise_task_scores(is_ood)

	lcs_jobs = []
	for idx, row in enumerate(rows):
		if not is_ood and str(row.get("data_source", "")).lower() == "lcs":
			question = str(row.get("prompt", ""))
			gt = str(row.get("reward_model", "").get("ground_truth", ""))
			pred = str(row.get("reward_model", "").get("answer", ""))
			lcs_jobs.append((idx, question, gt, pred))

	lcs_results = {}  # idx -> result dict

	if lcs_jobs:
		print(f"Running LCS in parallel... total {len(lcs_jobs)} items")
		with ThreadPoolExecutor(max_workers=16) as executor:
			future_map = {
				executor.submit(get_lcs_result, q, g, p): (idx, q, g, p)
				for (idx, q, g, p) in lcs_jobs
			}

			for future in tqdm(as_completed(future_map), total=len(future_map)):
				idx, q, g, p = future_map[future]
				try:
					ans_corr, basis_corr = future.result()
				except Exception as e:
					ans_corr, basis_corr = 0, 0
					print(f"LCS task {idx} failed: {e}")

				lcs_results[idx] = {
					"answer_correct": ans_corr,
					"basis_correct": basis_corr,
					"total": ans_corr + basis_corr
				}

	for idx, row in enumerate(rows):
		task_type = str(row.get("data_source", "")).lower()
		ground_truth = str(row.get("reward_model", "").get("ground_truth", ""))
		prediction = str(row.get("reward_model", "").get("answer", ""))

		if task_type == "lar" and not is_ood:
			metrics = score_lar_task(ground_truth, prediction)
			output_row = {**row, "score": metrics}
			for metric_name, value in metrics.items():
				task_scores[task_type][metric_name].append(value)

		elif task_type == "lcs" and not is_ood:
			result = lcs_results[idx]
			metrics = {
				"answer_correct": result["answer_correct"],
				"basis_correct": result["basis_correct"],
			}
			output_row = {**row, "score": metrics}
			task_scores["lcs"].append(metrics["answer_correct"] + metrics["basis_correct"])

		else:
			score = score_structured_task(task_type, ground_truth, prediction)
			output_row = {**row, "score": score}
			task_scores.setdefault(task_type, []).append(score)

		output_rows.append(output_row)

	write_jsonl(output_path, output_rows)

	summary_text, _ = summarise_scores(task_scores, is_ood=is_ood)
	latex_text, _ = print_latex_scores(task_scores, is_ood=is_ood)
	summary_path.write_text(summary_text + "\n" + latex_text, encoding="utf-8")


def evaluate(
	input_path: Path,
	output_path: Path,
	summary_path: Path,
	*,
	is_ood: bool = False,
) -> None:
	rows = read_jsonl(input_path)

	output_rows: List[Dict[str, object]] = []
	task_scores: Dict[str, object] = _initialise_task_scores(is_ood)

	for row in rows:
		task_type = str(row.get("data_source", "")).lower()
		if task_type not in SUPPORTED_TASKS:
			raise ValueError(f"Unsupported task type: {task_type}")

		ground_truth = str(row.get("reward_model", "").get("ground_truth", ""))
		prediction = str(row.get("reward_model", "").get("answer", ""))

		if task_type == "lar" and not is_ood:
			metrics = score_lar_task(ground_truth, prediction)
			output_row = {**row, "score": metrics}
			for metric_name, value in metrics.items():
				task_scores[task_type][metric_name].append(value)  # type: ignore[index]
		elif task_type == "lcs" and not is_ood:
			question = str(row.get("prompt", ""))
			answer_correct, basis_correct = get_lcs_result(question, ground_truth, prediction)
			metrics = {
				"answer_correct": answer_correct,
				"basis_correct": basis_correct,
			}
			output_row = {**row, "score": metrics}
			task_scores.setdefault(task_type, []).append(answer_correct + basis_correct)
		else:
			score = score_structured_task(task_type, ground_truth, prediction)
			output_row = {**row, "score": score}
			task_scores.setdefault(task_type, []).append(score)

		output_rows.append(output_row)

	write_jsonl(output_path, output_rows)

	summary_text, _ = summarise_scores(task_scores, is_ood=is_ood)
	summary_path.write_text(summary_text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate Legalsearch predictions")
	parser.add_argument("--input", type=Path, help="Path to the input predictions JSONL file")
	parser.add_argument("--ood", action="store_true", help="Use OOD scoring configuration")
	args = parser.parse_args()
	input_path = args.input
	if "results.jsonl" not in str(input_path):
		suffix = "_ood" if args.ood else ""
		output_path = str(input_path).replace(".jsonl", f"{suffix}_scores.jsonl")
		summary_path = str(input_path).replace(".jsonl", f"{suffix}_summary.txt")
	else:
		base = str(input_path).replace("results.jsonl", "")
		output_path = base + ("ood_scores.jsonl" if args.ood else "scores.jsonl")
		summary_path = base + ("ood_summary.txt" if args.ood else "summary.txt")
	parser.set_defaults(
		input=args.input,
		ood=args.ood,
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path(output_path),
		help="Path to write the scored JSONL output",
	)
	parser.add_argument(
		"--summary",
		type=Path,
		default=Path(summary_path),
		help="Path to write the textual summary",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.summary.parent.mkdir(parents=True, exist_ok=True)
	evaluate_multithread(args.input, args.output, args.summary, is_ood=args.ood)


if __name__ == "__main__":
	main()