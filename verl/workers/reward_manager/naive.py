from collections import defaultdict
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", max_workers=None) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            max_workers: Maximum number of threads for concurrent execution. If None, defaults to min(32, len(data) + 4).
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        # self.max_workers = max_workers
        self.max_workers = 32

    def _compute_single_score(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute score for a single data item.
        
        Args:
            args: Dictionary containing all arguments needed for compute_score
            
        Returns:
            Dictionary containing index, score, and metadata
        """
        index = args["index"]
        try:
            print(f"[DEBUG] start compute_score index={index}, data_source={args['data_source']}")
            score = self.compute_score(
                solution_str=args["response_str"],
                ground_truth=args["ground_truth"],
                val_type=args["val_type"],
                question=args["question"],
                data_source=args["data_source"],
            )
            print(f"[DEBUG] finished compute_score index={index}, score={score}")
            
            return {
                "index": args["index"],
                "score": score,
                "prompt_str": args["prompt_str"],
                "response_str": args["response_str"],
                "ground_truth": args["ground_truth"],
                "data_source": args["data_source"],
                "valid_response_length": args["valid_response_length"]
            }
        except Exception as e:
            print(f"Error computing score for index {args['index']}: {e}")
            # Return a default score in case of error
            return {
                "index": args["index"],
                "score": 0.0,
                "prompt_str": args["prompt_str"],
                "response_str": args["response_str"],
                "ground_truth": args["ground_truth"],
                "data_source": args["data_source"],
                "valid_response_length": args["valid_response_length"]
            }

    def __call__(self, data: DataProto, return_dict=False, val_type="legalsearch") -> Any:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Prepare arguments for concurrent execution
        compute_args = []
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            question = data_item.non_tensor_batch["reward_model"]["question"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            compute_args.append({
                "index": i,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "val_type": val_type,
                "question": question,
                "prompt_str": prompt_str,
                "data_source": data_source,
                "valid_response_length": valid_response_length,
                "extra_info": extra_info
            })

        # Execute compute_score concurrently
        max_workers = self.max_workers or min(32, len(data) + 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_args = {
                executor.submit(self._compute_single_score, args): args 
                for args in compute_args
            }
            
            # Collect results
            results = {}
            for future in as_completed(future_to_args):
                result = future.result()
                results[result["index"]] = result

        # Process results in original order
        already_print_data_sources = {}
        
        for i in range(len(data)):
            result = results[i]
            score = result["score"]
            
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, result["valid_response_length"] - 1] = reward

            # Print debug information
            data_source = result["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", result["prompt_str"])
                print("[response]", result["response_str"])
                print("[ground_truth]", result["ground_truth"])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor