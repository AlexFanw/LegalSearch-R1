# run on 8xH100
# make sure your current working directory is the root of the project
set -x
ps aux | grep ray | grep -v grep | awk '{print $2}' | xargs -r kill -9
ulimit -n 65535

PROJECT_DIR="$(pwd)"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
rm $PROJECT_DIR/user/cache/tool_context_log.jsonl
rm $PROJECT_DIR/core.*
CONFIG_PATH="$PROJECT_DIR/user"

export WANDB_API_KEY="your_wandb_api_key"
export VLLM_USE_V1=1
export PYTHONPATH="$PROJECT_DIR/user/tools"
# export CUDA_VISIBLE_DEVICES=6,7 # if you need to limit GPU usage

timestamp=$(date +"%Y%m%d_%H%M%S")
timestamp="debug"
project_name="train-legalsearch-r1"
experiment_name="qwen25-7b-eas01-lr1e-6-${timestamp}"

python3 -m verl.trainer.main_ppo_user \
    --config-path="$CONFIG_PATH" \
    --config-name='deepresearch_grpo' \
    algorithm.adv_estimator=grpo \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=28671 \
    +data.max_model_len=32767 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_NAME" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=10000000 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side='left' \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/user/deepresearch_tool_config_rag.yaml" \
    trainer.val_only=True \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.validation_data_dir="$PROJECT_DIR/evaluate" \
    data.train_files=$PROJECT_DIR/data/legalsearch_rag_train.parquet \
    data.val_files=$PROJECT_DIR/data/legalsearch_rag_test.parquet \
    2>&1 | tee $PROJECT_DIR/log/evaluate_${timestamp}_${project_name}_${experiment_name}.log