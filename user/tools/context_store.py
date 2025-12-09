# user/tools/context_store.py
import json
from pathlib import Path
from filelock import FileLock

JSONL_FILE = Path("./user/cache/tool_context_log.jsonl")
LOCK_FILE = JSONL_FILE.with_suffix(".lock")

def append_action_info(instance_id: str, action_info_dict: dict):
    with FileLock(str(LOCK_FILE), timeout=10):
        log_entry = {
            "instance_id": instance_id,
            **action_info_dict
        }
        with open(JSONL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def load_latest_context(instance_id: str):
    with FileLock(str(LOCK_FILE), timeout=10):
        if not JSONL_FILE.exists():
            return None
        with open(JSONL_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                obj = json.loads(line)
                if obj.get("instance_id") == instance_id:
                    return obj
            except json.JSONDecodeError:
                continue
        return None