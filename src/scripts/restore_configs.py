import json
from pathlib import Path

configs = [
    ("florence-vl-3b.json", "jiuhai/florence-vl-3b-sft"),
    ("florence-vl-8b.json", "jiuhai/florence-vl-8b-sft"),
    ("deepseek-vl2-tiny.json", "deepseek-ai/deepseek-vl2-tiny"),
    ("deepseek-vl2-small.json", "deepseek-ai/deepseek-vl2-small"),
    ("deepseek-vl2.json", "deepseek-ai/deepseek-vl2"),
    ("internvl2-1b.json", "OpenGVLab/InternVL2-1B"),
    ("internvl2-8b.json", "OpenGVLab/InternVL2-8B"),
    ("internvl2_5-26b.json", "OpenGVLab/InternVL2_5-26B"),
    ("ovis2.5-2b.json", "AIDC-AI/Ovis2.5-2B"),
    ("ovis2.5-9b.json", "AIDC-AI/Ovis2.5-9B"),
]

output_dir = Path("configs/remote")
output_dir.mkdir(parents=True, exist_ok=True)

for filename, model_id in configs:
    data = {
        "model_id": model_id,
        "dtype": "bfloat16",
        "device": "auto",
        "max_new_tokens": 32,
        "batch_size": 1,
        "trust_remote_code": True,
        "enabled": True
    }

    with open(output_dir / filename, "w") as f:
        json.dump(data, f, indent=2)
        print(f"Restored {filename}")
