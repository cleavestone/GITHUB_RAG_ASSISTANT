import yaml
from box import Box

def load_config(path: str = "config.yaml") -> Box:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Box(cfg)
