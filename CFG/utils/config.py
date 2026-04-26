import yaml


def merge_configs(global_config, specific_config):
    """
    Deep-merge a dataset-specific config into the global config.
    
    Dataset-specific values override global values. Nested dicts
    are merged recursively (one level deep).
    """
    merged = global_config.copy()
    for k, v in specific_config.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k].update(v)
        else:
            merged[k] = v
    return merged


def load_config(global_path, specific_path):
    """Load and merge global + dataset-specific config files."""
    with open(global_path, 'r') as f:
        global_cfg = yaml.safe_load(f)
    with open(specific_path, 'r') as f:
        specific_cfg = yaml.safe_load(f)
    return merge_configs(global_cfg, specific_cfg)
