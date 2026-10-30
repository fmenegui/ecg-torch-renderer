"""Resolve the published-code and article-table training protocols separately."""

from copy import deepcopy
from pathlib import Path

COMMIT = "97a15087d4abcda843da8c58ee74b1d8f47e6f9a"
WEIGHTS = {
    "unet_weights_07072025.pt": "17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b",
    "lead_name_unet_weights_07072025.pt": "840bd6bf2433ee6c22db67f57c861d9d427f29e10a32eeb334f0bcf061b175a2",
}


def training_config(upstream, profile, train_dir=None, validation_dir=None, device="cuda"):
    import yaml
    filename = "lead_name_unet.yml" if profile == "lead_text_public" else "unet.yml"
    config = yaml.safe_load((Path(upstream) / "src/config" / filename).read_text())
    config = deepcopy(config)
    if profile not in ["public_config", "paper_table4", "lead_text_public"]:
        raise ValueError("Unknown training protocol")
    if profile == "paper_table4":
        config["TRAIN"]["KWARGS"]["epochs"] = 45
        config["TRAIN"]["OPTIMIZER"]["KWARGS"]["weight_decay"] = 0.001
        config["TRAIN"]["LR_SCHEDULER"]["KWARGS"]["T_max"] = 20000
    if profile != "lead_text_public":
        if train_dir is None or validation_dir is None:
            raise ValueError("Training and original validation splits must be supplied")
        if Path(train_dir).resolve() == Path(validation_dir).resolve():
            raise ValueError("Training and validation directories must differ")
        for split, folder in [("TRAIN", train_dir), ("VAL", validation_dir), ("TEST", validation_dir)]:
            config["DATASET"][split]["KWARGS"]["data_path"] = str(Path(folder).resolve())
    config["TRAIN"]["KWARGS"]["device"] = device
    return config


def inference_config(upstream, preset, device, target_num_samples=None):
    import yaml
    if preset not in ["ahus", "george-moody-2024", "generic"]:
        raise ValueError("Use a named native preset")
    name = "inference_wrapper.yml" if preset == "generic" else f"inference_wrapper_{'ahus_testset' if preset == 'ahus' else preset}.yml"
    upstream = Path(upstream).resolve()
    config = yaml.safe_load((upstream / "src/config" / name).read_text())
    kwargs = config["MODEL"]["KWARGS"]
    kwargs["device"] = device
    components = kwargs["config"]
    components["LAYOUT_IDENTIFIER"]["KWARGS"]["device"] = device
    for node in components.values():
        for key in ["weight_path", "config_path", "unet_config_path", "unet_weight_path"]:
            if key in node:
                node[key] = str(upstream / node[key])
    if target_num_samples is not None:
        if not isinstance(target_num_samples, int) or target_num_samples <= 0:
            raise ValueError("An explicit positive public output sample count is required")
        components["LAYOUT_IDENTIFIER"]["KWARGS"]["target_num_samples"] = target_num_samples
    return config
