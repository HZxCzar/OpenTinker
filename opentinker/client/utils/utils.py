from verl import DataProto
import base64
import torch
import numpy as np
from typing import Dict, Any
from verl.trainer.main_ppo import create_rl_dataset
import os
from pathlib import Path
from omegaconf import DictConfig
import socket
import copy
import warnings
from collections.abc import Mapping

# ==================== Path Resolution ====================


def resolve_path(path: str, original_cwd: str = None) -> str:
    """Resolve a path to absolute path, handling both absolute and relative paths.

    This is necessary because Hydra changes the working directory to outputs/YYYY-MM-DD/HH-MM-SS/
    at runtime, causing relative paths to fail.

    Args:
        path: Path to resolve (can be absolute or relative)
        original_cwd: Original working directory (if None, attempts to get from Hydra)

    Returns:
        Absolute path
    """
    if not path:
        return path

    # If already absolute, return as-is
    if os.path.isabs(path):
        return path

    # Check if this looks like a HuggingFace model ID (namespace/model_name)
    # HuggingFace IDs: single slash, no common file extensions
    # Examples: "Qwen/Qwen2.5-3B", "meta-llama/Llama-2-7b-hf"
    if "/" in path:
        parts = path.split("/")
        # HuggingFace repo IDs have exactly 2 parts (namespace/model)
        # Check for common file extensions to exclude actual file paths
        file_extensions = (
            ".py",
            ".yaml",
            ".yml",
            ".json",
            ".parquet",
            ".txt",
            ".csv",
            ".bin",
            ".safetensors",
        )
        if len(parts) == 2 and not path.endswith(file_extensions):
            # Looks like a HuggingFace model ID, return as-is
            return path

    # Get original working directory
    if original_cwd is None:
        try:
            from hydra.utils import get_original_cwd

            original_cwd = get_original_cwd()
        except (ImportError, ValueError):
            # Fallback to current directory if not running under Hydra
            original_cwd = os.getcwd()

    # Resolve relative path from original working directory
    return str(Path(original_cwd) / path)


def resolve_paths_in_config(config: DictConfig, original_cwd: str = None) -> DictConfig:
    """Resolve common path fields in configuration to absolute paths.

    Args:
        config: Hydra configuration object
        original_cwd: Original working directory (if None, attempts to get from Hydra)

    Returns:
        Modified configuration with resolved paths
    """
    # Get original working directory once
    if original_cwd is None:
        try:
            from hydra.utils import get_original_cwd

            original_cwd = get_original_cwd()
        except (ImportError, ValueError):
            original_cwd = os.getcwd()

    # List of common path fields to resolve
    path_fields = [
        "data_path",
        "val_data_path",
        "tokenizer_path",
        "config_path",  # For reward function configs
        "checkpoint_path",
        "output_dir",
        "sweagent_config_path",
        "trajs_save_dir",
    ]

    # Resolve each path field if it exists
    for field in path_fields:
        if hasattr(config, field) and config[field] is not None:
            setattr(config, field, resolve_path(config[field], original_cwd))

    return config


# ==================== DataProto Serialization ====================


def serialize_dataproto(data: DataProto) -> Dict[str, Any]:
    """
    Serialize DataProto to JSON-compatible dict for HTTP transmission.

    Args:
        data: DataProto to serialize

    Returns:
        JSON-compatible dict
    """

    def serialize_tensor(t):
        """Serialize a single tensor to base64-encoded dict"""
        if isinstance(t, torch.Tensor):
            return {
                "__type__": "torch.Tensor",
                "__dtype__": str(t.dtype),
                "__shape__": list(t.shape),
                "__device__": str(t.device),
                "__data__": base64.b64encode(t.cpu().numpy().tobytes()).decode("utf-8"),
            }
        elif isinstance(t, np.ndarray):
            return {
                "__type__": "numpy.ndarray",
                "__dtype__": str(t.dtype),
                "__shape__": list(t.shape),
                "__data__": base64.b64encode(t.tobytes()).decode("utf-8"),
            }
        # Handle PIL Images for VL models
        elif hasattr(t, "save") and hasattr(t, "mode"):
            # This is a PIL Image
            import io

            buffer = io.BytesIO()
            # Convert to RGB if necessary (some formats like RGBA need conversion)
            if hasattr(t, "mode") and t.mode in ("RGBA", "P", "LA"):
                t = t.convert("RGB")
            t.save(buffer, format="PNG")
            return {
                "__type__": "PIL.Image",
                "__mode__": t.mode,
                "__size__": list(t.size),
                "__data__": base64.b64encode(buffer.getvalue()).decode("utf-8"),
            }
        return t

    def deep_serialize(obj):
        """Recursively serialize nested structures (dicts, lists) containing tensors/images."""
        if isinstance(obj, dict):
            return {k: deep_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_serialize(item) for item in obj]
        elif isinstance(obj, tuple):
            return [deep_serialize(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return serialize_tensor(obj)
        elif isinstance(obj, np.ndarray) and obj.dtype != object:
            return serialize_tensor(obj)
        elif hasattr(obj, "save") and hasattr(obj, "mode"):
            # PIL Image
            return serialize_tensor(obj)
        else:
            return obj

    # Serialize batch (TensorDict)
    serialized_batch = {}
    if data.batch is not None:
        for k, v in data.batch.items():
            serialized_batch[k] = serialize_tensor(v)

    # Serialize non_tensor_batch
    serialized_non_tensor = {}
    for k, v in data.non_tensor_batch.items():
        if isinstance(v, np.ndarray):
            # For object dtype arrays, convert to list (preserving structure)
            if v.dtype == object:
                # Recursively serialize any nested objects (dicts, lists, tensors, PIL Images)
                data_list = [deep_serialize(item) for item in v.flatten()]

                serialized_non_tensor[k] = {
                    "__type__": "numpy.ndarray",
                    "__dtype__": "object",
                    "__shape__": list(v.shape),
                    "__data__": data_list,
                }
            else:
                serialized_non_tensor[k] = serialize_tensor(v)
        else:
            serialized_non_tensor[k] = deep_serialize(v)

    return {
        "batch": serialized_batch,
        "non_tensor_batch": serialized_non_tensor,
        "meta_info": data.meta_info,
    }


def deserialize_dataproto(data_dict: Dict[str, Any]) -> DataProto:
    """
    Deserialize DataProto from JSON dict.

    Args:
        data_dict: Serialized DataProto dict

    Returns:
        DataProto instance
    """

    def deserialize_tensor(obj):
        """Deserialize a single tensor from base64-encoded dict"""
        if not isinstance(obj, dict) or "__type__" not in obj:
            return obj

        if obj["__type__"] == "torch.Tensor":
            dtype_str = obj["__dtype__"].replace("torch.", "")
            dtype = getattr(torch, dtype_str)
            shape = tuple(obj["__shape__"])
            data_bytes = base64.b64decode(obj["__data__"])
            array = np.frombuffer(
                data_bytes, dtype=np.dtype(str(dtype).replace("torch.", ""))
            ).copy()
            tensor = torch.from_numpy(array).reshape(shape)
            return tensor.to(dtype)

        elif obj["__type__"] == "numpy.ndarray":
            if obj["__dtype__"] == "object":
                # Reconstruct object array from list, recursively deserializing tensors
                data_list = obj["__data__"]
                deserialized_list = []
                for item in data_list:
                    # Recursively deserialize any nested tensors
                    deserialized_list.append(deserialize_tensor(item))

                shape = tuple(obj["__shape__"])
                array = np.array(deserialized_list, dtype=object).reshape(shape)
                return array
            else:
                dtype = np.dtype(obj["__dtype__"])
                shape = tuple(obj["__shape__"])
                data_bytes = base64.b64decode(obj["__data__"])
                array = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
                return array

        # Handle PIL Images for VL models
        elif obj["__type__"] == "PIL.Image":
            from PIL import Image
            import io

            data_bytes = base64.b64decode(obj["__data__"])
            buffer = io.BytesIO(data_bytes)
            return Image.open(buffer).copy()  # .copy() to detach from buffer

        return obj

    def deep_deserialize(obj):
        """Recursively deserialize nested structures (dicts, lists) containing serialized tensors/images."""
        if isinstance(obj, dict):
            if "__type__" in obj:
                # This is a serialized object (Tensor, ndarray, or PIL.Image)
                return deserialize_tensor(obj)
            else:
                # Regular dict, recursively deserialize values
                return {k: deep_deserialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_deserialize(item) for item in obj]
        else:
            return obj

    # Deserialize batch
    tensors = {}
    if "batch" in data_dict and data_dict["batch"]:
        for k, v in data_dict["batch"].items():
            tensors[k] = deserialize_tensor(v)

    # Deserialize non_tensor_batch
    non_tensors = {}
    if "non_tensor_batch" in data_dict:
        for k, v in data_dict["non_tensor_batch"].items():
            non_tensors[k] = deep_deserialize(v)

    # Get meta_info
    meta_info = data_dict.get("meta_info", {})

    return DataProto.from_dict(
        tensors=tensors, non_tensors=non_tensors, meta_info=meta_info
    )


def prepare_dataset(data_paths, data_config, tokenizer, is_train=True, max_samples=-1):
    dataset = create_rl_dataset(
        data_paths=data_paths,
        data_config=data_config,
        tokenizer=tokenizer,
        processor=None,
        is_train=is_train,
        max_samples=max_samples,
    )

    return dataset


def verify_raw_prompt_format(batch_dict):
    if "raw_prompt" not in batch_dict:
        raise ValueError(
            "Dataset must include 'raw_prompt' field for agent_loop mode.\n"
            "Make sure data.return_raw_chat=True in server config."
        )

    # Check format of first sample
    raw_prompt = batch_dict["raw_prompt"][0]
    if not isinstance(raw_prompt, (list, np.ndarray)):
        raise ValueError(f"raw_prompt must be list of messages, got {type(raw_prompt)}")

    # If numpy array, convert to list for checking
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()

    if not all(
        isinstance(msg, dict) and "role" in msg and "content" in msg
        for msg in raw_prompt
    ):
        raise ValueError(
            "raw_prompt messages must have 'role' and 'content' fields.\n"
            f"Got: {raw_prompt[0] if raw_prompt else 'empty'}"
        )

    print(
        f"✓ Batch format verified: raw_prompt field present with {len(batch_dict['raw_prompt'])} samples"
    )


def math_reward_function(data_source, solution_str, ground_truth, extra_info, **kwargs):
    """Custom reward function example.

    This is a simple example that gives 1.0 for correct answers, 0.0 otherwise.
    In practice, you would implement more sophisticated logic.
    """
    from verl.utils.reward_score import default_compute_score

    score = default_compute_score(data_source, solution_str, ground_truth, extra_info)
    return score


def find_free_port(host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))  # 端口 0 表示让 OS 分配
        return s.getsockname()[1]


def tolerant_merge_dicts(*dicts):
    """
    Recursively merge multiple dict-like objects.
    - Same key + both values are dict. recursively merge
    - Same key + type conflict. keep latter, emit warning
    """
    result = {}

    for d in dicts:
        if d is None:
            continue
        for k, v in d.items():
            if k not in result:
                result[k] = copy.deepcopy(v)
            else:
                old = result[k]
                if isinstance(old, Mapping) and isinstance(v, Mapping):
                    result[k] = tolerant_merge_dicts(old, v)
                else:
                    warnings.warn(
                        f"Config key '{k}' overridden. "
                        f"Old type={type(old)}, new type={type(v)}",
                        RuntimeWarning,
                    )
                    result[k] = copy.deepcopy(v)
    return result


if __name__ == "__main__":
    # test serialize and deserialize
    data = DataProto.from_dict(
        tensors={"a": torch.tensor([1, 2, 3])}, non_tensors={"b": np.array([4, 5, 6])}
    )
    data_dict = serialize_dataproto(data)
    data = deserialize_dataproto(data_dict)
    print(data)
