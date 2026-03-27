import importlib
import inspect
import json
import os
from typing import Iterable, TYPE_CHECKING

from lightewm.utils.config import ConfigNode

if TYPE_CHECKING:
    from lightewm.model.wan.pipeline import WanVideoPipeline


def import_class(class_path: str):
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _section_params_as_config(section, full_config=None, section_name: str | None = None):
    params = section.get("params", {})
    params_dict = params.to_dict() if isinstance(params, ConfigNode) else dict(params)
    component_config = ConfigNode.from_dict(params_dict)
    component_config.class_path = section.class_path
    component_config.section_name = section_name
    component_config.full_config = full_config
    return component_config


def _should_init_with_config(cls):
    signature = inspect.signature(cls.__init__)
    params = [p for p in signature.parameters.values() if p.name != "self"]
    if not params:
        return False
    first = params[0]
    if first.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
        return False
    return first.name == "config" or len(params) == 1


def instantiate_component(class_path: str, component_config: ConfigNode):
    cls = import_class(class_path)
    if _should_init_with_config(cls):
        return cls(component_config)
    kwargs = {k: v for k, v in component_config.items() if k not in {"class_path", "section_name", "full_config"}}
    return cls(**kwargs)


def instantiate_component_from_section(section, full_config, section_name: str):
    component_config = _section_params_as_config(section, full_config=full_config, section_name=section_name)
    component = instantiate_component(section.class_path, component_config)
    return component, component_config


def flatten_config_params(cfg: dict, sections: Iterable[str] = ("dataset", "model", "runner", "runtime")) -> dict:
    params = {}
    for section in sections:
        node = cfg.get(section)
        if node is None:
            continue
        node_params = node.get("params")
        if node_params is None:
            continue
        for key, value in node_params.items():
            params[key] = value
    return params


def _normalize_model_paths(model_paths):
    if model_paths is None:
        return []
    if isinstance(model_paths, str):
        stripped = model_paths.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return [model_paths]
    if isinstance(model_paths, list):
        return model_paths
    return [str(model_paths)]


def resolve_local_wan_tokenizer_path(model_paths):
    paths = _normalize_model_paths(model_paths)
    for path in paths:
        root = os.path.dirname(path)
        candidate = os.path.join(root, "google", "umt5-xxl")
        if os.path.exists(os.path.join(candidate, "tokenizer.json")) and os.path.exists(os.path.join(candidate, "spiece.model")):
            return candidate
    return None
