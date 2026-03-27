from __future__ import annotations

from typing import Any


class ConfigNode(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @classmethod
    def from_dict(cls, data: dict) -> "ConfigNode":
        node = cls()
        for key, value in data.items():
            node[key] = to_config_node(value)
        return node

    def to_dict(self) -> dict:
        out = {}
        for key, value in self.items():
            out[key] = from_config_node(value)
        return out


def to_config_node(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value
    if isinstance(value, dict):
        return ConfigNode.from_dict(value)
    if isinstance(value, list):
        return [to_config_node(item) for item in value]
    return value


def from_config_node(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value.to_dict()
    if isinstance(value, list):
        return [from_config_node(item) for item in value]
    return value
