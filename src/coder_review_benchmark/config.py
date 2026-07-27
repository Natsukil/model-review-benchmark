from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries without overriding exported variables."""
    env_path = path or ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def expand_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
    return pattern.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), value)


@dataclass(frozen=True)
class ModelProfile:
    id: str
    model_name: str
    base_url: str
    api_key: str
    parser: str
    max_output_tokens: int
    max_context_tokens: int
    max_concurrency: int
    send_auth: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    stream: bool = False
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    structured_output: bool = True


def get_model_profile(model_id: str, path: Path | None = None) -> ModelProfile:
    load_dotenv()
    data = load_yaml(path or ROOT / "configs" / "models.yaml")
    raw = data.get("models", {}).get(model_id)
    if not raw:
        raise KeyError(f"Unknown model profile: {model_id}")
    raw = {k: expand_env(v) for k, v in raw.items()}
    base_url = os.getenv(raw["base_url_env"], "").rstrip("/")
    if not base_url:
        raise RuntimeError(f"Set {raw['base_url_env']} before calling {model_id}")
    return ModelProfile(
        id=model_id,
        model_name=raw["model_name"],
        base_url=base_url,
        api_key=os.getenv(raw["api_key_env"], "dummy-key"),
        parser=raw.get("parser", "native_tool_calls"),
        max_output_tokens=int(raw.get("max_output_tokens", 4096)),
        max_context_tokens=int(raw.get("max_context_tokens", 32768)),
        max_concurrency=int(raw.get("max_concurrency", 1)),
        send_auth=raw.get("send_auth", True) is not False,
        temperature=float(raw.get("temperature", 0.0)),
        top_p=float(raw.get("top_p", 1.0)),
        seed=int(raw.get("seed", 42)),
        stream=bool(raw.get("stream", False)),
        repeat_penalty=float(raw.get("repeat_penalty", 1.0)),
        presence_penalty=float(raw.get("presence_penalty", 0.0)),
        frequency_penalty=float(raw.get("frequency_penalty", 0.0)),
        structured_output=(os.getenv("CBM_STRUCTURED_OUTPUT", "").lower() not in {"0", "false", "no"}) if os.getenv("CBM_STRUCTURED_OUTPUT") is not None else bool(raw.get("structured_output", True)),
    )


def suite_config(name: str = "balanced", path: Path | None = None) -> dict[str, Any]:
    data = load_yaml(path or ROOT / "configs" / "suites.yaml")
    if name not in data.get("suites", {}):
        raise KeyError(f"Unknown suite: {name}")
    return data["suites"][name]
