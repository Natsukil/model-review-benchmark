from __future__ import annotations

import hashlib
import json
import random
import subprocess
import os
import re
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import requests

from .config import ROOT


SOURCES = {
    "codereviewqa": {"repo_id": "Tomo-Melb/CodeReviewQA", "repo_type": "dataset"},
    "swe_multilingual": {"repo_id": "SWE-bench/SWE-bench_Multilingual", "repo_type": "dataset"},
    "swe_verified": {"repo_id": "SWE-bench/SWE-bench_Verified", "repo_type": "dataset"},
    "swe_review": {"repo_id": "SWE-Lego/SWE-Review-Bench", "repo_type": "dataset"},
    "multi_swe_mini": {"repo_id": "ByteDance-Seed/Multi-SWE-bench_mini", "repo_type": "dataset"},
    "martian": {"repo_id": "withmartian/code-review-benchmark", "repo_type": "github_zip"},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_sources(raw_dir: Path | None = None) -> dict[str, Any]:
    raw_dir = raw_dir or ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"sources": {}, "blocked": {}, "generated_by": "coder-review-benchmark"}
    insecure = os.getenv("CBM_ALLOW_INSECURE_DOWNLOAD", "0") == "1"
    verify = not insecure
    hf_endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install optional data dependencies: pip install -e '.[data]'") from exc
    for name, spec in SOURCES.items():
        target = raw_dir / name
        target.mkdir(parents=True, exist_ok=True)
        if spec["repo_type"] == "github_zip":
            if not any(target.iterdir()):
                url = f"https://github.com/{spec['repo_id']}/archive/refs/heads/main.zip"
                response = requests.get(url, verify=verify, timeout=120)
                response.raise_for_status()
                with zipfile.ZipFile(BytesIO(response.content)) as archive:
                    archive.extractall(target)
            path = str(target)
        else:
            try:
                path = snapshot_download(repo_id=spec["repo_id"], repo_type=spec["repo_type"], local_dir=str(target))
            except Exception as exc:
                if not insecure:
                    raise RuntimeError("Hugging Face TLS verification failed. Install the internal CA or set CBM_ALLOW_INSECURE_DOWNLOAD=1 for a one-off local download.") from exc
                api = f"{hf_endpoint}/api/datasets/{spec['repo_id']}/tree/main?recursive=true"
                listing = requests.get(api, verify=False, timeout=60)
                listing.raise_for_status()
                headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.getenv("HF_TOKEN") else {}
                for item in listing.json():
                    if item.get("type") != "file" or item.get("size", 0) > 400_000_000 or item.get("path", "").startswith((".", "README")):
                        continue
                    file_path = target / item["path"]
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    url = f"{hf_endpoint}/datasets/{spec['repo_id']}/resolve/main/{item['path']}?download=true"
                    data = requests.get(url, headers=headers, verify=verify, timeout=180, stream=True)
                    if data.status_code == 401:
                        manifest["blocked"][name] = "Hugging Face dataset access requires HF_TOKEN (CodeReviewQA is currently restricted)."
                        break
                    data.raise_for_status()
                    data.raise_for_status()
                    with file_path.open("wb") as output:
                        for block in data.iter_content(1024 * 1024):
                            if block:
                                output.write(block)
                path = str(target)
        files = [p for p in target.rglob("*") if p.is_file()]
        manifest["sources"][name] = {"repo_id": spec["repo_id"], "snapshot_path": path, "files": [{"path": str(p.relative_to(ROOT)), "sha256": _sha256(p)} for p in files[:2000]]}
    out = ROOT / "data" / "manifest.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _records(root: Path) -> Iterable[dict[str, Any]]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.strip():
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            yield obj
            elif path.suffix == ".json":
                obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(obj, list):
                    yield from (x for x in obj if isinstance(x, dict))
                elif isinstance(obj, dict):
                    yield obj
            elif path.suffix == ".parquet":
                import pyarrow.parquet as pq
                for obj in pq.read_table(path).to_pylist():
                    yield obj
        except Exception:
            continue


def _lang(record: dict[str, Any]) -> str:
    for key in ("language", "lang", "programming_language"):
        value = record.get(key)
        if value:
            return (str(value).lower().replace("csharp", "c#").replace("cpp", "c++").replace("js", "javascript").replace("ts", "typescript"))
    text = " ".join(str(record.get(key, "")) for key in ("patch", "model_patch", "test_patch"))
    if re.search(r"diff --git a/[^ ]+\.(?:cc|cpp|cxx)(?: |$)", text):
        return "c++"
    suffixes = {".tsx": "typescript", ".ts": "typescript", ".jsx": "javascript", ".js": "javascript", ".java": "java", ".go": "go", ".rs": "rust", ".php": "php", ".rb": "ruby", ".c": "c", ".h": "c", ".py": "python"}
    counts = {language: len(re.findall(r"diff --git a/[^ ]+" + re.escape(suffix) + r"(?: |$)", text)) for suffix, language in suffixes.items()}
    if counts:
        best = max(counts, key=counts.get)
        if counts[best] > 0:
            return best
    return "unknown"


def select_records(raw_root: Path, count: int, languages: dict[str, int], seed: int = 20260724) -> list[dict[str, Any]]:
    records = list(_records(raw_root))
    rng = random.Random(seed)
    rng.shuffle(records)
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for language, quota in languages.items():
        candidates = [(i, r) for i, r in enumerate(records) if i not in used and (_lang(r) == language or language == "unknown")]
        for i, record in candidates[:quota]:
            used.add(i)
            selected.append({"language": _lang(record), "record": record})
    if len(selected) < count:
        for i, record in enumerate(records):
            if i not in used:
                selected.append({"language": _lang(record), "record": record})
                if len(selected) >= count:
                    break
    return selected[:count]


def prepare_profile(profile: str = "balanced", raw_dir: Path | None = None) -> Path:
    raw_dir = raw_dir or ROOT / "data" / "raw"
    required_dirs = ["codereviewqa", "swe_multilingual", "swe_verified", "swe_review", "multi_swe_mini", "martian"]
    if not all(any((raw_dir / name).rglob("*.parquet")) or any((raw_dir / name).rglob("*.json")) or any((raw_dir / name).rglob("*.jsonl")) for name in required_dirs):
        download_sources(raw_dir)
    martian_golden_dirs = list((raw_dir / "martian").glob("*/offline/golden_comments"))
    if not martian_golden_dirs:
        raise RuntimeError("Martian golden_comments directory is missing")
    martian_records = list(_records(martian_golden_dirs[0]))
    if not martian_records or any(not {"url", "comments"}.issubset(record) for record in martian_records):
        raise RuntimeError("Martian golden comments contain an invalid record")

    selections = {
        "agentic": select_records(raw_dir / "multi_swe_mini", 80, {"c": 10, "c++": 10, "java": 10, "javascript": 10, "typescript": 10, "python": 10, "go": 10, "rust": 10}, 20260724),
        "codereviewqa": select_records(raw_dir / "codereviewqa", 140, {"c": 20, "c++": 20, "c#": 20, "java": 20, "javascript": 20, "python": 20, "go": 20}, 20260724),
        "martian": select_records_from_records(martian_records, 20, {"unknown": 20}, 20260724),
        "swe_review": select_records(raw_dir / "swe_review", 10, {"unknown": 10}, 20260724),
    }
    out = ROOT / "data" / "selections" / profile
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in selections.items():
        (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return out


def select_records_from_records(records: list[dict[str, Any]], count: int, languages: dict[str, int], seed: int = 20260724) -> list[dict[str, Any]]:
    """Select records from an already-filtered source (used for Martian goldens)."""
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    for language, quota in languages.items():
        candidates = [r for r in shuffled if language == "unknown" or _lang(r) == language]
        selected.extend({"language": _lang(record), "record": record} for record in candidates[:quota])
    if len(selected) < count:
        used = {id(row["record"]) for row in selected}
        selected.extend({"language": _lang(record), "record": record} for record in shuffled if id(record) not in used)
    return selected[:count]


def multi_swe_image_name(record: dict[str, Any]) -> str:
    """Return the image name used by the official Multi-SWE-bench harness."""
    try:
        org = str(record["org"])
        repo = str(record["repo"])
        number = int(record["number"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Multi-SWE-bench record is missing org/repo/number") from exc
    return f"mswebench/{org}_m_{repo}:pr-{number}".lower()


def _local_docker_images() -> set[str]:
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("docker executable not found")
    probe = subprocess.run(
        [docker, "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"failed to list Docker images: {(probe.stderr or probe.stdout).strip()}")
    return {line.strip().lower() for line in probe.stdout.splitlines() if line.strip()}


def prepare_agentic_local_profile(
    profile: str = "docker-local",
    count: int = 80,
    raw_dir: Path | None = None,
    seed: int = 20260724,
    instance_ids: set[str] | None = None,
) -> Path:
    """Select only Multi-SWE-bench records whose official image exists locally."""
    if count <= 0:
        raise ValueError("count must be positive")
    raw_dir = raw_dir or ROOT / "data" / "raw"
    records = list(_records(raw_dir / "multi_swe_mini"))
    if not records:
        raise RuntimeError("Multi-SWE-bench mini data is missing; run prepare/download first")

    local_images = _local_docker_images()
    available = [record for record in records if multi_swe_image_name(record) in local_images]
    if instance_ids:
        available = [record for record in available if record.get("instance_id") in instance_ids]
    if not available:
        suffix = f" and requested instances {sorted(instance_ids)}" if instance_ids else ""
        raise RuntimeError(f"none of the local mswebench images match the mini dataset{suffix}")

    languages = ["c", "c++", "java", "javascript", "typescript", "python", "go", "rust"]
    quota = max(1, count // len(languages))
    selected = select_records_from_records(
        available,
        min(count, len(available)),
        {language: quota for language in languages},
        seed,
    )
    for row in selected:
        row["image"] = multi_swe_image_name(row["record"])

    out = ROOT / "data" / "selections" / profile
    out.mkdir(parents=True, exist_ok=True)
    selection_path = out / "agentic.jsonl"
    selection_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n",
        encoding="utf-8",
    )
    language_counts: dict[str, int] = {}
    for row in selected:
        language_counts[row["language"]] = language_counts.get(row["language"], 0) + 1
    (out / "agentic_manifest.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "source": "multi_swe_mini",
                "selection_policy": "local_docker_images_only",
                "requested_instance_ids": sorted(instance_ids) if instance_ids else None,
                "requested_count": count,
                "available_local_records": len(available),
                "selected_count": len(selected),
                "language_counts": language_counts,
                "seed": seed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def _round_robin_sample(
    groups: dict[tuple[Any, ...], list[dict[str, Any]]],
    count: int,
    seed: int,
    identity_key: str | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    queues = {key: list(values) for key, values in groups.items()}
    for values in queues.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    keys = sorted(queues, key=lambda key: tuple(str(part) for part in key))
    while len(selected) < count:
        progressed = False
        for key in keys:
            queue = queues[key]
            while queue:
                record = queue.pop()
                identity = str(record.get(identity_key, "")) if identity_key else ""
                if identity_key and identity in used:
                    continue
                if identity_key:
                    used.add(identity)
                selected.append(record)
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def _martian_repo(record: dict[str, Any]) -> str:
    match = re.search(r"github\.com/([^/]+/[^/]+)/pull/", str(record.get("url", "")))
    return match.group(1).lower() if match else "unknown"


def prepare_review_profile(
    profile: str = "review-hour",
    swe_review_count: int = 500,
    martian_count: int = 50,
    max_patch_chars: int = 40_000,
    raw_dir: Path | None = None,
    seed: int = 20260724,
) -> Path:
    """Create a deterministic review-first profile for cross-model comparison."""
    if swe_review_count <= 0 or martian_count <= 0 or max_patch_chars <= 0:
        raise ValueError("review counts and max_patch_chars must be positive")
    raw_dir = raw_dir or ROOT / "data" / "raw"

    swe_records = [
        record
        for record in _records(raw_dir / "swe_review")
        if 0 < len(str(record.get("model_patch") or "")) <= max_patch_chars
    ]
    swe_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in swe_records:
        key = (record.get("generator_model", "unknown"), bool(record.get("resolved")))
        swe_groups.setdefault(key, []).append(record)
    selected_swe = _round_robin_sample(
        swe_groups,
        min(swe_review_count, len(swe_records)),
        seed,
        identity_key="instance_id",
    )

    golden_dirs = list((raw_dir / "martian").glob("*/offline/golden_comments"))
    if not golden_dirs:
        raise RuntimeError("Martian golden_comments directory is missing")
    martian_records = list(_records(golden_dirs[0]))
    martian_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in martian_records:
        key = (_martian_repo(record),)
        martian_groups.setdefault(key, []).append(record)
    selected_martian = _round_robin_sample(
        martian_groups,
        min(martian_count, len(martian_records)),
        seed + 1,
        identity_key="url",
    )

    out = ROOT / "data" / "selections" / profile
    out.mkdir(parents=True, exist_ok=True)
    wrapped_swe = [{"language": _lang(record), "record": record} for record in selected_swe]
    wrapped_martian = [{"language": _lang(record), "record": record} for record in selected_martian]
    for name, rows in (("swe_review", wrapped_swe), ("martian", wrapped_martian)):
        (out / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    generator_counts: dict[str, int] = {}
    resolved_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for record in selected_swe:
        generator = str(record.get("generator_model", "unknown"))
        generator_counts[generator] = generator_counts.get(generator, 0) + 1
        resolved = str(bool(record.get("resolved"))).lower()
        resolved_counts[resolved] = resolved_counts.get(resolved, 0) + 1
        difficulty = str(record.get("difficulty", "unknown"))
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    repo_counts: dict[str, int] = {}
    for record in selected_martian:
        repo = _martian_repo(record)
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
    (out / "review_manifest.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "selection_policy": "review_first_stratified",
                "seed": seed,
                "max_patch_chars": max_patch_chars,
                "swe_review": {
                    "selected_count": len(selected_swe),
                    "unique_instances": len({record.get("instance_id") for record in selected_swe}),
                    "generator_counts": generator_counts,
                    "resolved_counts": resolved_counts,
                    "difficulty_counts": difficulty_counts,
                },
                "martian": {
                    "selected_count": len(selected_martian),
                    "golden_comment_count": sum(len(record.get("comments") or []) for record in selected_martian),
                    "repo_counts": repo_counts,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def _write_selection(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"language": _lang(r), "record": r}, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def _selection_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    generators: dict[str, int] = {}
    resolved: dict[str, int] = {}
    difficulty: dict[str, int] = {}
    for record in records:
        generator = str(record.get("generator_model", "unknown"))
        generators[generator] = generators.get(generator, 0) + 1
        label = str(bool(record.get("resolved"))).lower()
        resolved[label] = resolved.get(label, 0) + 1
        level = str(record.get("difficulty", "unknown"))
        difficulty[level] = difficulty.get(level, 0) + 1
    ids = [str(r.get("instance_id", "")) for r in records]
    serialized = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records]
    return {"selected_count": len(records), "unique_instance_ids": len(set(ids)), "duplicate_instance_ids": len(ids) - len(set(ids)), "exact_duplicate_records": len(serialized) - len(set(serialized)), "generator_counts": generators, "resolved_counts": resolved, "difficulty_counts": difficulty}


def prepare_swe_v2_profiles(raw_dir: Path | None = None, *, seed: int = 20260724, max_patch_chars: int = 40_000, official_count: int = 1384) -> dict[str, Path]:
    """Build the two frozen, review-only SWE selections from already downloaded raw data."""
    raw_dir = raw_dir or ROOT / "data" / "raw"
    all_records = [r for r in _records(raw_dir / "swe_review") if str(r.get("model_patch") or "")]
    # The official protocol keeps every non-empty candidate. The balanced set
    # uses the same source but excludes pathological oversized patches.
    records = [r for r in all_records if len(str(r.get("model_patch") or "")) <= max_patch_chars]
    if len(all_records) < official_count:
        raise RuntimeError(f"need {official_count} non-empty candidate patches, found {len(all_records)}")
    official = sorted(all_records, key=lambda r: (str(r.get("instance_id", "")), str(r.get("generator_model", "")), json.dumps(r, sort_keys=True, ensure_ascii=False)))[:official_count]
    by_id: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_id.setdefault(str(record.get("instance_id", "")), []).append(record)
    # A single instance can have candidates from multiple generators with
    # different outcomes. Allocate the unresolved half first, then choose the
    # remaining resolved instances deterministically.
    def choose(label: bool, count: int, excluded: set[str]) -> list[dict[str, Any]]:
        candidates: dict[str, list[dict[str, Any]]] = {}
        for instance_id, values in by_id.items():
            if instance_id in excluded:
                continue
            matching = [r for r in values if bool(r.get("resolved")) is label]
            if matching:
                candidates[instance_id] = matching
        ordered_values = sorted(candidates.values(), key=lambda values: (len({bool(r.get("resolved")) for r in by_id.get(str(values[0].get("instance_id")), [])}), str(values[0].get("instance_id", ""))))
        rng = random.Random(seed + (1 if label else 0))
        exclusive = [values for values in ordered_values if len({bool(r.get("resolved")) for r in by_id.get(str(values[0].get("instance_id")), [])}) == 1]
        mixed = [values for values in ordered_values if values not in exclusive]
        rng.shuffle(exclusive); rng.shuffle(mixed)
        selected_values = (exclusive + mixed)[:count]
        return [sorted(values, key=lambda r: str(r.get("generator_model", "unknown")))[0] for values in selected_values]
    unresolved = choose(False, 250, set())
    used = {str(r.get("instance_id")) for r in unresolved}
    resolved = choose(True, 250, used)
    balanced = unresolved + resolved
    if len(balanced) != 500 or sum(bool(r.get("resolved")) for r in balanced) != 250:
        raise RuntimeError("cannot construct balanced 500 selection with 250 resolved and 250 unresolved unique instances")
    base = ROOT / "data" / "selections"
    outputs: dict[str, Path] = {}
    for profile, selected, policy in (("swe-review-balanced-500-v1", balanced, "stratified_round_robin_unique_instance_id"), ("swe-review-official-1384-v1", official, "all_non_empty_candidate_patches_no_instance_dedupe")):
        out = base / profile
        _write_selection(out / "swe_review.jsonl", selected)
        manifest = {"evaluation_version": "model-review-v2", "profile": profile, "suite": "swe_review", "seed": seed, "max_patch_chars": max_patch_chars if profile.startswith("swe-review-balanced") else None, "selection_policy": policy, **_selection_stats(selected)}
        manifest["expected_count"] = len(selected)
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs[profile] = out
    return outputs


def validate_selection(profile: str, suite: str = "swe_review", *, root: Path | None = None) -> dict[str, Any]:
    root = root or ROOT
    directory = root / "data" / "selections" / profile
    selection_path = directory / f"{suite}.jsonl"
    manifest_path = directory / "manifest.json"
    if not selection_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"selection and manifest are required under {directory}")
    rows = [json.loads(line) for line in selection_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [row.get("record", row) if isinstance(row, dict) else {} for row in rows]
    errors: list[str] = []
    if any(not isinstance(record, dict) for record in records):
        errors.append("record must be an object")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if suite == "swe_review":
        if any(not str(record.get("instance_id", "")) for record in records): errors.append("missing instance_id")
        if any(not str(record.get("model_patch", "")) for record in records): errors.append("empty model_patch")
        expected = int(manifest.get("expected_count", len(records)))
        if len(records) != expected: errors.append(f"count {len(records)} != manifest {expected}")
        if str(profile).startswith("swe-review-balanced") and len({str(r.get("instance_id")) for r in records}) != len(records): errors.append("duplicate instance_id")
        if str(profile).startswith("swe-review-balanced") and manifest.get("resolved_counts") != {"true": 250, "false": 250}: errors.append("balanced resolved distribution is not 250/250")
        actual = _selection_stats(records)
        for key in ("generator_counts", "resolved_counts", "difficulty_counts"):
            if manifest.get(key) != actual.get(key):
                errors.append(f"{key} does not match manifest")
    serialized = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records]
    if len(serialized) != len(set(serialized)): errors.append("duplicate records")
    result = {"valid": not errors, "profile": profile, "suite": suite, "count": len(records), "unique_instance_ids": len({str(r.get("instance_id", "")) for r in records}), "errors": errors}
    return result
