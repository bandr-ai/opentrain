"""Datasets: the general way to tell OpenTrain what a model should do.

A dataset is a set of examples, each a conversation whose last assistant turn is the reference
answer (it may be missing: a prompt-only row is labelled by a teacher later). It comes from
anywhere: a file you have (JSONL, JSON, CSV), a Hugging Face dataset, rows the agent writes or
generates from your description, or traces of an agent at work (coding sessions are one kind and
have their own, richer pipeline in ``opentrain.mine``).

On disk::

    .opentrain/datasets/<name>/meta.json     name, description, source, grader, counts, system prompt
    .opentrain/datasets/<name>/train.jsonl   the rows a trainer may see
    .opentrain/heldout/datasets/<name>.jsonl the held-out rows (sealed: only the evaluator reads them)

Every row has a stable ``id`` (a hash of its content unless given), so the split is deterministic
and the provenance check can prove that no held-out row reached the training data.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
PROMPT_KEYS = ("prompt", "input", "question", "instruction", "query", "text", "source", "problem", "user",
               "message", "ticket", "email", "content", "sentence", "document", "request", "body")
ANSWER_KEYS = ("completion", "output", "answer", "response", "target", "label", "solution", "chosen",
               "assistant", "reference", "category", "class", "reply", "summary", "translation", "result",
               "expected", "ground_truth")
SYSTEM_KEYS = ("system", "system_prompt")
MAX_ROWS = 200_000
HF_ROWS = "https://datasets-server.huggingface.co"


class DatasetError(ValueError):
    pass


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list) and all(isinstance(x, dict) and "text" in x for x in v):
        return "".join(str(x["text"]) for x in v)
    return json.dumps(v, ensure_ascii=False)


class _Row(dict):
    """A source row for templates, with ready-made fields for multiple choice: ``<list or dict>_lettered``
    ("A. ...\nB. ...") and ``<small int>_letter`` (0 -> "A")."""

    def __missing__(self, key: str) -> Any:
        base, _, kind = key.rpartition("_")
        v = dict.get(self, base)
        if kind == "lettered" and isinstance(v, (list, dict)):
            items = list(v.values()) if isinstance(v, dict) else v
            keys = list(v.keys()) if isinstance(v, dict) and all(len(str(k)) == 1 for k in v) else \
                [chr(65 + i) for i in range(len(items))]
            return "\n".join(f"{k}. {_text(x)}" for k, x in zip(keys, items))
        if kind == "letter" and isinstance(v, int) and 0 <= v < 26:
            return chr(65 + v)
        if kind == "letter" and isinstance(v, str) and v.isdigit() and int(v) < 26:
            return chr(65 + int(v))
        raise DatasetError(f"no column {key!r} for the template; have: {', '.join(sorted(self)[:15])}")


def apply_template(raw: dict[str, Any], input_template: str | None, output_template: str | None) -> dict[str, Any]:
    """Build the prompt and answer from a row's columns: ``"{question}\n\n{options_lettered}"``,
    ``"{answer_idx}"``, ``"{cop_letter}"``, ``"{choices[0]}"``. Columns only, no attributes."""
    out = dict(raw)
    for key, tpl in (("prompt", input_template), ("completion", output_template)):
        if not tpl:
            continue
        if "__" in tpl or re.search(r"\{[^}]*\.[A-Za-z_]", tpl):
            raise DatasetError("templates may name columns and indexes ({col}, {col[0]}, {col[key]}), not attributes")
        try:
            out[key] = tpl.format_map(_Row(raw))
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise DatasetError(f"template: {exc}") from None
    return out


def normalize(raw: dict[str, Any], *, input_key: str | None = None, output_key: str | None = None,
              system: str | None = None) -> dict[str, Any]:
    """One source row as ``{id, messages, meta}``. Raises DatasetError when no prompt can be found."""
    if not isinstance(raw, dict):
        raise DatasetError("a row must be an object")
    msgs: list[dict[str, str]] = []
    ik = ok = None
    convo = raw.get("messages") or raw.get("conversations") or raw.get("conversation")
    if isinstance(convo, list) and convo and not input_key:
        for m in convo:
            if not isinstance(m, dict):
                raise DatasetError("messages must be objects with role and content")
            role = m.get("role") or {"human": "user", "gpt": "assistant", "system": "system"}.get(m.get("from", ""))
            if role not in ("system", "user", "assistant"):
                continue  # tool turns are not part of a plain chat example
            msgs.append({"role": role, "content": _text(m.get("content", m.get("value")))})
    else:
        if not input_key and "rejected" in raw and not any(isinstance(raw.get(k), str) for k in PROMPT_KEYS):
            raise DatasetError("this is a preference dataset (chosen/rejected transcripts), not prompts with answers; "
                               "pick a dataset with prompt and answer columns, or name them with input_key/output_key")
        ik = input_key or _pick(raw, PROMPT_KEYS)
        ok = output_key or _pick(raw, ANSWER_KEYS, exclude=ik)
        if ik is None:  # unfamiliar names: with one or two text columns the meaning is still clear
            cols = [k for k, v in raw.items() if isinstance(v, str) and k != "id" and k not in SYSTEM_KEYS]
            if 1 <= len(cols) <= 2 and (ok is None or ok in cols):
                rest = [c for c in cols if c != ok]
                ik = rest[0] if rest else None
                if ok is None and len(cols) == 2:
                    ok = cols[1]
        if ik is None or raw.get(ik) in (None, ""):
            raise DatasetError(f"no prompt column among {', '.join(sorted(raw)[:12])}: say which one with input_key")
        sk = next((k for k in SYSTEM_KEYS if raw.get(k)), None)
        if sk:
            msgs.append({"role": "system", "content": _text(raw[sk])})
        prompt = _text(raw[ik])
        if not input_key and ik == "input" and isinstance(raw.get("instruction"), str) and raw["instruction"].strip():
            prompt = f"{raw['instruction'].strip()}\n\n{prompt}"  # Alpaca: the instruction, then its input
        msgs.append({"role": "user", "content": prompt})
        if ok and raw.get(ok) not in (None, ""):
            msgs.append({"role": "assistant", "content": _text(raw[ok])})
    if system and not any(m["role"] == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": system})
    if not any(m["role"] == "user" for m in msgs):
        raise DatasetError("a row needs at least one user turn")
    used = {"messages", "conversations", "conversation", "id", ik, ok, *PROMPT_KEYS, *ANSWER_KEYS, *SYSTEM_KEYS}
    meta = {k: v for k, v in raw.items() if k not in used and isinstance(v, (str, int, float, bool))}
    rid = str(raw.get("id") or "")[:80] or _sha(msgs)[:16]
    return {"id": rid, "messages": msgs, **({"meta": meta} if meta else {})}


def with_system(row: dict[str, Any], system: str | None) -> dict[str, Any]:
    """The row with the dataset's system prompt in front, unless the row has its own."""
    if not system or any(m["role"] == "system" for m in row["messages"]):
        return row
    return {**row, "messages": [{"role": "system", "content": system}, *row["messages"]]}


def _pick(raw: dict[str, Any], keys: tuple[str, ...], exclude: str | None = None) -> str | None:
    lower = {k.lower(): k for k in raw}
    for k in keys:
        real = lower.get(k)
        if real and real != exclude and raw.get(real) not in (None, ""):
            return real
    return None


def prompt_of(row: dict[str, Any]) -> list[dict[str, str]]:
    """The conversation up to (not including) the reference answer."""
    msgs = row["messages"]
    return msgs[:-1] if msgs and msgs[-1]["role"] == "assistant" else list(msgs)


def reference_of(row: dict[str, Any]) -> str | None:
    msgs = row["messages"]
    return msgs[-1]["content"] if msgs and msgs[-1]["role"] == "assistant" else None


def prompt_text(row: dict[str, Any]) -> str:
    return "\n\n".join(m["content"] for m in prompt_of(row) if m["role"] != "system")


# ---------------------------------------------------------------------- reading sources


def read_file(path: Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    if not path.exists():
        raise DatasetError(f"no such file: {path}")
    return parse_bytes(path.read_bytes(), path.name)


def parse_bytes(data: bytes, filename: str) -> list[dict[str, Any]]:
    text = data.decode("utf-8-sig", errors="replace")
    name = filename.lower()
    if name.endswith(".csv") or name.endswith(".tsv"):
        reader = csv.DictReader(io.StringIO(text), delimiter="\t" if name.endswith(".tsv") else ",")
        return [dict(r) for r in reader]
    stripped = text.lstrip()
    if stripped.startswith("["):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise DatasetError("a JSON file must hold a list of rows")
        return rows
    if stripped.startswith("{") and not name.endswith((".jsonl", ".ndjson")):
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            for k in ("rows", "data", "examples", "train"):
                if isinstance(obj.get(k), list):
                    return obj[k]
            return [obj]
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise DatasetError(f"line {n} is not JSON: {exc}") from None
    return rows


def _get_json(url: str, timeout: float = 30) -> Any:
    """GET from the Hugging Face Hub, signed with the person's token when they connected one (gated
    and private datasets need it); errors become a DatasetError that says what to do."""
    headers = {"User-Agent": "opentrain"}
    hf = (urllib.parse.urlparse(url).hostname or "").endswith("huggingface.co")
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")) if hf else None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    site = "Hugging Face" if hf else "Kaggle"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:  # noqa: S310
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404) and hf:
            hint = ("connect Hugging Face (/connect huggingface) and accept the dataset's terms on its page"
                    if not token else "accept the dataset's terms on its Hugging Face page")
            raise DatasetError(f"Hugging Face cannot show this dataset (HTTP {exc.code}): check the name, or if it "
                               f"is private or gated, {hint}") from None
        raise DatasetError(f"{site} answered HTTP {exc.code}; check the name, or try again in a minute") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DatasetError(f"could not reach {site}: {getattr(exc, 'reason', exc)}") from None


def read_hf(dataset: str, *, split: str | None = None, config: str | None = None, limit: int = 2000,
            fetch=_get_json) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows of a public Hugging Face dataset through the datasets server (no extra install)."""
    q = urllib.parse.quote
    dataset = dataset.strip().strip("/")
    splits = fetch(f"{HF_ROWS}/splits?dataset={q(dataset)}").get("splits") or []
    if not splits:
        raise DatasetError(f"Hugging Face has no viewable splits for {dataset!r} (private, gated or not converted)")
    pick = [s for s in splits if (config is None or s["config"] == config) and (split is None or s["split"] == split)]
    if not pick:
        avail = ", ".join(sorted({f"{s['config']}/{s['split']}" for s in splits}))
        raise DatasetError(f"no split {config or '*'}/{split or '*'} in {dataset}; available: {avail}")
    s = next((x for x in pick if x["split"] == "train"), pick[0])
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < limit:
        n = min(100, limit - len(rows))
        page = fetch(f"{HF_ROWS}/rows?dataset={q(dataset)}&config={q(s['config'])}&split={q(s['split'])}"
                     f"&offset={offset}&length={n}")
        got = [r["row"] for r in page.get("rows") or []]
        rows += got
        offset += len(got)
        if len(got) < n or offset >= (page.get("num_rows_total") or 0):
            break
    return rows, {"hf": dataset, "config": s["config"], "split": s["split"]}


HF_API = "https://huggingface.co/api"


def _tag(tags: list[str], prefix: str) -> list[str]:
    return [t.split(":", 1)[1] for t in tags if t.startswith(prefix + ":")]


def search_hf(query: str, limit: int = 8, fetch=_get_json) -> list[dict[str, Any]]:
    """Public Hugging Face datasets matching ``query``, most downloaded first."""
    q = urllib.parse.quote
    found = fetch(f"{HF_API}/datasets?search={q(query)}&sort=downloads&direction=-1&limit={int(limit) * 2}")
    out = []
    for d in found or []:
        if d.get("private") or d.get("disabled"):
            continue
        tags = d.get("tags") or []
        out.append({"source": "hf", "id": d["id"], "downloads": d.get("downloads", 0), "likes": d.get("likes", 0),
                    "gated": bool(d.get("gated")), "size": (_tag(tags, "size_categories") or [""])[0],
                    "license": (_tag(tags, "license") or [""])[0], "tasks": _tag(tags, "task_categories")[:4],
                    "languages": _tag(tags, "language")[:4],
                    "about": " ".join((d.get("description") or "").split())[:240]})
    return out[: int(limit)]


def search_kaggle(query: str, limit: int = 8, fetch=_get_json) -> list[dict[str, Any]]:
    """Public Kaggle datasets matching ``query``, most voted first (no key needed to search)."""
    found = fetch(f"{KAGGLE}/datasets/list?search={urllib.parse.quote(query)}&sortBy=votes")
    out = []
    for d in (found or [])[: int(limit)]:
        out.append({"source": "kaggle", "id": d.get("ref"), "title": d.get("title") or d.get("titleNullable"),
                    "downloads": d.get("downloadCount", 0), "votes": d.get("voteCount", 0),
                    "bytes": d.get("totalBytes") or d.get("totalBytesNullable"),
                    "license": d.get("licenseName") or d.get("licenseNameNullable") or "",
                    "usability": round(float(d.get("usabilityRating") or d.get("usabilityRatingNullable") or 0), 2),
                    "about": " ".join((d.get("subtitle") or d.get("subtitleNullable") or "").split())[:240]})
    return out


def _dtype(feature: Any) -> str:
    """A column's type as Hugging Face describes it (a list feature is a JSON list, not a dict)."""
    if isinstance(feature, list):
        return f"list[{_dtype(feature[0])}]" if feature else "list"
    if isinstance(feature, dict):
        if feature.get("dtype") or feature.get("_type") not in (None, "Sequence", "List", "LargeList"):
            return str(feature.get("dtype") or feature.get("_type"))
        inner = feature.get("feature")
        return f"list[{_dtype(inner)}]" if inner is not None else "struct"
    return str(feature)


def peek_hf(dataset: str, fetch=_get_json) -> dict[str, Any]:
    """Before importing: configs and splits with row counts, the columns, and three example rows."""
    q = urllib.parse.quote
    info = (fetch(f"{HF_ROWS}/info?dataset={q(dataset)}") or {}).get("dataset_info") or {}
    if not info:
        raise DatasetError(f"Hugging Face shows nothing for {dataset!r} (private, gated or not converted)")
    configs = {name: {"splits": {k: v.get("num_examples") for k, v in (c.get("splits") or {}).items()},
                      "columns": {k: _dtype(v) for k, v in (c.get("features") or {}).items()}}
               for name, c in list(info.items())[:8]}
    first = next(iter(configs))
    split = "train" if "train" in configs[first]["splits"] else next(iter(configs[first]["splits"]), "train")
    rows = fetch(f"{HF_ROWS}/first-rows?dataset={q(dataset)}&config={q(first)}&split={q(split)}") or {}
    sample = [{k: (str(v)[:400] if not isinstance(v, (int, float, bool)) else v) for k, v in r["row"].items()}
              for r in (rows.get("rows") or [])[:3]]
    return {"dataset": dataset, "configs": configs, "sample": {"config": first, "split": split, "rows": sample}}


KAGGLE = "https://www.kaggle.com/api/v1"
TABULAR = (".csv", ".tsv", ".jsonl", ".ndjson", ".json")


def kaggle_auth() -> dict[str, str]:
    """Kaggle's API auth: a token (``KAGGLE_API_TOKEN``, Bearer) or ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``
    (Basic), as Kaggle's own client reads them. ``/connect kaggle`` stores ``username:key`` or a token."""
    import base64
    import os

    token = os.environ.get("KAGGLE_API_TOKEN")
    key = os.environ.get("KAGGLE_KEY") or ""
    if not token and ":" in key and not os.environ.get("KAGGLE_USERNAME"):
        user, key = key.split(":", 1)
    else:
        user = os.environ.get("KAGGLE_USERNAME") or ""
    if token or (key and not user):
        return {"Authorization": f"Bearer {token or key}"}
    if user and key:
        return {"Authorization": "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()}
    raise DatasetError("Kaggle needs a key: /connect kaggle (username:key from kaggle.com/settings, or an API token)")


def read_kaggle(dataset: str, *, file: str | None = None, limit: int = 2000,
                fetch=None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows of a Kaggle dataset (``owner/name``): its zip is downloaded and one tabular file read (the
    one named, else the largest CSV/JSON/JSONL)."""
    import zipfile

    if not re.fullmatch(r"[\w.-]+/[\w.-]+", dataset or ""):
        raise DatasetError(f"{dataset!r} is not a Kaggle dataset id (owner/name)")
    if fetch is None:
        def fetch(url):
            req = urllib.request.Request(url, headers={"User-Agent": "opentrain", **kaggle_auth()})
            with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310 - fixed https host
                return r.read()
    try:
        blob = fetch(f"{KAGGLE}/datasets/download/{dataset}")
    except urllib.error.HTTPError as exc:
        raise DatasetError(f"Kaggle refused {dataset}: HTTP {exc.code} (private, a competition, or a bad key)") from None
    try:
        z = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        return parse_bytes(blob, file or "data.csv")[:limit], {"kaggle": dataset}
    files = [i for i in z.infolist() if i.filename.lower().endswith(TABULAR)]
    if file:
        files = [i for i in files if i.filename == file or i.filename.endswith("/" + file)]
    if not files:
        names = ", ".join(i.filename for i in z.infolist()[:20])
        raise DatasetError(f"no {'file ' + file if file else 'CSV/JSON/JSONL file'} in {dataset}; it has: {names}")
    pick = max(files, key=lambda i: i.file_size)
    rows = parse_bytes(z.read(pick), pick.filename)[:limit]
    return rows, {"kaggle": dataset, "file": pick.filename}


# ---------------------------------------------------------------------- the store


class Datasets:
    def __init__(self, state: Path):
        self.state = Path(state)
        self.root = self.state / "datasets"
        self.sealed = self.state / "heldout" / "datasets"

    def _dir(self, name: str) -> Path:
        if not NAME.match(name or ""):
            raise DatasetError("a dataset name is lowercase letters, digits, dot, dash or underscore")
        return self.root / name

    def names(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "meta.json").exists()) if self.root.exists() else []

    def exists(self, name: str) -> bool:
        return (self._dir(name) / "meta.json").exists()

    def meta(self, name: str) -> dict[str, Any]:
        p = self._dir(name) / "meta.json"
        if not p.exists():
            raise DatasetError(f"no dataset {name!r}" + (f"; have: {', '.join(self.names())}" if self.names() else ""))
        return json.loads(p.read_text())

    def list(self) -> list[dict[str, Any]]:
        return [self.meta(n) for n in self.names()]

    def _write_meta(self, name: str, meta: dict[str, Any]) -> dict[str, Any]:
        d = self._dir(name)
        d.mkdir(parents=True, exist_ok=True)
        meta["counts"] = {"train": _count(d / "train.jsonl"), "heldout": _count(self.sealed / f"{name}.jsonl")}
        meta["counts"]["labelled"] = sum(1 for r in self.rows(name) if reference_of(r) is not None)
        meta["updated"] = time.time()
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    def update_meta(self, name: str, **fields: Any) -> dict[str, Any]:
        m = self.meta(name)
        m.update({k: v for k, v in fields.items() if v is not None})
        return self._write_meta(name, m)

    def rows(self, name: str) -> list[dict[str, Any]]:
        """Training rows only; held-out rows are never returned here."""
        return _read(self._dir(name) / "train.jsonl")

    def heldout_ids(self, name: str) -> list[str]:
        return [r["id"] for r in _read(self.sealed / f"{name}.jsonl")]

    def heldout_rows(self, name: str) -> list[dict[str, Any]]:
        """For the evaluator only."""
        return _read(self.sealed / f"{name}.jsonl")

    def add(self, name: str, raw_rows: Iterable[dict[str, Any]], *, description: str | None = None,
            source: dict[str, Any] | None = None, input_key: str | None = None, output_key: str | None = None,
            system: str | None = None, heldout: float = 0.3, seed: int = 0) -> dict[str, Any]:
        """Normalize, de-duplicate and split rows into a dataset (created if new).

        New rows are split train/held-out by a seeded hash of their id, so the same row always
        lands on the same side. Rows already present (same id) are skipped.
        """
        d = self._dir(name)
        meta = self.meta(name) if self.exists(name) else {
            "name": name, "description": description or "", "created": time.time(), "sources": [],
            "grader": {"type": "judge"}, "system": system or ""}
        if description:
            meta["description"] = description
        seen = {r["id"] for r in self.rows(name)} | set(self.heldout_ids(name)) if self.exists(name) else set()
        train, held, errors, dup, skipped = [], [], [], 0, 0
        for i, raw in enumerate(raw_rows):
            if len(seen) + len(train) + len(held) >= MAX_ROWS:
                errors.append(f"stopped at {MAX_ROWS} rows")
                break
            try:
                row = normalize(raw, input_key=input_key, output_key=output_key, system=system or meta.get("system"))
            except (DatasetError, ValueError) as exc:
                msg = str(exc)
                if not any(e.endswith(msg) for e in errors) and len(errors) < 20:
                    errors.append(f"row {i + 1}: {msg}")
                skipped += 1
                continue
            if row["id"] in seen:
                dup += 1
                continue
            seen.add(row["id"])
            (held if _side(row["id"], seed) < heldout else train).append(row)
        if not train and not held and not self.exists(name):
            raise DatasetError("no usable rows" + (f" ({errors[0].split(': ', 1)[1]})" if errors else ""))
        d.mkdir(parents=True, exist_ok=True)
        self.sealed.mkdir(parents=True, exist_ok=True)
        _append(d / "train.jsonl", train)
        _append(self.sealed / f"{name}.jsonl", held)
        if source:
            meta.setdefault("sources", []).append({**source, "rows": len(train) + len(held), "time": time.time()})
        meta = self._write_meta(name, meta)
        return {"dataset": name, "added_train": len(train), "added_heldout": len(held), "duplicates": dup,
                "skipped": skipped, "errors": errors, "counts": meta["counts"]}

    def label(self, name: str, row_id: str, answer: str) -> None:
        """Set the reference answer of a training row (a teacher's or the user's)."""
        rows = self.rows(name)
        for r in rows:
            if r["id"] == row_id:
                if r["messages"][-1]["role"] == "assistant":
                    r["messages"][-1]["content"] = answer
                else:
                    r["messages"].append({"role": "assistant", "content": answer})
                break
        else:
            raise DatasetError(f"no training row {row_id} in {name}")
        _write(self._dir(name) / "train.jsonl", rows)
        self._write_meta(name, self.meta(name))

    def remove_rows(self, name: str, ids: list[str]) -> int:
        drop = set(ids)
        rows = self.rows(name)
        keep = [r for r in rows if r["id"] not in drop]
        _write(self._dir(name) / "train.jsonl", keep)
        held = self.heldout_rows(name)
        keep_h = [r for r in held if r["id"] not in drop]
        _write(self.sealed / f"{name}.jsonl", keep_h)
        self._write_meta(name, self.meta(name))
        return len(rows) - len(keep) + len(held) - len(keep_h)

    def delete(self, name: str) -> None:
        import shutil

        self.meta(name)
        shutil.rmtree(self._dir(name))
        (self.sealed / f"{name}.jsonl").unlink(missing_ok=True)

    def training_file(self, name: str, out: Path) -> dict[str, Any]:
        """Write the labelled training rows as chat examples; returns the manifest."""
        held = set(self.heldout_ids(name))
        system = self.meta(name).get("system")
        rows = [with_system(r, system) for r in self.rows(name) if reference_of(r) is not None and r["id"] not in held]
        if not rows:
            raise DatasetError(f"{name} has no labelled training rows (every row needs a reference answer)")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
        return {"dataset": name, "rows": [r["id"] for r in rows], "examples": len(rows), "tasks": [],
                "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def _side(row_id: str, seed: int) -> float:
    return int(hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _count(p: Path) -> int:
    return sum(1 for ln in p.read_text().splitlines() if ln.strip()) if p.exists() else 0


def _read(p: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def _append(p: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with open(p, "a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(p)


def sample(rows: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    if len(rows) <= n:
        return rows
    return random.Random(seed).sample(rows, n)
