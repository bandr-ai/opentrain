"""The four training recipes, run on one GPU (Modal, RunPod) or a CPU (tests): SFT, rejection
sampling, on-policy distillation and RL.

This file is shipped to the GPU as it stands, with ``graders.py`` next to it, so it imports only the
standard library at the top; torch, transformers and peft are imported inside, and vLLM only when it
is installed (it samples ~10x faster than ``generate``; without it, sampling falls back to
transformers, which is what the CPU tests use).

``run(spec, rows, out_dir, emit)`` trains a LoRA adapter and saves it to ``out_dir``. ``rows`` are
``{"prompt": [messages], "completion": [{"role": "assistant", "content": reference}]}``. ``emit``
receives ``{"event": "step", "step", "total", "loss", "reward"?, "seconds", "note"?}`` dicts.

The methods (``spec["method"]``), all on one LoRA adapter, one optimizer:

* ``sft``: cross-entropy on the reference answers (the completion tokens only).
* ``rejection`` (rejection-sampling fine-tuning, a.k.a. RFT / STaR): each round, sample ``k`` answers
  per prompt from the current model, grade them, keep the correct ones (at most 2 distinct per
  prompt) and fine-tune on those. The model learns from its own correct reasoning, not a reference's.
* ``opd`` (on-policy distillation): the student samples, a frozen teacher of the same tokenizer
  scores every sampled token, and the student is pushed toward the teacher on its own outputs:
  per-token advantage ``teacher_logprob - student_logprob`` (sampled reverse KL), as in
  Agarwal et al. 2024 (GKD) and Thinking Machines' "On-Policy Distillation" (2025).
* ``rl`` (GRPO): per prompt a group of ``k`` samples, reward from the grader, advantage = reward minus
  the group mean (no std division, Dr. GRPO, Liu et al. 2025), token-mean loss (DAPO, Yu et al. 2025),
  one optimizer step per batch, so the update is on-policy and needs no ratio clipping.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
import time
from typing import Any, Callable

METHODS = ("sft", "rejection", "opd", "rl")

# Defaults per method; spec["extra"] overrides any of them.
DEFAULTS: dict[str, dict[str, Any]] = {
    "sft": {},
    "rejection": {"rounds": 2, "samples": 8, "keep": 2, "temperature": 1.0, "max_new_tokens": 512},
    "opd": {"steps": 40, "prompts_per_step": 16, "samples": 2, "temperature": 1.0, "max_new_tokens": 512},
    "rl": {"steps": 40, "prompts_per_step": 16, "samples": 8, "temperature": 1.0, "max_new_tokens": 512},
}


def settings(spec: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULTS[spec.get("method", "sft")], **(spec.get("extra") or {})}


# ---------------------------------------------------------------------------- shared pieces


def render(tokenizer, messages: list[dict[str, Any]], *, generation: bool) -> list[int]:
    """Chat-template token ids. ``enable_thinking=False``: answers are trained and served without a
    thinking block (templates that do not know the flag ignore it)."""
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation,
                                        enable_thinking=False)
    if isinstance(out, dict) or hasattr(out, "input_ids"):  # transformers 5 may return a BatchEncoding
        out = out["input_ids"]
    return list(out)


def completion_ids(tokenizer, prompt: list[dict[str, Any]], answer: str) -> tuple[list[int], list[int]]:
    """(prompt ids, answer ids) where prompt + answer is exactly the rendered conversation, so the
    answer's tokens are the ones the template puts there (end-of-turn token included)."""
    p = render(tokenizer, prompt, generation=True)
    full = render(tokenizer, prompt + [{"role": "assistant", "content": answer}], generation=False)
    if full[:len(p)] != p:  # a template that rewrites earlier turns: fall back to plain concatenation
        a = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        return p, a
    a = full[len(p):]
    while a and a[-1] in _trailing(tokenizer):  # a trailing newline after the end-of-turn token
        a = a[:-1]
    return p, a


def _trailing(tokenizer) -> set[int]:
    return {i for i in tokenizer.encode("\n", add_special_tokens=False)}


def text_of(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")


def load_lm(model_id: str, dtype):
    """The model to train, as SFT (TRL) loads it: vision-language checkpoints (Qwen3.5, Gemma 3) with
    their image-text class, so the weights land where they are and the adapter serves on vLLM like an
    SFT one; text-only checkpoints with the causal-LM class. Text in, logits out, either way."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id)
    if getattr(cfg, "vision_config", None) is not None or (getattr(cfg, "text_config", None) is not None
                                                            and not hasattr(cfg, "vocab_size")):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)


class Policy:
    """The trainable model (base + LoRA), its optimizer, and a sampler that follows its weights."""

    def __init__(self, spec: dict[str, Any], out_dir: str, log: Callable[[str], None]):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer

        self.spec, self.out_dir, self.log = spec, out_dir, log
        self.stop = _stopper(None, log)
        self.s = settings(spec)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(spec["base_model"])
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.vllm = None
        self.version = 0
        self.lora_dir: str | None = None
        if self.device == "cuda" and spec.get("method") != "sft" and not self.s.get("no_vllm"):
            self._start_vllm()  # before the training model, so vLLM sizes its memory against a free GPU
        model = load_lm(spec["base_model"], self.dtype).to(self.device)
        model.config.use_cache = False
        if self.device == "cuda":
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        r = int(spec["lora_rank"])
        self.model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                                                      target_modules="all-linear", task_type="CAUSAL_LM"))
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=float(spec["learning_rate"]), betas=(0.9, 0.99),
                                     weight_decay=0.0)
        self.params = params

    # -- sampling

    def _start_vllm(self) -> None:
        try:
            from vllm import LLM
        except ImportError:
            self.log("vLLM is not installed: sampling with transformers (slower)")
            return
        mem = float(self.s.get("vllm_memory", 0.3 if self.spec.get("method") != "opd" else 0.25))
        rank = next(x for x in (8, 16, 32, 64, 128, 256) if x >= int(self.spec["lora_rank"]))
        self.vllm = LLM(model=self.spec["base_model"], enable_lora=True, max_lora_rank=rank, max_loras=1,
                        gpu_memory_utilization=mem, max_model_len=int(self.s.get("max_model_len", 4096)),
                        enforce_eager=True, seed=int(self.s.get("seed", 0)))
        self.log(f"vLLM sampler up ({mem:.0%} of GPU memory)")

    def sync(self) -> None:
        """Make the sampler follow the current weights: save the adapter, point vLLM at it."""
        self.version += 1
        if self.vllm is None:
            return
        d = os.path.join(tempfile.gettempdir(), "opentrain-policy", str(os.getpid()), str(self.version))
        self.model.save_pretrained(d)
        self.lora_dir = d

    def sample(self, prompts: list[list[int]], n: int, temperature: float, max_new: int) -> list[list[list[int]]]:
        """``n`` completions (token ids, end-of-turn included when the model stopped) per prompt."""
        if self.vllm is not None:
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest

            sp = SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=max_new,
                                seed=random.randint(0, 2**31 - 1))
            lora = LoRARequest(f"policy-{self.version}", self.version, self.lora_dir) if self.lora_dir else None
            outs = self.vllm.generate([{"prompt_token_ids": p} for p in prompts], sp, lora_request=lora,
                                      use_tqdm=False)
            return [[list(c.token_ids) for c in o.outputs] for o in outs]
        return self._hf_sample(prompts, n, temperature, max_new)

    def _hf_sample(self, prompts, n, temperature, max_new):
        import torch

        self.model.eval()
        self.model.config.use_cache = True
        out: list[list[list[int]]] = []
        try:
            for p in prompts:
                ids = torch.tensor([p], device=self.device)
                with torch.no_grad():
                    g = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                            temperature=max(temperature, 1e-3), top_p=1.0, top_k=0,
                                            max_new_tokens=max_new, num_return_sequences=n,
                                            pad_token_id=self.tok.pad_token_id)
                group = []
                for row in g[:, len(p):].tolist():
                    stop = self._stops()
                    cut = next((i + 1 for i, t in enumerate(row) if t in stop), len(row))
                    group.append(row[:cut])
                out.append(group)
        finally:
            self.model.config.use_cache = False
            self.model.train()
        return out

    def _stops(self) -> set[int]:
        stops = {self.tok.eos_token_id}
        gc = getattr(self.model, "generation_config", None)
        e = getattr(gc, "eos_token_id", None)
        stops |= set(e if isinstance(e, list) else [e] if e is not None else [])
        return {s for s in stops if s is not None}

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    # -- training

    def logprobs(self, model, seqs: list[tuple[list[int], list[int]]], grad: bool):
        """Per-token logprobs of each completion given its prompt, as a list of 1-D tensors. Sequences
        are right-padded into one batch, so this is one forward pass per call."""
        import torch

        if not seqs:
            return []
        width = max(len(p) + len(c) for p, c in seqs)
        pad = self.tok.pad_token_id
        ids = torch.tensor([p + c + [pad] * (width - len(p) - len(c)) for p, c in seqs], device=self.device)
        mask = torch.tensor([[1] * (len(p) + len(c)) + [0] * (width - len(p) - len(c)) for p, c in seqs],
                            device=self.device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        out = []
        with ctx:
            logits = model(input_ids=ids, attention_mask=mask).logits
            for i, (p, c) in enumerate(seqs):
                lg = logits[i, len(p) - 1:len(p) + len(c) - 1].float()
                tgt = torch.tensor(c, device=self.device)[:, None]
                out.append(torch.log_softmax(lg, -1).gather(-1, tgt)[:, 0])
        return out

    def no_grad_logprobs(self, model, seqs):
        micro = int(self.s.get("micro_batch", 8))
        return [lp for i in range(0, len(seqs), micro) for lp in self.logprobs(model, seqs[i:i + micro], False)]

    def step(self, seqs: list[tuple[list[int], list[int]]], advantages: list[Any]) -> float:
        """One optimizer step on ``-sum(adv * logprob) / tokens`` (token mean over the whole batch).
        ``advantages[i]`` is a number (whole sequence) or a per-token list. SFT is advantage 1."""
        import torch

        self.model.train()
        tokens = sum(len(c) for _, c in seqs) or 1
        total = 0.0
        self.opt.zero_grad(set_to_none=True)
        micro = int(self.s.get("micro_batch", 8))
        for i in range(0, len(seqs), micro):
            chunk = seqs[i:i + micro]
            lps = self.logprobs(self.model, chunk, grad=True)
            loss = 0
            for lp, adv in zip(lps, advantages[i:i + micro]):
                a = torch.tensor(adv, device=self.device, dtype=lp.dtype) if isinstance(adv, list) else adv
                loss = loss - (a * lp).sum()
            loss = loss / tokens
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()
        self.sync()
        return total

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def save(self) -> None:
        self.model.save_pretrained(self.out_dir)
        self.tok.save_pretrained(self.out_dir)


def cosine(step: int, total: int, lr: float, warmup: float = 0.05) -> float:
    w = max(1, int(total * warmup))
    if step < w:
        return lr * (step + 1) / w
    return lr * 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


# ---------------------------------------------------------------------------- the recipes


def run(spec: dict[str, Any], rows: list[dict[str, Any]], out_dir: str, emit: Callable[[dict], None],
        grade: Callable[[str, str, str], float] | None = None,
        should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Train; return a summary. ``grade(prompt_text, answer, reference) -> score in [0, 1]``.
    ``should_stop()`` is asked before each step: when it says yes (the budget ran out), training ends
    there and the adapter as it stands is saved, so the steps already paid for are kept."""
    method = spec.get("method", "sft")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; one of {', '.join(METHODS)}")
    if method in ("rejection", "rl") and grade is None:
        raise ValueError(f"{method} needs a grader that runs on the GPU (exact, contains, regex, json, number)")
    t0 = time.time()
    rng = random.Random(int(settings(spec).get("seed", 0)))

    def log(note: str) -> None:
        emit({"event": "log", "note": note, "seconds": round(time.time() - t0, 1)})

    pol = Policy(spec, out_dir, log)
    pol.stop = _stopper(should_stop, log)
    pol.sync()
    items = []
    for r in rows:
        ref = (r["completion"][-1].get("content") or "") if r.get("completion") else ""
        p, a = completion_ids(pol.tok, r["prompt"], ref)
        items.append({"prompt": r["prompt"], "text": text_of(r["prompt"]), "reference": ref, "p": p, "a": a})
    s = pol.s
    fn = {"sft": _sft, "rejection": _rejection, "opd": _opd, "rl": _rl}[method]
    summary = fn(pol, items, s, emit, grade, rng, t0)
    pol.save()
    return {**summary, **({"stopped_early": True} if pol.stop.fired else {}), "seconds": round(time.time() - t0, 1)}


def _stopper(should_stop: Callable[[], bool] | None, log: Callable[[str], None]):
    """``stop()`` is True once ``should_stop`` says so (checked at most every 10 s; errors mean no)."""
    state = {"at": 0.0}

    def stop() -> bool:
        if stop.fired:
            return True
        if should_stop is None or time.time() - state["at"] < 10:
            return False
        state["at"] = time.time()
        try:
            stop.fired = bool(should_stop())
        except Exception:  # noqa: BLE001 - a lookup that fails is not a request to stop
            return False
        if stop.fired:
            log("asked to stop early: saving the adapter as it stands")
        return stop.fired

    stop.fired = False
    return stop


def _emit_step(emit, t0, step, total, loss, **kw):
    emit({"event": "step", "step": step, "total": total, "loss": round(loss, 5),
          "seconds": round(time.time() - t0, 1), **kw})


def _batches(items, size, rng):
    order = list(items)
    rng.shuffle(order)
    return [order[i:i + size] for i in range(0, len(order), size)]


def _sft(pol, items, s, emit, grade, rng, t0):
    epochs, bs, lr = int(pol.spec["epochs"]), int(pol.spec["batch_size"]), float(pol.spec["learning_rate"])
    per = math.ceil(len(items) / bs)
    total, step = per * epochs, 0
    for _ in range(epochs):
        for b in _batches(items, bs, rng):
            if pol.stop():
                break
            pol.set_lr(cosine(step, total, lr))
            loss = pol.step([(x["p"], x["a"]) for x in b], [1.0] * len(b))
            step += 1
            _emit_step(emit, t0, step, total, loss)
    return {"method": "sft", "steps": step}


def _grade_groups(pol, batch, groups, grade):
    """Decoded answers and their scores, per prompt."""
    out = []
    for x, group in zip(batch, groups):
        answers = [pol.decode(c) for c in group]
        out.append([(c, a, float(grade(x["text"], a, x["reference"]))) for c, a in zip(group, answers)])
    return out


def _rejection(pol, items, s, emit, grade, rng, t0):
    rounds, k, keep = int(s["rounds"]), int(s["samples"]), int(s["keep"])
    bs, lr = int(pol.spec["batch_size"]), float(pol.spec["learning_rate"])
    step, history = 0, []
    est_total = rounds * math.ceil(len(items) * min(k, keep) / bs)
    for rnd in range(rounds):
        if pol.stop():
            break
        kept, solved, rewards = [], 0, []
        for chunk in [items[i:i + 64] for i in range(0, len(items), 64)]:
            groups = pol.sample([x["p"] for x in chunk], k, float(s["temperature"]), int(s["max_new_tokens"]))
            for x, graded in zip(chunk, _grade_groups(pol, chunk, groups, grade)):
                rewards += [g for _, _, g in graded]
                seen, good = set(), []
                for c, a, g in graded:
                    if g >= 1.0 and a.strip() not in seen:
                        seen.add(a.strip())
                        good.append(c)
                solved += bool(good)
                kept += [(x["p"], c) for c in good[:keep]]
        rate = sum(rewards) / max(1, len(rewards))
        history.append({"round": rnd + 1, "sample_accuracy": round(rate, 4), "prompts_solved": solved,
                        "kept": len(kept)})
        emit({"event": "log", "note": f"round {rnd + 1}: {rate:.1%} of samples correct, {solved}/{len(items)} "
                                      f"prompts solved, training on {len(kept)} answers",
              "seconds": round(time.time() - t0, 1)})
        if not kept:
            continue
        rng.shuffle(kept)
        batches = [kept[i:i + bs] for i in range(0, len(kept), bs)]
        for b in batches:
            pol.set_lr(cosine(min(step, est_total - 1), est_total, lr))
            loss = pol.step(b, [1.0] * len(b))
            step += 1
            _emit_step(emit, t0, step, max(est_total, step), loss, reward=round(rate, 4))
    if not step:
        raise RuntimeError("no sampled answer was correct in any round: the base model cannot solve these prompts "
                           "yet; train with sft (or opd) first")
    return {"method": "rejection", "steps": step, "rounds": history}


def _rl(pol, items, s, emit, grade, rng, t0):
    steps, n, k = int(s["steps"]), int(s["prompts_per_step"]), int(s["samples"])
    lr = float(pol.spec["learning_rate"])
    pool: list = []
    rewards_seen = []
    for step in range(steps):
        if pol.stop():
            break
        if len(pool) < n:
            pool += _batches(items, len(items), rng)[0]
        batch, pool = pool[:n], pool[n:]
        groups = pol.sample([x["p"] for x in batch], k, float(s["temperature"]), int(s["max_new_tokens"]))
        seqs, advs, rs = [], [], []
        for x, graded in zip(batch, _grade_groups(pol, batch, groups, grade)):
            r = [g for _, _, g in graded]
            rs += r
            mean = sum(r) / len(r)
            if max(r) - min(r) < 1e-6:
                continue  # every sample scored the same: no signal in this group
            for (c, _, g) in graded:
                if c:
                    seqs.append((x["p"], c))
                    advs.append(g - mean)
        reward = sum(rs) / max(1, len(rs))
        rewards_seen.append(reward)
        pol.set_lr(cosine(step, steps, lr, warmup=0.0))
        loss = pol.step(seqs, advs) if seqs else 0.0
        _emit_step(emit, t0, step + 1, steps, loss, reward=round(reward, 4),
                   note=f"{len(seqs)} samples with signal")
    return {"method": "rl", "steps": steps, "reward_first": rewards_seen[0], "reward_last": rewards_seen[-1]}


def _opd(pol, items, s, emit, grade, rng, t0):
    import torch
    from transformers import AutoTokenizer

    teacher_id = s.get("teacher")
    if not teacher_id:
        raise ValueError("opd needs a teacher model (extra.teacher), of the same tokenizer as the student")
    ttok = AutoTokenizer.from_pretrained(teacher_id)
    probe = "Natalia sold 48 clips in April. <|im_end|> 12 + 7 = 19 ✓"
    if ttok.encode(probe, add_special_tokens=False) != pol.tok.encode(probe, add_special_tokens=False):
        raise ValueError(f"teacher {teacher_id} tokenizes differently from {pol.spec['base_model']}; "
                         "on-policy distillation needs a teacher of the same family")
    teacher = load_lm(teacher_id, pol.dtype).to(pol.device).eval()
    emit({"event": "log", "note": f"teacher {teacher_id} loaded", "seconds": round(time.time() - t0, 1)})
    steps, n, k = int(s["steps"]), int(s["prompts_per_step"]), int(s["samples"])
    lr = float(pol.spec["learning_rate"])
    pool: list = []
    for step in range(steps):
        if pol.stop():
            break
        if len(pool) < n:
            pool += _batches(items, len(items), rng)[0]
        batch, pool = pool[:n], pool[n:]
        groups = pol.sample([x["p"] for x in batch], k, float(s["temperature"]), int(s["max_new_tokens"]))
        seqs = [(x["p"], c) for x, g in zip(batch, groups) for c in g if c]
        t_lp = pol.no_grad_logprobs(teacher, seqs)
        s_lp = pol.no_grad_logprobs(pol.model, seqs)
        advs = [(t - st).tolist() for t, st in zip(t_lp, s_lp)]
        kl = sum(float((st - t).sum()) for t, st in zip(t_lp, s_lp)) / max(1, sum(len(c) for _, c in seqs))
        rs = []
        if grade is not None:
            for x, g in zip(batch, groups):
                rs += [float(grade(x["text"], pol.decode(c), x["reference"])) for c in g]
        pol.set_lr(cosine(step, steps, lr, warmup=0.0))
        loss = pol.step(seqs, advs)
        extra = {"reward": round(sum(rs) / len(rs), 4)} if rs else {}
        _emit_step(emit, t0, step + 1, steps, loss, kl=round(kl, 5), **extra)
    del teacher
    torch.cuda.empty_cache() if pol.device == "cuda" else None
    return {"method": "opd", "steps": steps, "teacher": teacher_id}


# ---------------------------------------------------------------------------- a command line, for pods


def grader_fn(grader: dict[str, Any] | None, *, inprocess: bool = True):
    """The dataset's grader as ``(prompt, answer, reference) -> score``, if it runs here.

    A python reward runs in this process on a remote GPU (``inprocess``); on the user's own machine
    (the Fireworks loop) it runs in a sandboxed child process, as in evals, so it cannot reach the
    sealed rows. A reward that raises scores 0 for that sample instead of killing a paid job."""
    if not grader or grader.get("type") in ("judge", "command"):
        return None
    try:
        from graders import grade as g  # shipped next to this file
        from graders import load_reward
    except ImportError:
        from opentrain.graders import grade as g
        from opentrain.graders import load_reward
    if grader.get("type") == "python" and inprocess:
        fn = load_reward(grader["source"], grader.get("function", "reward"))
        score = lambda prompt, answer, ref: fn(prompt, answer, ref)[0]  # noqa: E731
    else:
        score = lambda prompt, answer, ref: g(grader, prompt, answer, ref).score  # noqa: E731
    failures = [0]

    def safe(prompt: str, answer: str, ref: str | None) -> float:
        try:
            return score(prompt, answer, ref)
        except Exception as exc:  # noqa: BLE001 - the user's reward; one bad sample is not the run
            failures[0] += 1
            if failures[0] in (1, 10, 100):
                print(f"opentrain: the grader failed {failures[0]} time(s), scored 0: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
            return 0.0

    return safe


def main(argv: list[str]) -> int:
    """``python gpu_train.py job.json out_dir``: job.json is {spec, rows}; events go to stdout."""
    job = json.loads(open(argv[0]).read())

    def emit(ev):
        print(json.dumps(ev), flush=True)

    try:
        summary = run(job["spec"], job["rows"], argv[1], emit, grader_fn(job["spec"].get("extra", {}).get("grader")))
        emit({"event": "done", "adapter": argv[1], **summary})
        return 0
    except Exception as e:  # noqa: BLE001
        emit({"event": "error", "error": f"{type(e).__name__}: {e}"})
        return 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
