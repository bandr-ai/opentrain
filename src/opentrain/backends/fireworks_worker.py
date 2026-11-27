"""The detached Fireworks training worker, run on Python 3.12 by ``uv run`` (see ``fireworks``).

``python -m opentrain.backends.fireworks_worker <job.json>``. Everything it learns goes into the job
file and log. It always closes its serverless session, so a crash does not hold a run slot.
"""

from __future__ import annotations

import math
import os
import random
import sys
from typing import Any

from . import JobStore, spec_from_dict, worker_main
from .tinker import datum_parts, hf_messages, hf_supervised_tokens, read_examples

TRAIN_URL = "https://api.fireworks.ai/training/v1/serverless"


def snapshot_name(job_id: str) -> str:
    """Fireworks caps checkpoint names at 17 characters."""
    return ("ot-" + job_id.rsplit("-", 1)[-1])[:17]


def run(store: JobStore, job: dict[str, Any]) -> None:
    if spec_from_dict(job["spec"]).method != "sft":
        return run_loop(store, job)
    return run_sft(store, job)


def run_sft(store: JobStore, job: dict[str, Any]) -> None:
    from fireworks.training import sdk
    from transformers import AutoTokenizer

    jid = job["job_id"]
    spec = spec_from_dict(job["spec"])
    price = float(job["usd_per_mtok"])
    base = f"accounts/fireworks/models/{job['fireworks_model']}"

    def log(line: str) -> None:
        store.log(jid, line)

    tokenizer = AutoTokenizer.from_pretrained(job["tokenizer"])
    datums, lengths = [], []
    for row in read_examples(spec.data):
        tokens, weights = hf_supervised_tokens(tokenizer, hf_messages(row["messages"]), row.get("tools"))
        inp, target, w = datum_parts(tokens, weights, spec.max_tokens)
        datums.append(sdk.Datum(model_input=sdk.ModelInput.from_ints(inp),
                                loss_fn_inputs={"target_tokens": target, "weights": w}))
        lengths.append(len(inp))
    log(f"{len(datums)} examples, {sum(lengths)} tokens per epoch ({job['tokenizer']})")

    service = sdk.FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url=TRAIN_URL)
    try:
        training = service.create_lora_training_client(base_model=base, rank=spec.lora_rank)
        log(f"serverless session ready: {base}, LoRA rank {spec.lora_rank}")
        rng = random.Random(spec.extra.get("seed", 0))
        order_ix = list(range(len(datums)))
        steps_per_epoch = math.ceil(len(datums) / spec.batch_size)
        total = steps_per_epoch * spec.epochs
        tokens_done, step = 0, 0
        for epoch in range(spec.epochs):
            rng.shuffle(order_ix)
            for b in range(steps_per_epoch):
                if store.cancel_requested(jid):
                    cost = round(tokens_done * price / 1e6, 6)
                    log(f"cancelled at step {step}/{total}")
                    store.update(jid, state="cancelled", cost_usd=cost,
                                 detail=f"cancelled at step {step}/{total}; ~${cost} accrued")
                    return
                ix = order_ix[b * spec.batch_size:(b + 1) * spec.batch_size]
                fwd = training.forward_backward([datums[i] for i in ix], "cross_entropy")
                opt = training.optim_step(sdk.AdamParams(learning_rate=spec.learning_rate))
                result = fwd.result()
                opt.result()
                step += 1
                tokens_done += sum(lengths[i] for i in ix)
                cost = round(tokens_done * price / 1e6, 6)
                loss = result.metrics.get("loss:sum")
                log(f"epoch {epoch + 1} step {step}/{total} loss={loss} tokens={tokens_done} ~${cost}")
                store.update(jid, progress=round(step / total, 4), cost_usd=cost, tokens=tokens_done, loss=loss,
                             detail=f"step {step}/{total}; ~${cost} ({tokens_done} tokens x ${price}/Mtok)")
        saved = training.save_weights_for_sampler(snapshot_name(jid)).result()
        cost = round(tokens_done * price / 1e6, 6)
        log(f"snapshot saved: {saved.path}")
        store.update(jid, state="succeeded", progress=1.0, checkpoint="fireworks-sampler://" + saved.path,
                     cost_usd=cost, detail=f"done: {total} steps, {tokens_done} tokens, ~${cost} "
                                           "(Fireworks billing is authoritative)")
    finally:
        try:
            service.delete_training_session()
        except Exception as e:  # noqa: BLE001 - the session expires on its own after 10 idle minutes
            log(f"could not close the session ({type(e).__name__}); it expires after 10 idle minutes")
        service.close()


def step_name(job_id: str, step: int) -> str:
    return (f"s{step}-" + job_id.rsplit("-", 1)[-1])[:17]


def run_loop(store: JobStore, job: dict[str, Any]) -> None:
    """Rejection sampling and RL on the Training API. Each step publishes a sampler snapshot, samples
    from it (``client.sample``, with logprobs), grades here, and trains: RL with the built-in
    ``importance_sampling`` loss (advantage = score minus the group mean, the sampler's logprobs as
    the anchor), rejection sampling with ``cross_entropy`` on the passing answers. MoE expert routes
    are not replayed (the importance ratio absorbs the drift)."""
    from fireworks.training import sdk
    from transformers import AutoTokenizer

    from .gpu_train import completion_ids, grader_fn, settings, text_of
    from .modal_backend import is_trace

    jid = job["job_id"]
    spec = spec_from_dict(job["spec"])
    s = settings({"method": spec.method, "extra": spec.extra})
    grade = grader_fn(spec.extra.get("grader"), inprocess=False)  # runs here, beside the sealed rows: sandboxed
    if grade is None:
        raise RuntimeError(f"{spec.method} needs a grader that runs in the loop (exact, contains, regex, json, number, python)")
    train_price, sample_price = float(job["usd_per_mtok"]), float(job.get("usd_per_mtok_sample") or 0)
    base = f"accounts/fireworks/models/{job['fireworks_model']}"
    tok = AutoTokenizer.from_pretrained(job["tokenizer"])
    items = []
    for row in read_examples(spec.data):
        if is_trace(row) or row.get("tools") or row["messages"][-1].get("tool_calls"):
            raise ValueError(f"{spec.method} samples single-turn answers; agent traces and tool-calling "
                             "examples train with sft")
        msgs = hf_messages(row["messages"])
        p, _ = completion_ids(tok, msgs[:-1], msgs[-1].get("content") or "")
        items.append({"p": p, "text": text_of(msgs[:-1]), "ref": msgs[-1].get("content") or ""})
    rng = random.Random(int(s.get("seed", 0)))
    usage = {"train": 0, "sample": 0}

    def cost() -> float:
        return round((usage["train"] * train_price + usage["sample"] * sample_price) / 1e6, 6)

    def log(line: str) -> None:
        store.log(jid, line)

    service = sdk.FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url=TRAIN_URL)
    adam = lambda lr: sdk.AdamParams(learning_rate=lr)  # noqa: E731
    norm = sdk.GradAccNormalization.NUM_LOSS_TOKENS
    try:
        training = service.create_lora_training_client(base_model=base, rank=spec.lora_rank)
        log(f"serverless session ready: {base}, LoRA rank {spec.lora_rank}, {spec.method}")
        version = [0]

        def sample(batch: list[dict[str, Any]], k: int) -> list[list[tuple[list[int], list[float]]]]:
            version[0] += 1
            snap = training.save_weights_for_sampler(step_name(jid, version[0])).result().path
            client = service.create_sampling_client(model_path=snap, base_model=base)
            params = sdk.SamplingParams(max_tokens=int(s["max_new_tokens"]), temperature=float(s["temperature"]))
            futs = [client.sample(prompt=sdk.ModelInput.from_ints(list(x["p"])), num_samples=k,
                                  sampling_params=params) for x in batch]
            out = []
            for x, f in zip(batch, futs):
                seqs = f.result().sequences
                group = [(list(q.tokens), [float(v) for v in (q.logprobs or [0.0] * len(q.tokens))]) for q in seqs]
                usage["sample"] += sum(len(x["p"]) + len(c) for c, _ in group)
                out.append(group)
            return out

        def graded(batch, groups):
            return [[(c, lp, float(grade(x["text"], tok.decode(c, skip_special_tokens=True), x["ref"])))
                     for c, lp in g] for x, g in zip(batch, groups)]

        if spec.method == "rl":
            steps, n, k = int(s["steps"]), int(s["prompts_per_step"]), int(s["samples"])
            pool: list = []
            for step in range(steps):
                if store.cancel_requested(jid):
                    store.update(jid, state="cancelled", cost_usd=cost(), detail=f"cancelled at step {step}/{steps}")
                    return
                if len(pool) < n:
                    extra = list(items)
                    rng.shuffle(extra)
                    pool += extra
                batch, pool = pool[:n], pool[n:]
                rows = graded(batch, sample(batch, k))
                datums, rewards = [], []
                for x, g in zip(batch, rows):
                    r = [sc for _, _, sc in g]
                    rewards += r
                    mean = sum(r) / len(r)
                    if max(r) - min(r) < 1e-6:
                        continue
                    for c, lp, sc in g:
                        if not c:
                            continue
                        full, pl = x["p"] + c, len(x["p"]) - 1
                        datums.append(sdk.Datum(model_input=sdk.ModelInput.from_ints(full[:-1]), loss_fn_inputs={
                            "target_tokens": full[1:], "logprobs": [0.0] * pl + lp[:len(c)],
                            "advantages": [0.0] * pl + [sc - mean] * len(c)}))
                        usage["train"] += len(full)
                reward = sum(rewards) / max(1, len(rewards))
                loss = None
                if datums:
                    fwd = training.forward_backward(datums, "importance_sampling")
                    opt = training.optim_step(adam(spec.learning_rate), grad_accumulation_normalization=norm)
                    loss = fwd.result().metrics.get("loss:sum")
                    opt.result()
                log(f"step {step + 1}/{steps} reward={reward:.3f} ({len(datums)} samples with signal) ~${cost()}")
                store.update(jid, progress=round((step + 1) / steps, 4), reward=round(reward, 4), cost_usd=cost(),
                             **({"loss": loss} if loss is not None else {}),
                             detail=f"step {step + 1}/{steps} · reward {reward:.0%} · ~${cost()}")
        else:  # rejection sampling
            rounds, k, keep, bs = int(s["rounds"]), int(s["samples"]), int(s["keep"]), spec.batch_size
            history = []
            for rnd in range(rounds):
                kept, rewards = [], []
                for i in range(0, len(items), 32):
                    if store.cancel_requested(jid):
                        store.update(jid, state="cancelled", cost_usd=cost(), detail=f"cancelled in round {rnd + 1}")
                        return
                    chunk = items[i:i + 32]
                    for x, g in zip(chunk, graded(chunk, sample(chunk, k))):
                        rewards += [sc for _, _, sc in g]
                        seen, good = set(), []
                        for c, _, sc in g:
                            if sc >= 1.0 and tuple(c) not in seen:
                                seen.add(tuple(c))
                                good.append(c)
                        kept += [(x["p"], c) for c in good[:keep]]
                rate = sum(rewards) / max(1, len(rewards))
                history.append({"round": rnd + 1, "sample_accuracy": round(rate, 4), "kept": len(kept)})
                log(f"round {rnd + 1}: {rate:.1%} of samples correct; training on {len(kept)} answers")
                rng.shuffle(kept)
                for b in range(0, len(kept), bs):
                    datums = []
                    for p, c in kept[b:b + bs]:
                        inp, target, w = datum_parts(p + c, [0.0] * len(p) + [1.0] * len(c), spec.max_tokens)
                        datums.append(sdk.Datum(model_input=sdk.ModelInput.from_ints(inp),
                                                loss_fn_inputs={"target_tokens": target, "weights": w}))
                        usage["train"] += len(inp)
                    fwd = training.forward_backward(datums, "cross_entropy")
                    opt = training.optim_step(adam(spec.learning_rate))
                    loss = fwd.result().metrics.get("loss:sum")
                    opt.result()
                    store.update(jid, loss=loss, reward=round(rate, 4), cost_usd=cost(),
                                 progress=round((rnd + (b + bs) / max(1, len(kept))) / rounds, 4),
                                 detail=f"round {rnd + 1}/{rounds} · {rate:.0%} correct · ~${cost()}")
            store.update(jid, rounds=history)
            if not any(h["kept"] for h in history):
                raise RuntimeError("no sampled answer was correct in any round: the base model cannot solve these "
                                   "prompts yet; train with sft first")
        saved = training.save_weights_for_sampler(snapshot_name(jid)).result()
        log(f"snapshot saved: {saved.path}")
        store.update(jid, state="succeeded", progress=1.0, checkpoint="fireworks-sampler://" + saved.path,
                     cost_usd=cost(), detail=f"done: {spec.method}, ~${cost()} (Fireworks billing is authoritative)")
    finally:
        try:
            service.delete_training_session()
        except Exception as e:  # noqa: BLE001
            log(f"could not close the session ({type(e).__name__}); it expires after 10 idle minutes")
        service.close()


if __name__ == "__main__":
    raise SystemExit(worker_main(sys.argv[1:], run))
