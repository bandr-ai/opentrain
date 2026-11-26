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


if __name__ == "__main__":
    raise SystemExit(worker_main(sys.argv[1:], run))
