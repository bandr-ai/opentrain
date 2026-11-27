"""The detached Tinker training worker: ``python -m opentrain.backends.tinker_worker <job.json>``.

It runs apart from the CLI so a multi-hour job is not tied to a terminal. Everything it learns
goes into the job file (progress, tokens, accrued cost, loss, checkpoint) and the job log; the
CLI only reads those. API calls follow the sync form documented for tinker 0.40 (each call
returns a future; ``.result()`` waits), see ``opentrain.backends.tinker`` for the doc links.
"""

from __future__ import annotations

import json
import math
import random
import sys
from typing import Any

from . import JobStore, missing_modules, spec_from_dict, worker_main
from .tinker import (
    datum_parts,
    hf_messages,
    hf_supervised_tokens,
    read_examples,
    split_example,
    tool_names_by_call_id,
)


def _cookbook_datums(rows: list[dict[str, Any]], tokenizer: Any, spec, log) -> list[Any]:
    """Datums built by tinker-cookbook's renderer for this model family (preferred path)."""
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.supervised.data import conversation_to_datum

    family = spec.base_model.split(":")[0]  # Inkling-2-Mini:peft:65536:free renders as Inkling-2-Mini
    name = spec.extra.get("renderer") or model_info.get_recommended_renderer_name(family)
    renderer = renderers.get_renderer(name, tokenizer, model_name=family)
    log(f"renderer {name} (tinker-cookbook)")
    out = []
    for row in rows:
        system, tools, msgs = split_example(row)
        if tools:
            conv = list(renderer.create_conversation_prefix_with_tools(tools, system))
        else:
            conv = [{"role": "system", "content": system}] if system else []
        names = tool_names_by_call_id(msgs)
        for m in msgs:
            msg: dict[str, Any] = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = []
                for c in m["tool_calls"]:
                    args = c["function"].get("arguments") or "{}"
                    body = renderers.ToolCall.FunctionBody(
                        name=c["function"]["name"], arguments=args if isinstance(args, str) else json.dumps(args))
                    msg["tool_calls"].append(renderers.ToolCall(id=c.get("id"), function=body))
            if m["role"] == "tool":
                msg["tool_call_id"] = m.get("tool_call_id", "")
                if m.get("tool_call_id") in names:
                    msg["name"] = names[m["tool_call_id"]]
            conv.append(msg)
        if getattr(renderer, "has_extension_property", True):
            out.append(conversation_to_datum(conv, renderer, max_length=spec.max_tokens,
                                             train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES))
        else:  # the template writes old turns differently (drops their reasoning): one example per turn,
            # each ending at the turn it trains, rendered as the model saw it when it produced that turn
            out += [conversation_to_datum(conv[:i + 1], renderer, max_length=spec.max_tokens,
                                          train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE)
                    for i, m in enumerate(conv) if m["role"] == "assistant"]
    return out


def tml_model(base_model: str) -> bool:
    """Thinking Machines' own models (Inkling): no HF tokenizer; tml-renderers renders them."""
    return base_model.startswith("thinkingmachines/")


def _tml_datums(rows: list[dict[str, Any]], tokenizer: Any, spec, log) -> list[Any]:
    """Datums for Inkling models from tml-renderers (TMLv0), one per training example it renders."""
    import torch
    from tinker_cookbook.supervised.common import datum_from_model_input_weights
    from tml_renderers import chat, renderers
    from tml_renderers.tinker import training_example_to_tinker_model_input_and_weights

    family = spec.base_model.split(":")[0].split("/")[-1]
    renderer = renderers.Inkling2() if family.startswith("Inkling-2") else renderers.Inkling()
    log(f"renderer {type(renderer).__name__} (tml-renderers)")
    out = []
    for row in rows:
        msgs = chat.OpenAIMessage.from_openai_chat_dicts(row["messages"], tools=row.get("tools") or None)
        for ex in renderer.render_for_sft(msgs):
            model_input, weights = training_example_to_tinker_model_input_and_weights(ex)
            out.append(datum_from_model_input_weights(model_input, torch.tensor(weights),
                                                      max_length=spec.max_tokens, reduction="mean"))
    return out


def _hf_datums(rows: list[dict[str, Any]], tokenizer: Any, spec, log) -> list[Any]:
    """Datums from the HF chat template, when tinker-cookbook is not installed."""
    import tinker

    log("renderer: HF tokenizer chat template (tinker-cookbook not installed)")
    out = []
    for row in rows:
        msgs = hf_messages(row["messages"])
        tokens, weights = hf_supervised_tokens(tokenizer, msgs, row.get("tools"))
        inp, target, w = datum_parts(tokens, weights, spec.max_tokens)
        out.append(tinker.Datum(model_input=tinker.ModelInput.from_ints(inp),
                                loss_fn_inputs={"target_tokens": target, "weights": w}))
    return out


def run(store: JobStore, job: dict[str, Any]) -> None:
    import tinker

    jid = job["job_id"]
    spec = spec_from_dict(job["spec"])
    price = float(job["usd_per_mtok"])

    def log(line: str) -> None:
        store.log(jid, line)

    service = tinker.ServiceClient()
    training = service.create_lora_training_client(base_model=spec.base_model, rank=spec.lora_rank).result()
    log(f"training client ready: {spec.base_model}, LoRA rank {spec.lora_rank}")
    try:
        log(f"console: {training.get_console_url()}")
    except Exception:  # noqa: BLE001 - a missing console link must not stop training
        pass
    rows = read_examples(spec.data)
    if tml_model(spec.base_model):  # Tinker has no tokenizer for these: get_tokenizer() raises
        tokenizer, build = None, _tml_datums
    else:
        tokenizer = training.get_tokenizer()
        build = _hf_datums if missing_modules("tinker_cookbook") else _cookbook_datums
    datums = build(rows, tokenizer, spec, log)
    def trains(d) -> bool:
        w = d.loss_fn_inputs["weights"]
        return any(float(x) for x in getattr(w, "data", w))

    kept = [d for d in datums if trains(d)]
    if len(kept) < len(datums):
        log(f"left out {len(datums) - len(kept)} example(s) whose answers start after max_tokens ({spec.max_tokens})")
    if not kept:
        raise RuntimeError(f"every example's answers start after max_tokens ({spec.max_tokens}): "
                           "raise max_tokens or shorten the examples")
    datums = kept
    log(f"{len(datums)} examples, {sum(d.model_input.length for d in datums)} tokens per epoch")

    rng = random.Random(spec.extra.get("seed", 0))
    steps_per_epoch = math.ceil(len(datums) / spec.batch_size)
    total = steps_per_epoch * spec.epochs
    tokens_done, step = 0, 0
    for epoch in range(spec.epochs):
        order = list(datums)
        rng.shuffle(order)
        for b in range(steps_per_epoch):
            if store.cancel_requested(jid):
                cost = round(tokens_done * price / 1e6, 4)
                log(f"cancelled at step {step}/{total}")
                store.update(jid, state="cancelled", cost_usd=cost,
                             detail=f"cancelled at step {step}/{total}; ~${cost} accrued (estimate)")
                return
            batch = order[b * spec.batch_size:(b + 1) * spec.batch_size]
            fwd = training.forward_backward(batch, "cross_entropy")
            opt = training.optim_step(tinker.AdamParams(learning_rate=spec.learning_rate))
            result = fwd.result()
            opt.result()
            step += 1
            tokens_done += sum(d.model_input.length for d in batch)
            cost = round(tokens_done * price / 1e6, 4)
            loss = result.metrics.get("loss:sum")
            log(f"epoch {epoch + 1} step {step}/{total} loss:sum={loss} tokens={tokens_done} ~${cost}")
            store.update(jid, progress=round(step / total, 4), cost_usd=cost, tokens=tokens_done,
                         loss=loss, detail=f"step {step}/{total}; ~${cost} from {tokens_done} tokens "
                                           f"x ${price}/Mtok (estimate; Tinker billing is authoritative)")

    saved = training.save_weights_for_sampler(name=jid).result()
    cost = round(tokens_done * price / 1e6, 4)
    log(f"sampler weights saved: {saved.path}")
    weights = None
    try:  # the adapter as files you can take anywhere: tinker checkpoint download <path>, or push-hf
        weights = training.save_weights_external(name=jid).result().path
        log(f"downloadable weights saved: {weights} (tinker checkpoint download {weights})")
    except Exception as exc:  # noqa: BLE001 - the model is trained and served either way
        log(f"could not save downloadable weights ({type(exc).__name__}: {exc}); the model still serves on Tinker")
    store.update(jid, state="succeeded", progress=1.0, checkpoint=saved.path, weights=weights, cost_usd=cost,
                 detail=f"done: {total} steps, {tokens_done} tokens, ~${cost} (estimate; Tinker billing "
                        "is authoritative)")


if __name__ == "__main__":
    raise SystemExit(worker_main(sys.argv[1:], run))
