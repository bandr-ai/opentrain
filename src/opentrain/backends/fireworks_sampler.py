"""Answers from a Fireworks sampler snapshot, run on Python 3.12 by ``uv run`` (see ``fireworks``).

``python -m opentrain.backends.fireworks_sampler <snapshot> <fireworks model id> <hf tokenizer>``

Prints ``{"ready": true}`` once the session is open, then answers one JSON request per stdin line
(``{"messages": [...], "max_tokens": n}``) with ``{"text", "prompt_tokens", "completion_tokens"}``
or ``{"error"}``. Exits, closing its session, when stdin closes or after five idle minutes.
"""

from __future__ import annotations

import json
import os
import select
import sys

IDLE_SECONDS = 300


def say(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main(argv: list[str]) -> int:
    snapshot, fid, tok_name = argv
    try:
        from fireworks.training import sdk
        from transformers import AutoTokenizer

        from .tinker import hf_messages

        tok = AutoTokenizer.from_pretrained(tok_name)
        service = sdk.FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                             base_url="https://api.fireworks.ai/training/v1/serverless")
        client = service.create_sampling_client(model_path=snapshot, base_model=f"accounts/fireworks/models/{fid}")
    except Exception as e:  # noqa: BLE001
        say({"error": f"{type(e).__name__}: {e}"[:500]})
        return 1
    say({"ready": True})
    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], IDLE_SECONDS)
            if not ready:
                return 0
            line = sys.stdin.readline()
            if not line:
                return 0
            try:
                req = json.loads(line)
                # render to text, then tokenize: apply_chat_template(tokenize=True) returns a
                # BatchEncoding on newer transformers and a list on older ones
                text = tok.apply_chat_template(hf_messages(req["messages"]), tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
                prompt = list(tok(text, add_special_tokens=False)["input_ids"])
                out = client.sample(prompt=sdk.ModelInput.from_ints(list(prompt)), num_samples=1,
                                    sampling_params=sdk.SamplingParams(max_tokens=int(req.get("max_tokens", 512)),
                                                                       temperature=0.0)).result()
                toks = list(out.sequences[0].tokens)
                say({"text": tok.decode(toks, skip_special_tokens=True).strip(),
                     "prompt_tokens": len(prompt), "completion_tokens": len(toks)})
            except Exception as e:  # noqa: BLE001 - one bad row must not end the session
                say({"error": f"{type(e).__name__}: {e}"[:500]})
    finally:
        try:
            service.delete_training_session()
        except Exception:  # noqa: BLE001
            pass
        service.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
