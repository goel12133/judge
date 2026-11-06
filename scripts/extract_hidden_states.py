"""Extract per-word hidden states (every layer) and model surprisal for Natural Stories.

Run where a GPU and Hugging Face are reachable (Colab free tier handles 8B in 8-bit):

    pip install -q transformers accelerate bitsandbytes
    git clone --depth 1 https://github.com/languageMIT/naturalstories data/raw/naturalstories
    python scripts/extract_hidden_states.py --model meta-llama/Llama-3.1-8B --out results/hs_llama31_8b.npz

Output .npz:
  item, zone            (N,)      alignment keys matching judge.naturalstories words
  surprisal             (N,)      model's own surprisal per word (sum over subtokens), nats
  layer_{l}             (N, d)    mean over the word's subtokens of layer-l hidden state
                                   (float16 to keep the file small; l = 0..L)
Context: each story is fed whole (they are < 2k tokens) so every word sees its full left
context; attention is causal, so hidden states at a word never see the future.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--out", default="results/hs.npz")
    ap.add_argument("--root", default="data/raw/naturalstories")
    ap.add_argument("--eightbit", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from judge import naturalstories as ns

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = dict(torch_dtype=torch.float16, device_map="auto")
    if args.eightbit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    t = ns.load(args.root)
    words = t.words.sort_values(["item", "zone"]).reset_index(drop=True)

    items, zones, surps = [], [], []
    layers: dict[int, list] = {}
    for item, grp in words.groupby("item", sort=True):
        ws = grp["word"].tolist()
        text = " ".join(ws)
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            out = model(**{k: v.to(model.device) for k, v in enc.items()}, output_hidden_states=True)
        logp = torch.log_softmax(out.logits[0].float(), -1)
        ids = enc["input_ids"][0]
        tok_surp = torch.zeros(len(ids))
        tok_surp[1:] = -logp[:-1].gather(1, ids[1:, None].to(logp.device))[:, 0].cpu()
        hs = [h[0].float().cpu() for h in out.hidden_states]  # L+1 of (T, d)

        # word spans in characters
        spans, cursor = [], 0
        for w in ws:
            spans.append((cursor, cursor + len(w)))
            cursor += len(w) + 1
        starts = np.array([s for s, _ in spans])
        # assign each token to the word containing its first non-space char
        tok_word = []
        for (a, b) in offsets:
            if b <= a:  # special tokens
                tok_word.append(-1)
                continue
            # skip leading space in the token's own span
            c = a
            while c < b and text[c] == " ":
                c += 1
            tok_word.append(int(np.searchsorted(starts, c, side="right") - 1))
        tok_word = np.array(tok_word)

        for wi, (_, row) in enumerate(grp.iterrows()):
            m = tok_word == wi
            items.append(int(row["item"])); zones.append(int(row["zone"]))
            if m.sum() == 0:
                surps.append(np.nan)
                for l, h in enumerate(hs):
                    layers.setdefault(l, []).append(np.full(h.shape[1], np.nan, np.float16))
                continue
            surps.append(float(tok_surp[m].sum()))
            for l, h in enumerate(hs):
                layers.setdefault(l, []).append(h[m].mean(0).numpy().astype(np.float16))
        print(f"story {item}: {len(ws)} words, {len(ids)} tokens")

    out = {"item": np.array(items), "zone": np.array(zones), "surprisal": np.array(surps)}
    for l, rows in layers.items():
        out[f"layer_{l}"] = np.stack(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("wrote", args.out, {k: v.shape for k, v in out.items() if k.startswith("layer_0")})


if __name__ == "__main__":
    main()
