"""Generate notebooks/judge_extract.ipynb — the one-click Colab extraction run.

Run: python notebooks/make_notebook.py
Then upload notebooks/judge_extract.ipynb to Colab, set runtime to T4 GPU, Run all.
"""
import json
from pathlib import Path

cells = []

def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# judge — hidden-state extraction (Colab)

One run: clones the repo, loads the chosen model (Qwen2.5-7B, Gemma-2-9B or Llama-3.1-8B) in 8-bit on a T4, extracts every layer's
hidden state per word for **Natural Stories** and per segment for **Sandman**, saves two
`.npz` files and downloads them to your machine.

Before running: **Runtime → Change runtime type → T4 GPU**. Then **Runtime → Run all**.
You'll be asked for a Hugging Face read token once. Qwen needs no approval; Gemma needs the
licence accepted on its model page (instant); Llama needs Meta's approval (usually under an hour).
Run the notebook once per model: change the dropdown in the model cell and Run all again. If the repo is private you'll also be asked
for a GitHub token; leave blank if it's public.
""")

code("""import torch, subprocess, sys
assert torch.cuda.is_available(), "No GPU: set Runtime → Change runtime type → T4 GPU"
print(torch.cuda.get_device_name(0))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.44", "accelerate", "bitsandbytes", "xlrd", "pandas", "scikit-learn"], check=True)
print("deps ok")""")

code("""from getpass import getpass
import os
hf = getpass("Hugging Face read token: ").strip()
os.environ["HF_TOKEN"] = hf
gh = getpass("GitHub token (blank if repo is public): ").strip()
url = f"https://{gh}@github.com/goel12133/judge" if gh else "https://github.com/goel12133/judge"
import subprocess
subprocess.run(["rm", "-rf", "judge"], check=True)
r = subprocess.run(["git", "clone", "-q", "--depth", "1", url, "judge"], capture_output=True, text=True)
print(r.stderr or "repo cloned")
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/languageMIT/naturalstories", "judge/data/raw/naturalstories"], check=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/eilab-gt/SuspensePerception", "judge/data/raw/SuspensePerception"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "judge"], check=True)
os.chdir("judge")
print("data ok")""")

code("""#@title Pick the model (run this notebook once per model)
MODEL = "Qwen/Qwen2.5-7B"  #@param ["Qwen/Qwen2.5-7B", "google/gemma-2-9b", "meta-llama/Llama-3.1-8B"]
TAG = {"Qwen/Qwen2.5-7B": "qwen25_7b", "google/gemma-2-9b": "gemma2_9b", "meta-llama/Llama-3.1-8B": "llama31_8b"}[MODEL]
from huggingface_hub import login
try:
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
except Exception as e:   # Qwen is ungated; only Gemma/Llama need a valid token
    print("HF login failed (fine for Qwen, fatal for Gemma/Llama):", str(e)[:120])
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto").eval()
print("model loaded; layers:", model.config.num_hidden_layers)

def forward_stats(enc, topk=0, chunk=256, window=4096, step=1024):
    \"\"\"Hidden states for every layer, per-token surprisal of the actual next token (nats)
    and optional top-k next-token log-probs, all on CPU.
    Memory: the LM head runs in `chunk`-token pieces so the vocab-sized logit matrix never
    sits on the GPU, and sequences longer than `window` are processed in sliding windows
    (each target block of `step` tokens sees the preceding window-step tokens of context),
    because SDPA on a T4 materialises the T x T attention (9.5 GB at 8.5k tokens).
    Natural Stories items (<2k tokens) always get full context; Sandman (~8.5k tokens) is windowed.\"\"\"
    all_ids = enc["input_ids"][0]
    T = len(all_ids)
    cap = getattr(model.config, "final_logit_softcapping", None)
    L = model.config.num_hidden_layers + 1
    hs = [[] for _ in range(L)]; surp = torch.zeros(T); tv, ti = [], []
    s = 0
    while s < T:
        e = min(T, s + (window if s == 0 else step))
        c0 = 0 if s == 0 else max(0, e - window)
        ids = all_ids[c0:e]
        with torch.no_grad():
            base = model.model(input_ids=ids[None].to(model.device), output_hidden_states=True)
            off = s - c0                       # first target position inside this window
            for l, h in enumerate(base.hidden_states): hs[l].append(h[0, off:].float().cpu())
            last = base.last_hidden_state[0, off:]
            for a in range(0, len(last), chunk):
                lg = model.lm_head(last[a:a + chunk]).float()
                if cap: lg = torch.tanh(lg / cap) * cap
                lp = torch.log_softmax(lg, -1)
                g = s + a                                     # global position of lp[0]
                nxt = all_ids[g + 1:g + 1 + len(lp)].to(lp.device)
                n = len(nxt)
                if n: surp[g + 1:g + 1 + n] = -lp[:n].gather(1, nxt[:, None])[:, 0].cpu()
                if topk:
                    v, i = torch.topk(lp, topk, dim=-1); tv.append(v.cpu()); ti.append(i.cpu())
                del lg, lp
            del base, last; torch.cuda.empty_cache()
        s = e
    hs = [torch.cat(x) for x in hs]
    top = (torch.cat(tv), torch.cat(ti)) if topk else None
    return hs, surp, top""")

md("## Natural Stories: per-word hidden states and model surprisal")

code("""import numpy as np
from judge import naturalstories as ns

def extract_words(words_by_item, out_path):
    items, zones, surps = [], [], []
    layers = {}
    for item, ws in words_by_item:
        text = " ".join(ws)
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"][0]
        hs, tok_surp, _ = forward_stats(enc)
        spans, cursor = [], 0
        for w in ws:
            spans.append((cursor, cursor + len(w))); cursor += len(w) + 1
        starts = np.array([s for s, _ in spans])
        tok_word = []
        for (a, b) in offsets:
            if b <= a: tok_word.append(-1); continue
            c = a
            while c < b and text[c] == " ": c += 1
            tok_word.append(int(np.searchsorted(starts, c, side="right") - 1))
        tok_word = np.array(tok_word)
        for wi in range(len(ws)):
            m = tok_word == wi
            items.append(item); zones.append(wi + 1)
            if m.sum() == 0:
                surps.append(np.nan)
                for l, h in enumerate(hs): layers.setdefault(l, []).append(np.full(h.shape[1], np.nan, np.float16))
                continue
            surps.append(float(tok_surp[m].sum()))
            for l, h in enumerate(hs): layers.setdefault(l, []).append(h[m].mean(0).numpy().astype(np.float16))
        print(f"item {item}: {len(ws)} words, {len(ids)} tokens")
        del hs; torch.cuda.empty_cache()
    out = {"item": np.array(items), "zone": np.array(zones), "surprisal": np.array(surps)}
    for l, rows in layers.items(): out[f"layer_{l}"] = np.stack(rows)
    np.savez_compressed(out_path, **out)
    print("wrote", out_path)

t = ns.load()
words = t.words.sort_values(["item", "zone"])
groups = [(int(item), grp["word"].tolist()) for item, grp in words.groupby("item", sort=True)]
# zone numbering in words.tsv is 1..n per story in order, matching wi+1 above
extract_words(groups, f"hs_naturalstories_{TAG}.npz")""")

md("## Sandman: per-segment hidden states (last token of each segment, full left context) and segment surprisal")

code("""from judge import sandman
s = sandman.load()
segs = s.segments_en   # English machine translation (German original is the stimulus; label as translated)
text_so_far = ""
seg_rows = {}; seg_surp = []; seg_idx = []
# Feed cumulative context so each segment's representation sees everything before it.
# Llama handles ~7k words; the full story is ~8.5k tokens, within 128k context.
full = "\\n\\n".join(segs)
enc = tok(full, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True)
offsets = enc.pop("offset_mapping")[0].tolist()
ids = enc["input_ids"][0]
hs, tok_surp, (topv, topi) = forward_stats(enc, topk=64)
# segment char spans
bounds, cur = [], 0
for seg in segs:
    bounds.append((cur, cur + len(seg))); cur += len(seg) + 2
seg_of_tok = np.array([next((i for i, (a, b) in enumerate(bounds) if a <= oa < b), -1) if ob > oa else -1 for oa, ob in offsets])
res = {"segment": np.arange(1, 66), "surprisal_sum": np.zeros(65), "surprisal_mean": np.zeros(65), "n_tokens": np.zeros(65)}
for i in range(65):
    m = seg_of_tok == i
    res["n_tokens"][i] = m.sum(); res["surprisal_sum"][i] = float(tok_surp[m].sum()); res["surprisal_mean"][i] = float(tok_surp[m].mean())
    last = np.where(m)[0].max()
    for l, h in enumerate(hs):
        res.setdefault(f"layer_{l}_last", []).append(h[last].numpy().astype(np.float16))
        res.setdefault(f"layer_{l}_mean", []).append(h[m].mean(0).numpy().astype(np.float16))
for k in list(res):
    if k.startswith("layer_"): res[k] = np.stack(res[k])
# also save the full per-token next-token log-prob top-k for belief-dynamics work (top 64 per position)
res["tok_segment"] = seg_of_tok; res["tok_surprisal"] = tok_surp.numpy()
res["tok_top64_logp"] = topv.numpy().astype(np.float16); res["tok_top64_ids"] = topi.numpy().astype(np.int32)
np.savez_compressed(f"hs_sandman_{TAG}.npz", **res)
print(f"wrote hs_sandman_{TAG}.npz; tokens:", len(ids))""")

md("## Quick analyses on the runtime (Sandman probe, memorisation), then download")

code("""import subprocess, sys, shutil, os
subprocess.run(["git", "pull", "-q"], check=False)
for cmd in (["scripts/run_sandman_model.py", "--hs", f"hs_sandman_{TAG}.npz", "--name", TAG],
            ["scripts/run_memorisation.py", "--sandman", f"hs_sandman_{TAG}.npz", "--naturalstories", f"hs_naturalstories_{TAG}.npz", "--name", TAG]):
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-1500:] if r.returncode else "")
shutil.make_archive(f"results_{TAG}", "zip", "results")
print(sorted(os.listdir("results")))""")

md("## Download: results zip (small) and the two hidden-state files (the Natural Stories one is ~1.7 GB; the full per-layer probe runs off-Colab)")

code("""from google.colab import files
files.download(f"results_{TAG}.zip")
files.download(f"hs_sandman_{TAG}.npz")
files.download(f"hs_naturalstories_{TAG}.npz")""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
Path(__file__).with_name("judge_extract.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote notebooks/judge_extract.ipynb")
