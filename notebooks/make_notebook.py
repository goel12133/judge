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
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, load_in_8bit=True, device_map="auto").eval()
print("model loaded; layers:", model.config.num_hidden_layers)""")

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
        with torch.no_grad():
            out = model(**{k: v.to(model.device) for k, v in enc.items()}, output_hidden_states=True)
        logp = torch.log_softmax(out.logits[0].float(), -1)
        ids = enc["input_ids"][0]
        tok_surp = torch.zeros(len(ids))
        tok_surp[1:] = -logp[:-1].gather(1, ids[1:, None].to(logp.device))[:, 0].cpu()
        hs = [h[0].float().cpu() for h in out.hidden_states]
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
        del out, hs, logp; torch.cuda.empty_cache()
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
with torch.no_grad():
    out = model(**{k: v.to(model.device) for k, v in enc.items()}, output_hidden_states=True)
logp = torch.log_softmax(out.logits[0].float(), -1)
ids = enc["input_ids"][0]
tok_surp = torch.zeros(len(ids)); tok_surp[1:] = -logp[:-1].gather(1, ids[1:, None].to(logp.device))[:, 0].cpu()
hs = [h[0].float().cpu() for h in out.hidden_states]
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
topv, topi = torch.topk(logp, 64, dim=-1)
res["tok_segment"] = seg_of_tok; res["tok_surprisal"] = tok_surp.numpy()
res["tok_top64_logp"] = topv.cpu().numpy().astype(np.float16); res["tok_top64_ids"] = topi.cpu().numpy().astype(np.int32)
np.savez_compressed(f"hs_sandman_{TAG}.npz", **res)
print(f"wrote hs_sandman_{TAG}.npz; tokens:", len(ids))""")

md("## Download the two files, then drop them into the chat")

code("""from google.colab import files
files.download(f"hs_naturalstories_{TAG}.npz")
files.download(f"hs_sandman_{TAG}.npz")""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
Path(__file__).with_name("judge_extract.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote notebooks/judge_extract.ipynb")
