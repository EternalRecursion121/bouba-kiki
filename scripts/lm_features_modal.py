"""Extract small-LM features for every item, on Modal.

For each item and prompt template, returns the last-token hidden state at a set of layers, plus a
zero-shot score: log P(" bouba") − log P(" kiki") as the answer to a direct question.

Usage (from the repo root):
  /root/vault-hallucination/.venv/bin/modal run scripts/lm_features_modal.py
Writes scripts/.lm_features.npz (gitignored via scripts/.cache*? no — see .gitignore entry below).
"""
import json, os
import modal

MODEL = os.environ.get("LM", "Qwen/Qwen2.5-1.5B-Instruct")
TEMPLATES = {
    "vibe": 'Think about the overall vibe of "{w}": its feel, sound, speed, texture and temperament. The vibe of "{w}" is',
    "plain": "{w}",
}
ASK = ('In the bouba/kiki effect, "bouba" means round, soft, slow, warm, mellow and "kiki" means sharp, '
       'spiky, quick, bright, crisp. Going by overall vibe, is "{w}" more bouba or kiki? Answer with one word.')

app = modal.App("bouba-kiki-lm-features")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.46.3", "accelerate", "numpy")


@app.function(image=image, gpu="A10G", timeout=20 * 60)
def extract(items: list[str], model_name: str) -> bytes:
    import io, numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda").eval()
    n_layers = model.config.num_hidden_layers
    layers = sorted({int(round(f * n_layers)) for f in (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)})
    out = {"layers": np.array(layers)}

    @torch.no_grad()
    def last_hidden(prompts):
        feats = []
        for i in range(0, len(prompts), 64):
            enc = tok(prompts[i:i + 64], return_tensors="pt", padding=True).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states
            feats.append(torch.stack([hs[l][:, -1, :] for l in layers], 1).float().cpu().numpy())
        return np.concatenate(feats).astype(np.float16)

    for name, t in TEMPLATES.items():
        out[name] = last_hidden([t.format(w=w) for w in items])

    # zero-shot: compare the first token of " bouba"/"bouba" vs " kiki"/"kiki" after the chat prompt
    ids_b = {tok.encode(s, add_special_tokens=False)[0] for s in ("bouba", " bouba", "Bouba", " Bouba")}
    ids_k = {tok.encode(s, add_special_tokens=False)[0] for s in ("kiki", " kiki", "Kiki", " Kiki")}
    prompts = [tok.apply_chat_template([{"role": "user", "content": ASK.format(w=w)}], tokenize=False, add_generation_prompt=True)
               for w in items]
    zs = []
    with torch.no_grad():
        for i in range(0, len(prompts), 64):
            enc = tok(prompts[i:i + 64], return_tensors="pt", padding=True).to("cuda")
            lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), -1)
            zs.append((torch.logsumexp(lp[:, list(ids_b)], -1) - torch.logsumexp(lp[:, list(ids_k)], -1)).cpu().numpy())
    out["zeroshot"] = np.concatenate(zs)
    buf = io.BytesIO(); np.savez_compressed(buf, **out); return buf.getvalue()


@app.local_entrypoint()
def main():
    items = json.load(open("scripts/.lm_items.json"))
    data = extract.remote(items, MODEL)
    path = f"scripts/.lm_features_{MODEL.split('/')[-1]}.npz"
    open(path, "wb").write(data)
    print(f"wrote {path} for {len(items)} items")
