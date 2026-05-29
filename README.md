# LIFE: Prompt-Induced Linguistic Fingerprints for LLM-Generated Fake News Detection

This is the official code of the paper *"Prompt-Induced Linguistic Fingerprints for
LLM-Generated Fake News Detection"*.

The core idea: for each news article, a malicious "write fake news" prompt is prepended
and a set of language models computes the per-token log-likelihood ("perplexity
fingerprint") of the text. A sequence classifier is then trained on those fingerprints to
detect LLM-generated fake news.

---

## Two ways to run this

**(A) Google Colab notebook — recommended, works out of the box.** `LIFE_colab.ipynb`
runs the whole pipeline on a free Colab GPU over a fixed, working subset of the method
(PolitiFact++, GPT-2 features, 4 classes). The bugs listed further down are already fixed
in the scripts it calls, and it avoids the multi-GPU mosec server entirely. See
[Quick start: Google Colab](#quick-start-google-colab-recommended) below.

**(B) The original server-based pipeline — research code, needs manual work.** The
multi-model mosec inference server (`backend_api.py` and friends) as released has hardcoded
paths, missing files, and undefined classes. It **will not run as-is**. Kept for reference
under [The original server-based pipeline](#the-original-server-based-pipeline-advanced).

---

## Quick start: Google Colab (recommended)

Runs **convert → key sentences → concatenate → features → train** on **PolitiFact++**
(~520 articles), classifying into 4 classes (human-fake / human-real / machine-fake /
machine-real) using **GPT-2** for the perplexity fingerprints. Fits a free-tier T4.

> **VLPFN is intentionally excluded:** its `content` has all punctuation stripped (real news
> too), so sentence segmentation — which the whole method relies on — cannot work.
> GossipCop++ works as well, but Step 1 takes hours on a T4; start with PolitiFact++.

**Steps:**
1. Upload the whole `LIFE` repo (including `dataset/data/`) to Google Drive, e.g. `MyDrive/LIFE`.
2. Open `LIFE_colab.ipynb` in Colab and set the runtime to **GPU** (Runtime → Change runtime type → T4).
3. Edit `PROJECT_DIR` in the path cell if you didn't use `MyDrive/LIFE`.
4. Run all cells top to bottom.

The notebook is only an orchestrator; the real work lives in these scripts, which are also
runnable standalone:

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `dataset/0_convert.py` | Convert PolitiFact++ `HF/HR/MF/MR.json` → `*_fake.jsonl` / `*_true.jsonl` with 4-class labels |
| 1 | `dataset/1_keySentenceExtraction.py` | Train BERT, extract top-20 key sentences |
| 2 | `dataset/2_concate.py` | Merge key sentences back by `(id, label)` |
| 3 | `dataset/3_gen_features_local.py` | GPT-2 log-likelihood features **in-process** (no server) |
| 4 | `LIFE_train/train.py` | Train the Transformer + CRF classifier |

Equivalent command line (run from the repo root; paths are examples):

```bash
python dataset/0_convert.py \
    --input_dir "dataset/data/Fakenews-dataset-main/Fakenews-dataset-main/Dataset/PolitiFact++" \
    --output_dir dataset/output_raw
python dataset/1_keySentenceExtraction.py \
    --data_dir dataset/output_raw \
    --output_file dataset/keySentence/important_sentences_top20.jsonl --gpu 0
python dataset/2_concate.py \
    --folder_path dataset/output_raw \
    --important_sentences_file dataset/keySentence/important_sentences_top20.jsonl
python dataset/3_gen_features_local.py \
    --input_dir dataset/output_raw --output_dir dataset/features --model gpt2 --gpu 0
python LIFE_train/train.py \
    --split_dataset --data_path dataset/features \
    --train_path dataset/train.jsonl --test_path dataset/test.jsonl \
    --model Transformer --num_train_epochs 2
```

**Label mapping (Step 0):** HF→`human_fake`, HR→`human_true`, MF→`gpt3.5_fake`,
MR→`gpt3.5_true`. To use only the two machine classes instead, drop HF/HR and re-comment
the `human_*` labels in `train.py`'s `en_labels`.

> **Re-runs:** Step 2 rewrites the files in `output_raw` in place (adds a `sentence` field),
> so re-run Step 0 before re-running Steps 1–3 from scratch.

> **fastNLP** (the CRF layer in `LIFE_train/model.py`) is the one fragile dependency — if
> Step 4 errors on `from fastNLP.modules.torch import ...`, run `pip install fastNLP==1.0.1`
> and restart the runtime.

---

## Installation

```bash
pip install -r requirements.txt
```

PyTorch with CUDA is best installed separately to match your CUDA version — see
<https://pytorch.org/get-started/locally/>.

After installing, download the NLTK sentence tokenizer data (used by steps 1 and 4).
Newer `nltk` needs `punkt_tab` as well as `punkt`:

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Datasets

The LLM-generated datasets used are **Politifact++** and **GossipCop++**; the human–LLM
mixed dataset is **VLFPN**. Download links:

- https://github.com/mbzuai-nlp/Fakenews-dataset
- https://www.dropbox.com/scl/fo/1kf2up2ge0v13izbr7z2e/h?rlkey=xzhm0dbmqevee8f76asz5cyuw&e=1&dl=0

The pipeline expects each input record to be a JSON line of the form
`{'text': ..., 'label': ..., 'id': ...}` in files ending `_fake.jsonl` / `_true.jsonl`.
The **raw downloaded data is not in this format** — PolitiFact++/GossipCop++ ship as JSON
dicts (`HF/HR/MF/MR.json`) and VLPFN as CSV. `dataset/0_convert.py` performs the conversion
for PolitiFact++ (see the [Colab quick start](#quick-start-google-colab-recommended)).

---

# The original server-based pipeline (advanced)

> The sections below document the **original** multi-model server pipeline as released.
> The [Colab quick start](#quick-start-google-colab-recommended) above does **not** need any
> of this. Bugs marked ✅ below are already fixed in the scripts the Colab path uses; bugs
> marked ⚠️ remain in the (Colab-unused) server stack.

## Pipeline overview

The README's original 3-step order is incomplete: step 3 (`3_gen_features.py`) is an HTTP
**client** that talks to model servers you must start first with `backend_api.py`. The
true order is:

```
1_keySentenceExtraction.py
        │
        ▼
2_concate.py
        │
        ▼
[ start backend_api.py server(s) ]   ◄── must be running for the next step
        │
        ▼
3_gen_features.py
        │
        ▼
LIFE_train/train.py
```

| Stage | File | What it does |
|-------|------|--------------|
| 1 | `dataset/1_keySentenceExtraction.py` | Trains/loads a BERT classifier, then removes one sentence at a time to find the **top-20 most impactful sentences** per article. Outputs `{id, sentence, index, label}` JSONL. |
| 2 | `dataset/2_concate.py` | Merges those key sentences back into the original news JSONL by matching on `(id, label)`. **⚠️ Overwrites the dataset files in place.** |
| — | `backend_api.py` + `backend_model.py` + `backend_utils.py` | A [mosec](https://github.com/mosecorg/mosec) inference server. Each LLM (GPT-2, GPT-Neo, GPT-J, Llama) computes per-token log-likelihood. `backend_utils.getPrompt()` prepends the fake-news prompt — the "Prompt-Induced" part of the method. |
| 3 | `dataset/3_gen_features.py` | Sends each article to the running server(s) and writes the log-likelihood token features to a JSONL. |
| 4 | `LIFE_train/train.py` + `dataloader.py` + `model.py` | Loads the feature files, pads/aligns per-token features, and trains a CNN/Transformer + CRF sequence classifier (BMES tagging over `gpt3.5_fake` / `gpt3.5_true`). |

---

## Running (server path)

Steps 0, 1, 2, and 4 use the **same commands as the
[Colab quick start](#quick-start-google-colab-recommended)**. The only difference in the
original pipeline is Step 3, which uses the mosec server instead of
`dataset/3_gen_features_local.py`. Apply the server-stack fixes from the sections below first.

### Step 3 — generate fingerprint features (server)

First start a backend inference server in a **separate terminal** (needs a GPU):

```bash
# from the repo root
python backend_api.py --port 6006 --timeout 30000 --model gpt2 --gpu 0
# --model can be one of the classes actually defined in backend_model.py:
#   gpt2, gptneo, gptj, llama   (see "Known issues" about the others)
```

Then run the feature generator (it posts to http://0.0.0.0:6006/inference):

```bash
cd dataset
python 3_gen_features.py --input_file <in.jsonl> --output_file <out.jsonl> --get_en_features
```

To use more than one model, start each on its own port (the client defaults are
6006=gpt2, 6007=gptneo, 6008=gptj, 6009=llama) and add them to `en_model_apis` in
`3_gen_features.py`.

### Step 4 — train

Same as the [Colab quick start](#quick-start-google-colab-recommended). Useful `train.py`
flags: `--model {Transformer,CNN,RNN}`, `--batch_size`, `--seq_len`, `--num_train_epochs`,
`--lr`, `--do_test`.

---

## Required configuration (must do before running)

These values are empty or hardcoded to the original author's machine and **must** be edited:

> The dataset scripts (`1_keySentenceExtraction.py`, `2_concate.py`) now take their paths as
> **CLI arguments** (see the Colab quick start) — the empty-path problems below are resolved.
> The remaining rows apply only to the server stack.

| File | Location | Problem | Action |
|------|----------|---------|--------|
| `backend_model.py` | `/home/wangchi/gpt2`, `/home/wangchi/gpt-neo-2.7b`, `/home/wangchi/gpt-j-6b`, `/home/wangchi/Llama-2-7B` | Hardcoded Linux model paths | Change to your local model weights (or HF ids) |
| `LIFE_train/train.py` | `--data_path` / `--train_path` / `--test_path` defaults | Hardcoded `/home/wangchi/...` paths | Pass correct paths on the CLI |

For the server stack you must also **download the LLM weights** referenced in
`backend_model.py` (GPT-2, GPT-Neo-2.7B, GPT-J-6B, Llama-2-7B).

---

## Known issues / bugs

1. ⚠️ **Missing file `backend_t5.py`** — imported by `backend_api.py` (`from backend_t5 import T5`). Not in the repo, so `backend_api.py` fails to import. Either supply this file or remove the T5 import/branch. *(The Colab path's `3_gen_features_local.py` avoids the server entirely.)*
2. ✅ **Missing file `backend_model_info.py`** — was imported by `LIFE_train/train.py`. **Fixed:** the unused import has been removed.
3. ⚠️ **Undefined model classes** — `backend_api.py` imports 8 classes (`SnifferWenZhongModel`, `SnifferSkyWorkModel`, `SnifferDaMoModel`, `SnifferChatGLMModel`, `SnifferAlpacaModel`, `SnifferDollyModel`, `SnifferStableLMRawModel`, `SnifferStableLMTunedModel`) that are **not defined** in `backend_model.py` (only Base/GPT2/GPTNeo/GPTJ/Llama exist). Remove the unused imports or implement the classes; only use defined `--model` values.
4. ✅ **`OUTPUT_FILE` NameError** — `dataset/1_keySentenceExtraction.py` referenced an undefined `OUTPUT_FILE`. **Fixed:** the output path is now the `--output_file` argument.
5. ⚠️ **Undefined `prompt` in Llama path** — `backend_utils.py` (`SPLlamaTokenizerPPLCalc.forward_calc_ppl`) references a variable `prompt` that is never assigned in that scope → `NameError` when using the Llama backend with a sentence. The GPT-2/BBPE path (used by the Colab pipeline) is unaffected; mirror its logic if you need Llama.
6. ✅ **`test()` signature mismatch** — `train.py` called `trainer.test(content_level_eval=...)` but `test()` takes no such argument. **Fixed:** the call is now `trainer.test()`.
7. ⚠️ **Path typo** — `train.py` default `--test_path` is `/home/wangchi/LIFET/dataset/test` (stray `T` in `LIFET`). Still present in the default; the Colab path passes `--test_path` explicitly to avoid it.

## Platform notes

- ✅ `transformers` `AdamW` — `train.py` now imports `AdamW` from `torch.optim` (it was removed from `transformers.optimization` in newer versions).
- ✅ `dataset/1_keySentenceExtraction.py` no longer hardcodes `CUDA_VISIBLE_DEVICES = '1'`; it takes `--gpu` (default `0`). The server-stack scripts still hardcode a device — adjust for your GPU layout.
- `bitsandbytes` (8-bit loading, server stack only) has limited Windows support; Linux/WSL2/Colab is recommended.
- `nltk` requires `punkt` (and `punkt_tab` on newer versions) — see [Installation](#installation).
