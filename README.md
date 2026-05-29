# LIFE: Prompt-Induced Linguistic Fingerprints for LLM-Generated Fake News Detection

This is the official code of the paper *"Prompt-Induced Linguistic Fingerprints for
LLM-Generated Fake News Detection"*.

The core idea: for each news article, a malicious "write fake news" prompt is prepended
and a set of language models computes the per-token log-likelihood ("perplexity
fingerprint") of the text. A sequence classifier is then trained on those fingerprints to
detect LLM-generated fake news.

---

## ⚠️ Read this before you start

This is research code released without all of its glue. **It will not run as-is** even
with every library installed. Before running anything you must fill in hardcoded paths,
supply two missing files, download model weights, and fix a few bugs. All of this is
documented in the [Required configuration](#required-configuration-must-do-before-running)
and [Known issues / bugs](#known-issues--bugs) sections below. Read them first.

Hardware: the backend models load with `load_in_8bit=True` and `device_map="auto"`, so a
**CUDA GPU is required** (plus `bitsandbytes`, which historically has poor Windows
support — Linux or WSL2 is recommended).

---

## Installation

```bash
pip install -r requirements.txt
```

PyTorch with CUDA is best installed separately to match your CUDA version — see
<https://pytorch.org/get-started/locally/>.

After installing, download the NLTK sentence tokenizer data (used by steps 1 and 4):

```python
python -c "import nltk; nltk.download('punkt')"
```

---

## Datasets

The LLM-generated datasets used are **Politifact++** and **GossipCop++**; the human–LLM
mixed dataset is **VLFPN**. Download links:

- https://github.com/mbzuai-nlp/Fakenews-dataset
- https://www.dropbox.com/scl/fo/1kf2up2ge0v13izbr7z2e/h?rlkey=xzhm0dbmqevee8f76asz5cyuw&e=1&dl=0

Each input record is a JSON line of the form `{'text': ..., 'label': ..., 'id': ...}`.
Filenames are expected to end in `_fake.jsonl` or `_true.jsonl`.

---

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

## Running

> Fill in all paths and apply the fixes from the sections below first.

### Step 1 & 2 — dataset preparation

```bash
cd dataset
python 1_keySentenceExtraction.py
python 2_concate.py
```

### Step 3 — generate fingerprint features

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

```bash
cd LIFE_train
python train.py \
    --split_dataset \
    --data_path <dir with feature jsonl files> \
    --train_path <train.jsonl> \
    --test_path <test.jsonl>
```

Useful `train.py` flags: `--model {Transformer,CNN,RNN}`, `--batch_size`, `--seq_len`,
`--num_train_epochs`, `--lr`, `--do_test`.

---

## Required configuration (must do before running)

These values are empty or hardcoded to the original author's machine and **must** be edited:

| File | Location | Problem | Action |
|------|----------|---------|--------|
| `dataset/1_keySentenceExtraction.py` | `DATA_DIR = ''` (line ~15) | Empty input folder | Point to your dataset folder |
| `dataset/2_concate.py` | `folder_path = ''` (line ~5) | Empty input folder | Point to your dataset folder |
| `dataset/2_concate.py` | `important_sentences_file` (line ~6) | Path to step-1 output | Point to step 1's output JSONL |
| `backend_model.py` | `/home/wangchi/gpt2`, `/home/wangchi/gpt-neo-2.7b`, `/home/wangchi/gpt-j-6b`, `/home/wangchi/Llama-2-7B` | Hardcoded Linux model paths | Change to your local model weights |
| `LIFE_train/train.py` | `--data_path` / `--train_path` / `--test_path` defaults (lines ~279–281) | Hardcoded `/home/wangchi/...` paths | Pass correct paths (or edit defaults) |

You must also **download the LLM weights** referenced in `backend_model.py` (GPT-2,
GPT-Neo-2.7B, GPT-J-6B, Llama-2-7B) to local directories.

---

## Known issues / bugs

The following must be fixed for the corresponding code path to run:

1. **Missing file `backend_t5.py`** — imported by `backend_api.py` (`from backend_t5 import T5`). Not in the repo, so `backend_api.py` fails to import. Either supply this file or remove the T5 import/branch.
2. **Missing file `backend_model_info.py`** — imported by `LIFE_train/train.py`. Not in the repo. Supply it or remove the import (it does not appear to be otherwise used).
3. **Undefined model classes** — `backend_api.py` imports 8 classes (`SnifferWenZhongModel`, `SnifferSkyWorkModel`, `SnifferDaMoModel`, `SnifferChatGLMModel`, `SnifferAlpacaModel`, `SnifferDollyModel`, `SnifferStableLMRawModel`, `SnifferStableLMTunedModel`) that are **not defined** in `backend_model.py` (only Base/GPT2/GPTNeo/GPTJ/Llama exist). The import fails. Remove the unused imports or implement the classes; only use `--model` values that are actually defined.
4. **`OUTPUT_FILE` NameError** — `dataset/1_keySentenceExtraction.py` (line ~152) writes to `OUTPUT_FILE`, which is never defined. Define it near `DATA_DIR`, e.g. `OUTPUT_FILE = 'keySentence/MF/important_sentences_top20.jsonl'`.
5. **Undefined `prompt` in Llama path** — `backend_utils.py` (line ~432, `SPLlamaTokenizerPPLCalc.forward_calc_ppl`) references a variable `prompt` that is never assigned in that scope → `NameError` when using the Llama backend with a sentence. (The GPT-2/BBPE path computes `prompt` correctly; mirror that logic.)
6. **`test()` signature mismatch** — `train.py` (line ~346) calls `trainer.test(content_level_eval=args.test_content)`, but `SupervisedTrainer.test()` takes no such argument → `TypeError` when running with `--do_test`. Remove the argument or add the parameter.
7. **Path typo** — `train.py` default `--test_path` is `/home/wangchi/LIFET/dataset/test` (note the stray `T` in `LIFET`). Pass `--test_path` explicitly to avoid it.

## Platform notes

- Several scripts hardcode `os.environ['CUDA_VISIBLE_DEVICES'] = '1'` — adjust for your GPU layout.
- `bitsandbytes` (8-bit loading) has limited Windows support; Linux or WSL2 is recommended.
- `nltk` requires the `punkt` data (see [Installation](#installation)).
- `transformers` must be old enough to still expose `AdamW` from `transformers.optimization` (used in `train.py`); on newer versions, switch to `torch.optim.AdamW`.
