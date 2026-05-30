# Project Memory — LIFE on Google Colab

Hand-off summary so a new session can pick up with full context. Last updated: 2026-05-29.

---

## 1. What this repo is

Official code for the paper *"Prompt-Induced Linguistic Fingerprints for LLM-Generated Fake
News Detection" (LIFE)*. Method: prepend a malicious "write fake news" prompt to each
article, have language model(s) compute per-token log-likelihood ("perplexity
fingerprints"), then train a sequence classifier (CNN/Transformer + CRF, BMES tagging) on
those fingerprints to detect LLM-generated fake news.

It was released as research code **without working glue**: hardcoded Linux paths, two
missing imported files, undefined model classes, a mosec client/server architecture, and
several runtime bugs.

## 2. Goal of our work

Make the pipeline **runnable on Google Colab** (free T4 GPU), since Colab is Linux+GPU and
solves the Windows/`bitsandbytes` blocker. User is on Windows 11 locally and **does not have
the Python libraries installed locally** — so nothing is run locally; all execution happens
on Colab by the user. We only do static checks (`py_compile`, JSON validation).

## 3. Key decisions (already made by the user)

- **Dataset:** PolitiFact++ first (~520 articles, fits free T4). GossipCop++ (~20k) deferred
  — its Step 1 is hours on a T4.
- **Label scheme: 4-class.** HF→`human_fake`, HR→`human_true`, MF→`gpt3.5_fake`,
  MR→`gpt3.5_true`.
- **VLPFN dropped.** Its CSV `content` has ALL punctuation stripped (verified, real news
  too), so `nltk` sentence segmentation — which the whole method depends on — collapses each
  article to one "sentence". Unusable for LIFE.
- **Approach "A2":** a notebook (`LIFE_colab.ipynb`) is a thin setup+orchestration layer;
  real logic stays in `.py` files. Steps 0 & 3 are new scripts; steps 1/2/4 are the original
  scripts with surgical bug fixes. The mosec server/HTTP path is replaced by in-process
  GPT-2 feature extraction (which also makes missing `backend_t5.py` and undefined Sniffer
  classes irrelevant).

## 4. Data layout (downloaded locally under `dataset/data/`)

- `dataset/data/Fakenews-dataset-main/Fakenews-dataset-main/Dataset/PolitiFact++/` —
  four files `HF.json`, `HR.json`, `MF.json`, `MR.json`. Each is a **single JSON dict**
  keyed `"0","1",...`, each value `{id, text, title, description}`. Counts: HF=97, MF=97,
  HR=194, MR=132 (=520). All records have `text`+`id`. **Punctuation intact** (good).
  (GossipCop++ sits in the same parent dir, same format.)
- `dataset/data/VLPFN/` — `train.csv`/`test.csv`/`dev.csv`. Columns:
  `, Unnamed: 0, content, folder, source, det_fake_label, idx`. **Dropped** (punctuation
  stripped). HF/MF/HR/MR meaning: machine-real (MR, paraphrased), machine-fake (MF),
  human-fake (HF), human-real (HR).

## 5. The pipeline (PolitiFact++, 4-class)

Order: **0 convert → 1 key sentences → 2 concatenate → 3 features (in-process) → 4 train**

Equivalent CLI (run from repo root; the notebook runs these via `!python`):
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
Notes:
- Step 0 emits `HF_fake.jsonl`/`MF_fake.jsonl`/`HR_true.jsonl`/`MR_true.jsonl`, lines
  `{id,text,label}`. The `_fake`/`_true` suffix drives Step 1's binary BERT;
  the fine `label` carries the 4-class signal.
- **Step 2 overwrites the files in `output_raw` in place** (adds a `sentence` field) — re-run
  Step 0 before re-running 1–3 from scratch.
- Step 4 uses 2 epochs as a smoke test; repo default is 50.

## 6. Files we created / changed

**New:**
- `dataset/0_convert.py` — converter (PolitiFact++ JSON dicts → JSONL, 4-class labels). CLI:
  `--input_dir`, `--output_dir`.
- `dataset/3_gen_features_local.py` — in-process GPT-2 feature extractor. Loads HF `gpt2`,
  builds `BBPETokenizerPPLCalc` from `backend_utils.py` (no mosec). Writes the exact JSONL
  keys `LIFE_train/dataloader.py` expects (`losses`, `begin_idx_list`,
  `sentence_ll_tokens_list`, `label_int`, `label`, `text`). CLI: `--input_dir`,
  `--output_dir`, `--model` (default gpt2), `--gpu` (default 0). Wraps scoring in
  `torch.no_grad()`.
- `LIFE_colab.ipynb` — orchestration notebook: nvidia-smi → mount Drive → set `PROJECT_DIR`
  (default `/content/drive/MyDrive/LIFE`) → `pip install -r requirements.txt` → nltk
  `punkt`+`punkt_tab` → steps 0–4.
- `requirements.txt` — all deps from the imports; **`fastNLP==1.0.1` is pinned** (the train
  code imports `fastNLP.modules.torch`, a 1.x API; 0.x fails). Torch/CUDA installed
  separately; transformers `AdamW` note.

**Edited (surgical):**
- `dataset/1_keySentenceExtraction.py` — added `argparse` (`--data_dir`, `--output_file`,
  `--gpu`); **defined the previously-undefined `OUTPUT_FILE`** (now `args.output_file`);
  GPU index now `--gpu` default `0` (was hardcoded `'1'`, which would hide Colab's only GPU).
- `dataset/2_concate.py` — `folder_path` / `important_sentences_file` now from `argparse`.
- `LIFE_train/train.py` — uncommented the 4 labels in `en_labels` (`human_fake`,
  `human_true`, `gpt3.5_fake`, `gpt3.5_true`); removed dead `import backend_model_info`
  (missing file); `AdamW` now from `torch.optim` (removed from `transformers.optimization`
  in new versions); fixed `trainer.test(content_level_eval=...)` → `trainer.test()`.
- `README.md` — restructured into (A) Colab quick start [recommended] and (B) original
  server pipeline [advanced]; known-issues list annotated ✅ fixed vs ⚠️ still-present.

## 7. Status

- **Pipeline runs end-to-end on Colab (PolitiFact++).** Steps 0→4 all execute. Two extra
  fixes were needed at runtime (see §7a). Step 1 took ~30 min for 520 articles.
- Smoke test = `--num_train_epochs 2`: Accuracy ~31–35% on the 4 classes (≈ majority-class
  baseline — HR is 194/520 ≈ 37%), train_loss falling 3.11→2.86. Plumbing confirmed; **not
  converged**. For real numbers run `--num_train_epochs 50` (repo default; still quick on a T4).
- Open question: whether GPT-2-only features on tiny PolitiFact++ reach the paper's accuracy
  (paper used a model ensemble). Not yet evaluated at full epochs.

### 7a. Colab-run fixes applied during execution (beyond the original plan)

1. `dataset/3_gen_features_local.py` — newer transformers (4.x, py3.12) no longer exports
   `bytes_to_unicode` from `transformers.models.gpt2.tokenization_gpt2`. Import is now
   try/except with a local fallback definition of the canonical GPT-2 byte→unicode map.
2. `LIFE_train/train.py` — PyTorch 2.6 changed `torch.load` default to `weights_only=True`,
   which can't unpickle the full saved model object (custom class). Both `torch.load(ckpt_name)`
   calls now pass `weights_only=False` (checkpoint is self-produced/trusted). NOTE: per-epoch
   saves succeed regardless; only the end-of-train reload + `--do_test` path needed this.

## 7b. Hardware

- Now running on **Colab A100** (Pro/Pro+), not T4. ~40–80 GB VRAM, much faster.
- Implications: GossipCop++ Step 1 (the expensive leave-one-sentence-out BERT pass) is now
  practical; can also swap Step 3's `--model gpt2` for larger feature LMs (e.g. `gpt-j`,
  `llama`) without 8-bit quantization, and raise `--batch_size`. Costs compute units.

## 8. Known risks / things to watch on first Colab run

- **fastNLP** (Step 4, `LIFE_train/model.py`): pin is a best guess. If the install line
  errors, try another `1.0.x`. If the import errors after install, **restart the Colab
  runtime** (import is cached) — do NOT re-run `requirements.txt`.
- nltk: needs `punkt` AND `punkt_tab` (train.py loads `tokenizers/punkt/english.pickle`).
- Step 1 is the slow step (a BERT forward pass per sentence per article).
- Checkpoints `gpt3.5_bert_model.pt` (step 1) and `linear_en.pt` (step 4) write to cwd =
  `PROJECT_DIR` on Drive (survive disconnects).

## 9. Untouched / out of scope

- Server stack (`backend_api.py`, `backend_model.py`, `backend_t5` [missing], `backend_utils.py`)
  is unused by the Colab path and left as-is. Remaining ⚠️ bugs there: missing `backend_t5.py`,
  8 undefined `Sniffer*` classes imported in `backend_api.py`, undefined `prompt` var in
  `backend_utils.py` Llama path, `LIFET` typo in `train.py`'s default `--test_path`.
- GossipCop++ and VLPFN not wired up.

## 7c. Realignment to the paper: binary MF-vs-MR + LLaMA2-7B (edits applied, run pending)

After reading the paper (§8b) we found the 4-class setup was wrong — the paper is **binary
MF (fake) vs MR (real)** with **LLaMA2-7B** reconstruction. gpt2 baseline was 51.7% (4-class,
50ep); the GPT-J-6B pass was abandoned (GPT-J is only a *generation* model in the paper).
User chose full faithful reproduction. Applied edits:
- `dataset/0_convert.py` — `--subset {all,llm}`; `llm` emits only MF_fake.jsonl + MR_true.jsonl.
- `dataset/1_keySentenceExtraction.py` — `--top_k` (default 10, paper's k) + `--model_path`
  (so the binary run trains a FRESH BERT extractor, not the stale 4-class one).
- `LIFE_train/train.py` — `en_labels` reverted to 2-class (gpt3.5_fake/gpt3.5_true = released default).
- `backend_utils.py` — fixed `SPLlamaTokenizerPPLCalc.forward_calc_ppl` (prepend malicious
  prompt, handle sentence LIST, remove undefined `prompt`) — mirrors the BBPE path.
- `dataset/3_gen_features_local.py` — `--scorer {bbpe,llama}` (+ `--dtype` from the gpt-j pass,
  kept); llama branch builds `SPLlamaTokenizerPPLCalc` with slow `LlamaTokenizer`(use_fast=False).
- `LIFE_colab.ipynb` — vars OUTPUT_BIN/FEATURES_LLAMA/BERT_CKPT/TRAIN_PATH/TEST_PATH; added an
  HF-login cell (LLaMA-2 gated); Step 0 `--subset llm`, Step 1 `--top_k 10 --model_path`,
  Step 3 `--model meta-llama/Llama-2-7b-hf --scorer llama --dtype bfloat16`, Step 4 binary 50ep.
- **Gating hit (run 1):** `meta-llama/Llama-2-7b-hf` is gated (403, needs Meta approval).
  Switched Step 3 to the **ungated** `NousResearch/Llama-2-7b-hf` mirror (same weights/tokenizer,
  no token). Notebook gating notes/login-cell softened to "optional".
- **Tokenizer fix (run 2):** mirror loaded fine, but `SPLlamaTokenizerPPLCalc.get_bbs_ll` crashed
  with `TokenizersBackend has no attribute sp_model` (newer transformers backend). Replaced
  `self.base_tokenizer.sp_model.IsByte(id)` with `token in self.byte_decoder` (byte tokens are
  the `<0x##>` strings). FIXED — path now runs end to end.

### ✅ RESULT (full faithful run, binary MF-vs-MR + LLaMA2-7B, PolitiFact++, 50 epochs)
- **Accuracy 85.5% / Macro-F1 80.8%.** Paper target 0.900 / 0.882 → ~4.5 / ~7.4 pts below.
- Big jump from the 51.7% (4-class/gpt2) setup; the realignment worked. Pipeline reproduces
  the paper's ballpark.

### Gap analysis / levers to close it (for next session, optional)
- **Tiny test set:** PolitiFact++ binary test ≈ 46 samples (20% of 229), so ~2 samples ≈ 4-5 pts.
  85.5 vs 90 is largely within variance — could re-run with different seeds / report mean±std.
- **Classifier divergence:** released `train.py` uses a CRF/BMES sequence-tagging head, while the
  paper describes a clean sigmoid + BCE binary head (Eq. 11-12). This is the most likely real gap.
- **Other knobs:** prompt template (getPrompt vs paper T1-T3, ~1% swing), top-k, BERT-extractor
  training, seeds.
- **Scale out:** GossipCop++ (k=15) for a larger/lower-variance comparison.

## 8b. Paper findings (read 2026-05-29 via locally-installed pypdf → paper_extracted.txt)

LIFE = WWW '26 (Chi Wang et al.). Key facts that contradict our current setup:
- **Task is BINARY (fake=1/real=0), BCE loss** (paper §3.1). Our 4-class setup is a
  mismatch → likely why we only got 51.7%.
- **PolitiFact++/GossipCop++ use ONLY the LLM pairs MF (fake) vs MR (real)** — NOT HF/HR.
  Paper Table 2: PolitiFact++ Fake=97(=MF), Real=132(=MR), Total=229; GossipCop++
  4084(MF)/4169(MR). HF/HR are the human *sources* + used only in the fingerprint study.
  → This equals the released `train.py` DEFAULT 2-class (`gpt3.5_fake`/`gpt3.5_true`);
  my uncommenting of `human_*` moved us away from the paper.
- **Reconstruction mLLM = LLaMA2-7B** (paper §4.1.3), NOT gpt2/gpt-j. (GPT-J is only a
  *generation* model for the GenFake generalization datasets, RQ2.)
- **top-k = 10** for PolitiFact++ (k=15 GossipCop++); our step 1 default is 20.
- Classifier hyperparams (lr 5e-5, wd 0.1, warmup 0.1) match our `train.py` defaults.
  BERT extractor: max 512 tokens, lr 0.001. Trained on a single RTX 4090.
- **Targets:** LIFE PolitiFact++ Acc 0.900 / F1 0.882; GossipCop++ 0.937 / 0.924;
  VLFPN 0.890 / 0.926.
- Malicious prompt matters most (biggest ablation drop); exact prompt only ~1% swing.
  Paper's T1–T3 templates differ from `backend_utils.getPrompt()` (DAN/short) but that's
  a minor factor.

To reproduce PolitiFact++ ~90% need: (A) binary MF-vs-MR, (B) LLaMA2-7B reconstruction
(gated HF + fix the undefined `prompt` bug in `SPLlamaTokenizerPPLCalc`), (C) top-k=10.

## 10. Likely next steps

- DONE: full faithful run → 85.5% / 80.8% (see §7c). Pipeline reproduces the paper's ballpark.
- To close the gap (optional): see §7c "Gap analysis" — biggest suspect is the CRF/BMES head
  vs the paper's sigmoid+BCE; also seed variance on the tiny test set.
- Optionally extend to GossipCop++ (k=15) and VLFPN (needs a clean, punctuated copy).
- The plan file is at `C:\Users\Admin\.claude\plans\squishy-cuddling-quiche.md`.
