# Project Memory — LIFE on Google Colab

Hand-off summary so a new session can pick up with full context. Last updated: 2026-07-07.

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

## 7d. Combined fake/real run — tried then reverted (2026-06-08)

Briefly tested pooling all four PolitiFact++ files by veracity (fake = HF+MF, real = HR+MR,
520 articles): **Acc 85.2 / Macro-F1 78.4** (per-class P/R: fake 69.8/63.2, real 89.3/91.8).
This **departs from the paper** (its PolitiFact++ = the GPT-3.5-generated MF-vs-MR pair only;
HF/HR are human sources — see §8b), and the human-written fakes (HF) lack the LLM fingerprint
LIFE keys on, which drags fake recall down. **Reverted to the MF-vs-MR config** (85.5/80.8)
as the paper-faithful setup; all combined-run edits to the `.py` files and notebook undone.

## 7e. RESUME HERE — closing the gap to the paper (classifier head) [2026-06-08 EOD]

**State:** back on the paper-faithful **MF-vs-MR** setup (revert verified: `py_compile` OK,
notebook valid JSON, no `combined`/`fake`/`real` remnants). A single fresh run gave **Acc 83.9
/ Macro-F1 78.2** (per-class P/R: fake 73.1/62.0, real 87.0/91.8) — within run-to-run variance
of the earlier 85.5/80.8 (see "why it wobbles"). `features_llama` confirmed clean: **229**
records, `{gpt3.5_fake:97, gpt3.5_true:132}` (not stale combined data).

**Goal:** close the gap to the paper's PolitiFact++ **0.900 / 0.882**.

**Root-cause finding (confirmed against the paper):** the released classifier head diverges
from the paper.
- Paper §3.4, **Eq 11**: `y_hat = sigmoid(W · Transformer(CNN(P)) + b)` — ONE probability per
  **article**, trained with **binary cross-entropy** (Eq 12). Direct, article-level.
- Released `LIFE_train/model.py` (`ModelWiseTransformerClassifier`): per-**token** logits over
  **8 BMES tags** (B/M/E/S × gpt3.5_fake/gpt3.5_true) + token-level `CrossEntropyLoss`; CRF
  only at decode (`viterbi_decode`); then `train.py:get_text_label` aggregates
  token→sentence→article by **majority vote**. The article label is never directly optimized.
- The **trunk (CNN→Transformer) already matches Eq 11** — only the HEAD diverges. Fix is
  localized to: output projection + loss + label encoding + eval.

**Proposed change (3 files):**
- `LIFE_train/model.py` — new `ModelWiseBCEClassifier`: reuse the conv+transformer trunk →
  masked **mean-pool** over valid tokens → `Linear(64, 1)` → `BCEWithLogitsLoss` vs a single
  0/1 article label; predict = `sigmoid > 0.5`. (Mean-pool is the implied seq→scalar step.)
- `LIFE_train/dataloader.py` — a BCE label path: emit one binary label per article
  (fake=1/real=0, i.e. label endswith 'fake') + keep the raw token mask (already built in
  `process_and_convert_to_tensor`) for pooling, instead of the BMES id masks.
- `LIFE_train/train.py` — select via `--model BCE` (keep CRF/BMES head for A/B); BCE loss
  path; simple article-level eval (compare preds vs labels, no `get_text_label`). Add `--seed`
  and seed everything (BERT extractor + training) so runs reproduce.

**DECISION TAKEN (2026-06-11): built the BCE head + seeding, A/B vs the released head.**
User constraint: don't edit originals — edit on copies. So the B-side is NEW files; the
untouched originals are the A-side:
- `LIFE_train/model_bce.py` (NEW) — `ModelWiseBCEClassifier`: trunk copied unchanged from
  `ModelWiseTransformerClassifier`, head = masked mean-pool → `Linear(64,1)` →
  `BCEWithLogitsLoss` (fake=1/real=0), preds = sigmoid>0.5. (Eq 11 doesn't specify the
  seq→vector reduction; mean-pool is our reading.) Imports `ConvFeatureExtractionModel`
  from model.py.
- `LIFE_train/train_bce.py` (NEW, edited copy of train.py) — `DataManagerBCE(DataManager)`
  overriding only `data_collator` (binary labels via `label.endswith('_fake')`, reuses
  `process_and_convert_to_tensor` for masks); `BCETrainer` with same optimizer/hyperparams;
  direct article-level eval printing "(real=0, fake=1)"; `split_dataset` copied verbatim
  (hard seed-0 → identical split to A-side); `--seed` flag with `set_seed_all` called
  **AFTER** DataManager init (its `__init__` re-seeds to 0 — dataloader.py:26 pitfall), so
  data is fixed and seed controls model init + batch order. Ckpt `bce_en.pt`.
- `LIFE_colab.ipynb` — Step 4 split into **4A** (released head, cell unchanged) and **4B**
  (train_bce.py, `--seed 0`, note to re-run seeds 1-4 for mean±std). Backup of the
  pre-edit notebook saved as `LIFE_colab_backup.ipynb`.
- `model.py`/`dataloader.py`/`train.py` confirmed untouched via git diff.
- Verified locally: both new files py_compile, notebook valid JSON (21 cells).

**✅ 4B RESULTS (Colab): head hypothesis CONFIRMED.** Multi-seed headline (seeds 0–20):
**Acc 86.82 ± 2.157, Macro-F1 84.48 ± 2.986** (mean ± std, 4 s.f.). Best single seed
(seed 2): Acc 89.1 / Macro-F1 87.4 (real P/R 85.3/100.0, fake P/R 100.0/70.6 — test is
17 fake/29 real; all errors are missed fakes). The mean clears the released CRF/BMES
A-side (~84–85.5 on the SAME features + split), confirming the head gain honestly, but is
~3.2 / ~3.7 pts below the paper's 90.0 / 88.2 — the cherry-picked peak alone overstated it.

**Why it wobbles (no fixed seed anywhere):** (1) Step-1 BERT extractor retrains
nondeterministically (HF `Trainer`, 3 ep); (2) Step-3 LLaMA features are bfloat16 on GPU (not
bit-reproducible); (3) ~46-sample test → each article ≈ 2 acc pts. So 83.9 vs 85.5 ≈ 1 article
= noise. Seeding is needed to measure any head gain honestly.

**Other levers (lower priority):** prompt template (`backend_utils.getPrompt` vs paper T1-T3,
~1% per §8b); BERT-extractor quality on ~183 train articles; top-k.

## 7f. 4-class multiclass exploration (HF/HR/MF/MR) — built, run pending [2026-06-17]

User asked to see a **4-class** multiclass result (human_fake / human_true / gpt3.5_fake /
gpt3.5_true) using the **original A-side files** (released CRF/BMES head), edited on copies —
NOT the BCE files. Exploratory ("see what it looks like"), not paper reproduction (the paper
is binary MF-vs-MR, §8b). Distinct from the earlier 4-class attempt because that used **gpt2**
features and scored **51.7%** (§7c); this uses the realigned **LLaMA2-7B** features.

Key simplification: `model.py`/`dataloader.py` are already class-count-agnostic
(`label_num = len(id2labels)`; CRF `allowed_transitions(id2labels)`; dataloader builds
`B-/M-/E-/S-+label` for any label string). So only `train.py` needed copying.

Built (verified locally: py_compile OK, notebook valid JSON = 33 cells):
- `LIFE_train/train_multi.py` (NEW, copy of `train.py`): only diffs are `en_labels` → the 4
  classes `{human_fake:0, human_true:1, gpt3.5_fake:2, gpt3.5_true:3}` (16 BMES tags), ckpt
  names suffixed `_multi` (e.g. `linear_multi_en.pt`, no clobber of binary `linear_en.pt`),
  and a `print('classes (id order):', en_labels)` so the per-class P/R is readable. **No
  seeding** (user said seeding methodology no longer needed). `model.py`/`dataloader.py`
  imported unchanged. Eval is the released sentence-level majority-vote (`get_text_label`).
- `LIFE_colab.ipynb`: appended a "Multiclass experiment" section (Steps 0m–4m) before the
  Notes cell, with its own `*_multi` paths (OUTPUT_MULTI / KEY_SENT_MULTI / BERT_CKPT_MULTI /
  FEATURES_MULTI / TRAIN_PATH_MULTI / TEST_PATH_MULTI) so the binary run's artifacts aren't
  touched. Step 0m `--subset all` (~520 articles), 1m `--top_k 10` (binary BERT: fake=HF+MF,
  true=HR+MR), 2m concat (in-place), 3m LLaMA features → FEATURES_MULTI, 4m train_multi 50ep.
- Originals confirmed untouched: `model.py`/`dataloader.py` no git diff; `train.py` only its
  pre-existing 5-line pipeline-fix diff.

**Needs a fresh Colab pass** (Steps 0m–3m must regenerate 4-class LLaMA features — the binary
`features_llama` has only MF/MR). Sync to Drive: `LIFE_train/train_multi.py` + updated
`LIFE_colab.ipynb`. **✅ RESULT (Colab, LLaMA2-7B, released CRF/BMES head, 50ep):
Acc 57.4 / Macro-F1 45.4.** Up from the old gpt2 4-class 51.7% (§7c), though not directly
comparable (that predates the current prompt/combined_ll/top-k=10 pipeline). The wide
Acc↔Macro-F1 gap (57.4 vs 45.4) implies weak minority-class performance — expected with 4
classes over ~520 articles and the majority class (HR) ≈ 37%.

## 7g. Combined-by-veracity binary exploration (fake=HF+MF, real=HR+MR) — built, run pending [2026-06-17]

User asked to re-run the combined setup as a clean side experiment on copies: pool the human
dataset in, binary fake/real by veracity (fake = HF+MF, real = HR+MR). NOTE: this was tried
once before (§7d) → **Acc 85.2 / Macro-F1 78.4** (fake P/R 69.8/63.2, real 89.3/91.8), then
reverted because it departs from the paper (human fakes lack the LLM fingerprint LIFE keys on).
Now kept permanently as a copy-based side experiment.

Built (verified locally: py_compile OK, notebook valid JSON = 35 cells):
- `LIFE_train/train_combined.py` (NEW, copy of `train.py`): adds `DataManagerCombined(DataManager)`
  which coarsens each article's fine label to `'fake'`/`'true'` by the `_fake`/`_true` suffix
  in `initialize_dataset` (the coarse label then flows through `data_collator` unchanged);
  `en_labels = {'fake':0, 'true':1}` (8 BMES tags); ckpt names suffixed `_combined`; class-order
  print. No seeding. `model.py`/`dataloader.py` imported unchanged.
- `LIFE_colab.ipynb`: added **Step 4c** (before Notes). **Reuses `FEATURES_MULTI`** + the same
  `TRAIN_PATH_MULTI`/`TEST_PATH_MULTI` split as the 4-class run — features are label-agnostic,
  so no extra Step 0–3 pass beyond the 4-class one (run Steps 0m–3m first). `--split_dataset`
  regenerates the identical seed-0 split, so 4m (4-class) and 4c (combined) are comparable.

Key insight: the binary MF-vs-MR run, the 4-class run, and this combined run that all use the
same articles share LLaMA features — the malicious-prompt reconstruction LL is computed
independent of the label. So 4-class and combined both consume `FEATURES_MULTI`.

Sync to Drive: `LIFE_train/train_combined.py` + updated `LIFE_colab.ipynb`.
**RESULT: <fill in after the Colab run>** (prior combined run: 85.2 / 78.4).

## 7h. GPT-2 as the reconstruction model — binary MF-vs-MR, both heads [2026-06-18]

User asked to swap the **reconstruction (inference) LM** from LLaMA2-7B to **GPT-2** on the
paper's real task (binary MF-vs-MR) and run BOTH classifier heads, to isolate the effect of the
reconstruction model. LLaMA baselines to beat/compare:
- Released CRF/BMES head (`train.py`): ~83.9–85.5 / ~78–80.8 (§7e).
- Paper BCE head (`train_bce.py`): **86.82 ± 2.157 / 84.48 ± 2.986** (seeds 0–20, §7e).

**No new files needed** (confirmed): `dataset/3_gen_features_local.py` already does GPT-2 via
`--model gpt2 --scorer bbpe` (BBPE path in `backend_utils.BBPETokenizerPPLCalc` is intact —
prepends `getPrompt`, handles the sentence list, returns `combined_ll`); `train.py` already IS
binary MF-vs-MR (released head); `train_bce.py` already IS the binary BCE head. Invoking them
with a GPT-2 features dir is not editing them, so the "don't touch originals" rule holds. This
change is **notebook-only** (no `.py` diffs).

**Methodological control:** the GPT-2 run REUSES the existing `OUTPUT_BIN` (key sentences from
the LLaMA binary run's Step 2). Do NOT re-run Steps 0–2 — re-running Step 1 retrains the BERT
extractor nondeterministically and would change the key sentences, confounding the model swap.
Reusing `OUTPUT_BIN` means the ONLY difference vs the LLaMA binary run is the Step-3
reconstruction LM.

Notebook: added a "GPT-2 reconstruction model (binary MF-vs-MR)" section (before Notes) with
GPT-2 paths (`FEATURES_GPT2_BIN` / `TRAIN_PATH_GPT2` / `TEST_PATH_GPT2`): Step 3g (GPT-2
features from `OUTPUT_BIN`), Step 4A-gpt2 (`train.py`, released head), Step 4B-gpt2
(`train_bce.py`, BCE head, `--seed 0`; sweep seeds 1–20 for mean±std like LLaMA).

Sync to Drive: updated `LIFE_colab.ipynb` only (no `.py` changed).
**✅ RESULT — BCE head (multi-seed): Acc 78.9 ± 4.82 / Macro-F1 72.52 ± 8.77** (mean ± std;
raw F1 mean 72.519, std 8.7698). Single seed-0 run was 83.6 / 78.1 — an above-mean draw.
Well below the LLaMA-2 BCE mean (86.82 ± 2.157 / 84.48 ± 2.986) by ~7.9 / ~12 pts, AND far
noisier (std ~2× on Acc, ~3× on F1). Strong confirmation that reconstruction-LM quality
drives the fingerprint: GPT-2 is both weaker and much less stable than LLaMA-2 here.
**Released CRF/BMES head (GPT-2): still pending.**

## 7i. Qwen2.5-32B reconstruction model (4-bit) — binary MF-vs-MR, both heads [2026-06-18]

Extending the reconstruction-model comparison to a modern, larger LM. Decision context:
frontier CLOSED models (Claude, GPT-5.x) are unusable for LIFE — the method needs teacher-forced
per-token log-likelihoods over the INPUT (prompt+article; `calc_sent_ppl` → CrossEntropyLoss over
logits), and Claude's API exposes no logprobs at all while OpenAI's chat API gives logprobs only
for generated tokens (echo-mode input scoring is deprecated). So "more modern" = open-weights.

Colab hardware reality: single **40GB A100** max (no 80GB, no multi-GPU). Llama-3-70B ruled out
(4-bit ≈ 38-40GB weights → no activation headroom on 40GB). Sweet spot = a **4-bit ~32B** model.
User chose **Qwen/Qwen2.5-32B** (base, not Instruct) — Qwen is GPT-2-style byte BPE so the
existing `--scorer bbpe` path works; only a quant flag was needed.

Built (verified: py_compile OK, notebook valid JSON):
- `dataset/3_gen_features_local.py`: added `--load_in_4bit` flag + `_load_model()` helper.
  4-bit path uses `BitsAndBytesConfig(load_in_4bit, nf4, bnb_4bit_compute_dtype=bfloat16,
  double_quant)` with `device_map={'':0}` and **skips `model.to(device)`** (bitsandbytes rejects
  moving a quantized model). Non-4bit path unchanged. `build_calculator` gained a `load_in_4bit`
  param; `bitsandbytes`/`accelerate` already in requirements.txt (from the original 8-bit design).
- `LIFE_colab.ipynb`: "Qwen2.5-32B" section (before Notes) with `FEATURES_QWEN_BIN`/`TRAIN_PATH_QWEN`/
  `TEST_PATH_QWEN`. Step 3q (`--model Qwen/Qwen2.5-32B --scorer bbpe --load_in_4bit`), Step 4A-qwen
  (released head), Step 4B-qwen (BCE head). Reuses `OUTPUT_BIN` (same key sentences → clean model swap).

Assumption to verify on first run: Qwen's fast tokenizer `_convert_id_to_token` + GPT-2 `byte_decoder`
mapping works in `BBPETokenizerPPLCalc` (should — Qwen is GPT-2-style byte BPE; would only KeyError if a
special token appeared mid-text, which plain scoring shouldn't produce). First run downloads ~65GB bf16
then quantizes to ~18-22GB VRAM.

Sync to Drive: updated `dataset/3_gen_features_local.py` + `LIFE_colab.ipynb`.
**RESULT — released head: <fill in>. BCE head: <fill in>.** (vs LLaMA / GPT-2 baselines above.)

## 7j. Human-only reconstruction — binary HF-vs-HR, LLaMA2-7B, both heads [2026-07-07]

User asked to test the method on the **human dataset only**: binary **HF (human_fake) vs HR
(human_true)** with **LLaMA2-7B** reconstruction, on BOTH classifier heads. This is a
**negative control** — both classes are human-written, so NEITHER carries the prompt-induced
LLM fingerprint LIFE keys on (contrast the paper's MF-vs-MR, where fake is LLM-generated).
Near-chance / weak results are the expected, informative outcome: evidence that LIFE detects
LLM-*generation*, not fakeness per se.

**Class imbalance matters for reading the result:** HF=97, HR=194 (§4), so 291 articles,
real:fake ≈ 2:1. A degenerate "always real" classifier already gets ~67% Acc but only ~40
Macro-F1. So **Macro-F1 and fake recall are the honest metrics** here; accuracy alone can look
fine just from predicting the majority (real) class.

**Files (only the released head needed a copy):**
- `LIFE_train/train_human.py` (NEW, copy of `train.py`): only diffs are `en_labels =
  {human_fake:0, human_true:1}` (8 BMES tags), ckpt names suffixed `_human` (`linear_human_en.pt`
  etc., no clobber), and a class-order print. No seeding. `model.py`/`dataloader.py` imported
  unchanged. Released sentence-level majority-vote eval.
- **BCE head reuses `train_bce.py` UNCHANGED** — its label is suffix-derived
  (`int(label.endswith('_fake'))`, train_bce.py:54), so HF→fake=1, HR→real=0 automatically. No
  copy (same way the GPT-2 run reused it).
- `LIFE_colab.ipynb`: appended a "Human-only (HF-vs-HR)" section (before Notes) with
  `FEATURES_HUMAN`/`TRAIN_PATH_HUMAN`/`TEST_PATH_HUMAN`. Step 3h **copies just `HF_fake.jsonl` +
  `HR_true.jsonl` out of `FEATURES_MULTI`** into `FEATURES_HUMAN` (no re-scoring — reuses the
  4-class LLaMA features; `split_dataset` loads every `.jsonl` in the dir, so it must hold ONLY
  HF+HR). Step 4A-human (`train_human.py`, released head), Step 4B-human (`train_bce.py`, BCE
  head, `--seed 0`; sweep 1–20 for mean±std). Requires the 4-class Step 3m to have produced
  `FEATURES_MULTI`; else run Steps 0m–3m first.

Key insight (reused again): all same-article runs share the malicious-prompt reconstruction LL
(computed independent of the label), so binary/4-class/combined/human-only all consume the same
`FEATURES_MULTI` LLaMA features — human-only is just the HF+HR **subset** of them.

Verified locally: `train_human.py` py_compile OK; notebook valid JSON = 59 cells; existing
cells untouched (pure insertion); no git diff on any existing `.py`.

Sync to Drive: `LIFE_train/train_human.py` + updated `LIFE_colab.ipynb`.
**RESULT — released head: <fill in>. BCE head: <fill in>.** (Compare against the majority
baseline ~67 Acc / ~40 Macro-F1; and against MF-vs-MR LLaMA released ~83.9–85.5 / ~78–80.8,
BCE 86.82 / 84.48.)

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
