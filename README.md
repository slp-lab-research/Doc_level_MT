# Enhancing Document-Level Machine Translation via Filtered Synthetic Corpora and Two-Stage LLM Adaptation

Supplementary code for the paper by **Ireh Kim, Tesia Sker, and Chanwoo Kim** (Department of Artificial Intelligence, Korea University).

LLMs are good at modeling context, which makes them a natural fit for document-level machine translation (MT). Two things hold them back: there is little document-level parallel data, and LLMs tend to hallucinate or omit content. This repository implements the paper's pipeline for English→German (En→De):

1. **Document-level data augmentation.** The CNN/Daily Mail summarization corpus is turned into document-level MT pairs with `Llama-3.1-8B-Instruct`.
2. **Multi-metric filtering.** Augmented pairs are scored with **sacreBLEU**, **COMET**, and **LaBSE-CosSim** against Google Translate pseudo references, and low-quality pairs are discarded.
3. **Two-stage fine-tuning.** `Llama-3.2-1B-Instruct` is fine-tuned with LoRA, first on sentence-level News Commentary v16 and then on the filtered document-level corpus.

```
CNN/Daily Mail ──► Llama-3.1-8B-Instruct translation ──┐
        │                                              ├─► merge ─► score ─► filter ─► Stage 2: document-level LoRA FT ─► evaluate
        └────────► Google Translate (pseudo reference) ┘                               ▲
News Commentary v16 (sentence-level) ──────────────────────► Stage 1: sentence-level LoRA FT ┘
```

## Repository structure

```
supplementary_submission/
├── data_augmentation/
│   ├── data_augmentation.py      # Samples 20K CNN/DM documents and translates them with Llama-3.1-8B-Instruct
│   └── cnn_google_translate.py   # Pseudo references for the same documents via Google Translate
├── preprocessing/
│   ├── merge_data.py             # Joins LLM translations and pseudo references by id; strips "Hier ist die Übersetzung ...:" prefixes
│   ├── data_scoring.py           # sacreBLEU / COMET / LaBSE-CosSim score per document
│   └── data_filtering.py         # Threshold filtering with every combination of the three metrics
├── fine-tuning/
│   ├── dataset.py                # Prompt formatting, label masking, padding/collation
│   ├── trainer.py                # Shared Trainer / Evaluator (sacreBLEU) / utilities
│   ├── sl_MT_finetuning.py       # Stage 1: sentence-level LoRA fine-tuning
│   ├── sl_dl_MT_finetuning.py    # Stage 2: continue the stage-1 LoRA model on document-level data (two-stage)
│   └── dlo_MT_finetuning.py      # Baseline: document-level-only LoRA fine-tuning
├── evaluation/
│   ├── evaluate_comet.py         # COMET for every result file in ../result/new
│   └── evaluate_labse.py         # LaBSE-CosSim for every result file in ../result/new
└── requirements.txt
```

## Requirements

- Python 3.10+, a CUDA GPU (bf16 and FlashAttention-2 are used for the 8B augmentation model)
- Access to the gated Hugging Face models `meta-llama/Meta-Llama-3.1-8B-Instruct` and `meta-llama/Llama-3.2-1B-Instruct` (`huggingface-cli login`)

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation        # used by data_augmentation.py
python -m spacy download en_core_web_trf           # sentence splitting for LaBSE-CosSim
python -m spacy download de_dep_news_trf
```

## Data

Every script uses paths relative to its own folder (`../data`, `../models`, `../result`), so **run each script from inside its folder**. The expected layout is:

```
data/
├── cnn_dailymail/                         # HF dataset saved with save_to_disk (e.g. load_dataset("cnn_dailymail", "3.0.0"))
│   ├── source_data/                       # (generated) sampled source articles
│   └── augmented_data/                    # (generated) Google Translate pseudo references
├── news_commentary_v16/
│   ├── News-Commentary.json               # sentence-level En–De pairs: [{"source": <en>, "label": <de>}, ...]
│   └── doc-level/merged_news.json         # document-level En–De pairs (evaluation): same format
├── augmented_data/                        # (generated) Llama-3.1-8B-Instruct translations
├── merged_data/train_re_merged.json       # (generated) {"id", "source", "label", "ref"}
└── filtered_data/                         # (generated) scored and filtered training sets
```

The two News Commentary v16 files are not built by this code. Convert the En–De portion of News Commentary v16 into the JSON format shown above: sentence pairs for stage 1, and documents for evaluation.

## Reproducing the pipeline

### 1. Document-level data augmentation

```bash
cd data_augmentation
python data_augmentation.py        # 20K random CNN/DM train documents (seed 42) -> ../data/augmented_data/*.json
python cnn_google_translate.py     # same documents -> ../data/cnn_dailymail/augmented_data/*.json
```

`data_augmentation.py` writes the sampled articles to `../data/cnn_dailymail/source_data/train_sampled.jsonl`. `cnn_google_translate.py` reads that file, so both translations cover the same documents. The augmentation prompt is the paper's (*"You are a translator. Please translate English to Deutsch."*), generated with `num_beams=5`.

### 2. Merge, score, and filter

```bash
cd preprocessing
python merge_data.py       # -> ../data/merged_data/train_re_merged.json
python data_scoring.py     # -> ../data/filtered_data/new_train_score_merged.json (adds "bleu", "comet", "cos")
python data_filtering.py   # -> ../data/filtered_data/new/train_*.json (7 metric combinations)
```

- **sacreBLEU** (13a tokenization) and **COMET** (`Unbabel/wmt22-comet-da`) compare the LLM translation with the Google Translate pseudo reference.
- **LaBSE-CosSim** is reference-free. Both source and translation are split into sentences with spaCy and embedded with `sentence-transformers/LaBSE`. The sentence embeddings are averaged per document, and the score is the cosine similarity of the two averages. This works around LaBSE's input-length limit.
- Thresholds are set in `data_filtering.py` (`thresholds = {'bleu': ..., 'comet': ..., 'cos': ...}`). The defaults are 30 / 0.70 / 0.80. The paper's best setting is **sacreBLEU ≥ 35, COMET ≥ 0.75, LaBSE-CosSim ≥ 0.85**. Output files are named after the thresholds, e.g. `train_both_bleu35_comet0.75_cos0.85.json`.

### 3. Two-stage fine-tuning

```bash
cd fine-tuning
python sl_MT_finetuning.py      # Stage 1 -> ../models/new/sl_<MMDDhhmm>_fine-tuned_llama3_1B_lora_epoch3
python sl_dl_MT_finetuning.py   # Stage 2 + evaluation -> ../models/new/dl_<...>, ../result/new/dl_<...>_result.json
python dlo_MT_finetuning.py     # (baseline) document-level only + evaluation
```

Before running stage 2 or the baseline, set these variables at the top of `main()`:

| Script | Variable | Set to |
|---|---|---|
| `sl_dl_MT_finetuning.py` | `model_path` | the stage-1 LoRA directory created by `sl_MT_finetuning.py` |
| `sl_dl_MT_finetuning.py`, `dlo_MT_finetuning.py` | `train_path` | a filtered file from step 2 (e.g. `../data/filtered_data/new/train_both_bleu35_comet0.75_cos0.85.json`), or `../data/filtered_data/new_train_score_merged.json` for the unfiltered setting |

Training and evaluation settings in the code:

| | Stage 1 (sentence-level) | Stage 2 / doc-level only |
|---|---|---|
| Base model | `Llama-3.2-1B-Instruct` | stage-1 LoRA / `Llama-3.2-1B-Instruct` |
| Training data | 20,000 random News Commentary v16 pairs | 2,000 random documents from the (filtered) augmented set |
| LoRA | r = 16, α = 16, dropout 0.1, `q_proj`, `v_proj` | same (stage 2 continues the stage-1 adapter) |
| Optimizer | AdamW, lr 5e-5, weight decay 0.01, linear schedule with 1% warm-up | same |
| Epochs / batch size | 3 / 3 | 3 / 1 |
| Evaluation | – | 1,500 human-labeled News Commentary v16 documents, sacreBLEU (13a) |

The loss covers only the target translation and its closing `<|eot_id|>`. The prompt and padding tokens are masked with `-100`. `random.seed(10)` is fixed, so every model is evaluated on the same 1,500 documents. Generation uses the base model's default generation config (`max_new_tokens=8192`).

### 4. Evaluation

The fine-tuning scripts print sacreBLEU and save `{"input", "output", "label"}` for every evaluation document to `../result/new/`. COMET and LaBSE-CosSim are computed on those files:

```bash
cd evaluation
python evaluate_comet.py
python evaluate_labse.py
```

The paper also reports the geometric mean of the three metrics: `(sacreBLEU × COMET × LaBSE-CosSim)^(1/3)`.

## Results reported in the paper

Two-stage fine-tuning vs. document-level-only training, both on 2,000 unfiltered augmented documents (Table 2):

| | Document-level only | Two-stage |
|---|---|---|
| sacreBLEU | 11.24 | **15.07** |
| COMET | 0.618 | **0.697** |
| LaBSE-CosSim | 0.764 | **0.852** |
| Geometric mean | 1.744 | **2.076** |

Two-stage fine-tuning with filtered data (Table 5, selected):

| Filtering (sacreBLEU & COMET & LaBSE-CosSim) | sacreBLEU | COMET | LaBSE-CosSim | Geometric mean |
|---|---|---|---|---|
| No filtering | 15.07 | 0.697 | 0.852 | 2.076 |
| 30 & 0.70 & 0.80 | 15.90 | 0.699 | 0.856 | 2.119 |
| **35 & 0.75 & 0.85** | **15.96** | **0.701** | **0.860** | **2.127** |

See the paper for the single-metric and two-metric filtering results (Tables 3 and 4) and the sizes of the filtered datasets (Table 1).

## Citation
```bash
@INPROCEEDINGS{11461046,
  author={Kim, Ireh and Sker, Tesia and Kim, Chanwoo},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Enhancing Document-Level Machine Translation via Filtered Synthetic Corpora and Two-Stage LLM Adaptation}, 
  year={2026},
  volume={},
  number={},
  pages={18582-18586},
  keywords={Filtering;Filters;Circuits and systems;Electronic mail;Postal services;Communications technology;Information and communication technology;Internet;Computer networks;Protocols;Machine Translation;Data Augmentation;Large Language Models},
  doi={10.1109/ICASSP55912.2026.11461046}}
```
