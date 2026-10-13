---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-classification
base_model: answerdotai/ModernBERT-large
tags:
- modernbert
- text-classification
- military
- message-triage
- decision-support
- offline
- calibrated
datasets:
- AI-Eval-Corp/cadence-flash-dataset
- PolyAI/banking77
- clinc/clinc_oos
- google/boolq
- nyu-mll/multi_nli
- deepset/prompt-injections
- jackhhao/jailbreak-classification
- LocalLLaMA/typed-decisions
model-index:
- name: cadence-flash
  results:
  - task: {type: text-classification, name: Military message type}
    dataset: {name: Cadence Flash Military Messages v1 (test, AI Eval curated benchmark), type: AI-Eval-Corp/cadence-flash-dataset}
    metrics:
    - {type: accuracy, value: 0.985, name: Type accuracy}
  - task: {type: text-classification, name: Military message precedence}
    dataset: {name: Cadence Flash Military Messages v1 (test, AI Eval curated benchmark), type: AI-Eval-Corp/cadence-flash-dataset}
    metrics:
    - {type: accuracy, value: 0.957, name: Precedence accuracy}
    - {type: recall, value: 0.995, name: Flash recall}
---

# Cadence Flash

**Cadence Flash** is an open-source decision model from **AI Eval Corp**. It is a 395M-parameter ModernBERT-large encoder, fine-tuned in two phases:

1. **Phase 1, public decision mix:** intent routing, out-of-scope detection, yes/no reading comprehension, natural language inference, prompt-injection detection, and typed decisions.
2. **Phase 2, military message traffic:** for each message, it predicts the **message type** (9 classes) and the **precedence** (Flash, Immediate, Priority, Routine). The training data was built from US Army doctrine cleared for public release (Distribution A).

It runs fully offline on a single consumer GPU or CPU, and its outputs are temperature-calibrated probabilities.

> All evaluation numbers on this card come from an **AI Eval curated benchmark** unless a row says otherwise.

## Intended use
- **Military message triage:** tag incoming reports and messages by type and precedence so staff can sort the queue faster.
- **Decision support with a human in the loop:** the model suggests a label and a confidence, and a trained operator makes the decision.
- Offline or disconnected settings where cloud models are not allowed.
- Test and evaluation research on small, calibrated, offline classifiers.

## Out-of-scope use
- Any autonomous action: targeting, engagement, fires, or weapons release, or anything else where the model's output is acted on with no human review.
- Classified, CUI, or operational traffic **without your own local evaluation first.** The model has seen only synthetic, unclassified messages.
- Medical triage decisions. The MEDEVAC labels describe message precedence and are not a clinical assessment.
- Languages other than English, and message formats that don't follow US Army conventions.

## Labels
| Head | Classes |
|---|---|
| Message type | SPOTREP, SALUTE, 9-line MEDEVAC, CCIR trigger, SITREP, FRAGO/order, RFI, routine admin, other |
| Precedence | Flash, Immediate, Priority, Routine |
| Phase 1 heads | Kept from phase 1 (intent, out-of-scope, yes/no, NLI, injection, typed decisions) |

## Training data
| Dataset | License | Use |
|---|---|---|
| BANKING77 (PolyAI/banking77) | CC-BY-4.0 | Phase 1 |
| CLINC150 (clinc/clinc_oos, plus) | CC-BY-3.0 | Phase 1 |
| BoolQ (google/boolq) | CC-BY-SA-3.0 | Phase 1 |
| MultiNLI (nyu-mll/multi_nli) | OANC license (permissive), CC-BY-3.0, CC-BY-SA-3.0, MIT | Phase 1 |
| deepset/prompt-injections | Apache-2.0 | Phase 1 |
| jackhhao/jailbreak-classification | Apache-2.0 | Phase 1 |
| LocalLLaMA/typed-decisions | Apache-2.0 | Phase 1 |
| [AI-Eval-Corp/cadence-flash-dataset](https://huggingface.co/datasets/AI-Eval-Corp/cadence-flash-dataset) (synthetic, this project) | CC-BY-4.0 | Phase 2, plus a ~20% phase 1 replay |

**Phase 2 provenance.**
- **Sources:** field structures and precedence definitions come from FM 6-99 *U.S. Army Report and Message Formats* (May 2021), FM 6-0 *Commander and Staff Organization and Operations* (May 2022), and ATP 4-02.2 *Medical Evacuation* (July 2019). FM 5-0 *Planning and Orders Production* (Nov 2024) supplied warning orders for the **test set only**, so one source document is held out of training entirely.
- **Distribution statement:** every source carries the cover statement *"DISTRIBUTION RESTRICTION: Approved for public release; distribution is unlimited."*
- **Generation:** units, call signs, grids, and times are fictional and come from templates. About half of the training set and all of the test set were rephrased by a US-made LLM (Google Gemini). Rewrites that added or dropped a precedence word, a type cue, or a number were rejected.
- **Checks:** near-duplicates were removed, and test items that matched a training item once numbers were masked were removed. An LLM judge screened a stratified sample (95.1% agreement), and the flagged items were removed. A human domain reviewer graded a 50-item sample (100% type, 96% precedence agreement).
- **What was not used:** personal or organizational email, operational traffic, classified material, CUI, or FOUO material.
- **Record fields:** every record lists its source document, page, publication date, and distribution statement.

**Size:** the dataset has 13,500 training and 1,866 test messages. The final checkpoint (**p2c**) trained on 23,824 examples: 15,010 military messages (including 6,546 rule-based short, shorthand, no-precedence-word, decoy, and balance augmentations built from training rows only, using FM 1-02.1-style abbreviations), plus 8,814 phase 1 replay examples (37%). Another 1,019 military messages were held out of training to fit the temperatures. No new LLM-generated text was added in p2c.

## Evaluation (AI Eval curated benchmark, network off)

**Phase 1** (public decision mix test set, n = 17,366)
| Model | Accuracy | ECE | typed_decisions acc |
|---|---|---|---|
| cadence-flash-p1 | 0.872 | 0.058 | 0.613 |
| cadence-flash (this release, p2c checkpoint) | 0.869 | 0.071 | 0.678 |

**Phase 2** (synthetic military held-out test, n = 1,866; temperature-scaled ECE)
| Metric | Target | cadence-flash |
|---|---|---|
| Message type accuracy | ≥ 0.85 | 0.985 |
| Precedence accuracy | ≥ 0.85 | 0.957 |
| Flash recall | ≥ 0.95 | 0.995 |
| ECE (worst head) | ≤ 0.08 | 0.022 (type 0.005, precedence 0.022) |
| Phase 1 accuracy retained | ≥ 0.862 | 0.869 |
| typed_decisions (secondary) | ≥ 0.66 | 0.678 |
| Human agreement, 30-item sample | ≥ 0.80 | pending |

All phase 2 targets were met. Phase 1 accuracy is 0.869 (0.872 before phase 2), and typed_decisions rose from 0.613 to 0.678.

**Short-message slices** (held-out, built only from test rows; precedence accuracy / Flash recall)
| Slice | n | Precedence accuracy | Flash recall |
|---|---|---|---|
| Short messages (all) | 677 | 0.866 | 0.989 |
| 8 words or fewer | | 0.814 | 0.985 |
| Shorthand | | 0.936 | 1.000 |
| Abbreviated | | 0.908 | 0.991 |
| Decoy (misleading precedence words) | | 0.95 | 1.00 |

**Known weak spots**
| Slice | Result |
|---|---|
| FRAGO/order recall | 0.871 (lowest type) |
| FM 5-0 warning orders (document held out of training), type and precedence both correct | 0.239 |

The FM 5-0 slice is the best guide to how the model handles formats it hasn't seen. Expect much lower accuracy on unfamiliar real-world formats than on the in-distribution test set. Per-source and per-class details are in `metrics.json`.

## Limitations
- **Synthetic data.** The phase 2 messages come from templates and an LLM's rephrasing. They are cleaner and more regular than real traffic, so accuracy on real messages will be lower. Evaluate locally before use.
- **Small "other" test set.** Only 83 test items are "other," with none at Routine precedence, so "other" results are noisy. In training, "other" also leans toward Flash.
- **FRAGO/WARNO weakness.** FRAGO/order is the weakest type (recall 0.871). On warning orders from FM 5-0, a document held out of training, only 23.9% get both type and precedence right, so formats outside training generalize poorly.
- **Labels follow a rubric.** Precedence follows an owner-approved rubric: urgent and urgent-surgical casualties are Flash, and admin with a near-term deadline is Priority. Local SOPs may differ.
- **Judge-screened, partly human-reviewed.** Most items weren't reviewed by a person. The LLM judge and the human reviewer disagreed on some precedence calls.
- **Short messages improved, but still need review.** The p2c checkpoint added rule-based short, shorthand, abbreviated, no-precedence-word, and decoy training examples, all derived from the same Distribution A templates. Short messages now score 0.866 precedence accuracy with 0.989 Flash recall (0.814 / 0.985 at 8 words or fewer), and phrases like "troops in contact" and "TIC" now come back Flash. Short messages are still the weakest input, so keep a person in the loop for safety-critical short traffic.
- **Phase 1 calibration.** ECE on the phase 1 test set is 0.071 (0.058 before phase 2). The military heads are well calibrated (ECE 0.005 type, 0.022 precedence).
- Only English and US Army formats.

## How to use
The checkpoint is a single 317-label classification layer, grouped into 27 heads with a softmax per head. `label_map.json` maps each head to its label indices, and `temperature.json` holds the calibration temperatures. `predict.py` in this repo does the selection for you:

```bash
pip install torch "transformers>=4.48" huggingface_hub
huggingface-cli download AI-Eval-Corp/cadence-flash --local-dir cadence-flash
python cadence-flash/predict.py "SPOTREP. Squad-size element engaging our position with small arms at 31U NB 1234 5678." --model cadence-flash
```

```python
import sys; sys.path.insert(0, "cadence-flash")
from predict import load, predict
tok, model, lm, T = load("cadence-flash")
r = predict("9-line MEDEVAC request. Line 3: 1A (urgent). Line 6: E - enemy troops in area.", tok, model, lm, T)
print(r["type"]["label"], r["type"]["prob"], r["precedence"]["label"], r["precedence"]["prob"])
```

Each head returns `label`, `prob`, and the full `probs` distribution. To get phase 1 heads, pass `heads=(...)` using the head names in `label_map.json`. No `trust_remote_code` is needed.

## Citation and attribution
Please cite this model, the datasets listed above (with their licenses), and the US Army publications: FM 6-99 (2021), FM 6-0 (2022), ATP 4-02.2 (2019), FM 5-0 (2024).

**Author:** AI Eval Corp · License: Apache-2.0 · Code: https://github.com/OGMavenTandE/cadence-flash
