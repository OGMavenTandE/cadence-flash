# Cadence Flash

An open-source, offline decision model from **AI Eval Corp**. It is ModernBERT-large (395M parameters), fine-tuned on a public decision mix and then on synthetic military message traffic built from US Army doctrine cleared for public release (Distribution A). It labels each message with its **type** (9 classes) and **precedence** (Flash, Immediate, Priority, Routine), with calibrated confidence, as decision support with a human in the loop.

- Model weights and card: https://huggingface.co/AI-Eval-Corp/cadence-flash (weights coming soon)
- Dataset: https://huggingface.co/datasets/AI-Eval-Corp/cadence-flash-dataset
- Results (AI Eval curated benchmark, full held-out test): type accuracy 0.987, precedence 0.958, Flash recall 1.000, phase 1 retained 0.865. Weak spots: FRAGO recall 0.888, held-out FM 5-0 warning orders 0.248 (type and precedence both correct)

## Repository layout
```
data_build/   phase 2 synthetic data pipeline (templates, Gemini rephrasing, drift check, dedupe/leakage check, LLM judge)
  sources.json   distribution statement evidence for each doctrine source
training/     phase 1 data download, phase 2 training and evaluation (two heads, temperature scaling, phase 1 re-score)
LICENSE       Apache-2.0
NOTICE        attributions for every dataset
```

## Reproduce
1. `python training/download_phase1_data.py` downloads the public phase 1 data and ModernBERT-large, with no token needed.
2. Download the doctrine PDFs listed in `data_build/sources.json`, then run `GEMINI_API_KEY=... python data_build/build_full.py`, then `python data_build/judge_full.py`. The full build cost about $13 in Gemini API usage.
3. `python training/train_p2.py --init <phase1 checkpoint> --replay 2700`. This took about 3 hours on an RTX 4070.

## Intended use and limits
This is decision support only, with a human reviewing every output. It is not for autonomous targeting or engagement. All training messages are synthetic, so evaluate it on your own data before use. Full details are in the model card.

## License
Code and weights are Apache-2.0. Dataset attributions are in NOTICE.
