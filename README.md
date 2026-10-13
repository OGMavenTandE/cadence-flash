# Cadence Flash

An open-source, offline decision model from **AI Eval Corp**. It is ModernBERT-large (395M parameters), fine-tuned on a public decision mix and then on synthetic military message traffic built from US Army doctrine cleared for public release (Distribution A). It labels each message with its **type** (9 classes) and **precedence** (Flash, Immediate, Priority, Routine), with calibrated confidence, as decision support with a human in the loop.

- Model weights and card: https://huggingface.co/AI-Eval-Corp/cadence-flash
- Dataset: https://huggingface.co/datasets/AI-Eval-Corp/cadence-flash-dataset
- Results (AI Eval curated benchmark, full held-out test): type accuracy 0.987, precedence 0.958, Flash recall 1.000, phase 1 retained 0.865. Weak spots: FRAGO recall 0.888, held-out FM 5-0 warning orders 0.248 (type and precedence both correct)

## Repository layout
```
data_build/   phase 2 synthetic data pipeline (templates, Gemini rephrasing, drift check, dedupe/leakage check, LLM judge)
  sources.json   distribution statement evidence for each doctrine source
scripts/      download.py (phase 1 data + base model), build_mix.py (phase 1 mix), train.py (phase 1), train_p2.py / train_p2b.py (phase 2, final = p2b)
training/     launcher_p2.ps1 (detached Windows run)
predict.py    inference: type + precedence with calibrated probabilities
LICENSE       Apache-2.0
NOTICE        attributions for every dataset
```

## Reproduce
1. `python scripts/download.py`, then `python scripts/build_mix.py`, then `python scripts/train.py` (phase 1).
2. Download the doctrine PDFs listed in `data_build/sources.json`, then run `GEMINI_API_KEY=... python data_build/build_full.py`, then `python data_build/judge_full.py`. The full build cost about $13 in Gemini API usage.
3. `python scripts/train_p2.py` (from phase 1), then `python scripts/train_p2b.py` (from p2, the final checkpoint). The p2b run took about 28 minutes of training plus 8 minutes of evaluation on an RTX 4070.

## Quick start
```bash
huggingface-cli download AI-Eval-Corp/cadence-flash --local-dir cadence-flash
python predict.py "SPOTREP. Squad-size element engaging our position with small arms at 31U NB 1234 5678." --model cadence-flash
```

## Intended use and limits
This is decision support only, with a human reviewing every output. It is not for autonomous targeting or engagement. All training messages are synthetic and formatted. Short or shorthand messages (for example "troops in contact") can get confidently wrong precedence, so send them to a person. Evaluate it on your own data before use. Full details are in the model card.

## License
Code and weights are Apache-2.0. Dataset attributions are in NOTICE.
