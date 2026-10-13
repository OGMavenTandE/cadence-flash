# Cadence Flash

An open-source, offline decision model from **AI Eval Corp**. It is ModernBERT-large (395M parameters), fine-tuned on a public decision mix and then on synthetic military message traffic built from US Army doctrine cleared for public release (Distribution A). It labels each message with its **type** (9 classes) and **precedence** (Flash, Immediate, Priority, Routine), with calibrated confidence, as decision support with a human in the loop.

- Model weights and card: https://huggingface.co/AI-Eval-Corp/cadence-flash
- Dataset: https://huggingface.co/datasets/AI-Eval-Corp/cadence-flash-dataset
- Results (AI Eval curated benchmark, full held-out test, p2c checkpoint): type accuracy 0.985, precedence 0.957, Flash recall 0.995, ECE 0.005 type / 0.022 precedence, phase 1 retained 0.869 (ECE 0.071), typed_decisions 0.678. Short messages (n=677): precedence 0.866, Flash recall 0.989. Weak spots: FRAGO recall 0.871, held-out FM 5-0 warning orders 0.239 (type and precedence both correct). Full model card: [MODEL_CARD.md](MODEL_CARD.md)

## Repository layout
```
data_build/   phase 2 synthetic data pipeline (templates, Gemini rephrasing, drift check, dedupe/leakage check, LLM judge)
  sources.json   distribution statement evidence for each doctrine source
scripts/      download.py (phase 1 data + base model), build_mix.py (phase 1 mix), train.py (phase 1), train_p2.py / train_p2b.py / build_p2c.py + train_p2c.py (phase 2, final = p2c)
training/     launcher_p2.ps1 (detached Windows run)
predict.py    inference: type + precedence with calibrated probabilities
LICENSE       Apache-2.0
NOTICE        attributions for every dataset
```

## Reproduce
1. `python scripts/download.py`, then `python scripts/build_mix.py`, then `python scripts/train.py` (phase 1).
2. Download the doctrine PDFs listed in `data_build/sources.json`, then run `GEMINI_API_KEY=... python data_build/build_full.py`, then `python data_build/judge_full.py`. The full build cost about $13 in Gemini API usage.
3. `python scripts/train_p2.py` (from phase 1), then `python scripts/train_p2b.py`, then `python scripts/build_p2c.py` and `python scripts/train_p2c.py` (the final checkpoint, which adds short-message and shorthand robustness). The p2c run took about 37 minutes of training plus 2 minutes of evaluation on an RTX 4070.

## Quick start
```bash
huggingface-cli download AI-Eval-Corp/cadence-flash --local-dir cadence-flash
python predict.py "SPOTREP. Squad-size element engaging our position with small arms at 31U NB 1234 5678." --model cadence-flash
```

## Intended use and limits
This is decision support only, with a human reviewing every output. It is not for autonomous targeting or engagement. All training messages are synthetic and formatted. Short and shorthand messages are much improved in p2c ("troops in contact" and "TIC" now come back Flash), but they are still the weakest input, so keep a person in the loop for safety-critical short traffic. Evaluate it on your own data before use. Full details are in the model card.

## License
Code and weights are Apache-2.0. Dataset attributions are in NOTICE.
