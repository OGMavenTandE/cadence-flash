# Review

Security fixes and deletions below keep the published inference path: a local directory, `model.safetensors`, and positive finite values under `temperatures` in `temperature.json`. `predict()` still returns the same label and probability fields. README text, the model card, and public URLs are unchanged.

No committed secret, token, password, or key was found in the tree or in any git revision (pattern scan only; history was not rewritten). If the old Gemini client was run, rotate that API key. It was sent as a URL query parameter, so it can be in provider or proxy logs. The value was not in this repository.

## Security findings

### High — Gemini API key was placed in the request URL

The old client in `data_build/gen_lib.py` (previously lines 3–6) built `...generateContent?key=<GEMINI_API_KEY>`. `data_build/gen.py` (previously lines 4–6) had a second copy. urllib puts that URL in errors, and the API host and any proxy log the query string. urllib also copies custom headers onto redirects, including redirects to another host.

Fixed in `data_build/gen_lib.py` lines 3–26. The key is sent as the `x-goog-api-key` header. The model id must match `[A-Za-z0-9._-]{1,64}`. Redirects are accepted only for `https://generativelanguage.googleapis.com`. A key containing CR, LF, or NUL is rejected. If an exception string contains the key, that string is rewritten with the key removed. The JSON body, host, path, 180s timeout, and retry sleeps are unchanged. The second client was removed with `gen.py`.

### High — `scripts/probe.py` executed whatever `predict.py` was in the checkpoint

Previously line 1 did `sys.path.insert(0, sys.argv[1])` and then `import predict`. A downloaded checkpoint that contains `predict.py` (the published model repo does) ran that file as code.

Fixed in `scripts/probe.py` lines 1–5. The script loads this repository's `predict.py` by file path. The printed fields are unchanged: text, type label, type probability, precedence label, precedence probability, Flash probability.

### Medium — checkpoint load could run pickle weights or fetch remote code

`from_pretrained` will load `pytorch_model.bin` with pickle, which is code execution, and a non-local path can be fetched from the Hub. `scripts/train_p2.py`, `scripts/train_p2b.py`, and `scripts/train_p2c.py` did not pass `trust_remote_code=False`.

Loads now pass `trust_remote_code=False`, `local_files_only=True`, and `use_safetensors=True`. Saves pass `safe_serialization=True`.

- `predict.py` lines 6–10
- `scripts/train.py` lines 22–23 and 73
- `scripts/train_p2.py` lines 45–46 and 144
- `scripts/train_p2b.py` lines 49–50 and 144
- `scripts/train_p2c.py` lines 66–67 and 201
- `p2d/standing_eval.py` lines 82–83

The published checkpoint is `ModernBertForSequenceClassification` with `model.safetensors` and no `auto_map`. A directory that only has pickle weights now fails instead of executing them.

### Medium — calibration temperatures were used unchecked

`predict.py` lines 12–15 divide logits by the temperature. Zero, negative, NaN, or infinite values change the softmax. Values now have to be finite numbers greater than 0. The published `mil_type` and `mil_prec` temperatures are finite and positive, so those probabilities stay the same.

### Low — judge CSV could be a spreadsheet formula

`data_build/judge_full.py` lines 34–40 write the judge reason and the message text to `judge_full_flags.csv`. A cell starting with `=`, `+`, `-`, `@`, tab, or CR is a formula in a spreadsheet. Those cells are now prefixed with an apostrophe. Rows that do not start with those characters are unchanged. The training jsonl is not passed through this helper.

### Low — a local API key file was not ignored

`.gitignore` lines 9–10 ignore `.env` and `.env.*`. No secret was in the tree to remove.

## Removals

- `data_build/gen.py` (pilot generator, `p2pilot` ids, `pilot.json`). Nothing imported it. The README reproduce path is `build_full.py`, then `judge_full.py`. This file was the extra API-key-in-URL client.
- `data_build/judge.py`. It only scored `pilot.json`. `judge_full.py` is the judge on the reproduce path.
- Unused imports: `sys` in `scripts/train.py`, `scripts/download.py`, and `data_build/build_full.py`; `dtg` in `data_build/backfill_other.py`; `re` in `data_build/judge_full.py`; `random`, `templates`, and the thread pool in `data_build/gen_lib.py`.
- `scripts/train_p2c.py`: `import re as _re` was never used (`re` is still used for the abbreviation pattern).
- `scripts/probe.py`: a top-2 list was computed and never printed.
- `p2d/build_p2d.py` lines 98–99: `T2` was assigned from a `re.sub` and then assigned again before use. `re.sub` does not draw from the RNG, so later template choices are the same.

`data_build/backfill_other.py` stays. It is not named in the README, and it is what appends the extra `other` rows the published set was built with.

`data_build/sources.json` stays, including the entries for documents that were rejected. That file is the distribution-statement record.

## Other safe fixes

- `scripts/train_p2c.py` lines 79–81 and `scripts/build_p2c.py` lines 136–138 referenced `p2c/chief_inputs.json`, which was never committed. If that file exists it is still read. Otherwise the script reads the committed `p2c/probe_inputs.json`, which is the list `p2d/build_p2d.py` and `p2d/standing_eval.py` already use.
- `training/launcher_p2.ps1` lines 1–8 were two `param` blocks, which PowerShell rejects, so `-Smoke`, `-Epochs`, `-MaxLen`, and `-MaxMinutes` could not bind. They are one block now. The script was not executed.
- `p2d/standing_eval.py` freeze text now says `p2d/frozen`, the directory the script writes. Gate thresholds are unchanged.

## Changes not made

- Pin `requirements.txt` and the Hub revisions in `scripts/download.py`. The five dependencies are unpinned (`torch>=2.2`, `transformers>=4.48`, `pandas`, `numpy`, `huggingface_hub`), and `download.py` takes the current dataset and ModernBERT revisions. A pin changes what a later install or download gets. This environment has no GPU replay to confirm one. `download.py` already removes HF token variables and passes `token=False`.
- Merge `scripts/train_p2.py`, `scripts/train_p2b.py`, and `scripts/train_p2c.py`. The training loop is repeated, and the mix, oversampling, augmentations, and gates are not. A shared module needs a GPU run before it is safe.
- Edit the Ollama block in `training/launcher_p2.ps1` lines 17–23. It stops `qwen2.5:3b-instruct` and `phi4-mini`, then reloads Qwen to hold VRAM. Qwen is a non-US model. Removing those calls changes that machine's GPU state after the job. Line 23 also clears `CUDA_VISIBLE_DEVICES`. The busy-process filter names `standard_bench`, `pilot.py`, and `rescore`, which are not in this repo.
- Teach `predict.py` line 12 to also accept the flat `{mil_type, mil_prec}` object written at `scripts/train_p2.py` line 145, `scripts/train_p2b.py` line 145, and `scripts/train_p2c.py` line 202. The published file and `predict.py` use a `temperatures` object. `p2d/standing_eval.py` line 81 reads the flat object. Accepting both would change the loader.
- Close the spend-cap race in `data_build/build_full.py` lines 45–47. The $30 check sits outside the lock, so several batches can pass it together. A stricter cap can change which rows are rephrased. The reported full build was about $13.
- Replace SHA-1 at `p2d/standing_eval.py` line 49. It only orders the frozen public subsample. Integrity hashes in that file are SHA-256. A different hash changes which 300 rows per source are frozen.
- Cap characters before tokenization in `predict()`. The tokenizer already truncates to 256 tokens. A character cap would change long messages.
- Resolve `../final` from the script directory in `data_build/judge_full.py` line 43 and `data_build/backfill_other.py` lines 23 and 64. Those paths are relative to the current directory. Changing them changes a run that was started elsewhere.
- Edit the model card's `sys.path.insert` quick start. That is the published import of `predict.py` shipped in the model bundle. The card text was left as it is.

No dependency was added or removed. No new service or model was added.

## Checks

- Before and after: `python3 -m compileall -q -f .` exited 0.
- The repo has no test suite, linter config, or CI workflow. `ruff`, `flake8`, `pytest`, `pyflakes`, `mypy`, and `pylint` are not installed. PowerShell is not installed, so `training/launcher_p2.ps1` was not executed.
- `torch` and `transformers` are not installed, so training, `scripts/download.py`, and inference against the real checkpoint were not run. No Gemini call was made. No deploy script is in this repo, and none was run. Nothing was pushed to Hugging Face. The Hub was read to confirm the published file list: `model.safetensors`, no custom code, positive finite temperatures.
- Stdlib checks passed: the Gemini URL contains no key and the JSON body matches the previous client; off-host and non-https redirects are refused; a key containing a carriage return is rejected; exception text that contains the key is redacted; `predict.load` requires a directory, safetensors, and local files, and rejects 0, negative, infinite, and non-numeric temperatures; ordinary CSV cells are unchanged.

## Counts

Tracked files, with a final partial line counted. Code is `*.py` and `*.ps1`. Dependencies are the non-comment lines in `requirements.txt`. `REVIEW.md` is included in files and in all lines, and it is not code.

| | Before | After |
|---|---:|---:|
| Files | 29 | 28 |
| All lines | 2149 | 2198 |
| Code lines (py + ps1) | 1677 | 1627 |
| Dependencies | 5 | 5 |
