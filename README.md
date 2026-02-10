# Who's That Pokémon Bench (VLM)

Minimal benchmark scaffold for silhouette-style Pokémon identification.

## Data layout

```
data/
  data/imagesHQ            # source HQ images
  silhouette/          # per-Pokémon silhouette folders
  metadata.csv         # image_path -> label (relative paths)
  extra/               # optional auxiliary data
```

### Zero-shot evaluation (no train/val splits)

We evaluate VLMs directly on the silhouettes with the prompt:

"Who's that Pokemon?"

Use the inference template to generate a predictions CSV (paths are resolved relative to `--base-dir`):

```
python src/scripts/vlm_infer_template.py \
  --input-csv data/metadata.csv \
  --output-csv data/predictions_vlm.csv \
  --prompt "Who's that Pokemon?" \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --base-dir data
```

Model lists are tracked in [configs/vlm_models.yaml](configs/vlm_models.yaml).

### Proprietary VLMs (GPT/Claude/Grok/Gemini via OpenRouter)

```
python src/scripts/vlm_infer_openrouter.py \
  --input-csv data/metadata.csv \
  --output-csv data/predictions_proprietary.csv \
  --prompt "Who's that Pokemon?" \
  --model-id openai/gpt-4o-mini \
  --base-dir data
```

Set your API key via environment variables:
`OPENROUTER_API_KEY`.

### Evaluate predictions

```
PYTHONPATH=src python src/scripts/eval_predictions.py \
  --pred-csv data/predictions_vlm.csv
```

## Folder structure

- `configs/` model lists and configs
- `data/` dataset files and silhouettes
- `src/bench/` shared utilities
- `src/scripts/` inference + evaluation scripts

## Notes

- This scaffold does **not** ship Pokémon images.
- Please confirm any dataset licensing before public release.

## Leaderboard

See [LEADERBOARD.md](LEADERBOARD.md).

## Benchmark card

See [BENCHMARK_CARD.md](BENCHMARK_CARD.md).

## Results schema

See [RESULTS_SCHEMA.md](RESULTS_SCHEMA.md).

## Metadata CSV format

Columns (recommended):
- `image_path`: **relative path** to the image (no absolute paths)
- `label`: canonical Pokémon name (used for evaluation)
- `gen`: integer generation (1-9)
- `form`: optional form name (e.g., Alolan, Galarian)

If no metadata is provided, labels are inferred from filenames.
