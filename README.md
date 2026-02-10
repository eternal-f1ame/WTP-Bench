# WTP-Bench: Who's That Pokémon?

**A Visual Recognition Benchmark for Vision-Language Models**

*Can frontier multimodal models identify Pokémon from official artwork -- and from silhouettes alone?*

WTP-Bench evaluates 27 vision-language models across **1,160 Pokémon** spanning all 8 generations, including regional forms, Mega Evolutions, and Gigantamax variants. Inspired by the iconic anime segment, models face both high-quality artwork and pure silhouettes in a zero-shot "Who's That Pokémon?" challenge.

> **Aaditya Baranwal** -- University of Central Florida, CRCV

---

## Highlights

- **1,160 Pokémon** -- the most comprehensive VLM recognition benchmark covering base forms, Megas, Gigantamax, Alolan, Galarian, Hisuian, and Paldean variants
- **Dual evaluation modes** -- official artwork (HQ) and silhouette-only (Shadow Ball) to isolate shape understanding from color/texture reliance
- **Tiered scoring** -- Master Ball (exact form match), Ultra Ball (correct species), Great Ball (correct evolutionary line) -- revealing where models actually fail
- **27 models benchmarked** -- 7 proprietary (GPT-4.1, Gemini 2.5 Pro/Flash, Claude 4.5 Sonnet/Haiku, Grok 4, GPT-4o-mini) and 20 open-weight (Qwen, LLaVA, InternVL, Ovis, PaLIGemma, Florence-VL, and more)

## Key Results

| | Model | Master Ball | Ultra Ball | Great Ball | Shadow Ball :star: |
|---|---|---|---|---|---|
| 1 | GPT-4.1 | **79.0%** | 84.2% | 84.7% | 49.1% |
| 2 | Gemini 2.5 Pro | 78.2% | **88.1%** | **91.4%** | 23.0% |
| 3 | Gemini 2.5 Flash | 77.2% | 83.0% | 86.2% | 42.6% |
| 4 | Claude 4.5 Sonnet | 65.1% | 79.2% | 81.9% | **53.4%** |
| 5 | GPT-4o-mini | 49.3% | 62.0% | 68.7% | 29.0% |
| -- | Best Open-Weight (Qwen3-VL 4B) | 13.4% | 18.5% | 22.4% | 2.2% |

Full leaderboard with all 27 models available on the [project website](https://eternal-f1ame.github.io/WTP-Bench/).

## Key Findings

1. **Proprietary models dominate** -- the top 7 models are all proprietary; the best open-weight model (Qwen3-VL 4B) scores only 13.4% exact match
2. **Silhouettes humble everyone** -- GPT-4.1 drops from 79% to 49% without color; Claude 4.5 Sonnet is the only model exceeding 50% on silhouettes
3. **Massive Gen 1 hallucination bias** -- when open-weight models guess wrong, 57–79% of guesses are Gen 1 Pokémon (only 19.6% of the dataset)
4. **Form blindness** -- most models score near 0% on Mega/Gmax/regional forms despite decent base form accuracy; 162 Pokémon get zero correct across all models
5. **Generation gap** -- proprietary models average 73% on Gen 2 but only 39% on Gen 8; open-weight models collapse after Gen 1

## Evaluation Tiers

| Tier | Criteria | Example |
|---|---|---|
| **Shadow Ball** | Any tier, but on silhouettes only | Shape-only recognition |
| **Master Ball** | Exact match -- species, form, variant | `charizard-mega-x` |
| **Ultra Ball** | Correct species, wrong/missing form | Predicted `vulpix` for `vulpix-alola` |
| **Great Ball** | Same evolutionary line | Predicted `charmeleon` for `charizard` |


## Quick Start

### Installation

```bash
git clone https://github.com/eternal-f1ame/WTP-Bench.git
cd WTP-Bench
pip install -r requirements.txt
```

### Run Inference -- Open-Weight Models

```bash
python src/scripts/vlm_infer_template.py \
  --input-csv data/metadata.csv \
  --output-csv predictions/my_model_hq.csv \
  --prompt "Who's that Pokemon?" \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --base-dir data
```

### Run Inference -- Proprietary Models (via OpenRouter)

```bash
export OPENROUTER_API_KEY="your-key-here"

python src/scripts/vlm_infer_openrouter.py \
  --input-csv data/metadata.csv \
  --output-csv predictions/gpt5_hq.csv \
  --prompt "Who's that Pokemon?" \
  --model-id openai/gpt-5 \
  --base-dir data
```

### Evaluate Predictions

```bash
PYTHONPATH=src python src/scripts/eval_predictions.py \
  --pred-csv predictions/my_model_hq.csv
```

### Batch Run All Models

```bash
python src/scripts/run_all_models.py --config-dir configs/legacy
```

## Project Structure

```
WTP-Bench/
├── configs/
│   ├── legacy/          # Open-weight model configs (22 models)
│   ├── proprietary/     # Proprietary model configs via OpenRouter
│   └── remote/          # Remote/API-served open models
├── data/
│   ├── imagesHQ/        # Official artwork (1,160 images)
│   ├── silhouette/      # Silhouette versions
│   ├── metadata.csv     # Image paths, labels, generations, forms
│   ├── pokemon_types.json
│   └── evolution_families.json
├── predictions/         # Model prediction CSVs
├── results/
│   ├── eval_hq.csv      # HQ leaderboard
│   ├── eval_silhouette.csv
│   └── analysis_full.txt
├── src/
│   ├── bench/           # Shared utilities (label matching, I/O)
│   └── scripts/         # Inference and evaluation scripts
├── website/             # Project website with interactive leaderboard
└── requirements.txt
```

## Dataset

Images are not included in this repository. The dataset is available on HuggingFace:

```python
from datasets import load_dataset

wtp = load_dataset('aaditya-baranwal/wtp-bench', split='test')
```

Each sample contains: `image`, `pokemon_name`, `generation`, `form`, and `difficulty`.

## Citation

```bibtex
@misc{baranwal2026wtpbench,
  title        = {WTP-Bench: Who's That Pok\'{e}mon? A Visual Recognition
                  Benchmark for Multimodal Models},
  author       = {Baranwal, Aaditya},
  year         = {2026},
  howpublished = {\url{https://github.com/eternal-f1ame/WTP-Bench}},
  note         = {University of Central Florida, CRCV}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

Pokémon and all related names, images, and assets are trademarks and © of Nintendo, Game Freak, and The Pokémon Company. This project is not affiliated with or endorsed by any of these entities.
