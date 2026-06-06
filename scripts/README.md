# Component Conversion Pipeline

Convert every `downloads/<component_id>/` folder into a schema JSON draft, CAD-generation instructions, and review metadata:

```bash
python scripts/convert_downloads.py --output outputs
```

Useful options:

```bash
python scripts/convert_downloads.py --limit 10 --output outputs_smoke
python scripts/convert_downloads.py --sample 5 --sample-seed 20260605 --output outputs_random5
python scripts/convert_downloads.py --component-id 110300011570 --component-id 110300026020 --output outputs_selected
python scripts/convert_downloads.py --downloads downloads --output outputs
python scripts/convert_downloads.py --limit 10 --output outputs_llm_smoke --use-ollama --ollama-model gemma4:latest --ollama-max-images 2 --progress
python scripts/convert_downloads.py --output outputs_llm --use-ollama --ollama-model gemma4:latest --ollama-max-images 1 --resume --progress
```

Each processed component gets:

- `component.json`: draft object for `component_schema.json`.
- `cad_instructions.md`: manufacturing-oriented CAD brief.
- `review.json`: confidence, missing evidence, validation errors, and source traceability.

Batch indexes:

- `outputs/summary.csv`
- `outputs/review_queue.jsonl`
- `outputs/ready.jsonl`
- `outputs/failures.json`

The pipeline targets exact-manufacturing readiness. It does not invent hidden geometry, GD&T, tolerances, or finishes; those gaps are recorded as `needs_review:*` flags.

For higher-quality CAD briefs, use `--use-ollama`. The current recommended local model is `gemma4:latest`, because it supports vision and gives stable JSON through Ollama's API. Use `--resume` for long runs so interrupted batches can continue without reprocessing completed components.

## CadQuery QC Gate

Score generated `cadquery_script.py` files against their `component.json` and downloaded source drawings/specs:

```bash
python scripts/qc_cadquery_outputs.py \
  --outputs outputs_openai_compare_10/gpt-5.4-mini \
  --downloads downloads \
  --model gpt-5.4-mini \
  --output qc_openai_mini
```

For deterministic-only checks without OpenAI vision review:

```bash
python scripts/qc_cadquery_outputs.py --outputs outputs_openai_compare_10/gpt-5.4-mini --downloads downloads --output qc_openai_mini --skip-vision
```

The QC command writes one `qc.json` per component plus `qc_summary.csv` and `qc_summary.json`. Each scorecard includes schema/static checks, CadQuery execution metrics, source dimension checks, drawing-match status, `overall_status`, and a compact `repair_prompt`.
