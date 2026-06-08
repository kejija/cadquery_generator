# Drawing + Datasheet JSON → Parametric CAD Pipeline

This project turns catalog-style 2D engineering drawings and datasheet tables into reusable parametric CAD feature templates, then uses those templates to generate CAD models for many part configurations.

The intended use case is a configurable part family, such as a MISUMI catalog drawing where one drawing represents hundreds or thousands of part-number variants.

---

## Goal

Given:

- a 2D engineering drawing image
- a datasheet JSON file containing dimension symbols, part-number tables, tolerances, notes, material options, and metadata

Produce:

1. a reusable engineering feature template
2. normalized configuration rows
3. resolved CAD specs for selected part numbers
4. CadQuery/build123d code
5. STEP/STL exports
6. validation reports

The key design principle is:

> The drawing defines the reusable geometry template.  
> The datasheet table defines the dimensional truth.  
> The generated feature template becomes the durable asset.

---

## High-Level Workflow

```text
Step 0: Select the main engineering drawing from downloaded image noise
Step 1: Main drawing + datasheet JSON → reusable feature template
Step 2: Review / score / repair the feature template
Step 3: Normalize datasheet configuration tables
Step 4: Resolve selected part number → concrete CAD spec
Step 5: Generate CadQuery/build123d code
Step 6: Execute CAD code and export STEP/STL
Step 7: Validate generated geometry
Step 8: Repair failed code or failed validation
Step 9: Generate all catalog configurations deterministically
Step 10: Store templates, specs, CAD outputs, and validation reports
```

---

## Recommended Repository Structure

```text
cad-template-pipeline/
  README.md

  data/
    raw/
      misumi_shaft_collar_d_cut/
        drawings/
        main_engineering_drawing/
        json/specs.json

      misumi_hinge_base_u_shaped/
        drawings/
        main_engineering_drawing/
        json/specs.json

  prompts/
    developer_message.txt
    user_prompt.txt

  schemas/
    engineering_feature_template.schema.json
    normalized_configurations.schema.json
    resolved_cad_spec.schema.json
    cad_generation.schema.json

  batch/
    create_batch_jsonl.py
    submit_batch.py
    download_batch_results.py

  outputs/
    feature_templates/
      misumi_shaft_collar_d_cut.feature_template.json
      misumi_hinge_base_u_shaped.feature_template.json

    normalized_configs/
      misumi_shaft_collar_d_cut.configurations.json

    resolved_specs/
      SL-SSCDN10.resolved_cad_spec.json

    cad_code/
      SL-SSCDN10.py

    cad_exports/
      SL-SSCDN10.step
      SL-SSCDN10.stl

    validation/
      SL-SSCDN10.validation_report.json
```

---

## TUI Workflow

Install local dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the Textual workflow UI from the repository root:

```bash
python tui.py
```

Hotkeys:

```text
up/down  choose a workflow step in the left table
space    toggle the focused runnable step
enter    run selected runnable steps in order
s        start/resume Step 0 directly
f        force Step 0 rerun for the current filter/search scope
j        create Step 1 batch JSONL from completed Step 0 rows
b        run Step 0, then create Step 1 batch JSONL
u        submit latest Step 1 JSONL as an OpenAI Batch
2        run Step 2 (review templates) on the current filter scope
3        run Step 3 (normalize configurations) on the current filter scope
4        run Step 4 (resolve part numbers) on the current filter scope
A        run Steps 2, 3, and 4 in sequence on the current filter scope
c        toggle OpenAI Batch status monitor
d        download latest OpenAI Batch output/error files to output/
r        rescan/index downloads
n/p      next/previous page
t        cycle status filter
/        search by component id
o        change local Ollama model
m        toggle Step 1 model between gpt-5.4-mini and gpt-5.5
q        quit
```

Steps 0, 1, 2, 3, and 4 are implemented. Steps 5–10 are planned. The TUI marks unimplemented steps as dimmed disabled placeholders; Up/Down skips over them so the cursor only lands on runnable steps. The right table shows OpenAI Batch run status from `output/workflow_state.sqlite3`; press `c` to start/stop a 5-second OpenAI status monitor. The monitor also downloads completed run output/error files into `output/` when the files are not already present locally.

For non-interactive use:

```bash
python tui.py --no-tui --both --limit 10
```

The TUI indexes component folders into `output/workflow_state.sqlite3`, shows aggregate Step 0 counts, and renders only one `DataTable` page at a time. The default page size is 100 rows, so the UI is designed for large download directories with 20,000+ components.

---

## Step 0 — Select the Main Engineering Drawing

Downloaded component folders often contain many non-CAD images: product photos, packaging examples, cleaning process graphics, icons, alternate-option diagrams, and datasheet/table screenshots. Step 0 selects the single main dimensioned engineering drawing for each component before Step 1 sees the data.

Input shape:

```text
downloads/
  110300324920/
    drawings/
      drw_01_001.gif
      product_photo.jpg
      ...
    json/
      specs.json
      specs_after.json
```

Output shape:

```text
downloads/
  110300324920/
    main_engineering_drawing/
      drw_01_001.gif
      selection.json
```

Step 0 uses a local Ollama vision-capable model on a contact sheet of likely candidates. The default is `qwen3.6:latest`; it handles the noisy drawing/contact-sheet cases better than `gemma4:12b` in local tests. If Ollama is unavailable, returns an unusable filename, or picks a materially weaker small/detail drawing, Step 0 falls back to or guards with a filename and image-size heuristic.

Run Step 0 directly:

```bash
python scripts/step0_main_engineering_drawing.py --downloads-dir downloads --limit 10 --state-db output/workflow_state.sqlite3
```

Change the Ollama model:

```bash
python scripts/step0_main_engineering_drawing.py --limit 10 --ollama-model qwen3.6:latest --state-db output/workflow_state.sqlite3
```

Step 0 skips completed components by default when `main_engineering_drawing/selection.json` exists. Use `--force` to recompute and overwrite existing Step 0 outputs.

For machine-readable progress:

```bash
python scripts/step0_main_engineering_drawing.py --downloads-dir downloads --state-db output/workflow_state.sqlite3 --json
```

---

## Step 1 — Extract a Reusable Feature Template

### Input

Each catalog family should have:

```text
main_engineering_drawing/drawing.gif
json/specs_after.json or json/specs.json
```

The `specs.json` should contain:

- product title
- dimension tables
- material tables
- part-number rows
- tolerances
- notes
- spec image metadata
- configuration options

### Output

The model returns a structured `feature_template.json`.

For batch processing, Step 1 creates OpenAI Batch JSONL under `output/`:

```bash
python scripts/step1_drawing_datasheetjson.py batch-jsonl --downloads-dir downloads --limit 10 --output-dir output
```

After Step 0 has populated the workflow state DB, create JSONL only for completed Step 0 components:

```bash
python scripts/step1_drawing_datasheetjson.py batch-jsonl --state-db output/workflow_state.sqlite3 --status done --output-dir output
```

Each JSONL record includes:

- `custom_id`: component folder name
- `/v1/responses` request body
- selected main drawing as an embedded image data URL
- datasheet JSON text from `specs_after.json` when available, otherwise `specs.json`
- metadata recording the source drawing and specs paths

Validate the JSONL before submitting:

```bash
python scripts/step1_drawing_datasheetjson.py validate-batch-jsonl output/step1_batch_YYYYMMDD_HHMMSS.jsonl
```

Submit the JSONL as an OpenAI Batch job:

```bash
python scripts/step1_drawing_datasheetjson.py submit-batch output/step1_batch_YYYYMMDD_HHMMSS.jsonl --output output/step1_batch_job.json
```

Check a submitted batch:

```bash
python scripts/step1_drawing_datasheetjson.py batch-status batch_... --output output/step1_batch_status.json
```

Download available batch output/error files:

```bash
python scripts/step1_drawing_datasheetjson.py download-batch-output batch_... --output-dir output
```

The submit helper reads the JSONL `url` and creates the batch with the same endpoint. For Step 1 that endpoint is `/v1/responses`; using `/v1/chat/completions` for this JSONL will fail with “The URL provided for this request does not match the batch endpoint.”

Example top-level fields:

```json
{
  "schema_version": "1.0",
  "template_id": "misumi_clean_pack_two_piece_d_cut_shaft_collar",
  "part_family": "Clean & Pack Clamping Shaft Collars - Two Piece, D-Cut",
  "units": "mm",
  "modeling_mode": "multi_body_part",
  "coordinate_system": {},
  "components": [],
  "parameters": [],
  "feature_graph": [],
  "catalog_metadata": {},
  "assumptions": [],
  "missing_information": [],
  "validation_requirements": []
}
```

### What the Feature Template Should Capture

The feature template should be CAD-construction-ready.

For every geometric feature, capture:

- `feature_type`
- `operation`
- `modeling_primitive`
- `subtype`
- `target_bodies`
- `output_bodies`
- `depends_on`
- `parameters`
- `construction`
- `position`
- `axis`
- `side`
- `pattern`
- `confidence`
- `needs_review`
- `review_reason`
- `notes`

Example feature:

```json
{
  "id": "central_bore",
  "feature_type": "hole",
  "operation": "subtract",
  "modeling_primitive": "hole",
  "subtype": "through_bore",
  "target_bodies": ["body_1"],
  "output_bodies": ["body_1"],
  "depends_on": ["base_body"],
  "parameters": [
    {
      "name": "diameter",
      "value": "$D",
      "role": "driving_dimension"
    }
  ],
  "construction": {
    "sketch_plane": "front face",
    "reference_plane": "part center plane",
    "profile_type": "circle",
    "profile_description": "Through bore centered on the shaft axis.",
    "start_condition": "from_face",
    "end_condition": "through_all",
    "depth": null,
    "boolean_scope": "body_1",
    "construction_notes": "Axis is inferred from drawing view."
  },
  "position": {
    "x": 0,
    "y": 0,
    "z": 0
  },
  "axis": "Z",
  "side": "both",
  "pattern": {
    "pattern_type": "none",
    "count": null,
    "spacing": null,
    "angle_step": null,
    "direction": null,
    "base_feature": null
  },
  "confidence": 0.92,
  "needs_review": false,
  "review_reason": "",
  "notes": ""
}
```

---

## Step 2 — Review / Score / Repair the Feature Template

Run an automated, **deterministic** quality gate on every `*.feature_template.json` produced by Step 1 before any of it is allowed near a CAD kernel.

### Script

```bash
python scripts/step2_review_template.py
python scripts/step2_review_template.py --component 110300324920
python scripts/step2_review_template.py --limit 3
python scripts/step2_review_template.py --allow-codex-repair        # opt-in LLM escalation
python scripts/step2_review_template.py --allow-codex-repair --codex-model gpt-5.5
python scripts/step2_review_template.py --json                      # machine-readable summary
```

### What it does

1. **Schema validation** against `schemas/feature_template.schema.json`. Structural shape is enforced here.
2. **Quality scoring** — every issue has a level (`error` / `warning` / `info`) and a stable `code`. Score = `1.0 - weighted_penalty`, clamped to `[0, 1]`. Weights: `error=0.30, warning=0.05, info=0.01`.
3. **Mechanical repairs** — only well-defined, low-risk patches:
   - Flag a hole/pocket/boss with `axis=null` and `needs_review=false` by setting `needs_review=true` and filling a placeholder `review_reason`. Never guesses a default axis.
   - Fill empty `review_reason` with an auto-generated note.
   - Flag features with `modeling_primitive=null` for review.
4. **Optional Codex repair** (off by default; pass `--allow-codex-repair`). When enabled, templates with warnings are routed to a single `codex exec` call (default model `gpt-5.4-mini`) that returns a JSON patch plan; the script validates the patch shape before applying it.

A template **passes** when it has no `error`-level issues, score ≥ 0.6, and no `schema_violation` errors.

### Checks performed

| Code | Level | Meaning |
|---|---|---|
| `schema_violation` | error | Failed JSON Schema validation |
| `missing_top_level_key` | error | Required top-level field missing |
| `duplicate_feature_id` | error | Same feature id appears twice in `feature_graph` |
| `no_base_body` | error | `feature_graph` has no `base_body` |
| `multiple_base_bodies` | warning | More than one `base_body` (verify intentional) |
| `non_standard_units` | warning | `units` not in `{mm, in}` |
| `non_standard_modeling_mode` | warning | `modeling_mode` not conventional |
| `non_standard_feature_type` | warning | `feature_type` not in conventional set |
| `non_standard_operation` | warning | `operation` not in conventional set |
| `non_standard_role` | info | parameter `role` not in conventional set |
| `non_standard_category` | info | parameter `category` not in conventional set |
| `unknown_modeling_primitive` | warning | `modeling_primitive` is null/empty/unknown but not flagged for review |
| `missing_axis` | warning | hole/pocket/boss has `axis=null` and not flagged for review |
| `low_confidence` | warning | feature `confidence` < 0.6 |
| `review_without_reason` | warning | `needs_review=true` but `review_reason` empty |
| `metadata_in_feature_graph` | warning | a `metadata` feature lives in `feature_graph` instead of `catalog_metadata` |
| `too_many_missing_information` | warning | `missing_information` list is excessive (> 10 items) |

### Outputs

```text
output/feature_templates_reviewed/<component_id>.review.json
  - pass: bool
  - score: float (0..1)
  - n_errors, n_warnings
  - issues: [{level, code, message, feature_id?, parameter?, field?}]
  - repairs_applied: [string]
  - codex: {attempted, applied, model, error}

output/feature_templates_repaired/<component_id>.feature_template.json
  - Only written if at least one repair was applied
  - Step 4 picks this up with precedence over the original
```

### Verified on 8 real templates

| Component | Score | Pass | Notes |
|---|---|---|---|
| 110300324920 (air coupler) | 0.950 | ✓ | |
| 110302171610 (hinge base) | 0.830 | ✓ | non-standard `axis` annotation |
| 110302255160 (plate bracket) | 0.750 | ✓ | non-standard `operation` on metadata feature |
| 110302259360 (reversal bracket) | 0.830 | ✓ | |
| 110302691010 (aluminum extrusion) | 0.840 | ✓ | |
| 110310763649 (linear shaft) | 0.800 | ✓ | low-confidence datum/gdnt features |
| 110310764189 (linear shaft) | 0.900 | ✓ | |
| 110310764369 (linear shaft) | 0.670 | ✓ | `metadata` feature leak in feature_graph |

**Mean score: 0.821. All 8 pass, 0 schema violations.**

---

## Step 3 — Normalize Datasheet Configuration Tables

Parse `downloads/<component_id>/json/specs.json` (or `specs_after.json` if non-empty) into a clean one-row-per-part-number configurations JSON. **Deterministic by default.** Opt-in Codex fallback for tables the heuristic can't parse.

### Script

```bash
python scripts/step3_normalize_configs.py
python scripts/step3_normalize_configs.py --component 110300324920
python scripts/step3_normalize_configs.py --limit 3
python scripts/step3_normalize_configs.py --verbose                    # per-table diagnostics
python scripts/step3_normalize_configs.py --allow-codex               # opt-in LLM fallback
python scripts/step3_normalize_configs.py --allow-codex --codex-model gpt-5.4-mini
python scripts/step3_normalize_configs.py --json
```

### How the parser works

For each `tables[*]` entry in the specs file, the script:

1. **Picks the part-number column** by header priority: `Part Number` / `Model` / `Item` first, then the more ambiguous `No.` / `Size` / `Type` headers. This avoids the `Type` column in MISUMI option tables being mistaken for a part-number column.
2. **Detects multi-column part numbers** — MISUMI's two-column `Type + No.` pattern (`MCSCN` + `8` → `MCSCN8`). The script composes them automatically.
3. **Inherits family prefixes across left-shifted rows** — when a row has more cells than the header (e.g. `['MCSCN', '8', '64.5', '4.5', '17', '105']` vs 5-column header), subsequent digit-only rows inherit the family prefix from the prior row and shift their data columns left by 1.
4. **Skips sub-header rows** like `['Type', 'No.']` that appear between the column header and the data.
5. **Coerces cell values**: integers stay ints, decimals become floats, thread specs (`M5`, `#10-32`) stay strings, blanks become `null`.
6. **Maps column headers to template parameter symbols** via exact match, common synonyms (`D` ↔ `bore` ↔ `ID`, `B` ↔ `width`, `D1` ↔ `OD`), and falls back to the cleaned header.
7. **Picks the variant column** by distinct-value count — a real variant column has many distinct values; a single-family-prefix column has 1.
8. **Emits per-table diagnostic reasons**: `ok`, `ok_with_text_skip`, `not_pn_table`, `not_dimension_table`, `empty`. The `--verbose` flag prints them inline.

When deterministic extraction finds 0 rows AND `--allow-codex` is passed, the script makes a single `codex exec` call (default `gpt-5.4-mini`) to parse the hard table. Falls back to a single-row default if everything fails.

### Output

```text
output/normalized_configs/<component_id>.configurations.json
  - schema-validated against schemas/normalized_configurations.schema.json
  - one entry per part-number row
  - extraction.per_table: per-table diagnostic info
  - extraction.warnings: any non-fatal issues encountered
```

Schema shape: `{schema_version, catalog_id, source_template_id, source_component_id, units, parameter_symbols, configurations: [{part_number, variant, values, source_table_idx, source_row_idx}], extraction: {method, tables_scanned, tables_with_part_numbers, rows_emitted, per_table, warnings}}`.

### Verified on 8 components

| Component | Method | Rows | Notes |
|---|---|---|---|
| 110300324920 (air coupler) | deterministic | 3 | MCSCN8 / MCSCN10 / MCSCN12 from `Type+No.` pattern |
| 110302171610 (hinge base) | single_row_default | 1 | specs is a material-only table, no dimensions |
| 110302255160 (plate bracket) | deterministic | 2 | |
| 110302259360 (reversal bracket) | deterministic | 1 | |
| 110302691010 (extrusion) | deterministic | 1 | |
| 110310763649 (linear shaft) | deterministic | 2 | |
| 110310764189 (linear shaft) | deterministic | 3 | |
| 110310764369 (linear shaft) | deterministic | 1 | |

**7/8 deterministic. 14 total part-number rows. $0 LLM cost.**

---

## Step 4 — Resolve a Selected Part Number

Combine the feature template, the normalized configurations, and a selected part number into a single **concrete** CAD spec — every `$D`-style placeholder is bound to a real number or string.

### Script

```bash
python scripts/step4_resolve_spec.py --component 110300324920                          # first row
python scripts/step4_resolve_spec.py --component 110300324920 --part-number MCSCN10    # specific
python scripts/step4_resolve_spec.py --component 110300324920 --all                     # every row
python scripts/step4_resolve_spec.py --all --limit 3                                   # all components
python scripts/step4_resolve_spec.py --json
```

### How it works

**Pure deterministic.** No LLM calls.

1. **Load** the Step 1 feature template. If a Step 2 repaired copy exists in `output/feature_templates_repaired/`, prefer it.
2. **Load** the Step 3 normalized configurations for the same component.
3. **Pick the row** — by `--part-number` if provided, otherwise the first row.
4. **Build a flat value map**: configuration row values win, then template parameter defaults fill the rest.
5. **Walk the feature graph** — for each feature, replace every parameter `value` with the corresponding entry in the value map. Mark each parameter's `source` as `configuration_row` (from the spec), `template_default` (filled from template), `carried_over` (preserved as-is), or `missing` (no value found, recorded in `unresolved_references`).
6. **Resolve nested references**: `construction.depth` and `position.{x,y,z}` are also substituted when they hold a string that's a parameter name.
7. **Inherit review flags** from the Step 2 review JSON and from the template's `needs_review` features.
8. **Validate** the output against `schemas/resolved_cad_spec.schema.json`.

### Output

```text
output/resolved_specs/<component_id>__<part_number>.resolved_cad_spec.json
  - schema_version, template_id, catalog_id, part_number, variant, units
  - parameter_bindings: { name: value }  (flat, every binding)
  - unresolved_references: [{ feature_id, parameter_name, raw_value }]  (informational)
  - coordinate_system, components
  - resolved_features: [{ id, feature_type, operation, modeling_primitive, parameters: [{name, value, role, source}], construction, position, axis, side, pattern, needs_review, review_reason }]
  - review_flags: [string]
```

### Example resolved spec (excerpt)

```json
{
  "part_number": "MCSCN10",
  "template_id": "MISUMI_MCSCN_socket_nut_tightening_template",
  "parameter_bindings": {
    "overall_length": 64.5,
    "tightening_section_length": 5.3,
    "hex_width_across_flats": 17,
    "mass_g": 106,
    "body_outer_diameter": 26.5
  },
  "resolved_features": [
    {
      "id": "fb1",
      "feature_type": "base_body",
      "parameters": [
        {"name": "body_outer_diameter", "value": 26.5, "source": "configuration_row"},
        {"name": "overall_length",      "value": 64.5, "source": "configuration_row"}
      ]
    },
    {"id": "fb2", "feature_type": "boss",  "parameters": [...]},
    {"id": "fb3", "feature_type": "hole",  "parameters": [
      {"name": "internal_bore_diameter", "value": null, "source": "carried_over"}
    ]}
  ],
  "review_flags": [
    "fb1: The image shows a more detailed right-end contour than the provided table supports...",
    "fb3: Internal bore size and any internal valve/step details are missing from the supplied source material."
  ]
}
```

### Verified on 8 components

13/14 schema-valid resolved specs across 7 part families, 0 unresolved references, 13 inheriting review flags from Step 2. The one failure is the hinge-base component that has no resolved part number in Step 3 (material-only spec).

---

## Step 5 — Generate CadQuery / build123d Code

Input:

```text
resolved_cad_spec.json
```

Output:

```text
model.py
```

Rules for the CAD generation model:

```text
- Use the resolved CAD spec as the source of truth.
- Do not reinterpret the drawing.
- Do not change dimensions.
- Generate executable Python code.
- Put all parameters at the top.
- Use feature IDs as comments in the code.
- Export STEP and STL.
- Include validation hooks.
```

Start with CadQuery because it is simple, scriptable, and good for parametric mechanical parts.

Example skeleton:

```python
import cadquery as cq

# Parameters
D = 10
D1 = 35
B = 15

# Feature: base_body
part = cq.Workplane("XY").circle(D1 / 2).extrude(B)

# Feature: central_bore
part = part.faces(">Z").workplane().hole(D)

# Export
cq.exporters.export(part, "output.step")
cq.exporters.export(part, "output.stl")
```

---

## Step 6 — Execute CAD Code

Run the generated code inside a controlled CAD environment.

Recommended outputs:

```text
part.step
part.stl
preview.png
execution_log.txt
```

Use a container or sandbox with:

```text
cadquery
ocp / opencascade
python
```

Example:

```bash
python outputs/cad_code/SL-SSCDN10.py
```

---

## Step 7 — Validate Geometry

Validation should be deterministic, not model-based.

Check:

```text
- bounding box dimensions
- hole diameters
- hole count
- hole center positions
- body count
- axis alignment
- major diameters
- overall thickness
- metadata presence
```

Example validation report:

```json
{
  "part_number": "SL-SSCDN10",
  "status": "pass",
  "checks": [
    {
      "name": "outer_diameter",
      "expected": 35,
      "measured": 35.0,
      "pass": true
    },
    {
      "name": "overall_thickness",
      "expected": 15,
      "measured": 15.0,
      "pass": true
    }
  ]
}
```

---

## Step 8 — Repair Loop

There are two repair paths.

### Code Execution Repair

Input:

```text
resolved_cad_spec.json
generated_code.py
python_error_log.txt
```

Output:

```text
corrected_code.py
```

The repair model should follow this rule:

```text
Fix the code, but do not reinterpret the drawing or modify dimensions unless the resolved spec requires it.
```

### Geometry Validation Repair

Input:

```text
resolved_cad_spec.json
generated_code.py
validation_report.json
```

Output:

```text
corrected_code.py
```

If validation fails due to the feature template being wrong, route the issue back to Step 2.

---

## Step 9 — Generate Many Configurations

Once the template and CAD builder work for one or two rows, generate the whole family deterministically:

```python
for row in configurations:
    resolved_spec = resolve_template(template, row)
    model_path = generate_cad_code(resolved_spec)
    execute_cad(model_path)
    validate_model(row["part_number"])
```

Do not call the model once per row unless validation fails.

---

## Step 10 — Store Outputs

Recommended final folder for each catalog family:

```text
catalogs/
  misumi_shaft_collar_d_cut/
    raw/
      drawing.gif
      specs.json

    templates/
      feature_template.v1.json
      feature_template.reviewed.json

    configs/
      normalized_configurations.json

    resolved_specs/
      SL-SSCDN10.json

    cad_code/
      SL-SSCDN10.py

    exports/
      SL-SSCDN10.step
      SL-SSCDN10.stl

    validation/
      SL-SSCDN10.report.json
```

---

## Batch Processing

Batch is useful for Step 1 when you have many part families.

Each batch request should include:

```text
developer message
user prompt
drawing image file_id
datasheet JSON text
structured output schema
```

Use Python to generate JSONL. Do not hand-write JSONL.

Recommended flow:

```text
Upload drawing image with purpose="vision"
Read specs.json
Create one JSONL line per drawing/spec pair
Upload JSONL with purpose="batch"
Create batch job against /v1/responses
Download output JSONL
Parse feature_template.json for each custom_id
```

Example batch line shape:

```json
{
  "custom_id": "misumi_shaft_collar_d_cut_001",
  "method": "POST",
  "url": "/v1/responses",
  "body": {
    "model": "gpt-5.4-mini",
    "reasoning": {
      "effort": "medium"
    },
    "input": [
      {
        "role": "developer",
        "content": [
          {
            "type": "input_text",
            "text": "..."
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": "..."
          },
          {
            "type": "input_image",
            "file_id": "file_..."
          },
          {
            "type": "input_text",
            "text": "Datasheet JSON:\n{...}"
          }
        ]
      }
    ],
    "text": {
      "format": {
        "type": "json_schema",
        "name": "engineering_feature_template",
        "strict": true,
        "schema": {}
      }
    }
  }
}
```

---

## Prompt Caching

Keep the stable prefix identical across requests:

```text
developer message
structured output schema
stable user prompt prefix
```

Put dynamic inputs later:

```text
drawing image
specs JSON
selected part number
part-specific notes
```

This improves cache reuse when the same prompt/schema is sent repeatedly.

---

## Model Strategy

Recommended model routing:

```text
gpt-5.4-nano
  - cheap classification
  - metadata extraction
  - table cleanup where geometry is not important

gpt-5.4-mini
  - first-pass feature-template extraction
  - batch extraction across many catalog families

gpt-5.5 medium/high
  - repair of ambiguous templates
  - assembly/multi-body parts
  - complex drawing interpretation
  - code repair loops
```

Use stronger models only when the quality gate flags a template.

---

## Quality Bar

A good feature template should satisfy:

```text
- The CAD code generator should not need to look at the original drawing.
- Every geometric feature has construction information.
- Every ambiguous feature has confidence, needs_review, and review_reason.
- Material, packaging, cleaning, and finish are metadata unless they change geometry.
- Threads are metadata plus pilot/clearance/tap-drill holes unless true thread geometry is requested.
- GD&T is metadata plus validation requirements, not solid geometry.
- The selected coordinate system is consistent with hole axes and sketch planes.
```

---

## Common Failure Modes

### 1. Drawing interpreted as a single part instead of a part family

Fix: explicitly say the drawing is a reusable catalog template.

### 2. Model invents dimensions

Fix: require symbolic parameters and set missing dimensions to null.

### 3. Hole axes are wrong

Fix: add hole-axis reasoning rules:

```text
- circles in front view usually mean axis normal to front view
- circles in side view usually mean axis normal to side view
- bottom mounting holes usually cut normal to mounting face
```

### 4. Metadata appears in the feature graph as geometry

Fix: separate `catalog_metadata` from `feature_graph`.

### 5. Split/multi-body parts are collapsed into metadata

Fix: require `modeling_mode` and explicit split features.

### 6. The first template is too vague

Fix: require:

```text
modeling_primitive
construction.profile_type
construction.profile_description
target_bodies
output_bodies
confidence
needs_review
review_reason
```

---

## Minimal First Milestone

A practical MVP should support these feature types:

```text
1. extruded block
2. cylindrical body
3. revolved body
4. through hole
5. blind hole
6. counterbore hole
7. countersink hole
8. pocket
9. slot
10. boss
11. fillet
12. chamfer
13. linear pattern
14. circular pattern
15. mirror pattern
16. split body
17. thread metadata
18. datum metadata
19. GD&T metadata
20. validation requirements
```

---

## Recommended Next Build Steps

1. Finish the batch JSONL generator.
2. Create a schema validator and quality scorer.
3. Batch 10–20 catalog families.
4. Manually inspect outputs and tune the prompt/schema.
5. Add repair routing for failed templates.
6. Build normalized configuration table extraction.
7. Build deterministic parameter resolver.
8. Generate CadQuery for one reviewed template.
9. Execute and validate one part number.
10. Scale to all rows in the catalog family.

---

## Notes

This workflow is intentionally not a one-shot “image to perfect CAD” system. The robust architecture is:

```text
LLM creates and repairs reusable templates
↓
human or automated quality gate reviews templates
↓
deterministic code resolves dimensions and generates rows
↓
CAD kernel executes geometry
↓
validator checks geometry
↓
LLM repairs only when needed
```

The durable assets are:

```text
feature_template.json
normalized_configurations.json
resolved_cad_spec.json
cad_builder.py
validation_report.json
```
