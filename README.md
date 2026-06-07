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

Only Step 0 and Step 1 are implemented today. The TUI shows later pipeline stages as dimmed disabled placeholders, and Up/Down skips over them so the cursor only lands on runnable steps. The right table shows OpenAI Batch run status from `output/workflow_state.sqlite3`; press `c` to start/stop a 5-second OpenAI status monitor. The monitor also downloads completed run output/error files into `output/` when the files are not already present locally.

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

Do not directly generate CAD from the first model output.

Run an automated quality gate first.

Recommended checks:

```text
- Schema validates
- Exactly one or more base_body features exist
- No geometric feature has modeling_primitive = unknown unless needs_review = true
- No hole has axis = null unless needs_review = true
- Confidence is not too low
- Missing information list is not excessive
- Metadata is not mixed into geometry
- Hole axes are plausible given the coordinate system
- Features with ambiguous construction have review_reason filled
```

Example scoring rule:

```python
def score_template(template: dict) -> dict:
    issues = []

    feature_graph = template.get("feature_graph", [])
    base_features = [f for f in feature_graph if f.get("feature_type") == "base_body"]

    if not base_features:
        issues.append("No base_body feature found.")

    for feature in feature_graph:
        if feature.get("feature_type") in {"hole", "pocket", "boss", "slot"}:
            if feature.get("modeling_primitive") == "unknown":
                issues.append(f"{feature['id']} has unknown modeling primitive.")

            if feature.get("axis") is None and not feature.get("needs_review"):
                issues.append(f"{feature['id']} has no axis but is not marked for review.")

            if feature.get("confidence", 1.0) < 0.6:
                issues.append(f"{feature['id']} has low confidence.")

    return {
        "pass": len(issues) == 0,
        "issues": issues
    }
```

Route weak templates to a refinement call, ideally using a stronger model.

---

## Step 3 — Normalize Datasheet Configuration Tables

The raw datasheet JSON is often messy. Normalize it into clean part-number rows.

### Input

```json
{
  "tables": [
    {
      "idx": 2,
      "rows": [
        ["Model", "D1", "B", "M", "d", "R", "H", "M1", "P", "h", "W", "X", "Y"],
        ["SL-SSCDN", "10", "35", "15", "M5", "5.5", "4.5", "15", "M4", "10", "7", "1.5", "11.5", "6"]
      ]
    }
  ]
}
```

### Output

```json
{
  "catalog_id": "misumi_clean_pack_two_piece_d_cut_shaft_collar",
  "units": "mm",
  "configurations": [
    {
      "part_number": "SL-SSCDN10",
      "variant": "standard_separate",
      "D": 10,
      "D1": 35,
      "B": 15,
      "M": "M5",
      "d": 5.5,
      "R": 4.5,
      "H": 15,
      "M1": "M4",
      "P": 10,
      "h": 7,
      "W": 1.5,
      "X": 11.5,
      "Y": 6
    }
  ]
}
```

This stage should eventually become deterministic Python. Use an LLM only to bootstrap difficult table parsing.

---

## Step 4 — Resolve a Selected Part Number

Take:

```text
feature_template.json
+ normalized_configurations.json
+ selected_part_number
```

Return:

```text
resolved_cad_spec.json
```

Example:

```json
{
  "part_number": "SL-SSCDN10",
  "template_id": "misumi_clean_pack_two_piece_d_cut_shaft_collar",
  "units": "mm",
  "resolved_parameters": {
    "D": 10,
    "D1": 35,
    "B": 15,
    "M": "M5",
    "d": 5.5
  },
  "features": []
}
```

This step should mostly be deterministic:

```python
def resolve_parameter_value(value, row):
    if isinstance(value, str) and value.startswith("$"):
        key = value[1:]
        return row.get(key)
    return value
```

Do not let the model reinterpret the drawing at this stage.

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
