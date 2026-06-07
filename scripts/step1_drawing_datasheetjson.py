import argparse
import json
import base64
import mimetypes
import os
from datetime import datetime
from pathlib import Path
import sys
import urllib.error
import urllib.request
import uuid

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.workflow_state import DEFAULT_STATE_DB, completed_component_dirs, connect_state_db, upsert_batch_job

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_BATCH_OUTPUT_DIR = "output"
RESPONSES_BATCH_ENDPOINT = "/v1/responses"
SUPPORTED_BATCH_ENDPOINTS = {
    RESPONSES_BATCH_ENDPOINT,
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/completions",
    "/v1/moderations",
}


DEVELOPER_MESSAGE = """
You are a senior mechanical CAD automation engineer specializing in converting 2D catalog drawings and datasheet tables into reusable parametric CAD feature templates.

You are given:
1. A 2D engineering drawing image for a configurable catalog part family.
2. A JSON datasheet table with dimension symbols, values, tolerances, notes, materials, variants, and part-number configurations.

Your job is to create a reusable parametric engineering feature template, not final CAD code.

Core rules:
- Treat the drawing as a reusable template for a part family, not as one fixed part.
- Treat the JSON datasheet as the source of dimensional truth.
- Use the drawing to infer feature layout, view relationships, orientation, operations, and missing geometry intent.
- Do not generate final CAD code.
- Do not silently invent dimensions.
- Use symbolic parameters such as $D, $D1, $B, $W, $H, $P, $X, $Y, $T.
- Preserve ambiguity explicitly, but always provide the best proposed geometric interpretation.
- Separate geometric features from catalog/process metadata.

For every geometric feature, provide enough construction information that a CAD code generator can build it without reading the drawing again.

Each geometric feature should identify:
- feature type
- operation: base, add, subtract, split, transform, metadata
- modeling primitive: extrude, revolve, cut_extrude, cut_revolve, hole, pocket, split_body, fillet, chamfer, thread_metadata
- sketch plane or reference plane
- profile definition when applicable
- axis and direction
- start side or face
- depth or end condition
- symbolic driving parameters
- target body or component
- dependencies
- confidence score
- needs_review flag

Assembly and multi-body rules:
- If the part is an assembly or multi-body part, explicitly choose one modeling mode:
  1. simplified_single_body
  2. multi_body_part
  3. assembly_with_separate_components
- If geometry is split into multiple bodies, identify the resulting bodies and which features apply to each body.
- Do not collapse a meaningful split, clamp interface, or mating interface into metadata only unless the schema requires a simplified model.

Hole-axis rules:
- Before assigning hole axes, reason from the selected coordinate system.
- Holes visible as circles in the front view usually have axes normal to the front view.
- Holes visible as circles in the side view usually have axes normal to the side view.
- Bottom mounting holes usually cut through the mounting face normal unless the drawing clearly shows otherwise.
- If the axis is inferred rather than explicit, set needs_review=true.

Thread rules:
- Represent threads as metadata plus pilot/clearance/tap-drill holes unless true thread geometry is explicitly requested.
- If a screw feature includes clearance, counterbore, and tapped mate geometry, split those into separate features when possible.

GD&T and tolerance rules:
- Represent GD&T as metadata and validation requirements, not solid geometry.
- Preserve size tolerances and fit callouts as metadata linked to the relevant feature.
- Include validation checks for dimensions that should be measurable from generated CAD.

Ambiguity rules:
- Do not use vague phrases like “drawing-dependent” by themselves.
- If a feature is ambiguous, still provide a proposed interpretation, set needs_review=true, give a confidence score, and explain the unresolved point.
- If a required dimension is missing, use null for that value and list it in missing_information.

Output rules:
- Output only valid JSON matching the requested schema.
- Do not include markdown, explanations, code fences, or comments outside the JSON.
"""


USER_PROMPT = """
Attached are:
1. The 2D engineering drawing image.
2. A datasheet JSON object.

Create a reusable parametric engineering feature template for this catalog part family.

Focus on CAD-construction readiness:
- identify the modeling mode
- define base body construction
- define all subtractive/additive features
- define hole/thread/counterbore features separately where possible
- define split/multi-body behavior if applicable
- preserve all symbolic dimensions from the datasheet
- mark ambiguous features with needs_review=true and confidence score
- keep material, cleaning, packaging, and surface treatment as metadata, not geometry

A good output should be usable by a second model or deterministic code generator to produce CadQuery/build123d code without looking at the drawing again.

Before creating the feature graph, use the datasheet JSON to identify all dimension symbols that appear in the datasheet tables. Every geometric symbol used in the drawing should appear in the parameters array unless it is truly absent from the datasheet.

Return only the structured JSON required by the schema.
"""


ENGINEERING_FEATURE_TEMPLATE_FORMAT = {
    "type": "json_schema",
    "name": "engineering_feature_template",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "template_id",
            "part_family",
            "units",
            "modeling_mode",
            "coordinate_system",
            "components",
            "parameters",
            "feature_graph",
            "catalog_metadata",
            "assumptions",
            "missing_information",
            "validation_requirements"
        ],
        "properties": {
            "schema_version": {"type": "string"},
            "template_id": {"type": "string"},
            "part_family": {"type": "string"},
            "units": {"type": "string", "enum": ["mm", "inch"]},
            "modeling_mode": {
                "type": "string",
                "enum": [
                    "simplified_single_body",
                    "multi_body_part",
                    "assembly_with_separate_components",
                    "unknown"
                ]
            },
            "coordinate_system": {
                "type": "object",
                "additionalProperties": False,
                "required": ["origin", "x_axis", "y_axis", "z_axis"],
                "properties": {
                    "origin": {"type": "string"},
                    "x_axis": {"type": "string"},
                    "y_axis": {"type": "string"},
                    "z_axis": {"type": "string"}
                }
            },
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "id",
                        "name",
                        "component_type",
                        "description",
                        "parent_component",
                        "created_by_feature"
                    ],
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "component_type": {
                            "type": "string",
                            "enum": [
                                "single_body",
                                "solid_body",
                                "component",
                                "assembly",
                                "metadata"
                            ]
                        },
                        "description": {"type": "string"},
                        "parent_component": {"type": ["string", "null"]},
                        "created_by_feature": {"type": ["string", "null"]}
                    }
                }
            },
            "parameters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "name",
                        "symbol",
                        "description",
                        "parameter_type",
                        "category",
                        "source",
                        "default_value",
                        "units",
                        "tolerance",
                        "notes"
                    ],
                    "properties": {
                        "name": {"type": "string"},
                        "symbol": {"type": "string"},
                        "description": {"type": "string"},
                        "parameter_type": {
                            "type": "string",
                            "enum": [
                                "length",
                                "diameter",
                                "radius",
                                "angle",
                                "thread",
                                "tolerance",
                                "enum",
                                "integer",
                                "boolean",
                                "material",
                                "surface_finish",
                                "unknown"
                            ]
                        },
                        "category": {
                            "type": "string",
                            "enum": [
                                "geometry",
                                "manufacturing",
                                "catalog",
                                "material",
                                "surface_finish",
                                "packaging",
                                "validation",
                                "unknown"
                            ]
                        },
                        "source": {
                            "type": "string",
                            "enum": [
                                "datasheet",
                                "drawing",
                                "inferred",
                                "constant",
                                "missing"
                            ]
                        },
                        "default_value": {
                            "type": ["number", "string", "boolean", "null"]
                        },
                        "units": {"type": ["string", "null"]},
                        "tolerance": {"type": ["string", "null"]},
                        "notes": {"type": "string"}
                    }
                }
            },
            "feature_graph": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "id",
                        "feature_type",
                        "operation",
                        "modeling_primitive",
                        "subtype",
                        "target_bodies",
                        "output_bodies",
                        "depends_on",
                        "parameters",
                        "construction",
                        "position",
                        "axis",
                        "side",
                        "pattern",
                        "confidence",
                        "needs_review",
                        "review_reason",
                        "notes"
                    ],
                    "properties": {
                        "id": {"type": "string"},
                        "feature_type": {
                            "type": "string",
                            "enum": [
                                "base_body",
                                "hole",
                                "pocket",
                                "boss",
                                "slot",
                                "split",
                                "pattern",
                                "fillet",
                                "chamfer",
                                "thread",
                                "datum",
                                "gdnt",
                                "metadata"
                            ]
                        },
                        "operation": {
                            "type": "string",
                            "enum": [
                                "base",
                                "add",
                                "subtract",
                                "split",
                                "metadata",
                                "transform"
                            ]
                        },
                        "modeling_primitive": {
                            "type": "string",
                            "enum": [
                                "extrude",
                                "revolve",
                                "sweep",
                                "loft",
                                "cut_extrude",
                                "cut_revolve",
                                "hole",
                                "counterbore_hole",
                                "countersink_hole",
                                "pocket",
                                "split_body",
                                "fillet",
                                "chamfer",
                                "thread_metadata",
                                "pattern",
                                "metadata",
                                "unknown"
                            ]
                        },
                        "subtype": {"type": "string"},
                        "target_bodies": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "output_bodies": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "parameters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["name", "value", "role"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "value": {
                                        "type": [
                                            "string",
                                            "number",
                                            "boolean",
                                            "null"
                                        ]
                                    },
                                    "role": {
                                        "type": "string",
                                        "enum": [
                                            "driving_dimension",
                                            "derived_dimension",
                                            "reference_dimension",
                                            "tolerance",
                                            "metadata",
                                            "unknown"
                                        ]
                                    }
                                }
                            }
                        },
                        "construction": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "sketch_plane",
                                "reference_plane",
                                "profile_type",
                                "profile_description",
                                "start_condition",
                                "end_condition",
                                "depth",
                                "boolean_scope",
                                "construction_notes"
                            ],
                            "properties": {
                                "sketch_plane": {"type": ["string", "null"]},
                                "reference_plane": {"type": ["string", "null"]},
                                "profile_type": {
                                    "type": ["string", "null"],
                                    "enum": [
                                        "circle",
                                        "rectangle",
                                        "rounded_rectangle",
                                        "obround",
                                        "d_cut_circle",
                                        "annulus",
                                        "slot",
                                        "custom_2d_profile",
                                        "line_or_plane",
                                        "none",
                                        "unknown",
                                        None
                                    ]
                                },
                                "profile_description": {"type": "string"},
                                "start_condition": {
                                    "type": ["string", "null"],
                                    "enum": [
                                        "from_sketch_plane",
                                        "from_face",
                                        "midplane",
                                        "offset",
                                        "through_all",
                                        "none",
                                        "unknown",
                                        None
                                    ]
                                },
                                "end_condition": {
                                    "type": ["string", "null"],
                                    "enum": [
                                        "blind",
                                        "through_all",
                                        "up_to_face",
                                        "up_to_body",
                                        "symmetric",
                                        "none",
                                        "unknown",
                                        None
                                    ]
                                },
                                "depth": {
                                    "type": ["string", "number", "null"]
                                },
                                "boolean_scope": {"type": ["string", "null"]},
                                "construction_notes": {"type": "string"}
                            }
                        },
                        "position": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["x", "y", "z"],
                            "properties": {
                                "x": {"type": ["string", "number", "null"]},
                                "y": {"type": ["string", "number", "null"]},
                                "z": {"type": ["string", "number", "null"]}
                            }
                        },
                        "axis": {
                            "type": ["string", "null"],
                            "enum": [
                                "X",
                                "Y",
                                "Z",
                                "-X",
                                "-Y",
                                "-Z",
                                "radial",
                                "normal_to_face",
                                None
                            ]
                        },
                        "side": {
                            "type": ["string", "null"],
                            "enum": [
                                "front",
                                "back",
                                "left",
                                "right",
                                "top",
                                "bottom",
                                "both",
                                "component_specific",
                                None
                            ]
                        },
                        "pattern": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "pattern_type",
                                "count",
                                "spacing",
                                "angle_step",
                                "direction",
                                "base_feature"
                            ],
                            "properties": {
                                "pattern_type": {
                                    "type": ["string", "null"],
                                    "enum": [
                                        "none",
                                        "symmetric_pair",
                                        "linear",
                                        "rectangular",
                                        "circular",
                                        "mirror",
                                        None
                                    ]
                                },
                                "count": {"type": ["integer", "null"]},
                                "spacing": {
                                    "type": ["string", "number", "null"]
                                },
                                "angle_step": {
                                    "type": ["string", "number", "null"]
                                },
                                "direction": {
                                    "type": ["string", "null"],
                                    "enum": [
                                        "X",
                                        "Y",
                                        "Z",
                                        "radial",
                                        "angular",
                                        None
                                    ]
                                },
                                "base_feature": {"type": ["string", "null"]}
                            }
                        },
                        "confidence": {"type": "number"},
                        "needs_review": {"type": "boolean"},
                        "review_reason": {"type": "string"},
                        "notes": {"type": "string"}
                    }
                }
            },
            "catalog_metadata": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "material_options",
                    "surface_treatment_options",
                    "cleaning_options",
                    "packaging_options",
                    "accessories",
                    "notes"
                ],
                "properties": {
                    "material_options": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "surface_treatment_options": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "cleaning_options": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "packaging_options": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "accessories": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "notes": {"type": "string"}
                }
            },
            "assumptions": {
                "type": "array",
                "items": {"type": "string"}
            },
            "missing_information": {
                "type": "array",
                "items": {"type": "string"}
            },
            "validation_requirements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "check",
                        "target_feature",
                        "target_body",
                        "expected",
                        "tolerance",
                        "validation_method"
                    ],
                    "properties": {
                        "check": {"type": "string"},
                        "target_feature": {"type": "string"},
                        "target_body": {"type": ["string", "null"]},
                        "expected": {"type": "string"},
                        "tolerance": {
                            "type": ["string", "number", "null"]
                        },
                        "validation_method": {
                            "type": "string",
                            "enum": [
                                "bounding_box_measurement",
                                "diameter_measurement",
                                "distance_measurement",
                                "hole_count",
                                "axis_alignment",
                                "body_count",
                                "metadata_match",
                                "manual_review",
                                "unknown"
                            ]
                        }
                    }
                }
            }
        }
    }
}


def upload_drawing_for_vision(path: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    uploaded = client.files.create(
        file=open(path, "rb"),
        purpose="vision"
    )
    return uploaded.id


def extract_output_text(response) -> str:
    """
    Robust extraction for Responses API text output.
    """
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)

    return "\n".join(chunks)


def create_feature_template(
    drawing_path: str,
    specs_json_path: str,
    model: str = DEFAULT_MODEL
) -> dict:
    from openai import OpenAI

    client = OpenAI()
    drawing_file_id = upload_drawing_for_vision(drawing_path)

    specs_json = json.loads(Path(specs_json_path).read_text(encoding="utf-8"))
    specs_json_text = json.dumps(specs_json, ensure_ascii=False)

    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        prompt_cache_key="cad_feature_template_extraction_v1",
        prompt_cache_retention="24h",
        input=[
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": DEVELOPER_MESSAGE
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": USER_PROMPT
                    },
                    {
                        "type": "input_image",
                        "file_id": drawing_file_id
                    },
                    {
                        "type": "input_text",
                        "text": "Datasheet JSON:\n" + specs_json_text
                    }
                ]
            }
        ],
        text={
            "format": ENGINEERING_FEATURE_TEMPLATE_FORMAT
        }
    )

    output_text = extract_output_text(response)
    parsed = json.loads(output_text)

    cached_tokens = (
        getattr(response, "usage", None)
        and getattr(response.usage, "prompt_tokens_details", None)
        and getattr(response.usage.prompt_tokens_details, "cached_tokens", None)
    )

    print("response_id:", response.id)
    print("cached_tokens:", cached_tokens)

    return parsed


def image_data_url(path: str | Path) -> str:
    path = Path(path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def find_specs_json(component_dir: str | Path) -> Path:
    component_dir = Path(component_dir)
    json_dir = component_dir / "json"
    candidates = [json_dir / "specs_after.json", json_dir / "specs.json"]
    usable = [candidate for candidate in candidates if candidate.exists() and specs_json_has_content(candidate)]
    if usable:
        return usable[0]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No specs_after.json or specs.json found in {json_dir}")


def specs_json_has_content(path: str | Path) -> bool:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    for key in ("tables", "specImages", "filters", "configFields"):
        value = data.get(key)
        if isinstance(value, list) and value:
            return True
        if isinstance(value, dict) and value:
            return True
    return False


def find_main_drawing(component_dir: str | Path) -> Path:
    component_dir = Path(component_dir)
    main_dir = component_dir / "main_engineering_drawing"
    images = sorted(
        path
        for path in main_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".gif", ".png", ".jpg", ".jpeg", ".webp"}
    ) if main_dir.exists() else []
    if not images:
        raise FileNotFoundError(
            f"No main engineering drawing found in {main_dir}. "
            "Run scripts/step0_main_engineering_drawing.py before Step 1."
        )
    return images[0]


def build_responses_body(
    drawing_path: str | Path,
    specs_json_path: str | Path,
    model: str = DEFAULT_MODEL,
) -> dict:
    specs_json = json.loads(Path(specs_json_path).read_text(encoding="utf-8"))
    specs_json_text = json.dumps(specs_json, ensure_ascii=False)

    return {
        "model": model,
        "reasoning": {"effort": "medium"},
        "prompt_cache_key": "cad_feature_template_extraction_v1",
        "prompt_cache_retention": "24h",
        "input": [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": DEVELOPER_MESSAGE,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": USER_PROMPT,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url(drawing_path),
                    },
                    {
                        "type": "input_text",
                        "text": "Datasheet JSON:\n" + specs_json_text,
                    },
                ],
            },
        ],
        "text": {
            "format": ENGINEERING_FEATURE_TEMPLATE_FORMAT,
        },
    }


def build_batch_jsonl_record(
    component_dir: str | Path,
    model: str = DEFAULT_MODEL,
) -> dict:
    component_dir = Path(component_dir)
    drawing_path = find_main_drawing(component_dir)
    specs_json_path = find_specs_json(component_dir)

    body = build_responses_body(
        drawing_path=drawing_path,
        specs_json_path=specs_json_path,
        model=model,
    )

    body.setdefault("metadata", {})
    body["metadata"].update(
        {
            "component_id": component_dir.name,
            "drawing_path": str(drawing_path),
            "specs_json_path": str(specs_json_path),
        }
    )

    return {
        "custom_id": component_dir.name,
        "method": "POST",
        "url": RESPONSES_BATCH_ENDPOINT,
        "body": body,
    }


def create_batch_jsonl(
    component_dirs: list[str | Path],
    output_dir: str | Path = DEFAULT_BATCH_OUTPUT_DIR,
    model: str = DEFAULT_MODEL,
    filename: str | None = None,
) -> Path:
    if not component_dirs:
        raise ValueError("At least one component directory is required")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"step1_batch_{timestamp}.jsonl"

    output_path = output_dir / filename
    with output_path.open("w", encoding="utf-8") as handle:
        for component_dir in component_dirs:
            record = build_batch_jsonl_record(component_dir=component_dir, model=model)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def iter_jsonl_records(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number}: JSONL record must be an object")
            yield line_number, record


def validate_batch_jsonl(path: str | Path, endpoint: str | None = None) -> tuple[str, int]:
    detected_endpoint: str | None = None
    count = 0

    for line_number, record in iter_jsonl_records(path):
        count += 1
        method = record.get("method")
        url = record.get("url")
        if method != "POST":
            raise ValueError(f"Line {line_number}: method must be POST, got {method!r}")
        if url not in SUPPORTED_BATCH_ENDPOINTS:
            raise ValueError(
                f"Line {line_number}: unsupported batch url {url!r}; "
                f"expected one of {sorted(SUPPORTED_BATCH_ENDPOINTS)}"
            )
        if detected_endpoint is None:
            detected_endpoint = url
        elif url != detected_endpoint:
            raise ValueError(
                f"Line {line_number}: url {url!r} does not match first line url {detected_endpoint!r}. "
                "A batch file can only target one endpoint."
            )
        if endpoint is not None and url != endpoint:
            raise ValueError(
                f"Line {line_number}: url {url!r} does not match requested batch endpoint {endpoint!r}. "
                f"Create the batch with endpoint={url!r}, or regenerate the JSONL for {endpoint!r}."
            )

    if count == 0 or detected_endpoint is None:
        raise ValueError(f"No JSONL records found in {path}")

    return detected_endpoint, count


def serialize_openai_object(value) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return json.loads(json.dumps(value, default=lambda obj: getattr(obj, "__dict__", str(obj))))


def openai_json_request(path: str, *, api_key: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Authorization": f"Bearer {api_key}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"https://api.openai.com/v1{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc


def openai_bytes_request(path: str, *, api_key: str) -> bytes:
    request = urllib.request.Request(
        f"https://api.openai.com/v1{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc


def upload_batch_file(path: str | Path, *, api_key: str) -> dict:
    path = Path(path)
    boundary = "----cadworkflow" + uuid.uuid4().hex
    parts: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        parts.append(
            (
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n"
            ).encode("utf-8")
        )

    def add_file(name: str, file_path: Path) -> None:
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        parts.append(
            (
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{file_path.name}\"\r\n"
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
            + file_path.read_bytes()
            + b"\r\n"
        )

    add_field("purpose", "batch")
    add_file("file", path)
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        "https://api.openai.com/v1/files",
        data=b"".join(parts),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI file upload error {exc.code}: {body}") from exc


def submit_batch_jsonl(
    jsonl_path: str | Path,
    *,
    endpoint: str | None = None,
    completion_window: str = "24h",
) -> dict:
    detected_endpoint, _ = validate_batch_jsonl(jsonl_path, endpoint=endpoint)
    batch_endpoint = endpoint or detected_endpoint
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    uploaded = upload_batch_file(jsonl_path, api_key=api_key)
    return openai_json_request(
        "/batches",
        api_key=api_key,
        method="POST",
        payload={
            "input_file_id": uploaded["id"],
            "endpoint": batch_endpoint,
            "completion_window": completion_window,
        },
    )


def get_batch(batch_id: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return openai_json_request(f"/batches/{batch_id}", api_key=api_key)


def download_file_content(file_id: str, output_path: str | Path) -> Path:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(openai_bytes_request(f"/files/{file_id}/content", api_key=api_key))
    return output_path


def download_batch_outputs(batch_id: str, output_dir: str | Path = DEFAULT_BATCH_OUTPUT_DIR) -> dict:
    batch = get_batch(batch_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {"batch": batch, "output_path": None, "error_path": None}

    output_file_id = batch.get("output_file_id")
    if output_file_id:
        result["output_path"] = str(download_file_content(output_file_id, output_dir / f"{batch_id}_output.jsonl"))

    error_file_id = batch.get("error_file_id")
    if error_file_id:
        result["error_path"] = str(download_file_content(error_file_id, output_dir / f"{batch_id}_error.jsonl"))

    return result


def discover_component_dirs(downloads_dir: str | Path, limit: int | None = None) -> list[Path]:
    downloads_dir = Path(downloads_dir)
    components = sorted(path for path in downloads_dir.iterdir() if path.is_dir())
    if limit is not None:
        components = components[:limit]
    return components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1: drawing + datasheet JSON -> feature-template batch requests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    direct = subparsers.add_parser("direct", help="Run one immediate Responses API extraction.")
    direct.add_argument("--drawing", required=True, type=Path)
    direct.add_argument("--specs", required=True, type=Path)
    direct.add_argument("--model", default=DEFAULT_MODEL)
    direct.add_argument("--output", default="feature_template_output.json", type=Path)

    batch = subparsers.add_parser("batch-jsonl", help="Create OpenAI Batch JSONL for selected components.")
    batch.add_argument("--downloads-dir", default="downloads", type=Path)
    batch.add_argument("--component-dir", action="append", default=[], type=Path)
    batch.add_argument("--limit", type=int)
    batch.add_argument("--output-dir", default=DEFAULT_BATCH_OUTPUT_DIR, type=Path)
    batch.add_argument("--filename")
    batch.add_argument("--model", default=DEFAULT_MODEL)
    batch.add_argument("--state-db", type=Path)
    batch.add_argument("--status", default="done", choices=["all", "pending", "error", "done", "skipped"])

    validate = subparsers.add_parser("validate-batch-jsonl", help="Validate JSONL endpoint consistency before upload.")
    validate.add_argument("jsonl", type=Path)
    validate.add_argument("--endpoint", choices=sorted(SUPPORTED_BATCH_ENDPOINTS))

    submit = subparsers.add_parser("submit-batch", help="Upload JSONL and create an OpenAI Batch with the matching endpoint.")
    submit.add_argument("jsonl", type=Path)
    submit.add_argument("--endpoint", choices=sorted(SUPPORTED_BATCH_ENDPOINTS))
    submit.add_argument("--completion-window", default="24h")
    submit.add_argument("--output", type=Path, help="Optional path to save the created batch object JSON.")
    submit.add_argument("--state-db", default=DEFAULT_STATE_DB, type=Path)

    status = subparsers.add_parser("batch-status", help="Fetch an OpenAI Batch job status.")
    status.add_argument("batch_id")
    status.add_argument("--output", type=Path, help="Optional path to save the batch object JSON.")
    status.add_argument("--state-db", default=DEFAULT_STATE_DB, type=Path)

    download = subparsers.add_parser("download-batch-output", help="Download available output/error files for a batch.")
    download.add_argument("batch_id")
    download.add_argument("--output-dir", default=DEFAULT_BATCH_OUTPUT_DIR, type=Path)
    download.add_argument("--state-db", default=DEFAULT_STATE_DB, type=Path)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "direct":
        result = create_feature_template(
            drawing_path=str(args.drawing),
            specs_json_path=str(args.specs),
            model=args.model,
        )

        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "validate-batch-jsonl":
        endpoint, count = validate_batch_jsonl(args.jsonl, endpoint=args.endpoint)
        print(json.dumps({"ok": True, "path": str(args.jsonl), "endpoint": endpoint, "records": count}))
        return 0

    if args.command == "submit-batch":
        batch = submit_batch_jsonl(
            args.jsonl,
            endpoint=args.endpoint,
            completion_window=args.completion_window,
        )
        with connect_state_db(args.state_db) as conn:
            upsert_batch_job(conn, batch, jsonl_path=args.jsonl)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(batch, indent=2, ensure_ascii=False))
        return 0

    if args.command == "batch-status":
        batch = get_batch(args.batch_id)
        with connect_state_db(args.state_db) as conn:
            upsert_batch_job(conn, batch)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(batch, indent=2, ensure_ascii=False))
        return 0

    if args.command == "download-batch-output":
        result = download_batch_outputs(args.batch_id, args.output_dir)
        batch = result["batch"]
        with connect_state_db(args.state_db) as conn:
            upsert_batch_job(
                conn,
                batch,
                output_path=result.get("output_path"),
                error_path=result.get("error_path"),
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.state_db:
        conn = connect_state_db(args.state_db or DEFAULT_STATE_DB)
        components = completed_component_dirs(conn, status=args.status, limit=args.limit)
        conn.close()
    elif args.component_dir:
        components = args.component_dir
        if args.limit is not None:
            components = components[: args.limit]
    else:
        components = discover_component_dirs(args.downloads_dir, args.limit)

    output_path = create_batch_jsonl(
        component_dirs=components,
        output_dir=args.output_dir,
        model=args.model,
        filename=args.filename,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
