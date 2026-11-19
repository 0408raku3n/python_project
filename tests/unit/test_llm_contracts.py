"""Executable contract checks for the four version-1 LLM responses.

Run with: python -m pytest tests/unit/test_llm_contracts.py
"""

import copy
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator
import pydot
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "docs" / "llm" / "schemas"
EXAMPLE_DIR = ROOT / "examples" / "llm"
CONTRACTS = (
    ("intent-decision.schema.json", "intent-classification.json"),
    ("extraction-result.schema.json", "knowledge-extraction.json"),
    ("answer-result.schema.json", "answer-generation.json"),
    ("change-explanation.schema.json", "change-explanation.json"),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def schema_validator(filename: str) -> Draft202012Validator:
    schema = read_json(SCHEMA_DIR / filename)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def quote_unwrap(value: str) -> str:
    return value[1:-1] if value.startswith('"') and value.endswith('"') else value


def source_text_from_spec() -> str:
    spec = (ROOT / "docs" / "semantic-model.md").read_text(encoding="utf-8")
    example_section = spec.split("## 9. Пример построения модели", maxsplit=1)[1]
    return next(line[2:] for line in example_section.splitlines() if line.startswith("> "))


@pytest.mark.parametrize("schema_name,example_name", CONTRACTS)
def test_example_validates_against_draft_2020_12(schema_name: str, example_name: str) -> None:
    schema = read_json(SCHEMA_DIR / schema_name)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    schema_validator(schema_name).validate(read_json(EXAMPLE_DIR / example_name))


@pytest.mark.parametrize("schema_name,_example_name", CONTRACTS)
def test_no_schema_object_accepts_unexpected_properties(
    schema_name: str, _example_name: str
) -> None:
    schema = read_json(SCHEMA_DIR / schema_name)

    def visit(definition: object) -> None:
        if isinstance(definition, dict):
            if definition.get("type") == "object":
                assert definition.get("additionalProperties") is False
            for value in definition.values():
                visit(value)
        elif isinstance(definition, list):
            for value in definition:
                visit(value)

    visit(schema)


@pytest.mark.parametrize(
    "schema_name,example_name,mutate",
    [
        ("intent-decision.schema.json", "intent-classification.json", lambda d: d.update(category="OTHER")),
        ("intent-decision.schema.json", "intent-classification.json", lambda d: d.update(confidence=-0.01)),
        ("intent-decision.schema.json", "intent-classification.json", lambda d: d.update(confidence=1.01)),
        ("intent-decision.schema.json", "intent-classification.json", lambda d: d.update(confidence="0.5")),
        ("intent-decision.schema.json", "intent-classification.json", lambda d: d.update(confidence=True)),
        ("intent-decision.schema.json", "intent-classification.json", lambda d: d.update(reason=None)),
        ("intent-decision.schema.json", "intent-classification.json", lambda d: d.update(extra="no")),
        ("intent-decision.schema.json", "intent-classification.json", lambda d: d.update(needs_clarification=True)),
        ("intent-decision.schema.json", "intent-classification.json", lambda d: d.update(clarification_question="Not allowed")),
        ("intent-decision.schema.json", "intent-classification.json", lambda d: d.update(category="CLARIFICATION_REQUIRED", needs_clarification=True)),
        ("intent-decision.schema.json", "intent-classification.json", lambda d: d.update(category="CLARIFICATION_REQUIRED", clarification_question="Question?")),
        ("extraction-result.schema.json", "knowledge-extraction.json", lambda d: d.pop("warnings")),
        ("extraction-result.schema.json", "knowledge-extraction.json", lambda d: d.update(source_ref="assigned_by_python")),
        ("extraction-result.schema.json", "knowledge-extraction.json", lambda d: d["entities"][0].update(type="mystery")),
        ("extraction-result.schema.json", "knowledge-extraction.json", lambda d: d["entities"][0].update(value_kind="year")),
        ("extraction-result.schema.json", "knowledge-extraction.json", lambda d: d["entities"][3].pop("value")),
        ("extraction-result.schema.json", "knowledge-extraction.json", lambda d: d["entities"][3].update(value_kind="money")),
        ("extraction-result.schema.json", "knowledge-extraction.json", lambda d: d["entities"][0].update(source_ref="forbidden")),
        ("extraction-result.schema.json", "knowledge-extraction.json", lambda d: d["relations"][0].update(relation="Has Event")),
        ("extraction-result.schema.json", "knowledge-extraction.json", lambda d: d["relations"][0].update(confidence=1.2)),
        ("extraction-result.schema.json", "knowledge-extraction.json", lambda d: d["relations"][0].update(confidence="0.98")),
        ("extraction-result.schema.json", "knowledge-extraction.json", lambda d: d["relations"][0].update(source_ref="forbidden")),
        ("answer-result.schema.json", "answer-generation.json", lambda d: d.update(used_node_ids=[], used_edge_ids=[])),
        ("answer-result.schema.json", "answer-generation.json", lambda d: d.update(status="INSUFFICIENT_CONTEXT")),
        ("answer-result.schema.json", "answer-generation.json", lambda d: d.update(status="OTHER")),
        ("answer-result.schema.json", "answer-generation.json", lambda d: d.update(answer=None)),
        ("answer-result.schema.json", "answer-generation.json", lambda d: d.update(used_node_ids=["x", "x"])),
        ("change-explanation.schema.json", "change-explanation.json", lambda d: d.update(items=[])),
        ("change-explanation.schema.json", "change-explanation.json", lambda d: d.update(status="NO_CHANGES")),
        ("change-explanation.schema.json", "change-explanation.json", lambda d: d["items"][0].update(kind="renamed")),
        ("change-explanation.schema.json", "change-explanation.json", lambda d: d["items"][0].pop("text")),
        ("change-explanation.schema.json", "change-explanation.json", lambda d: d["items"][0].update(object_id="not_in_response")),
    ],
)
def test_invalid_contract_data_is_rejected(schema_name: str, example_name: str, mutate) -> None:
    instance = copy.deepcopy(read_json(EXAMPLE_DIR / example_name))
    mutate(instance)
    assert not schema_validator(schema_name).is_valid(instance)


def test_conditional_valid_branches() -> None:
    intent = read_json(EXAMPLE_DIR / "intent-classification.json")
    intent.update(category="CLARIFICATION_REQUIRED", needs_clarification=True)
    intent["clarification_question"] = "Какое действие требуется?"
    schema_validator("intent-decision.schema.json").validate(intent)

    extraction = {"contract_version": "1", "entities": [], "relations": [], "warnings": []}
    schema_validator("extraction-result.schema.json").validate(extraction)

    answer = {
        "contract_version": "1",
        "status": "INSUFFICIENT_CONTEXT",
        "answer": "В модели недостаточно информации для ответа.",
        "used_node_ids": [],
        "used_edge_ids": [],
    }
    schema_validator("answer-result.schema.json").validate(answer)

    explanation = {
        "contract_version": "1",
        "status": "NO_CHANGES",
        "message": "Изменений нет.",
        "items": [],
    }
    schema_validator("change-explanation.schema.json").validate(explanation)


def test_extraction_matches_source_and_reference_dot() -> None:
    example = read_json(EXAMPLE_DIR / "knowledge-extraction.json")
    source_text = source_text_from_spec()
    entities = example["entities"]
    relations = example["relations"]
    assert (len(entities), len(relations)) == (8, 7)
    assert "source_ref" not in example

    entity_ids = [entity["local_id"] for entity in entities]
    relation_ids = [relation["local_id"] for relation in relations]
    assert len(set(entity_ids)) == len(entity_ids)
    assert len(set(relation_ids)) == len(relation_ids)
    assert all(entity["mention"] in source_text for entity in entities)
    assert all(relation["evidence"] in source_text for relation in relations)
    assert all(
        relation["source_local_id"] in entity_ids
        and relation["target_local_id"] in entity_ids
        for relation in relations
    )

    graph = pydot.graph_from_dot_file(str(ROOT / "examples" / "example-model.dot"))[0]
    dot_nodes = {
        quote_unwrap(node.get_name()): {
            key: quote_unwrap(value) for key, value in node.get_attributes().items()
        }
        for node in graph.get_nodes()
        if node.get_name() != "graph"
    }
    assert len(dot_nodes) == 8
    assert len(graph.get_edges()) == 7

    local_to_graph_id = {}
    for entity in entities:
        matching_nodes = [
            graph_id
            for graph_id, attrs in dot_nodes.items()
            if attrs["label"] == entity["name"] and attrs["type"] == entity["type"]
        ]
        assert len(matching_nodes) == 1, entity
        graph_id = matching_nodes[0]
        local_to_graph_id[entity["local_id"]] = graph_id
        if entity["type"] == "value":
            assert dot_nodes[graph_id]["value_kind"] == entity["value_kind"]
            assert dot_nodes[graph_id]["value"] == entity["value"]

    expected_edges = {
        (
            quote_unwrap(edge.get_source()),
            quote_unwrap(edge.get_destination()),
            quote_unwrap(edge.get_attributes()["label"]),
            quote_unwrap(edge.get_attributes()["evidence"]),
        )
        for edge in graph.get_edges()
    }
    extracted_edges = {
        (
            local_to_graph_id[relation["source_local_id"]],
            local_to_graph_id[relation["target_local_id"]],
            relation["relation"],
            relation["evidence"],
        )
        for relation in relations
    }
    assert extracted_edges == expected_edges


def test_answer_references_only_graph_elements() -> None:
    answer = read_json(EXAMPLE_DIR / "answer-generation.json")
    graph = pydot.graph_from_dot_file(str(ROOT / "examples" / "example-model.dot"))[0]
    node_ids = {quote_unwrap(node.get_name()) for node in graph.get_nodes()}
    edge_ids = {quote_unwrap(edge.get_attributes()["id"]) for edge in graph.get_edges()}
    assert set(answer["used_node_ids"]) <= node_ids
    assert set(answer["used_edge_ids"]) <= edge_ids


def test_change_explanation_covers_documented_applied_changes() -> None:
    contracts_doc = (ROOT / "docs" / "llm-contracts.md").read_text(encoding="utf-8")
    section = contracts_doc.split("### 7.1. Запрос", maxsplit=1)[1]
    request = json.loads(re.search(r"```json\s*(\{.*?\})\s*```", section, re.DOTALL).group(1))
    explanation = read_json(EXAMPLE_DIR / "change-explanation.json")
    expected = {change["change_id"]: change["kind"] for change in request["applied_changes"]}
    actual = {item["change_id"]: item["kind"] for item in explanation["items"]}
    assert len(actual) == len(explanation["items"])
    assert actual == expected
