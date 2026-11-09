"""Модульные тесты моделей патчей."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from semantic_knowledge.domain.models import NodeType, ValueKind
from semantic_knowledge.domain.patches import (
    AddEdge,
    AddNode,
    DeleteEdge,
    DeleteNode,
    GraphPatch,
    UpdateEdge,
    UpdateNode,
)


def test_add_node_builds_domain_node() -> None:
    operation = AddNode(id="person_1", label="Анна Смирнова", type="person")

    assert operation.op == "add_node"
    assert operation.type is NodeType.PERSON
    assert operation.to_node().name == "Анна Смирнова"
    assert operation.to_node().type is NodeType.PERSON


def test_add_value_node_requires_value_fields() -> None:
    operation = AddNode(
        id="value_1",
        label="2018 год",
        type=NodeType.VALUE,
        value_kind="year",
        value="2018",
    )

    assert operation.value_kind is ValueKind.YEAR
    assert operation.to_node().value == "2018"

    with pytest.raises(ValueError, match="value_kind и value"):
        AddNode(id="value_2", label="Год", type=NodeType.VALUE)


def test_add_regular_node_forbids_value_fields() -> None:
    with pytest.raises(ValueError, match="только для узла"):
        AddNode(
            id="person_1",
            label="Анна",
            type=NodeType.PERSON,
            value_kind=ValueKind.TEXT,
            value="активен",
        )


def test_update_node_accepts_known_attributes() -> None:
    operation = UpdateNode(
        id="person_1",
        attributes={"label": "Анна Петрова", "description": "Новое описание"},
    )

    assert operation.op == "update_node"
    assert operation.attributes["label"] == "Анна Петрова"


@pytest.mark.parametrize(
    "attributes",
    [
        {},
        {"name": "Анна"},
        {"label": ""},
        {"type": "unknown"},
        {"value_kind": "unknown"},
        {"confidence": 0.5},
    ],
)
def test_update_node_rejects_invalid_attributes(attributes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        UpdateNode(id="person_1", attributes=attributes)  # type: ignore[arg-type]


def test_update_node_attributes_are_immutable() -> None:
    operation = UpdateNode(id="person_1", attributes={"label": "Анна"})

    with pytest.raises(TypeError):
        operation.attributes["label"] = "Другое"  # type: ignore[index]


def test_delete_node_validates_id() -> None:
    assert DeleteNode(id="person_1").op == "delete_node"

    with pytest.raises(ValueError, match="id узла"):
        DeleteNode(id="1person")


def test_add_edge_builds_domain_edge() -> None:
    operation = AddEdge(
        id="edge_1",
        source="person_1",
        target="org_sever",
        label="works_at",
        evidence="Анна работает в компании",
        source_ref="doc_1",
        confidence=0.8,
    )

    edge = operation.to_edge()

    assert operation.op == "add_edge"
    assert edge.relation == "works_at"
    assert edge.confidence == pytest.approx(0.8)


@pytest.mark.parametrize("relation", ["works at", "Works_at", ""])
def test_add_edge_rejects_invalid_relation(relation: str) -> None:
    with pytest.raises(ValueError, match="relation"):
        AddEdge(id="edge_1", source="person_1", target="org_sever", label=relation)


def test_update_edge_accepts_known_attributes() -> None:
    operation = UpdateEdge(
        id="edge_1",
        attributes={"label": "employed_by", "confidence": 0.7},
    )

    assert operation.op == "update_edge"
    assert operation.attributes["confidence"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    "attributes",
    [
        {},
        {"source": "person_2"},
        {"type": "person"},
        {"label": ""},
        {"confidence": "0.5"},
        {"confidence": 1.5},
    ],
)
def test_update_edge_rejects_invalid_attributes(attributes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        UpdateEdge(id="edge_1", attributes=attributes)  # type: ignore[arg-type]


def test_delete_edge_validates_id() -> None:
    assert DeleteEdge(id="edge_1").op == "delete_edge"

    with pytest.raises(ValueError, match="id связи"):
        DeleteEdge(id="edge 1")


def test_graph_patch_accepts_operations() -> None:
    patch = GraphPatch(
        base_revision=0,
        operations=[
            AddNode(id="person_1", label="Анна", type="person"),
            AddEdge(id="edge_1", source="person_1", target="person_1", label="knows"),
        ],
    )

    assert isinstance(patch.operations, tuple)
    assert patch.operations[0].op == "add_node"
    assert patch.operations[1].op == "add_edge"


def test_graph_patch_rejects_empty_operations() -> None:
    with pytest.raises(ValueError, match="хотя бы одну операцию"):
        GraphPatch(base_revision=0, operations=[])


def test_graph_patch_rejects_unknown_operation() -> None:
    with pytest.raises(TypeError, match="неизвестная операция"):
        GraphPatch(base_revision=0, operations=["add_node"])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("revision", "error"),
    [(-1, ValueError), (True, TypeError), ("0", TypeError)],
)
def test_graph_patch_rejects_invalid_revision(
    revision: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match="base_revision"):
        GraphPatch(
            base_revision=revision,  # type: ignore[arg-type]
            operations=[DeleteNode(id="person_1")],
        )


def test_graph_patch_is_frozen() -> None:
    patch = GraphPatch(base_revision=0, operations=[DeleteNode(id="person_1")])

    with pytest.raises(FrozenInstanceError):
        patch.base_revision = 1  # type: ignore[misc]
