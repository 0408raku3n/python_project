"""Модульные тесты доменных моделей графа."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from semantic_knowledge.domain.models import (
    NodeType,
    SemanticEdge,
    SemanticGraph,
    SemanticNode,
    ValueKind,
)


def make_node(node_id: str = "person_1", name: str = "Анна Смирнова") -> SemanticNode:
    return SemanticNode(id=node_id, name=name, type=NodeType.PERSON)


def make_edge(edge_id: str = "edge_1") -> SemanticEdge:
    return SemanticEdge(
        id=edge_id,
        source_node_id="person_1",
        target_node_id="org_sever",
        relation="works_at",
    )


def test_node_accepts_valid_fields() -> None:
    node = SemanticNode(
        id="person_1",
        name="Анна Смирнова",
        type="person",
        description="Основатель компании",
    )

    assert node.type is NodeType.PERSON
    assert node.value_kind is None
    assert node.value is None


@pytest.mark.parametrize("node_id", ["1person", "_person", "person 1", "", "человек"])
def test_node_rejects_invalid_id(node_id: str) -> None:
    with pytest.raises(ValueError, match="id узла"):
        make_node(node_id=node_id)


@pytest.mark.parametrize("name", ["", "   ", None])
def test_node_rejects_empty_name(name: object) -> None:
    with pytest.raises((TypeError, ValueError), match="name"):
        SemanticNode(id="person_1", name=name, type=NodeType.PERSON)  # type: ignore[arg-type]


def test_node_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="тип узла"):
        SemanticNode(id="unit_1", name="Неизвестно", type="unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value_kind", "value"),
    [
        (ValueKind.TEXT, "активен"),
        (ValueKind.INTEGER, "12"),
        (ValueKind.INTEGER, "-3"),
        (ValueKind.DECIMAL, "12.5"),
        (ValueKind.BOOLEAN, "true"),
        (ValueKind.BOOLEAN, "false"),
        (ValueKind.YEAR, "2018"),
        (ValueKind.DATE, "2018-04-12"),
        (ValueKind.DATETIME, "2018-04-12T10:30:00+03:00"),
        (ValueKind.QUANTITY, "12 kg"),
    ],
)
def test_value_node_accepts_canonical_values(value_kind: ValueKind, value: str) -> None:
    node = SemanticNode(
        id="value_1",
        name="Значение",
        type=NodeType.VALUE,
        value_kind=value_kind,
        value=value,
    )

    assert node.value_kind is value_kind
    assert node.value == value


@pytest.mark.parametrize(
    ("value_kind", "value"),
    [
        (ValueKind.TEXT, ""),
        (ValueKind.INTEGER, "12.0"),
        (ValueKind.INTEGER, "01"),
        (ValueKind.DECIMAL, "1e3"),
        (ValueKind.DECIMAL, ".5"),
        (ValueKind.BOOLEAN, "True"),
        (ValueKind.YEAR, "18"),
        (ValueKind.DATE, "12.04.2018"),
        (ValueKind.DATETIME, "2018-04-12T10:30:00"),
        (ValueKind.QUANTITY, "12"),
    ],
)
def test_value_node_rejects_invalid_values(value_kind: ValueKind, value: str) -> None:
    with pytest.raises(ValueError):
        SemanticNode(
            id="value_1",
            name="Значение",
            type=NodeType.VALUE,
            value_kind=value_kind,
            value=value,
        )


def test_value_node_requires_value_kind_and_value() -> None:
    with pytest.raises(ValueError, match="value_kind и value"):
        SemanticNode(id="value_1", name="2018 год", type=NodeType.VALUE)


def test_regular_node_forbids_value_fields() -> None:
    with pytest.raises(ValueError, match="только для узла"):
        SemanticNode(
            id="person_1",
            name="Анна",
            type=NodeType.PERSON,
            value_kind=ValueKind.TEXT,
            value="активен",
        )


def test_edge_accepts_metadata() -> None:
    edge = SemanticEdge(
        id="edge_1",
        source_node_id="person_1",
        target_node_id="org_sever",
        relation="works_at",
        evidence="Анна работает в компании «Север»",
        source_ref="doc_1",
        confidence=0.9,
    )

    assert edge.confidence == pytest.approx(0.9)
    assert edge.evidence == "Анна работает в компании «Север»"


@pytest.mark.parametrize("relation", ["works at", "Works_at", "1works", "_works", ""])
def test_edge_rejects_invalid_relation(relation: str) -> None:
    with pytest.raises(ValueError, match="relation"):
        SemanticEdge(
            id="edge_1",
            source_node_id="person_1",
            target_node_id="org_sever",
            relation=relation,
        )


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0, 1])
def test_edge_accepts_confidence_in_range(confidence: float) -> None:
    edge = SemanticEdge(
        id="edge_1",
        source_node_id="person_1",
        target_node_id="org_sever",
        relation="works_at",
        confidence=confidence,
    )

    assert isinstance(edge.confidence, float)
    assert 0.0 <= edge.confidence <= 1.0


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_edge_rejects_invalid_confidence(confidence: object) -> None:
    with pytest.raises((TypeError, ValueError), match="confidence"):
        SemanticEdge(
            id="edge_1",
            source_node_id="person_1",
            target_node_id="org_sever",
            relation="works_at",
            confidence=confidence,  # type: ignore[arg-type]
        )


def test_graph_accepts_nodes_and_edges() -> None:
    node = make_node()
    edge = make_edge()
    graph = SemanticGraph(nodes=[node], edges=[edge])

    assert graph.schema_version == "1"
    assert graph.revision == 0
    assert isinstance(graph.nodes, tuple)
    assert isinstance(graph.edges, tuple)


def test_graph_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        SemanticGraph(schema_version="2")
    with pytest.raises(ValueError, match="revision"):
        SemanticGraph(revision=-1)
    with pytest.raises(TypeError, match="revision"):
        SemanticGraph(revision=True)  # type: ignore[arg-type]


def test_graph_rejects_wrong_collection_items() -> None:
    with pytest.raises(TypeError, match="nodes"):
        SemanticGraph(nodes=[{"id": "person_1"}])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="edges"):
        SemanticGraph(edges=[make_node()])  # type: ignore[list-item]


def test_graph_accessors_and_incident_edges() -> None:
    person = make_node()
    organization = SemanticNode(
        id="org_sever",
        name="Компания «Север»",
        type=NodeType.ORGANIZATION,
    )
    edge = make_edge()
    graph = SemanticGraph(nodes=(person, organization), edges=(edge,))

    assert graph.node("person_1") == person
    assert graph.edge("edge_1") == edge
    assert graph.has_node("org_sever")
    assert not graph.has_node("missing")
    assert graph.has_edge("edge_1")
    assert not graph.has_edge("missing")
    assert graph.incident_edges("person_1") == (edge,)
    assert graph.incident_edges("org_sever") == (edge,)

    with pytest.raises(KeyError):
        graph.node("missing")
    with pytest.raises(KeyError):
        graph.edge("missing")


def test_graph_copy_is_independent() -> None:
    graph = SemanticGraph(nodes=(make_node(),), edges=(make_edge(),))
    copied = graph.copy()

    assert copied is not graph
    assert copied == graph
    assert copied.nodes == graph.nodes
    assert copied.edges == graph.edges


def test_graph_with_revision_returns_copy() -> None:
    graph = SemanticGraph(revision=3)
    updated = graph.with_revision(4)

    assert graph.revision == 3
    assert updated.revision == 4


def test_models_are_frozen() -> None:
    node = make_node()

    with pytest.raises(FrozenInstanceError):
        node.name = "Другое имя"  # type: ignore[misc]
