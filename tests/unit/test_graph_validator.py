"""Модульные тесты GraphValidator."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from semantic_knowledge.domain.models import (
    NodeType,
    SemanticEdge,
    SemanticGraph,
    SemanticNode,
    ValueKind,
)
from semantic_knowledge.graph.validator import (
    GraphDiagnosticCode,
    GraphObjectKind,
    GraphValidationError,
    GraphValidator,
)


def make_node(
    node_id: str = "a",
    name: str = "Узел A",
    node_type: NodeType = NodeType.PERSON,
) -> SemanticNode:
    return SemanticNode(id=node_id, name=name, type=node_type)


def make_edge(
    edge_id: str = "e1",
    source: str = "a",
    target: str = "b",
) -> SemanticEdge:
    return SemanticEdge(
        id=edge_id,
        source_node_id=source,
        target_node_id=target,
        relation="knows",
    )


def test_empty_graph_is_valid() -> None:
    assert GraphValidator().validate(SemanticGraph()) == []


def test_graph_with_cycle_and_self_loop_is_valid() -> None:
    graph = SemanticGraph(
        nodes=(make_node("a"), make_node("b")),
        edges=(
            make_edge("e1", "a", "b"),
            make_edge("e2", "b", "a"),
            make_edge("e3", "a", "a"),
        ),
    )

    assert GraphValidator().validate(graph) == []


def test_duplicate_node_id_is_reported_once() -> None:
    graph = SemanticGraph(nodes=(make_node("a"), make_node("a", "Другой A")))

    diagnostics = GraphValidator().validate(graph)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code is GraphDiagnosticCode.DUPLICATE_ID
    assert diagnostic.object_kind is GraphObjectKind.NODE
    assert diagnostic.object_id == "a"
    assert "идентификатор узла" in diagnostic.message


def test_duplicate_edge_id_is_reported_once() -> None:
    graph = SemanticGraph(
        nodes=(make_node("a"), make_node("b")),
        edges=(make_edge("e1", "a", "b"), make_edge("e1", "b", "a")),
    )

    diagnostics = GraphValidator().validate(graph)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code is GraphDiagnosticCode.DUPLICATE_ID
    assert diagnostic.object_kind is GraphObjectKind.EDGE
    assert diagnostic.object_id == "e1"


def test_node_and_edge_id_namespaces_are_separate() -> None:
    graph = SemanticGraph(
        nodes=(make_node("same"), make_node("b")),
        edges=(make_edge("same", "same", "b"),),
    )

    assert GraphValidator().validate(graph) == []


def test_missing_source_node_is_reported() -> None:
    graph = SemanticGraph(
        nodes=(make_node("b"),),
        edges=(make_edge("e1", "a", "b"),),
    )

    diagnostics = GraphValidator().validate(graph)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code is GraphDiagnosticCode.MISSING_NODE
    assert diagnostic.object_kind is GraphObjectKind.EDGE
    assert diagnostic.object_id == "e1"
    assert diagnostic.field == "source_node_id"
    assert diagnostic.related_id == "a"


def test_missing_target_node_is_reported() -> None:
    graph = SemanticGraph(
        nodes=(make_node("a"),),
        edges=(make_edge("e1", "a", "b"),),
    )

    diagnostics = GraphValidator().validate(graph)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code is GraphDiagnosticCode.MISSING_NODE
    assert diagnostic.field == "target_node_id"
    assert diagnostic.related_id == "b"


def test_both_missing_endpoints_are_reported() -> None:
    graph = SemanticGraph(edges=(make_edge("e1", "a", "b"),))

    diagnostics = GraphValidator().validate(graph)

    assert [item.field for item in diagnostics] == ["source_node_id", "target_node_id"]
    assert {item.related_id for item in diagnostics} == {"a", "b"}


@pytest.mark.parametrize(
    ("mutate", "expected_message"),
    [
        (lambda node: object.__setattr__(node, "name", ""), "name"),
        (lambda node: object.__setattr__(node, "type", "unknown"), "тип узла"),
        (lambda node: object.__setattr__(node, "id", "1bad"), "id узла"),
    ],
)
def test_invalid_node_attributes_are_reported(
    mutate: Callable[[SemanticNode], None],
    expected_message: str,
) -> None:
    node = make_node()
    mutate(node)
    graph = SemanticGraph(nodes=(node,))

    diagnostics = GraphValidator().validate(graph)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code is GraphDiagnosticCode.INVALID_ATTRIBUTE
    assert diagnostic.object_kind is GraphObjectKind.NODE
    assert diagnostic.object_id == node.id
    assert expected_message in diagnostic.message


def test_invalid_value_node_is_reported() -> None:
    node = SemanticNode(
        id="v1",
        name="2018 год",
        type=NodeType.VALUE,
        value_kind=ValueKind.YEAR,
        value="2018",
    )
    object.__setattr__(node, "value", "18")
    graph = SemanticGraph(nodes=(node,))

    diagnostics = GraphValidator().validate(graph)

    assert len(diagnostics) == 1
    assert diagnostics[0].code is GraphDiagnosticCode.INVALID_ATTRIBUTE
    assert diagnostics[0].object_kind is GraphObjectKind.NODE
    assert "value" in diagnostics[0].message


@pytest.mark.parametrize(
    ("mutate", "expected_message"),
    [
        (lambda edge: object.__setattr__(edge, "relation", "works at"), "relation"),
        (lambda edge: object.__setattr__(edge, "confidence", float("nan")), "confidence"),
    ],
)
def test_invalid_edge_attributes_are_reported(
    mutate: Callable[[SemanticEdge], None],
    expected_message: str,
) -> None:
    edge = make_edge()
    mutate(edge)
    graph = SemanticGraph(
        nodes=(make_node("a"), make_node("b")),
        edges=(edge,),
    )

    diagnostics = GraphValidator().validate(graph)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code is GraphDiagnosticCode.INVALID_ATTRIBUTE
    assert diagnostic.object_kind is GraphObjectKind.EDGE
    assert diagnostic.object_id == "e1"
    assert expected_message in diagnostic.message


def test_unsupported_schema_version_is_reported() -> None:
    graph = SemanticGraph()
    object.__setattr__(graph, "schema_version", "2")

    diagnostics = GraphValidator().validate(graph)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code is GraphDiagnosticCode.INVALID_ATTRIBUTE
    assert diagnostic.object_kind is GraphObjectKind.GRAPH
    assert diagnostic.field == "schema_version"
    assert "schema_version" in diagnostic.message


@pytest.mark.parametrize("revision", [-1, True])
def test_invalid_revision_is_reported(revision: object) -> None:
    graph = SemanticGraph()
    object.__setattr__(graph, "revision", revision)

    diagnostics = GraphValidator().validate(graph)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code is GraphDiagnosticCode.INVALID_ATTRIBUTE
    assert diagnostic.object_kind is GraphObjectKind.GRAPH
    assert diagnostic.field == "revision"
    assert "revision" in diagnostic.message


def test_invalid_collection_item_is_reported() -> None:
    graph = SemanticGraph()
    object.__setattr__(graph, "nodes", (object(),))

    diagnostics = GraphValidator().validate(graph)

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code is GraphDiagnosticCode.INVALID_GRAPH
    assert diagnostic.object_kind is GraphObjectKind.GRAPH
    assert diagnostic.field == "nodes"


def test_multiple_diagnostics_are_returned() -> None:
    graph = SemanticGraph(
        nodes=(make_node("a"), make_node("a")),
        edges=(make_edge("e1", "a", "missing"),),
    )

    diagnostics = GraphValidator().validate(graph)

    assert [item.code for item in diagnostics] == [
        GraphDiagnosticCode.DUPLICATE_ID,
        GraphDiagnosticCode.MISSING_NODE,
    ]


def test_validate_or_raise_accepts_valid_graph() -> None:
    GraphValidator().validate_or_raise(SemanticGraph())


def test_validate_or_raise_rejects_invalid_graph() -> None:
    graph = SemanticGraph(nodes=(make_node("a"), make_node("a")))

    with pytest.raises(GraphValidationError) as error:
        GraphValidator().validate_or_raise(graph)

    assert len(error.value.diagnostics) == 1
    assert error.value.diagnostics[0].code is GraphDiagnosticCode.DUPLICATE_ID
    assert "DUPLICATE_ID" in str(error.value)
