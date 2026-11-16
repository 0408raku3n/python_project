"""Проверки чтения и записи графа DOT."""

from pathlib import Path

import pytest

from semantic_knowledge.domain.models import NodeType, SemanticEdge, SemanticGraph, SemanticNode
from semantic_knowledge.storage.dot_codec import DotCodec, DotCodecError


def test_examples_round_trip() -> None:
    codec = DotCodec()
    for filename in ("examples/dot/valid.dot", "examples/example-model.dot"):
        graph = codec.parse(Path(filename).read_text(encoding="utf-8"))
        assert codec.parse(codec.serialize(graph)) == graph


def test_unicode_escapes_and_order() -> None:
    codec = DotCodec()
    graph = SemanticGraph(
        nodes=(
            SemanticNode(id="z", name='Кириллица "ёлка"\nстрока', type=NodeType.PERSON),
            SemanticNode(id="a", name="Путь C:\\docs", type=NodeType.CONCEPT),
        )
    )
    text = codec.serialize(graph)
    result = codec.parse(text)
    assert result.nodes == tuple(sorted(graph.nodes, key=lambda n: n.id))
    assert codec.serialize(result) == text


@pytest.mark.parametrize(
    "filename", ["missing-node.dot", "duplicate-edge-id.dot", "unsupported-subgraph.dot"]
)
def test_invalid_examples(filename: str) -> None:
    with pytest.raises(DotCodecError):
        DotCodec().parse(Path("examples/dot", filename).read_text(encoding="utf-8"))


def test_duplicate_attribute_rejected() -> None:
    text = 'digraph "semantic_graph" { graph [revision="0", revision="1", schema_version="1"]; }'
    with pytest.raises(DotCodecError, match="повтор"):
        DotCodec().parse(text)


def test_implicit_node_rejected() -> None:
    text = Path("examples/dot/missing-node.dot").read_text(encoding="utf-8")
    with pytest.raises(DotCodecError, match="MISSING_NODE"):
        DotCodec().parse(text)


def test_confidence_round_trip_small_and_negative_zero() -> None:
    codec = DotCodec()
    node = SemanticNode(id="a", name="A", type=NodeType.CONCEPT)
    for confidence in (1e-7, -0.0):
        edge = SemanticEdge(
            id="e",
            source_node_id="a",
            target_node_id="a",
            relation="related",
            confidence=confidence,
        )
        graph = SemanticGraph(nodes=(node,), edges=(edge,))
        assert codec.parse(codec.serialize(graph)) == graph
