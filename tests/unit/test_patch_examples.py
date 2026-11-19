"""Проверка примеров патчей на соответствие доменным моделям."""

import json
from pathlib import Path

from semantic_knowledge.domain.models import validate_relation
from semantic_knowledge.domain.patches import AddEdge, AddNode, DeleteEdge, UpdateEdge
from semantic_knowledge.storage.dot_codec import DotCodec


def test_patch_examples() -> None:
    graph = DotCodec().parse(Path("examples/dot/valid.dot").read_text(encoding="utf-8"))
    for path in Path("examples/patches").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["base_revision"] == graph.revision
        nodes = {node.id for node in graph.nodes}
        edges = {edge.id for edge in graph.edges}
        missing = False
        for operation in data["operations"]:
            fields = dict(operation)
            op = fields.pop("op")
            if op == "add_node":
                node = AddNode(**fields)
                assert node.id not in nodes
                nodes.add(node.id)
            elif op == "add_edge":
                edge = AddEdge(**fields)
                assert edge.id not in edges
                missing = edge.source not in nodes or edge.target not in nodes
                edges.add(edge.id)
            elif op == "update_edge":
                update = UpdateEdge(**fields)
                assert update.id in edges
                validate_relation(update.attributes["label"])
            elif op == "delete_edge":
                deletion = DeleteEdge(**fields)
                assert deletion.id in edges
                edges.remove(deletion.id)
            else:
                raise AssertionError(op)
        assert missing == (path.name == "invalid-missing-node.json")
