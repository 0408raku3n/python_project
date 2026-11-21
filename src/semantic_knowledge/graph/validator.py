"""Проверка инвариантов семантического графа."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from semantic_knowledge.domain.models import (
    SCHEMA_VERSION,
    SemanticEdge,
    SemanticGraph,
    SemanticNode,
)


class GraphDiagnosticCode(StrEnum):
    """Коды диагностики GraphValidator."""

    DUPLICATE_ID = "DUPLICATE_ID"
    MISSING_NODE = "MISSING_NODE"
    INVALID_ATTRIBUTE = "INVALID_ATTRIBUTE"
    INVALID_GRAPH = "INVALID_GRAPH"


class GraphObjectKind(StrEnum):
    """Вид объекта, к которому относится диагностика."""

    GRAPH = "graph"
    NODE = "node"
    EDGE = "edge"


@dataclass(frozen=True, slots=True)
class GraphDiagnostic:
    """Структурированное сообщение о нарушении инварианта графа."""

    code: GraphDiagnosticCode
    message: str
    object_kind: GraphObjectKind
    object_id: str | None = None
    field: str | None = None
    related_id: str | None = None

    def __str__(self) -> str:
        location = self.object_kind.value
        if self.object_id is not None:
            location += f" {self.object_id!r}"
        if self.field is not None:
            location += f".{self.field}"
        return f"{self.code.value}: {self.message} ({location})"


class GraphValidationError(ValueError):
    """Исключение с полным списком диагностик невалидного графа."""

    def __init__(self, diagnostics: Sequence[GraphDiagnostic]) -> None:
        self.diagnostics = tuple(diagnostics)
        details = "; ".join(str(item) for item in self.diagnostics)
        super().__init__(details or "граф не прошёл валидацию")


class GraphValidator:
    """Проверяет доменные инварианты графа без изменения входного объекта."""

    def validate(self, graph: SemanticGraph) -> list[GraphDiagnostic]:
        """Вернуть все обнаруженные нарушения; пустой список означает успех."""

        diagnostics: list[GraphDiagnostic] = []
        self._validate_metadata(graph, diagnostics)

        nodes = list(graph.nodes)
        edges = list(graph.edges)
        self._validate_nodes(nodes, diagnostics)
        self._validate_edges(edges, diagnostics)
        self._validate_duplicate_node_ids(nodes, diagnostics)
        self._validate_duplicate_edge_ids(edges, diagnostics)
        self._validate_edge_endpoints(edges, nodes, diagnostics)
        return diagnostics

    def validate_or_raise(self, graph: SemanticGraph) -> None:
        """Проверить граф перед сохранением и выбросить ошибку при нарушениях."""

        diagnostics = self.validate(graph)
        if diagnostics:
            raise GraphValidationError(diagnostics)

    def _validate_metadata(
        self,
        graph: SemanticGraph,
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        if graph.schema_version != SCHEMA_VERSION:
            diagnostics.append(
                GraphDiagnostic(
                    code=GraphDiagnosticCode.INVALID_ATTRIBUTE,
                    message=(
                        f"неподдерживаемая schema_version: {graph.schema_version!r}; "
                        f"ожидается {SCHEMA_VERSION!r}"
                    ),
                    object_kind=GraphObjectKind.GRAPH,
                    field="schema_version",
                )
            )

        if (
            isinstance(graph.revision, bool)
            or not isinstance(graph.revision, int)
            or graph.revision < 0
        ):
            diagnostics.append(
                GraphDiagnostic(
                    code=GraphDiagnosticCode.INVALID_ATTRIBUTE,
                    message=(
                        "revision должен быть неотрицательным целым числом: "
                        f"{graph.revision!r}"
                    ),
                    object_kind=GraphObjectKind.GRAPH,
                    field="revision",
                )
            )

    def _validate_nodes(
        self,
        nodes: Iterable[SemanticNode],
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        for node in nodes:
            if not isinstance(node, SemanticNode):
                diagnostics.append(
                    GraphDiagnostic(
                        code=GraphDiagnosticCode.INVALID_GRAPH,
                        message=(
                            "nodes должен содержать только SemanticNode: "
                            f"{node!r}"
                        ),
                        object_kind=GraphObjectKind.GRAPH,
                        field="nodes",
                    )
                )
                continue
            try:
                SemanticNode(
                    id=node.id,
                    name=node.name,
                    type=node.type,
                    description=node.description,
                    value_kind=node.value_kind,
                    value=node.value,
                )
            except (TypeError, ValueError) as error:
                diagnostics.append(
                    GraphDiagnostic(
                        code=GraphDiagnosticCode.INVALID_ATTRIBUTE,
                        message=str(error),
                        object_kind=GraphObjectKind.NODE,
                        object_id=node.id if isinstance(node.id, str) else None,
                    )
                )

    def _validate_edges(
        self,
        edges: Iterable[SemanticEdge],
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        for edge in edges:
            if not isinstance(edge, SemanticEdge):
                diagnostics.append(
                    GraphDiagnostic(
                        code=GraphDiagnosticCode.INVALID_GRAPH,
                        message=(
                            "edges должен содержать только SemanticEdge: "
                            f"{edge!r}"
                        ),
                        object_kind=GraphObjectKind.GRAPH,
                        field="edges",
                    )
                )
                continue
            try:
                SemanticEdge(
                    id=edge.id,
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    relation=edge.relation,
                    evidence=edge.evidence,
                    source_ref=edge.source_ref,
                    confidence=edge.confidence,
                )
            except (TypeError, ValueError) as error:
                diagnostics.append(
                    GraphDiagnostic(
                        code=GraphDiagnosticCode.INVALID_ATTRIBUTE,
                        message=str(error),
                        object_kind=GraphObjectKind.EDGE,
                        object_id=edge.id if isinstance(edge.id, str) else None,
                    )
                )

    def _validate_duplicate_node_ids(
        self,
        nodes: Iterable[SemanticNode],
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        self._validate_duplicate_ids(
            (
                node.id
                for node in nodes
                if isinstance(node, SemanticNode) and isinstance(node.id, str)
            ),
            GraphObjectKind.NODE,
            "идентификатор узла",
            diagnostics,
        )

    def _validate_duplicate_edge_ids(
        self,
        edges: Iterable[SemanticEdge],
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        self._validate_duplicate_ids(
            (
                edge.id
                for edge in edges
                if isinstance(edge, SemanticEdge) and isinstance(edge.id, str)
            ),
            GraphObjectKind.EDGE,
            "идентификатор связи",
            diagnostics,
        )

    @staticmethod
    def _validate_duplicate_ids(
        ids: Iterable[str],
        kind: GraphObjectKind,
        label: str,
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        seen: set[str] = set()
        reported: set[str] = set()
        for object_id in ids:
            if object_id in seen and object_id not in reported:
                diagnostics.append(
                    GraphDiagnostic(
                        code=GraphDiagnosticCode.DUPLICATE_ID,
                        message=f"повтор {label}: {object_id!r}",
                        object_kind=kind,
                        object_id=object_id,
                    )
                )
                reported.add(object_id)
            seen.add(object_id)

    def _validate_edge_endpoints(
        self,
        edges: Iterable[SemanticEdge],
        nodes: Iterable[SemanticNode],
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        node_ids = {
            node.id
            for node in nodes
            if isinstance(node, SemanticNode) and isinstance(node.id, str)
        }
        for edge in edges:
            if not isinstance(edge, SemanticEdge):
                continue
            for field, node_id in (
                ("source_node_id", edge.source_node_id),
                ("target_node_id", edge.target_node_id),
            ):
                if not isinstance(node_id, str) or node_id in node_ids:
                    continue
                diagnostics.append(
                    GraphDiagnostic(
                        code=GraphDiagnosticCode.MISSING_NODE,
                        message=(
                            f"связь {edge.id!r} ссылается на отсутствующий узел "
                            f"{node_id!r} ({field})"
                        ),
                        object_kind=GraphObjectKind.EDGE,
                        object_id=edge.id if isinstance(edge.id, str) else None,
                        field=field,
                        related_id=node_id,
                    )
                )
