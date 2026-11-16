"""Чтение и детерминированная сериализация поддерживаемого подмножества DOT."""

from __future__ import annotations

import re
from decimal import Decimal

import pydot

from semantic_knowledge.domain.models import (
    NodeType,
    SemanticEdge,
    SemanticGraph,
    SemanticNode,
    ValueKind,
)


class DotCodecError(ValueError):
    """Ошибка чтения или сериализации DOT."""


# Токенизация сохраняет повторы, которые pydot объединяет при разборе.
_TOKEN = re.compile(
    r"\s+|//[^\r\n]*|/\*[\s\S]*?\*/|\#[^\r\n]*|"
    r'"(?:\\.|[^"\\])*"|->|[A-Za-z_][A-Za-z0-9_]*|[{}\[\]=,;]'
)


def _tokens(text: str) -> list[str]:
    result: list[str] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match:
            raise DotCodecError(f"INVALID_DOT: неподдерживаемый символ в позиции {pos}")
        token = match.group()
        if not (token.isspace() or token.startswith(("//", "/*", "#"))):
            result.append(token)
        pos = match.end()
    return result


def _decode(value: str) -> str:
    if not (value.startswith('"') and value.endswith('"')):
        raise DotCodecError(f"INVALID_ATTRIBUTE: требуется строка в кавычках: {value!r}")
    out: list[str] = []
    i = 1
    while i < len(value) - 1:
        ch = value[i]
        if ch == "\\":
            i += 1
            if i >= len(value) - 1 or value[i] not in {'"', "\\", "n"}:
                raise DotCodecError("INVALID_ATTRIBUTE: недопустимая escape-последовательность")
            out.append("\n" if value[i] == "n" else value[i])
        elif ord(ch) < 32 or ord(ch) == 127:
            raise DotCodecError("INVALID_ATTRIBUTE: управляющий символ в строке")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _quote(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if any((ord(c) < 32 and c != "\n") or ord(c) == 127 for c in value):
        raise DotCodecError("INVALID_ATTRIBUTE: управляющий символ в строке")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


class _Reader:
    def __init__(self, text: str) -> None:
        self.tokens = _tokens(text)
        self.index = 0

    def take(self) -> str:
        if self.index >= len(self.tokens):
            raise DotCodecError("INVALID_DOT: неожиданный конец файла")
        token = self.tokens[self.index]
        self.index += 1
        return token

    def expect(self, expected: str) -> None:
        value = self.take()
        if value != expected:
            raise DotCodecError(f"UNSUPPORTED_DOT: ожидалось {expected!r}, получено {value!r}")

    def attrs(self) -> dict[str, str]:
        self.expect("[")
        attrs: dict[str, str] = {}
        while True:
            name = self.take()
            if name == "]":
                return attrs
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise DotCodecError(f"INVALID_ATTRIBUTE: неверное имя {name!r}")
            self.expect("=")
            value = _decode(self.take())
            if name in attrs:
                raise DotCodecError(f"INVALID_ATTRIBUTE: повтор атрибута {name}")
            attrs[name] = value
            if self.index < len(self.tokens) and self.tokens[self.index] in {",", ";"}:
                self.index += 1


def _fields(attrs: dict[str, str], required: set[str], optional: set[str]) -> None:
    missing = required - attrs.keys()
    unknown = attrs.keys() - required - optional
    if missing or unknown:
        raise DotCodecError(
            f"INVALID_ATTRIBUTE: отсутствуют {sorted(missing)}, неизвестны {sorted(unknown)}"
        )


def _check_graph(nodes: list[SemanticNode], edges: list[SemanticEdge]) -> None:
    node_ids = [n.id for n in nodes]
    edge_ids = [e.id for e in edges]
    if len(set(node_ids)) != len(node_ids) or len(set(edge_ids)) != len(edge_ids):
        raise DotCodecError("DUPLICATE_ID: повтор идентификатора узла или связи")
    for edge in edges:
        if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
            raise DotCodecError(f"MISSING_NODE: связь {edge.id} ссылается на отсутствующий узел")


class DotCodec:
    """Преобразование между DOT-текстом и доменной моделью без файлового ввода-вывода."""

    def parse(self, text: str) -> SemanticGraph:
        """Прочитать DOT, проверив структуру до объединения атрибутов парсером."""
        try:
            reader = _Reader(text)
            reader.expect("digraph")
            if _decode(reader.take()) != "semantic_graph":
                raise DotCodecError("UNSUPPORTED_DOT: неверное имя графа")
            reader.expect("{")
            graph_attrs: dict[str, str] | None = None
            nodes: list[SemanticNode] = []
            edges: list[SemanticEdge] = []
            while True:
                token = reader.take()
                if token == "}":
                    break
                if token == "graph":
                    if graph_attrs is not None:
                        raise DotCodecError("INVALID_ATTRIBUTE: повтор атрибутов графа")
                    graph_attrs = reader.attrs()
                else:
                    source = _decode(token)
                    if reader.index < len(reader.tokens) and reader.tokens[reader.index] == "->":
                        reader.take()
                        target = _decode(reader.take())
                        attrs = reader.attrs()
                        _fields(attrs, {"id", "label"}, {"evidence", "source_ref", "confidence"})
                        raw_confidence = attrs.get("confidence")
                        if raw_confidence is not None and not re.fullmatch(
                            r"(?:0(?:\.[0-9]+)?|1(?:\.0+)?)", raw_confidence
                        ):
                            raise DotCodecError("INVALID_ATTRIBUTE: неверное значение confidence")
                        confidence = float(raw_confidence) if raw_confidence is not None else None
                        edges.append(
                            SemanticEdge(
                                id=attrs["id"],
                                source_node_id=source,
                                target_node_id=target,
                                relation=attrs["label"],
                                evidence=attrs.get("evidence"),
                                source_ref=attrs.get("source_ref"),
                                confidence=confidence,
                            )
                        )
                    else:
                        attrs = reader.attrs()
                        _fields(attrs, {"label", "type"}, {"description", "value_kind", "value"})
                        kind = ValueKind(attrs["value_kind"]) if "value_kind" in attrs else None
                        nodes.append(
                            SemanticNode(
                                id=source,
                                name=attrs["label"],
                                type=NodeType(attrs["type"]),
                                description=attrs.get("description"),
                                value_kind=kind,
                                value=attrs.get("value"),
                            )
                        )
                if reader.index < len(reader.tokens) and reader.tokens[reader.index] == ";":
                    reader.take()
            if reader.index != len(reader.tokens):
                raise DotCodecError("UNSUPPORTED_DOT: данные после графа")
            if graph_attrs is None:
                raise DotCodecError("INVALID_ATTRIBUTE: отсутствуют атрибуты графа")
            _fields(graph_attrs, {"revision", "schema_version"}, set())
            if not re.fullmatch(r"0|[1-9][0-9]*", graph_attrs["revision"]):
                raise DotCodecError("INVALID_ATTRIBUTE: неверная ревизия")
            _check_graph(nodes, edges)
            parsed = pydot.graph_from_dot_data(text)
            if parsed is None or len(parsed) != 1:
                raise DotCodecError("INVALID_DOT: не удалось разобрать граф")
            return SemanticGraph(
                schema_version=graph_attrs["schema_version"],
                revision=int(graph_attrs["revision"]),
                nodes=tuple(nodes),
                edges=tuple(edges),
            )
        except DotCodecError:
            raise
        except (ValueError, TypeError, KeyError) as error:
            raise DotCodecError(f"INVALID_ATTRIBUTE: {error}") from error

    def serialize(self, graph: SemanticGraph) -> str:
        """Вернуть канонический DOT с сохранением полей модели."""
        _check_graph(list(graph.nodes), list(graph.edges))
        lines = [
            'digraph "semantic_graph" {',
            (
                f"  graph [revision={_quote(str(graph.revision))}, "
                f"schema_version={_quote(graph.schema_version)}];"
            ),
        ]

        def attributes(values: dict[str, str]) -> str:
            return ", ".join(f"{key}={_quote(value)}" for key, value in sorted(values.items()))

        for node in sorted(graph.nodes, key=lambda n: n.id):
            attrs = {"label": node.name, "type": node.type.value}
            if node.description is not None:
                attrs["description"] = node.description
            if node.value_kind is not None:
                attrs["value_kind"] = node.value_kind.value
            if node.value is not None:
                attrs["value"] = node.value
            lines.append(f"  {_quote(node.id)} [{attributes(attrs)}];")
        for edge in sorted(graph.edges, key=lambda e: (e.source_node_id, e.target_node_id, e.id)):
            attrs = {"id": edge.id, "label": edge.relation}
            if edge.evidence is not None:
                attrs["evidence"] = edge.evidence
            if edge.source_ref is not None:
                attrs["source_ref"] = edge.source_ref
            if edge.confidence is not None:
                attrs["confidence"] = (
                    "0" if edge.confidence == 0 else format(Decimal(str(edge.confidence)), "f")
                )
            lines.append(
                f"  {_quote(edge.source_node_id)} -> {_quote(edge.target_node_id)} "
                f"[{attributes(attrs)}];"
            )
        return "\n".join([*lines, "}", ""])
