"""Доменные модели семантического графа.

Модели неизменяемы и не зависят от библиотек сериализации. Проверяются
обязательные поля, типы и локальные ограничения. Инварианты уровня графа
(уникальность идентификаторов и ссылочная целостность) проверяет
``GraphValidator``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

NODE_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
RELATION_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
INTEGER_PATTERN = re.compile(r"^-?(0|[1-9][0-9]*)$")
DECIMAL_PATTERN = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?$")
YEAR_PATTERN = re.compile(r"^[0-9]{4}$")
QUANTITY_PATTERN = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?\s+\S+$")

SCHEMA_VERSION = "1"


class NodeType(StrEnum):
    """Допустимые семантические типы узлов."""

    OBJECT = "object"
    PERSON = "person"
    ORGANIZATION = "organization"
    EVENT = "event"
    CONCEPT = "concept"
    VALUE = "value"


class ValueKind(StrEnum):
    """Допустимые типы самостоятельных значений."""

    TEXT = "text"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    YEAR = "year"
    DATE = "date"
    DATETIME = "datetime"
    QUANTITY = "quantity"


def validate_identifier(value: object, field: str) -> str:
    """Проверить идентификатор узла или связи и вернуть его строкой."""

    if not isinstance(value, str) or not NODE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field} должен соответствовать шаблону [A-Za-z][A-Za-z0-9_-]*: {value!r}"
        )
    return value


def validate_relation(value: object) -> str:
    """Проверить код отношения в snake_case."""

    if not isinstance(value, str) or not RELATION_PATTERN.fullmatch(value):
        raise ValueError(
            f"relation должен соответствовать шаблону [a-z][a-z0-9_]*: {value!r}"
        )
    return value


def _require_non_empty_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} должен быть строкой: {value!r}")
    if not value.strip():
        raise ValueError(f"{field} должен быть непустой строкой: {value!r}")
    return value


def _optional_text(value: object, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field} должен быть строкой или None: {value!r}")
    return value


def _coerce_node_type(value: object) -> NodeType:
    if isinstance(value, NodeType):
        return value
    if isinstance(value, str):
        try:
            return NodeType(value)
        except ValueError as error:
            raise ValueError(f"неизвестный тип узла: {value!r}") from error
    raise TypeError(f"тип узла должен быть NodeType или строкой: {value!r}")


def _coerce_value_kind(value: object) -> ValueKind:
    if isinstance(value, ValueKind):
        return value
    if isinstance(value, str):
        try:
            return ValueKind(value)
        except ValueError as error:
            raise ValueError(f"неизвестный тип значения: {value!r}") from error
    raise TypeError(f"тип значения должен быть ValueKind или строкой: {value!r}")


def _validate_value(value_kind: ValueKind, value: object) -> str:
    text = _require_non_empty_text(value, "value")

    if value_kind is ValueKind.TEXT:
        return text
    if value_kind is ValueKind.INTEGER:
        if not INTEGER_PATTERN.fullmatch(text):
            raise ValueError(f"value должно быть целым числом: {text!r}")
        return text
    if value_kind is ValueKind.DECIMAL:
        if not DECIMAL_PATTERN.fullmatch(text):
            raise ValueError(f"value должно быть десятичным числом: {text!r}")
        return text
    if value_kind is ValueKind.BOOLEAN:
        if text not in {"true", "false"}:
            raise ValueError(f"value должно быть 'true' или 'false': {text!r}")
        return text
    if value_kind is ValueKind.YEAR:
        if not YEAR_PATTERN.fullmatch(text):
            raise ValueError(f"value должно быть годом из четырёх цифр: {text!r}")
        return text
    if value_kind is ValueKind.DATE:
        try:
            date.fromisoformat(text)
        except ValueError as error:
            raise ValueError(f"value должно быть датой ISO 8601: {text!r}") from error
        return text
    if value_kind is ValueKind.DATETIME:
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as error:
            raise ValueError(
                f"value должно быть датой и временем ISO 8601: {text!r}"
            ) from error
        if parsed.tzinfo is None:
            raise ValueError(f"value должно содержать часовой пояс: {text!r}")
        return text
    if value_kind is ValueKind.QUANTITY:
        if not QUANTITY_PATTERN.fullmatch(text):
            raise ValueError(f"value должно быть величиной с единицей измерения: {text!r}")
        return text
    raise ValueError(f"неизвестный тип значения: {value_kind!r}")


@dataclass(frozen=True, slots=True)
class SemanticNode:
    """Сущность или самостоятельное значение семантического графа."""

    id: str
    name: str
    type: NodeType
    description: str | None = None
    value_kind: ValueKind | None = None
    value: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.id, "id узла")
        _require_non_empty_text(self.name, "name")
        node_type = _coerce_node_type(self.type)
        object.__setattr__(self, "type", node_type)
        _optional_text(self.description, "description")

        if node_type is NodeType.VALUE:
            if self.value_kind is None or self.value is None:
                raise ValueError("для узла type='value' обязательны value_kind и value")
            value_kind = _coerce_value_kind(self.value_kind)
            object.__setattr__(self, "value_kind", value_kind)
            _validate_value(value_kind, self.value)
        elif self.value_kind is not None or self.value is not None:
            raise ValueError("value_kind и value допустимы только для узла type='value'")


@dataclass(frozen=True, slots=True)
class SemanticEdge:
    """Направленное утверждение «субъект — отношение — объект»."""

    id: str
    source_node_id: str
    target_node_id: str
    relation: str
    evidence: str | None = None
    source_ref: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.id, "id связи")
        validate_identifier(self.source_node_id, "source_node_id")
        validate_identifier(self.target_node_id, "target_node_id")
        validate_relation(self.relation)
        _optional_text(self.evidence, "evidence")
        _optional_text(self.source_ref, "source_ref")

        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise TypeError(f"confidence должен быть числом: {self.confidence!r}")
            confidence = float(self.confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence должен быть в диапазоне 0.0..1.0: {confidence!r}")
            object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class SemanticGraph:
    """Семантический граф; структура неизменяема после создания."""

    schema_version: str = SCHEMA_VERSION
    revision: int = 0
    nodes: tuple[SemanticNode, ...] = ()
    edges: tuple[SemanticEdge, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"поддерживается только schema_version={SCHEMA_VERSION!r}: "
                f"{self.schema_version!r}"
            )
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError(f"revision должен быть целым числом: {self.revision!r}")
        if self.revision < 0:
            raise ValueError(f"revision не может быть отрицательным: {self.revision!r}")

        nodes = self._as_tuple(self.nodes, SemanticNode, "nodes")
        edges = self._as_tuple(self.edges, SemanticEdge, "edges")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

    @staticmethod
    def _as_tuple[T](
        values: Iterable[T], expected: type[T], field: str
    ) -> tuple[T, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{field} должен быть коллекцией объектов")
        try:
            items = tuple(values)
        except TypeError as error:
            raise TypeError(f"{field} должен быть коллекцией объектов") from error
        for item in items:
            if not isinstance(item, expected):
                raise TypeError(
                    f"{field} может содержать только {expected.__name__}: {item!r}"
                )
        return items

    def node(self, node_id: str) -> SemanticNode:
        """Вернуть узел по идентификатору; для валидного графа ID уникальны."""

        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"узел не найден: {node_id!r}")

    def edge(self, edge_id: str) -> SemanticEdge:
        """Вернуть связь по идентификатору; для валидного графа ID уникальны."""

        for edge in self.edges:
            if edge.id == edge_id:
                return edge
        raise KeyError(f"связь не найдена: {edge_id!r}")

    def has_node(self, node_id: str) -> bool:
        return any(node.id == node_id for node in self.nodes)

    def has_edge(self, edge_id: str) -> bool:
        return any(edge.id == edge_id for edge in self.edges)

    def incident_edges(self, node_id: str) -> tuple[SemanticEdge, ...]:
        """Вернуть все входящие и исходящие связи указанного узла."""

        return tuple(
            edge
            for edge in self.edges
            if edge.source_node_id == node_id or edge.target_node_id == node_id
        )

    def copy(self) -> SemanticGraph:
        """Вернуть независимую копию: модели неизменяемы, поэтому достаточно новых кортежей."""

        return SemanticGraph(
            schema_version=self.schema_version,
            revision=self.revision,
            nodes=tuple(self.nodes),
            edges=tuple(self.edges),
        )

    def with_revision(self, revision: int) -> SemanticGraph:
        """Вернуть копию графа с другой ревизией."""

        return SemanticGraph(
            schema_version=self.schema_version,
            revision=revision,
            nodes=tuple(self.nodes),
            edges=tuple(self.edges),
        )
