"""Модели операций патча и контейнер изменений семантического графа.

Формат операций соответствует ``docs/graph-patches.md`` с расширением под
``docs/semantic-model.md``: ``add_node`` требует обязательный ``type``, а
``attributes`` допускает все поля соответствующей доменной модели.

Модели проверяют только собственную структуру. Существование объектов,
совпадение ревизии и конфликты операций проверяет ``PatchValidator``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from .models import (
    NodeType,
    SemanticEdge,
    SemanticNode,
    ValueKind,
    validate_identifier,
)

type AttributeValue = str | int | float

NODE_ATTRIBUTE_NAMES = frozenset({"label", "type", "description", "value_kind", "value"})
EDGE_ATTRIBUTE_NAMES = frozenset({"label", "evidence", "source_ref", "confidence"})


def _validate_revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"base_revision должен быть целым числом: {value!r}")
    if value < 0:
        raise ValueError(f"base_revision не может быть отрицательным: {value!r}")
    return value


def _validate_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"confidence должен быть числом: {value!r}")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence должен быть в диапазоне 0.0..1.0: {confidence!r}")
    return confidence


def _validate_attribute_value(name: str, value: object) -> AttributeValue:
    if name == "label":
        if not isinstance(value, str):
            raise TypeError(f"label должен быть строкой: {value!r}")
        if not value.strip():
            raise ValueError(f"label должен быть непустой строкой: {value!r}")
        return value
    if name == "type":
        if not isinstance(value, str):
            raise TypeError(f"type должен быть строкой: {value!r}")
        try:
            NodeType(value)
        except ValueError as error:
            raise ValueError(f"неизвестный тип узла: {value!r}") from error
        return value
    if name == "value_kind":
        if not isinstance(value, str):
            raise TypeError(f"value_kind должен быть строкой: {value!r}")
        try:
            ValueKind(value)
        except ValueError as error:
            raise ValueError(f"неизвестный тип значения: {value!r}") from error
        return value
    if name == "confidence":
        return _validate_confidence(value)
    if name in {"description", "evidence", "source_ref", "value"}:
        if not isinstance(value, str):
            raise TypeError(f"{name} должен быть строкой: {value!r}")
        return value
    raise ValueError(f"неизвестный атрибут: {name!r}")


def _validate_attributes(
    attributes: object,
    allowed: frozenset[str],
    field: str,
) -> Mapping[str, AttributeValue]:
    if not isinstance(attributes, Mapping):
        raise TypeError(f"{field} должен быть объектом атрибутов: {attributes!r}")
    if not attributes:
        raise ValueError(f"{field} не может быть пустым")

    normalized: dict[str, AttributeValue] = {}
    for name, value in attributes.items():
        if not isinstance(name, str) or name not in allowed:
            raise ValueError(f"неизвестный атрибут: {name!r}")
        normalized[name] = _validate_attribute_value(name, value)
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class AddNode:
    """Добавить узел с новым идентификатором."""

    op: ClassVar[str] = "add_node"

    id: str
    label: str
    type: NodeType
    description: str | None = None
    value_kind: ValueKind | None = None
    value: str | None = None

    def __post_init__(self) -> None:
        node = self.to_node()
        object.__setattr__(self, "type", node.type)
        object.__setattr__(self, "value_kind", node.value_kind)

    def to_node(self) -> SemanticNode:
        """Собрать доменный узел и проверить его поля."""

        return SemanticNode(
            id=self.id,
            name=self.label,
            type=self.type,
            description=self.description,
            value_kind=self.value_kind,
            value=self.value,
        )


@dataclass(frozen=True, slots=True)
class UpdateNode:
    """Изменить атрибуты существующего узла."""

    op: ClassVar[str] = "update_node"

    id: str
    attributes: Mapping[str, AttributeValue]

    def __post_init__(self) -> None:
        validate_identifier(self.id, "id узла")
        attributes = _validate_attributes(self.attributes, NODE_ATTRIBUTE_NAMES, "attributes")
        object.__setattr__(self, "attributes", attributes)


@dataclass(frozen=True, slots=True)
class DeleteNode:
    """Удалить существующий узел без инцидентных связей."""

    op: ClassVar[str] = "delete_node"

    id: str

    def __post_init__(self) -> None:
        validate_identifier(self.id, "id узла")


@dataclass(frozen=True, slots=True)
class AddEdge:
    """Добавить связь с новым идентификатором между существующими узлами."""

    op: ClassVar[str] = "add_edge"

    id: str
    source: str
    target: str
    label: str
    evidence: str | None = None
    source_ref: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        edge = self.to_edge()
        object.__setattr__(self, "confidence", edge.confidence)

    def to_edge(self) -> SemanticEdge:
        """Собрать доменную связь и проверить её поля."""

        return SemanticEdge(
            id=self.id,
            source_node_id=self.source,
            target_node_id=self.target,
            relation=self.label,
            evidence=self.evidence,
            source_ref=self.source_ref,
            confidence=self.confidence,
        )


@dataclass(frozen=True, slots=True)
class UpdateEdge:
    """Изменить атрибуты существующей связи."""

    op: ClassVar[str] = "update_edge"

    id: str
    attributes: Mapping[str, AttributeValue]

    def __post_init__(self) -> None:
        validate_identifier(self.id, "id связи")
        attributes = _validate_attributes(self.attributes, EDGE_ATTRIBUTE_NAMES, "attributes")
        object.__setattr__(self, "attributes", attributes)


@dataclass(frozen=True, slots=True)
class DeleteEdge:
    """Удалить существующую связь."""

    op: ClassVar[str] = "delete_edge"

    id: str

    def __post_init__(self) -> None:
        validate_identifier(self.id, "id связи")


type PatchOperation = AddNode | UpdateNode | DeleteNode | AddEdge | UpdateEdge | DeleteEdge


@dataclass(frozen=True, slots=True)
class GraphPatch:
    """Упорядоченный набор операций, применяемый к одной ревизии графа."""

    base_revision: int
    operations: tuple[PatchOperation, ...]

    def __post_init__(self) -> None:
        _validate_revision(self.base_revision)
        operations = tuple(self.operations)
        if not operations:
            raise ValueError("патч должен содержать хотя бы одну операцию")
        for operation in operations:
            if not isinstance(
                operation,
                (AddNode, UpdateNode, DeleteNode, AddEdge, UpdateEdge, DeleteEdge),
            ):
                raise TypeError(f"неизвестная операция патча: {operation!r}")
        object.__setattr__(self, "operations", operations)
