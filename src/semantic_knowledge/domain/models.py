"""Доменные модели семантического графа и ответов LLM.

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
LLM_CONTRACT_VERSION = "1"


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


class IntentCategory(StrEnum):
    """Категории намерения пользователя из контракта LLM."""

    QUESTION = "QUESTION"
    UPDATE = "UPDATE"
    INGEST = "INGEST"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


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


def _validate_contract_version(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"contract_version должен быть строкой: {value!r}")
    if value != LLM_CONTRACT_VERSION:
        raise ValueError(
            f"поддерживается только contract_version={LLM_CONTRACT_VERSION!r}: {value!r}"
        )
    return value


def _validate_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"confidence должен быть числом: {value!r}")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence должен быть в диапазоне 0.0..1.0: {confidence!r}")
    return confidence


def _coerce_intent_category(value: object) -> IntentCategory:
    if isinstance(value, IntentCategory):
        return value
    if isinstance(value, str):
        try:
            return IntentCategory(value)
        except ValueError as error:
            raise ValueError(f"неизвестная категория намерения: {value!r}") from error
    raise TypeError(f"категория намерения должна быть IntentCategory или строкой: {value!r}")


def _ensure_unique(values: Iterable[str], field: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"повтор {field}: {value!r}")
        seen.add(value)


def _as_tuple[T](values: Iterable[T], expected: type[T], field: str) -> tuple[T, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} должен быть коллекцией объектов")
    try:
        items = tuple(values)
    except TypeError as error:
        raise TypeError(f"{field} должен быть коллекцией объектов") from error
    for item in items:
        if not isinstance(item, expected):
            raise TypeError(f"{field} может содержать только {expected.__name__}: {item!r}")
    return items


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
            object.__setattr__(self, "confidence", _validate_confidence(self.confidence))


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

        nodes = _as_tuple(self.nodes, SemanticNode, "nodes")
        edges = _as_tuple(self.edges, SemanticEdge, "edges")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

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


@dataclass(frozen=True, slots=True)
class IntentDecision:
    """Результат классификации намерения пользователя."""

    category: IntentCategory
    confidence: float
    reason: str
    needs_clarification: bool
    clarification_question: str | None = None
    contract_version: str = LLM_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _validate_contract_version(self.contract_version)
        category = _coerce_intent_category(self.category)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "confidence", _validate_confidence(self.confidence))
        _require_non_empty_text(self.reason, "reason")

        if not isinstance(self.needs_clarification, bool):
            raise TypeError(
                f"needs_clarification должен быть bool: {self.needs_clarification!r}"
            )

        requires_clarification = category is IntentCategory.CLARIFICATION_REQUIRED
        if self.needs_clarification is not requires_clarification:
            raise ValueError(
                "needs_clarification должен быть true ровно для категории "
                "CLARIFICATION_REQUIRED"
            )

        if requires_clarification:
            _require_non_empty_text(
                self.clarification_question,
                "clarification_question",
            )
        elif self.clarification_question is not None:
            raise ValueError(
                "clarification_question допустим только для категории "
                "CLARIFICATION_REQUIRED"
            )


@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    """Кандидат сущности из ответа LLM; local_id действует только внутри ответа."""

    local_id: str
    name: str
    type: NodeType
    mention: str
    description: str | None = None
    value_kind: ValueKind | None = None
    value: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_text(self.local_id, "local_id")
        _require_non_empty_text(self.name, "name")
        node_type = _coerce_node_type(self.type)
        object.__setattr__(self, "type", node_type)
        _require_non_empty_text(self.mention, "mention")

        if self.description is not None:
            _require_non_empty_text(self.description, "description")

        if node_type is NodeType.VALUE:
            if self.value_kind is None or self.value is None:
                raise ValueError(
                    "для сущности type='value' обязательны value_kind и value"
                )
            value_kind = _coerce_value_kind(self.value_kind)
            object.__setattr__(self, "value_kind", value_kind)
            _validate_value(value_kind, self.value)
        elif self.value_kind is not None or self.value is not None:
            raise ValueError(
                "value_kind и value допустимы только для сущности type='value'"
            )


@dataclass(frozen=True, slots=True)
class ExtractedRelation:
    """Кандидат связи между сущностями одного ответа LLM."""

    local_id: str
    source_local_id: str
    target_local_id: str
    relation: str
    evidence: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        _require_non_empty_text(self.local_id, "local_id")
        _require_non_empty_text(self.source_local_id, "source_local_id")
        _require_non_empty_text(self.target_local_id, "target_local_id")
        validate_relation(self.relation)
        _require_non_empty_text(self.evidence, "evidence")

        if self.confidence is not None:
            object.__setattr__(self, "confidence", _validate_confidence(self.confidence))


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Структурированный результат извлечения знаний из одного источника."""

    entities: tuple[ExtractedEntity, ...] = ()
    relations: tuple[ExtractedRelation, ...] = ()
    warnings: tuple[str, ...] = ()
    contract_version: str = LLM_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _validate_contract_version(self.contract_version)
        entities = _as_tuple(self.entities, ExtractedEntity, "entities")
        relations = _as_tuple(self.relations, ExtractedRelation, "relations")
        warnings = _as_tuple(self.warnings, str, "warnings")
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "warnings", warnings)

        _ensure_unique((entity.local_id for entity in entities), "local_id сущности")
        _ensure_unique((relation.local_id for relation in relations), "local_id связи")

        entity_ids = {entity.local_id for entity in entities}
        for relation in relations:
            missing = {
                node_id
                for node_id in (relation.source_local_id, relation.target_local_id)
                if node_id not in entity_ids
            }
            if missing:
                missing_ids = ", ".join(repr(node_id) for node_id in sorted(missing))
                raise ValueError(
                    f"связь {relation.local_id!r} ссылается на отсутствующие "
                    f"local_id: {missing_ids}"
                )
