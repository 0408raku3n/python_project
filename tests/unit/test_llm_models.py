"""Модульные тесты доменных моделей ответов LLM."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from semantic_knowledge.domain.models import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    IntentCategory,
    IntentDecision,
    NodeType,
    ValueKind,
)


def make_entity(local_id: str = "e1") -> ExtractedEntity:
    return ExtractedEntity(
        local_id=local_id,
        name="Анна Смирнова",
        type=NodeType.PERSON,
        mention="Анной Смирновой",
    )


def make_relation(
    local_id: str = "r1",
    source_local_id: str = "e1",
    target_local_id: str = "e2",
) -> ExtractedRelation:
    return ExtractedRelation(
        local_id=local_id,
        source_local_id=source_local_id,
        target_local_id=target_local_id,
        relation="works_at",
        evidence="Анна работает в компании «Север»",
    )


def test_intent_decision_accepts_valid_non_clarification() -> None:
    decision = IntentDecision(
        category="QUESTION",
        confidence=0.9,
        reason="Пользователь задаёт вопрос",
        needs_clarification=False,
    )

    assert decision.category is IntentCategory.QUESTION
    assert decision.confidence == pytest.approx(0.9)
    assert decision.contract_version == "1"
    assert decision.clarification_question is None


def test_intent_decision_accepts_clarification() -> None:
    decision = IntentDecision(
        category=IntentCategory.CLARIFICATION_REQUIRED,
        confidence=0.4,
        reason="Неясно, какой объект изменить",
        needs_clarification=True,
        clarification_question="Какой объект вы хотите изменить?",
    )

    assert decision.category is IntentCategory.CLARIFICATION_REQUIRED
    assert decision.needs_clarification is True


def test_intent_decision_rejects_unknown_category() -> None:
    with pytest.raises(ValueError, match="категория намерения"):
        IntentDecision(
            category="OTHER",  # type: ignore[arg-type]
            confidence=0.5,
            reason="Неизвестно",
            needs_clarification=False,
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_intent_decision_rejects_invalid_confidence(confidence: object) -> None:
    with pytest.raises((TypeError, ValueError), match="confidence"):
        IntentDecision(
            category=IntentCategory.QUESTION,
            confidence=confidence,  # type: ignore[arg-type]
            reason="Вопрос",
            needs_clarification=False,
        )


@pytest.mark.parametrize("reason", ["", "   ", None])
def test_intent_decision_rejects_empty_reason(reason: object) -> None:
    with pytest.raises((TypeError, ValueError), match="reason"):
        IntentDecision(
            category=IntentCategory.QUESTION,
            confidence=0.5,
            reason=reason,  # type: ignore[arg-type]
            needs_clarification=False,
        )


@pytest.mark.parametrize(
    ("category", "needs_clarification", "question", "match"),
    [
        (IntentCategory.CLARIFICATION_REQUIRED, False, "Уточните", "needs_clarification"),
        (IntentCategory.QUESTION, True, None, "needs_clarification"),
        (IntentCategory.CLARIFICATION_REQUIRED, True, None, "clarification_question"),
        (IntentCategory.CLARIFICATION_REQUIRED, True, "   ", "clarification_question"),
        (IntentCategory.QUESTION, False, "Уточните", "clarification_question"),
    ],
)
def test_intent_decision_rejects_inconsistent_clarification(
    category: IntentCategory,
    needs_clarification: bool,
    question: str | None,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        IntentDecision(
            category=category,
            confidence=0.5,
            reason="Причина",
            needs_clarification=needs_clarification,
            clarification_question=question,
        )


@pytest.mark.parametrize("contract_version", ["2", 1, None])
def test_intent_decision_rejects_invalid_contract_version(
    contract_version: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="contract_version"):
        IntentDecision(
            category=IntentCategory.QUESTION,
            confidence=0.5,
            reason="Вопрос",
            needs_clarification=False,
            contract_version=contract_version,  # type: ignore[arg-type]
        )


def test_intent_decision_is_frozen() -> None:
    decision = IntentDecision(
        category=IntentCategory.QUESTION,
        confidence=0.5,
        reason="Вопрос",
        needs_clarification=False,
    )

    with pytest.raises(FrozenInstanceError):
        decision.category = IntentCategory.UNKNOWN  # type: ignore[misc]


def test_extracted_entity_accepts_valid_entity() -> None:
    entity = ExtractedEntity(
        local_id="e1",
        name="Анна Смирнова",
        type="person",
        mention="Анной Смирновой",
        description="Основатель компании",
    )

    assert entity.type is NodeType.PERSON
    assert entity.value_kind is None


def test_extracted_entity_accepts_value() -> None:
    entity = ExtractedEntity(
        local_id="e1",
        name="2018 год",
        type=NodeType.VALUE,
        mention="2018 году",
        value_kind=ValueKind.YEAR,
        value="2018",
    )

    assert entity.value_kind is ValueKind.YEAR
    assert entity.value == "2018"


def test_extracted_entity_requires_value_fields() -> None:
    with pytest.raises(ValueError, match="value_kind и value"):
        ExtractedEntity(
            local_id="e1",
            name="2018 год",
            type=NodeType.VALUE,
            mention="2018 году",
        )


def test_extracted_entity_validates_value() -> None:
    with pytest.raises(ValueError, match="value"):
        ExtractedEntity(
            local_id="e1",
            name="2018 год",
            type=NodeType.VALUE,
            mention="2018 году",
            value_kind=ValueKind.YEAR,
            value="18",
        )


def test_extracted_entity_forbids_value_fields_for_regular_types() -> None:
    with pytest.raises(ValueError, match="только для сущности"):
        ExtractedEntity(
            local_id="e1",
            name="Анна",
            type=NodeType.PERSON,
            mention="Анной",
            value_kind=ValueKind.TEXT,
            value="активен",
        )


@pytest.mark.parametrize("field", ["local_id", "name", "mention", "description"])
def test_extracted_entity_rejects_empty_text_fields(field: str) -> None:
    values: dict[str, object] = {
        "local_id": "e1",
        "name": "Анна",
        "type": NodeType.PERSON,
        "mention": "Анной",
        "description": "Основатель",
    }
    values[field] = ""

    with pytest.raises(ValueError, match=field):
        ExtractedEntity(**values)  # type: ignore[arg-type]


def test_extracted_entity_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="тип узла"):
        ExtractedEntity(
            local_id="e1",
            name="Неизвестно",
            type="unknown",  # type: ignore[arg-type]
            mention="Неизвестно",
        )


def test_extracted_relation_accepts_metadata() -> None:
    relation = ExtractedRelation(
        local_id="r1",
        source_local_id="e1",
        target_local_id="e2",
        relation="works_at",
        evidence="Анна работает в компании «Север»",
        confidence=1,
    )

    assert relation.confidence == pytest.approx(1.0)
    assert relation.evidence == "Анна работает в компании «Север»"


@pytest.mark.parametrize("relation", ["works at", "Works_at", "1works", "_works", ""])
def test_extracted_relation_rejects_invalid_relation(relation: str) -> None:
    with pytest.raises(ValueError, match="relation"):
        ExtractedRelation(
            local_id="r1",
            source_local_id="e1",
            target_local_id="e2",
            relation=relation,
            evidence="Анна работает",
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_extracted_relation_rejects_invalid_confidence(confidence: object) -> None:
    with pytest.raises((TypeError, ValueError), match="confidence"):
        ExtractedRelation(
            local_id="r1",
            source_local_id="e1",
            target_local_id="e2",
            relation="works_at",
            evidence="Анна работает",
            confidence=confidence,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field", ["local_id", "source_local_id", "target_local_id", "evidence"])
def test_extracted_relation_rejects_empty_text_fields(field: str) -> None:
    values: dict[str, object] = {
        "local_id": "r1",
        "source_local_id": "e1",
        "target_local_id": "e2",
        "relation": "works_at",
        "evidence": "Анна работает",
    }
    values[field] = ""

    with pytest.raises(ValueError, match=field):
        ExtractedRelation(**values)  # type: ignore[arg-type]


def test_extraction_result_accepts_entities_and_relations() -> None:
    entities = [
        make_entity("e1"),
        ExtractedEntity(
            local_id="e2",
            name="Компания «Север»",
            type=NodeType.ORGANIZATION,
            mention="компании «Север»",
        ),
    ]
    relation = make_relation()
    result = ExtractionResult(entities=entities, relations=[relation])

    assert isinstance(result.entities, tuple)
    assert isinstance(result.relations, tuple)
    assert result.entities == tuple(entities)
    assert result.relations == (relation,)
    assert result.contract_version == "1"


def test_extraction_result_allows_empty_result() -> None:
    result = ExtractionResult()

    assert result.entities == ()
    assert result.relations == ()
    assert result.warnings == ()


def test_extraction_result_rejects_duplicate_entity_ids() -> None:
    with pytest.raises(ValueError, match="local_id сущности"):
        ExtractionResult(entities=(make_entity("e1"), make_entity("e1")))


def test_extraction_result_rejects_duplicate_relation_ids() -> None:
    entities = (make_entity("e1"), make_entity("e2"))
    relations = (make_relation("r1"), make_relation("r1"))

    with pytest.raises(ValueError, match="local_id связи"):
        ExtractionResult(entities=entities, relations=relations)


@pytest.mark.parametrize(
    ("source_local_id", "target_local_id"),
    [("missing", "e2"), ("e1", "missing")],
)
def test_extraction_result_rejects_missing_entity_references(
    source_local_id: str,
    target_local_id: str,
) -> None:
    entities = (make_entity("e1"), make_entity("e2"))
    relation = make_relation(
        source_local_id=source_local_id,
        target_local_id=target_local_id,
    )

    with pytest.raises(ValueError, match="отсутствующие"):
        ExtractionResult(entities=entities, relations=(relation,))


def test_extraction_result_allows_self_and_parallel_relations() -> None:
    entity = make_entity("e1")
    self_relation = make_relation("r1", "e1", "e1")
    parallel_relation = make_relation("r2", "e1", "e1")

    result = ExtractionResult(
        entities=(entity,),
        relations=(self_relation, parallel_relation),
    )

    assert result.relations == (self_relation, parallel_relation)


def test_extraction_result_rejects_wrong_collection_items() -> None:
    with pytest.raises(TypeError, match="entities"):
        ExtractionResult(entities=(object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="relations"):
        ExtractionResult(relations=(make_entity(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="warnings"):
        ExtractionResult(warnings=(1,))  # type: ignore[arg-type]


@pytest.mark.parametrize("contract_version", ["2", 1, None])
def test_extraction_result_rejects_invalid_contract_version(
    contract_version: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="contract_version"):
        ExtractionResult(contract_version=contract_version)  # type: ignore[arg-type]


def test_extraction_result_is_frozen() -> None:
    result = ExtractionResult()

    with pytest.raises(FrozenInstanceError):
        result.warnings = ("warning",)  # type: ignore[misc]
