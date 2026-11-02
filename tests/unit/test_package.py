"""Проверка доступности установленного пакета."""


def test_package_import() -> None:
    import semantic_knowledge

    assert semantic_knowledge.__name__ == "semantic_knowledge"
