import pytest

from app.sql.validator import SQLValidationError, validate_select_only


def test_allows_select():
    assert validate_select_only("SELECT id FROM invoices LIMIT 10") == (
        "SELECT id FROM invoices LIMIT 10"
    )


def test_allows_with():
    sql = "WITH x AS (SELECT 1 AS n) SELECT * FROM x"
    assert validate_select_only(sql) == sql


def test_rejects_insert():
    with pytest.raises(SQLValidationError):
        validate_select_only("INSERT INTO invoices VALUES (1)")


def test_rejects_multi_statement():
    with pytest.raises(SQLValidationError):
        validate_select_only("SELECT 1; DROP TABLE invoices")


def test_rejects_drop():
    with pytest.raises(SQLValidationError):
        validate_select_only("DROP TABLE invoices")