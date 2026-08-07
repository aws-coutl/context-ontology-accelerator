# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for xsd_for_column numeric type inference."""

import pytest
from coa_ontology.inducer.strategies.base import xsd_for, xsd_for_column
from rdflib import XSD

pytestmark = pytest.mark.unit


class TestXsdFor:
    """Original xsd_for behavior unchanged."""

    def test_integer(self):
        assert xsd_for("INT") == XSD.integer

    def test_varchar(self):
        assert xsd_for("VARCHAR(255)") == XSD.string

    def test_decimal(self):
        assert xsd_for("DECIMAL(10,2)") == XSD.decimal

    def test_float(self):
        assert xsd_for("FLOAT") == XSD.float

    def test_unknown_defaults_to_string(self):
        assert xsd_for("UNKNOWN_TYPE") == XSD.string


class TestXsdForColumn:
    """xsd_for_column adds column-name heuristic for numeric VARCHAR columns."""

    # Columns that should be inferred as numeric despite VARCHAR type
    @pytest.mark.parametrize(
        "col_name",
        [
            "amount",
            "total",
            "spent",
            "remaining",
            "price",
            "cost",
            "revenue",
            "profit",
            "salary",
            "income",
            "balance",
            "quantity",
            "score",
            "rating",
            "consumption",
            "lat",
            "lng",
            "longitude",
            "latitude",
        ],
    )
    def test_numeric_column_names_override_varchar(self, col_name):
        result = xsd_for_column("VARCHAR(255)", col_name)
        assert result == XSD.decimal, f"{col_name} should be xsd:decimal, got {result}"

    # Compound column names with numeric suffix/prefix
    @pytest.mark.parametrize(
        "col_name",
        [
            "total_amount",
            "net_revenue",
            "budget_spent",
            "account_balance",
            "amount_paid",
            "score_final",
        ],
    )
    def test_compound_numeric_names_override_varchar(self, col_name):
        result = xsd_for_column("VARCHAR(255)", col_name)
        assert result == XSD.decimal, f"{col_name} should be xsd:decimal, got {result}"

    # Columns that should remain string even with VARCHAR
    @pytest.mark.parametrize(
        "col_name",
        [
            "name",
            "description",
            "status",
            "category",
            "type",
            "email",
            "phone",
            "address",
            "country",
            "currency",
        ],
    )
    def test_non_numeric_columns_stay_string(self, col_name):
        result = xsd_for_column("VARCHAR(255)", col_name)
        assert result == XSD.string, f"{col_name} should be xsd:string, got {result}"

    # Already-numeric SQL types should stay numeric regardless of name
    def test_integer_type_ignores_name(self):
        assert xsd_for_column("INT", "name") == XSD.integer

    def test_float_type_ignores_name(self):
        assert xsd_for_column("FLOAT", "description") == XSD.float

    # Edge cases
    def test_empty_column_name(self):
        assert xsd_for_column("VARCHAR", "") == XSD.string

    def test_none_data_type(self):
        assert xsd_for_column("", "amount") == XSD.decimal
