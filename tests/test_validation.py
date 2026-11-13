from datetime import date

import pytest

from ringr_agents.validation import (
    InvalidDataError,
    optional_iso_date,
    optional_positive_amount,
    optional_text,
)

TODAY = date(2026, 11, 10)


class TestOptionalIsoDate:
    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_missing_values_are_none(self, value: object) -> None:
        assert optional_iso_date(value, field="d") is None

    def test_valid_date_is_returned_normalized(self) -> None:
        assert optional_iso_date(" 2026-11-20 ", field="d") == "2026-11-20"

    def test_today_is_accepted(self) -> None:
        assert optional_iso_date("2026-11-10", field="d", not_before=TODAY) == "2026-11-10"

    def test_past_date_is_rejected(self) -> None:
        with pytest.raises(InvalidDataError, match="ya ha pasado"):
            optional_iso_date("2026-11-09", field="d", not_before=TODAY)

    @pytest.mark.parametrize("value", ["20261120", "2026/11/20", "20-11-2026", "2026-11-20T10:00"])
    def test_wrong_format_is_rejected(self, value: str) -> None:
        with pytest.raises(InvalidDataError, match="formato no válido"):
            optional_iso_date(value, field="d")

    @pytest.mark.parametrize("value", ["2026-02-30", "2026-13-01", "2026-00-10"])
    def test_impossible_date_is_rejected(self, value: str) -> None:
        with pytest.raises(InvalidDataError, match="no existe"):
            optional_iso_date(value, field="d")

    @pytest.mark.parametrize("value", [20261120, 2026.11, ["2026-11-20"]])
    def test_non_string_is_rejected(self, value: object) -> None:
        with pytest.raises(InvalidDataError, match="se esperaba un texto"):
            optional_iso_date(value, field="d")

    def test_error_carries_field_name(self) -> None:
        with pytest.raises(InvalidDataError) as error:
            optional_iso_date("nope", field="commitment_date")
        assert error.value.field == "commitment_date"


class TestOptionalPositiveAmount:
    @pytest.mark.parametrize("value", [None, "", "  "])
    def test_missing_values_are_none(self, value: object) -> None:
        assert optional_positive_amount(value, field="a") is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (150, 150.0),
            (150.5, 150.5),
            (0.1, 0.1),
            (10.005, 10.01),  # redondeo half-up, no el "banker's rounding" de round()
            (99.994, 99.99),
            ("150", 150.0),
            ("150.5", 150.5),
            ("150,50", 150.5),
            (" 75,9 ", 75.9),
            (1_000_000_000, 1e9),  # el máximo admitido
            ("1000000000,00", 1e9),
        ],
    )
    def test_valid_amounts_are_normalized(self, value: object, expected: float) -> None:
        assert optional_positive_amount(value, field="a") == expected

    @pytest.mark.parametrize("value", [0, 0.0, -5, -0.01, 0.004])
    def test_zero_or_negative_is_rejected(self, value: float) -> None:
        with pytest.raises(InvalidDataError, match="mayor que 0"):
            optional_positive_amount(value, field="a")

    @pytest.mark.parametrize("value", [True, False])
    def test_booleans_are_rejected(self, value: bool) -> None:
        with pytest.raises(InvalidDataError, match="booleano"):
            optional_positive_amount(value, field="a")

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_is_rejected(self, value: float) -> None:
        with pytest.raises(InvalidDataError, match="finito"):
            optional_positive_amount(value, field="a")

    @pytest.mark.parametrize("value", ["1.500", "1.500,50", "150€", "-5", "cien", "1e3"])
    def test_ambiguous_or_malformed_text_is_rejected(self, value: str) -> None:
        with pytest.raises(InvalidDataError, match="formato de importe"):
            optional_positive_amount(value, field="a")

    # 10**5000 es un 1 con 5000 ceros. Lleva `id` porque pytest no puede
    # escribirlo como texto para nombrar el test (pasa de 4300 cifras).
    @pytest.mark.parametrize("value", [1e30, pytest.param(10**5000, id="10**5000")])
    def test_huge_number_is_rejected(self, value: float) -> None:
        with pytest.raises(InvalidDataError, match="fuera de rango"):
            optional_positive_amount(value, field="a")

    @pytest.mark.parametrize("value", [1_000_000_000.01, 10**12, 1e16, "1000000000,01"])
    def test_amount_above_the_maximum_is_rejected(self, value: object) -> None:
        with pytest.raises(InvalidDataError, match="máximo"):
            optional_positive_amount(value, field="a")

    @pytest.mark.parametrize("value", [[150], {"amount": 150}])
    def test_other_types_are_rejected(self, value: object) -> None:
        with pytest.raises(InvalidDataError, match="se esperaba un número"):
            optional_positive_amount(value, field="a")


class TestOptionalText:
    @pytest.mark.parametrize("value", [None, "", " \n\t "])
    def test_missing_values_are_none(self, value: object) -> None:
        assert optional_text(value, field="t") is None

    def test_text_is_stripped(self) -> None:
        assert optional_text("  Cambiar tarifa \n", field="t") == "Cambiar tarifa"

    def test_non_string_is_rejected(self) -> None:
        with pytest.raises(InvalidDataError, match="se esperaba un texto"):
            optional_text(42, field="t")
