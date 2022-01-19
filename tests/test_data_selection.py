
import pytest

from src.data_selection import FieldHelper

def test_FieldHelper():

    dpath_schema = '../data/schema'
    field_helper = FieldHelper(dpath_schema)

    assert set(field_helper.get_fields_from_categories([345])) == set()

    assert set(field_helper.get_fields_from_categories([120])) == {
        20138, 20240,
    }

    assert set(field_helper.get_fields_from_categories([1001])) == {
        31, 21003, 34, 52, 54, 53, 21000, 189,
    }

    assert set(field_helper.get_fields_from_categories([100083])) == {
        30512, 30513, 30502, 30503, 30522, 30523, 30532, 30533,
        30510, 30515, 30500, 30505, 30520, 30525, 30530, 30535,
    }

    assert len(field_helper.get_fields_from_categories([100026, 116])) == 108 + 56
