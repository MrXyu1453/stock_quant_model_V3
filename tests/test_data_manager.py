import pytest
from app.data_manager import get_and_process_data, validate_code

def test_validate_code():
    assert validate_code('600519.SH') == True
    assert validate_code('000001.SZ') == True
    assert validate_code('123456') == True
    assert validate_code('abcdefg') == False