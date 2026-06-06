def add(a, b):
    return a + b

def test_addition():
    assert add(2, 3) == 6

def test_subtraction():
    result = 10 - 4
    assert result == 6

def test_string():
    assert "hello".upper() == "HELLO"
