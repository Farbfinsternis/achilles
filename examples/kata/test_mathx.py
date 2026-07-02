"""A tiny failing test so you can watch the act -> verify -> commit loop work
end to end without any real project. Run from this folder:

    python -m pytest -q          # red: mathx does not exist yet
    python ../../__main__.py "make the failing tests pass" --verify "python -m pytest -q" -y

Achilles should plan a step or two, write mathx.py, and turn the bar green.
"""

from mathx import add, is_even


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_is_even():
    assert is_even(4) is True
    assert is_even(7) is False
