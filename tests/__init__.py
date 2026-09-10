"""Test package.

An ``__init__.py`` so that shared helpers in ``conftest.py`` -- the config-writing
fixture builder and the reference-docs skip marker -- can be imported by name. Marks in
particular cannot be fixtures, so they have to be importable.
"""
