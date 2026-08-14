"""Placing conftest.py at the project root puts the root on sys.path, so tests
can `import model.config`. Without it pytest only adds tests/ to the path and the
package import fails.
"""
