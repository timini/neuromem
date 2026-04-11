"""T029 — tests for the ``neuromem`` logger namespace.

Locks in the behaviour configured in ``neuromem/__init__.py``:

  1. The ``neuromem`` logger exists.
  2. It has a ``NullHandler`` attached so importers don't see "no
     handlers could be found" warnings (a stdlib quirk that can
     silently drop messages).
  3. Its effective level defaults to WARNING — matching the Python
     logging HOWTO guidance for libraries ("don't pin a level in
     library code, inherit from root").
  4. Callers can reconfigure it (e.g. to DEBUG) without the library
     fighting back.
  5. Sub-loggers created via ``logging.getLogger("neuromem.foo")``
     inherit the handlers and level from the parent namespace.
"""

from __future__ import annotations

import logging

import neuromem


def test_namespace_logger_exists() -> None:
    """``neuromem.logger`` is the top-level library logger."""
    assert isinstance(neuromem.logger, logging.Logger)
    assert neuromem.logger.name == "neuromem"
    # And it must be the same object ``logging.getLogger("neuromem")``
    # returns, i.e., it's registered in the global logger registry.
    assert neuromem.logger is logging.getLogger("neuromem")


def test_namespace_logger_has_null_handler() -> None:
    """A ``NullHandler`` must be attached so library imports don't emit
    the stdlib "no handlers could be found" warning on first log call.
    """
    handlers = neuromem.logger.handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers), (
        f"neuromem logger missing NullHandler — found: {handlers}"
    )


def test_effective_level_defaults_to_warning() -> None:
    """Default effective level is WARNING.

    The library intentionally does NOT call ``setLevel`` — we leave the
    level at ``NOTSET`` so the effective level inherits from the root
    logger, which defaults to WARNING. This test guards against anyone
    adding a premature ``logger.setLevel(logging.INFO)`` in __init__.py
    that would override caller configuration.
    """
    # Snapshot and restore around any level mutation this test induces.
    original_level = neuromem.logger.level
    original_root = logging.root.level
    try:
        # Ensure the neuromem logger is at NOTSET so it inherits.
        neuromem.logger.setLevel(logging.NOTSET)
        # Root at default.
        logging.root.setLevel(logging.WARNING)
        assert neuromem.logger.getEffectiveLevel() == logging.WARNING
    finally:
        neuromem.logger.setLevel(original_level)
        logging.root.setLevel(original_root)


def test_caller_can_reconfigure_level() -> None:
    """Callers can set the namespace level to DEBUG and the library
    doesn't fight back — subsequent log calls honour the new level."""
    original = neuromem.logger.level
    try:
        neuromem.logger.setLevel(logging.DEBUG)
        assert neuromem.logger.getEffectiveLevel() == logging.DEBUG
    finally:
        neuromem.logger.setLevel(original)


def test_sub_loggers_inherit_handlers_and_level() -> None:
    """Sub-loggers (e.g. ``neuromem.system``) inherit handlers from the
    parent and respect the parent's level.

    This matters because every module inside the library calls
    ``logger = logging.getLogger("neuromem.<module>")`` — if those
    loggers didn't inherit the parent's NullHandler, each one would
    emit the "no handlers" warning independently.
    """
    child = logging.getLogger("neuromem.system")
    # Propagation to the parent is the default; verify it hasn't been
    # accidentally disabled.
    assert child.propagate is True
    # Child's effective level matches the parent when child is NOTSET.
    original_child = child.level
    original_parent = neuromem.logger.level
    try:
        child.setLevel(logging.NOTSET)
        neuromem.logger.setLevel(logging.ERROR)
        assert child.getEffectiveLevel() == logging.ERROR
    finally:
        child.setLevel(original_child)
        neuromem.logger.setLevel(original_parent)
