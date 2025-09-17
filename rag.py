"""
Backward-compatible CLI shim.

Delegates to the modular package entrypoint at rag/cli.py
while coexisting with the package directory named 'rag'.
"""
import importlib
import pathlib
import sys
import types

if __name__ == "__main__":

    pkg_dir = pathlib.Path(__file__).parent / "rag"

    if "rag" not in sys.modules:
        rag_pkg = types.ModuleType("rag")
        # Mark as a package by setting __path__
        rag_pkg.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
        sys.modules["rag"] = rag_pkg

    # Now import and run the real CLI
    cli = importlib.import_module("rag.cli")
    cli.main()
