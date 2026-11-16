"""Keep default smoke outputs separate from formal experiments."""

from pathlib import Path


def is_smoke_path(path):
    """Identify data or results already placed under the smoke directory."""
    return Path(path).resolve().is_relative_to(Path("outputs/smoke").resolve())


def resolve_output_root(configured, override=None, *, smoke=False):
    """Honor an explicit output path; otherwise route smoke runs to their own root."""
    if override is not None:
        return Path(override)
    output = Path(configured)
    if not smoke:
        return output
    try:
        relative = output.resolve().relative_to(Path("outputs/experiment").resolve())
    except ValueError:
        relative = Path(output.name)
    if relative.name == "..":
        raise ValueError("A smoke output requires a named destination")
    return Path("outputs/smoke") / relative
