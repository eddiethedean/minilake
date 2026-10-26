"""Resolve Unity Catalog volume URIs to their on-disk storage."""

import shutil
from pathlib import Path

from minilake.config import settings
from minilake.errors import DatabricksError

_VOLUMES_PREFIX = "/Volumes"


def _merge_legacy_volume(legacy_path: Path, volume_root: Path) -> bool:
    """Move legacy Files/DBFS volume data into its UC volume directory.

    Preflight the entire tree before moving anything. A conflicting path leaves
    both copies untouched and stops startup with a message for the operator.
    """
    legacy_dbfs_root = settings.data_dir / "dbfs"
    current = legacy_dbfs_root
    if current.is_symlink():
        raise RuntimeError(f"Cannot migrate legacy volume data through symlink {current}")
    for part in legacy_path.relative_to(legacy_dbfs_root).parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"Cannot migrate legacy volume data through symlink {current}")

    if not legacy_path.exists():
        return False
    if not legacy_path.is_dir():
        raise RuntimeError(f"Cannot migrate legacy volume data at {legacy_path}: expected a directory")

    def check(source: Path, destination: Path) -> None:
        for child in source.iterdir():
            target = destination / child.name
            if child.is_symlink():
                raise RuntimeError(f"Cannot migrate legacy volume data: symlink at {child}")
            if target.exists() or target.is_symlink():
                if not child.is_dir() or not target.is_dir() or target.is_symlink():
                    raise RuntimeError(
                        f"Cannot migrate legacy volume data: both legacy and current paths contain {target}"
                    )
            if child.is_dir():
                check(child, target)

    check(legacy_path, volume_root)

    def move_contents(source: Path, destination: Path) -> None:
        for child in source.iterdir():
            target = destination / child.name
            if child.is_dir():
                target.mkdir(exist_ok=True)
                move_contents(child, target)
                child.rmdir()
            else:
                shutil.move(str(child), str(target))

    move_contents(legacy_path, volume_root)
    legacy_path.rmdir()
    return True


def migrate_legacy_volume_data() -> int:
    """Run the one-time migration of old `/Volumes` uploads into UC directories."""
    volumes_root = (settings.data_dir / "volumes").resolve()
    migration_marker = volumes_root / ".legacy_volume_migration_complete"
    if migration_marker.exists():
        return 0

    volumes_root.mkdir(parents=True, exist_ok=True)
    migrated = 0
    for catalog_path in volumes_root.iterdir():
        if catalog_path.is_symlink() or not catalog_path.is_dir():
            continue
        for schema_path in catalog_path.iterdir():
            if schema_path.is_symlink() or not schema_path.is_dir():
                continue
            for volume_root in schema_path.iterdir():
                if volume_root.is_symlink() or not volume_root.is_dir():
                    continue
                legacy_path = (
                    settings.data_dir / "dbfs" / "Volumes" / catalog_path.name / schema_path.name / volume_root.name
                )
                if _merge_legacy_volume(legacy_path, volume_root):
                    migrated += 1

    # Keep later volume creations from adopting files left by volumes deleted
    # before this upgrade. The marker lives inside volumes/ so a full reset
    # removes it along with the migrated storage.
    migration_marker.touch()
    return migrated


def resolve_volume_path(path: str) -> Path | None:
    """Resolve a `/Volumes/...` URI to its Unity Catalog volume directory.

    Returns ``None`` for paths outside the volume namespace so Files and DBFS can
    use their ordinary shared root. Volume URIs must name an existing volume;
    this prevents deleted or unknown volumes from becoming generic DBFS paths.
    """
    normalized = "/" + path.lstrip("/")
    if normalized != _VOLUMES_PREFIX and not normalized.startswith(_VOLUMES_PREFIX + "/"):
        return None

    suffix = normalized[len(_VOLUMES_PREFIX) :]
    if suffix.startswith("/"):
        suffix = suffix[1:]
    if suffix.endswith("/"):
        suffix = suffix[:-1]
    parts = suffix.split("/")
    if len(parts) < 3 or any(part in {"", ".", ".."} for part in parts[:3]):
        raise DatabricksError(error_code="INVALID_REQUEST", message="Invalid volume path", status_code=400)

    catalog_name, schema_name, volume_name = parts[:3]
    volumes_root = (settings.data_dir / "volumes").resolve()
    volume_root = volumes_root.joinpath(catalog_name, schema_name, volume_name)

    # The catalog/schema/volume components identify the volume itself; do not
    # follow a symlink there to a different volume or outside the volume tree.
    current = volumes_root
    for part in (catalog_name, schema_name, volume_name):
        current = current / part
        if current.is_symlink():
            raise DatabricksError(error_code="INVALID_REQUEST", message="Invalid volume path", status_code=400)

    volume_root = volume_root.resolve()
    if volumes_root not in volume_root.parents or not volume_root.is_dir():
        full_name = f"{catalog_name}.{schema_name}.{volume_name}"
        raise DatabricksError(
            error_code="RESOURCE_DOES_NOT_EXIST",
            message=f"Volume '{full_name}' not found",
            status_code=404,
        )

    target = volume_root.joinpath(*parts[3:]).resolve()
    if target != volume_root and volume_root not in target.parents:
        raise DatabricksError(error_code="INVALID_REQUEST", message="Invalid path", status_code=400)
    return target
