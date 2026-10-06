"""Canonical names and checkpoint naming metadata for 4D-VGGT."""

from __future__ import annotations

import argparse
import json
from typing import Final


DISPLAY_NAME: Final = "4D-VGGT"
DISTRIBUTION_NAME: Final = "4d-vggt"
IMPORT_PACKAGE: Final = "four_d_vggt"
CLI_NAME: Final = "4d-vggt"
GEOMETRY_PROFILE: Final = "4d-vggt-geometry-v1"
ARCHITECTURE_SCHEMA_VERSION: Final = 2
CHECKPOINT_NAMING_SCHEMA_VERSION: Final = 2


def checkpoint_metadata() -> dict[str, str | int]:
    """Return the required naming fields for newly saved checkpoints."""
    return {
        "model_name": DISPLAY_NAME,
        "naming_schema_version": CHECKPOINT_NAMING_SCHEMA_VERSION,
        "distribution_name": DISTRIBUTION_NAME,
        "import_package": IMPORT_PACKAGE,
        "release": "version-1.0",
    }


def main() -> None:
    """Expose the canonical project metadata through the 4d-vggt CLI."""
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=f"{DISPLAY_NAME} command line interface",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("metadata", help="print canonical project metadata")
    args = parser.parse_args()

    if args.command == "metadata":
        print(json.dumps(checkpoint_metadata(), indent=2, sort_keys=True))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
