from __future__ import annotations

from .catalog import format_catalog_lines


def main() -> int:
    print("=== Example Model Zoo (Source References) ===")
    for line in format_catalog_lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
