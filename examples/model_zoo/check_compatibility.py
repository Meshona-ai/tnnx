from __future__ import annotations

from .compatibility import readiness_for_all


def main() -> int:
    print("=== Model Zoo Transpile Readiness ===")
    for item in readiness_for_all():
        status = "ready" if item.ready else "blocked"
        print(f"{item.name}: {status}")
        print(f"  smoke example: {item.example_status}")
        if item.missing_ops:
            print(f"  missing base ops: {', '.join(item.missing_ops)}")
        elif not item.ready:
            print("  missing base ops: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
