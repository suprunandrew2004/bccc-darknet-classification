import hashlib
import sys
from pathlib import Path

SUMS = Path("./data/SHA256SUMS.txt")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ok = True
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        path = SUMS.parent / name
        if not path.exists():
            print(f"НЕТ ФАЙЛА  {path}")
            ok = False
            continue
        real = sha256(path)
        status = "OK" if real == digest else "НЕ СОВПАДАЕТ"
        ok &= real == digest
        print(f"{status:12s}{path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
