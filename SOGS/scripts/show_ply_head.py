#!/usr/bin/env python3
"""Show the first N lines of a .PLY file.

- If the PLY is ASCII, prints the first N text lines.
- If the PLY is binary (little or big endian), prints the header and a hex/byte preview of the following bytes.

Usage:
    playground/outputs/concatenated_output.ply

"""

import argparse
import sys
from pathlib import Path

DEFAULT_LINES = 100


def is_ascii_ply(header_text: str) -> bool:
    # PLY header contains a line `format ascii 1.0` for ASCII files.
    return "format ascii" in header_text.lower()


def read_ply_header(f) -> str:
    # Read lines until 'end_header' (inclusive) or EOF.
    header_lines = []
    while True:
        line = f.readline()
        if not line:
            break
        # If opened in binary mode, decode bytes to str for header parsing
        if isinstance(line, bytes):
            try:
                s = line.decode("utf-8", errors="replace")
            except Exception:
                s = line.decode("latin-1", errors="replace")
        else:
            s = line
        header_lines.append(s)
        if s.strip().lower() == "end_header":
            break
    return "".join(header_lines)


def show_ascii_head(path: Path, lines: int) -> None:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i in range(lines):
            line = f.readline()
            if not line:
                break
            print(line.rstrip("\n"))


def show_binary_preview(path: Path, header: str, preview_bytes: int = 256) -> None:
    print(header.rstrip("\n"))
    print("\n[Binary PLY detected] Showing next {} bytes as hex preview:".format(preview_bytes))
    with path.open("rb") as f:
        # skip header bytes
        # we need to find the end of header (b'end_header') in the file
        data = f.read()
        idx = data.lower().find(b"end_header")
        if idx >= 0:
            # move past the end_header line
            # find the newline after end_header
            rem = data[idx:]
            # find first newline after end_header
            nl = rem.find(b"\n")
            start = idx + nl + 1 if nl >= 0 else idx + len(b"end_header")
        else:
            start = 0
        preview = data[start : start + preview_bytes]
        # print hex groups
        hex_str = " ".join(f"{b:02x}" for b in preview)
        print(hex_str)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Show the first N lines of a .PLY file (handles ASCII and binary headers).")
    parser.add_argument("path", type=Path, help="Path to .ply file")
    parser.add_argument("-n", "--lines", type=int, default=DEFAULT_LINES, help="Number of lines to show for ASCII PLY (default: 100)")
    parser.add_argument("--preview-bytes", type=int, default=256, help="When binary, how many bytes to show as hex preview (default: 256)")
    args = parser.parse_args(argv)

    path = args.path
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    # Try to open as text first and read header
    try:
        with path.open("rb") as f:
            header = read_ply_header(f)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    if is_ascii_ply(header):
        show_ascii_head(path, args.lines)
    else:
        # binary PLY: print header (already read) then preview bytes
        print(header.rstrip("\n"))
        print()
        show_binary_preview(path, header, preview_bytes=args.preview_bytes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
