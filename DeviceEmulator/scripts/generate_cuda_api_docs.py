#!/usr/bin/env python3
"""Generate offline CUDA Driver/Runtime API references from headers.

Usage:
  python3 scripts/generate_cuda_api_docs.py --cuda-include /usr/local/cuda/include --out docs/cuda
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def extract_macro_value(lines: Iterable[str], macro: str) -> Optional[str]:
    pattern = re.compile(rf"^#define\s+{re.escape(macro)}\s+(\d+)")
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            return match.group(1)
    return None


def clean_comment(comment_lines: List[str]) -> str:
    cleaned: List[str] = []
    for line in comment_lines:
        stripped = line.strip()
        if stripped.startswith("/**"):
            stripped = stripped[3:]
        if stripped.endswith("*/"):
            stripped = stripped[:-2]
        if stripped.startswith("*"):
            stripped = stripped[1:]
        cleaned.append(stripped.strip())
    text = "\n".join(cleaned).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_function_name(prototype: str) -> Optional[str]:
    before_paren = prototype.split("(", 1)[0]
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", before_paren)
    if not tokens:
        return None
    name = tokens[-1]
    if name in {"CUDARTAPI", "CUDAAPI"}:
        return None
    return name


def parse_header(
    header_path: Path,
    api_macro: str,
) -> Tuple[List[Tuple[str, str, str]], Optional[str]]:
    lines = header_path.read_text(errors="ignore").splitlines()
    entries: List[Tuple[str, str, str]] = []
    version_macro = "CUDA_VERSION" if api_macro == "CUDAAPI" else "CUDART_VERSION"
    version = extract_macro_value(lines, version_macro)

    pending_comment: Optional[List[str]] = None
    pending_comment_end = -10

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("/**"):
            comment_lines = [line]
            i += 1
            while i < len(lines):
                comment_lines.append(lines[i])
                if "*/" in lines[i]:
                    break
                i += 1
            pending_comment = comment_lines
            pending_comment_end = i
            i += 1
            continue

        if pending_comment and stripped and (i - pending_comment_end) > 3:
            pending_comment = None

        if (
            api_macro in line
            and not stripped.startswith("#")
            and "typedef" not in stripped
        ):
            proto_lines = [line.rstrip()]
            j = i + 1
            while j < len(lines) and ";" not in proto_lines[-1]:
                proto_lines.append(lines[j].rstrip())
                j += 1
            prototype = " ".join(part.strip() for part in proto_lines)
            prototype = re.sub(r"\s+", " ", prototype).strip()
            func_name = extract_function_name(prototype)
            if func_name:
                doc = ""
                if pending_comment and (i - pending_comment_end) <= 3:
                    doc = clean_comment(pending_comment)
                entries.append((func_name, prototype, doc))
            pending_comment = None
            i = j
            continue

        i += 1

    return entries, version


def write_api_doc(
    output_path: Path,
    title: str,
    header_path: Path,
    version: Optional[str],
    entries: List[Tuple[str, str, str]],
) -> None:
    header_line = f"Generated from `{header_path}`"
    if version:
        header_line += f" (version {version})"

    lines: List[str] = [f"# {title}", "", f"_{header_line}._", ""]
    lines.append(
        "This file is auto-generated from NVIDIA CUDA headers. "
        "See the CUDA license for redistribution terms."
    )
    lines.append("")
    lines.append(f"Total functions: **{len(entries)}**")
    lines.append("")

    for name, prototype, doc in entries:
        lines.append(f"## {name}")
        lines.append("")
        if doc:
            lines.append(doc)
            lines.append("")
        lines.append("```c")
        lines.append(prototype)
        lines.append("```")
        lines.append("")

    output_path.write_text("\n".join(lines))


def write_readme(
    output_path: Path,
    driver_version: Optional[str],
    runtime_version: Optional[str],
    driver_count: int,
    runtime_count: int,
) -> None:
    lines = [
        "# CUDA API Offline Reference",
        "",
        "This directory contains auto-generated Markdown references for the CUDA",
        "Driver API and Runtime API. It is intended for offline search by agents",
        "without relying on web access.",
        "",
        "## Contents",
        "",
        (
            "- [CUDA Driver API](driver_api.md)"
            f" — {driver_count} functions"
            + (f" (CUDA_VERSION={driver_version})" if driver_version else "")
        ),
        (
            "- [CUDA Runtime API](runtime_api.md)"
            f" — {runtime_count} functions"
            + (f" (CUDART_VERSION={runtime_version})" if runtime_version else "")
        ),
        "",
        "## Regenerate",
        "",
        "Run inside the CUDA-enabled environment (Docker image) so the headers",
        "match the CUDA version:",
        "",
        "```bash",
        "python3 scripts/generate_cuda_api_docs.py \\",
        "  --cuda-include /usr/local/cuda/include \\",
        "  --out docs/cuda",
        "```",
        "## Notes",
        "",
        "- The content is derived from NVIDIA CUDA headers; ensure your usage",
        "  complies with the CUDA license agreement.",
        "- Use repository search (e.g., `cuInit`, `cudaMalloc`) for quick lookup.",
    ]
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CUDA Driver/Runtime API Markdown references."
    )
    parser.add_argument(
        "--cuda-include",
        default="/usr/local/cuda/include",
        help="CUDA include directory containing cuda.h and cuda_runtime_api.h",
    )
    parser.add_argument(
        "--out",
        default="docs/cuda",
        help="Output directory for generated docs",
    )
    args = parser.parse_args()

    include_dir = Path(args.cuda_include)
    driver_header = include_dir / "cuda.h"
    runtime_header = include_dir / "cuda_runtime_api.h"

    if not driver_header.exists():
        raise FileNotFoundError(f"Missing driver header: {driver_header}")
    if not runtime_header.exists():
        raise FileNotFoundError(f"Missing runtime header: {runtime_header}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    driver_entries, driver_version = parse_header(driver_header, "CUDAAPI")
    runtime_entries, runtime_version = parse_header(runtime_header, "CUDARTAPI")

    write_api_doc(
        out_dir / "driver_api.md",
        "CUDA Driver API",
        driver_header,
        driver_version,
        driver_entries,
    )
    write_api_doc(
        out_dir / "runtime_api.md",
        "CUDA Runtime API",
        runtime_header,
        runtime_version,
        runtime_entries,
    )
    write_readme(
        out_dir / "README.md",
        driver_version,
        runtime_version,
        len(driver_entries),
        len(runtime_entries),
    )

    print("Generated CUDA API docs:")
    print(f"- {out_dir / 'README.md'}")
    print(f"- {out_dir / 'driver_api.md'}")
    print(f"- {out_dir / 'runtime_api.md'}")


if __name__ == "__main__":
    main()