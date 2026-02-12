from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ICARTTInfo:
    path: Path
    header_length: int
    ffi: str


@dataclass(frozen=True)
class VariableDef:
    name: str
    unit: Optional[str] = None
    description: Optional[str] = None
    missing: Optional[float] = None  # per-variable missing, if known


class ICARTTReader:
    """
    General ICARTT/ICT reader.

    Goals:
      - Robustly read the data table (CSV-like) for typical ICT files.
      - Avoid file-specific assumptions (campaign/platform/column names).
      - Provide best-effort metadata parsing (especially for FFI=1001) but
        never let metadata parsing prevent data extraction.

    Notes:
      - Many airborne ICT files are FFI=1001 (1D time series), but other FFIs exist.
      - Header length is always the first token on the first line in the files you've shown.
    """

    def __init__(self, path: PathLike):
        self.path = Path(path)
        self.info = self._read_info()

    # ----------------------------
    # Core: file format info
    # ----------------------------
    def _read_info(self) -> ICARTTInfo:
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            line1 = f.readline().strip()

        parts = [p.strip() for p in line1.split(",")]
        if len(parts) < 2:
            raise ValueError(f"Unexpected ICARTT first line format: {line1!r}")

        header_length = int(parts[0])
        ffi = parts[1]
        return ICARTTInfo(path=self.path, header_length=header_length, ffi=ffi)

    def read_header_lines(self) -> List[str]:
        """Return the raw header lines (including line 1)."""
        n = self.info.header_length
        lines: List[str] = []
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(n):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
        return lines

    # ----------------------------
    # Minimal assumptions: table extraction
    # ----------------------------
    def read_table(
        self,
        *,
        na_values: Optional[List[Union[str, float, int]]] = None,
        strip_colnames: bool = True,
    ) -> pd.DataFrame:
        """
        Extract the data table.

        Strategy:
          - Most ICARTT files place the column header row at line `header_length`.
          - So we skip `header_length - 1` lines and let pandas treat the next line as header.

        This is general and does not depend on campaign/platform.
        """
        skiprows = max(self.info.header_length - 1, 0)

        # Many ICT files use -9999, -99999, etc., but we won't assume; allow caller to pass.
        # We'll also attempt to auto-detect common missing indicators from header if possible.
        if na_values is None:
            na_values = self._guess_missing_values()

        df = pd.read_csv(
            self.path,
            skiprows=skiprows,
            sep=",",
            engine="python",
            na_values=na_values,
        )

        if strip_colnames:
            df.columns = [str(c).strip() for c in df.columns]

        return df

    # ----------------------------
    # Best-effort metadata parsing
    # ----------------------------
    def read_metadata(self) -> Dict[str, str]:
        """
        Best-effort metadata extraction.

        Returns a dict of key metadata fields when the header matches common ICARTT layouts.
        If parsing fails, returns what it can without throwing.
        """
        lines = self.read_header_lines()
        meta: Dict[str, str] = {}

        # Common ICARTT: line indices below assume a conventional layout often used with FFI=1001.
        # We'll guard everything with length checks.
        def safe(i: int) -> str:
            return lines[i].strip() if 0 <= i < len(lines) else ""

        meta["path"] = str(self.path)
        meta["header_length"] = str(self.info.header_length)
        meta["ffi"] = self.info.ffi

        # These are common but not guaranteed. Keep them best-effort.
        meta["pi"] = safe(1)
        meta["organization"] = safe(2)
        meta["data_description"] = safe(3)
        meta["mission"] = safe(4)
        meta["volume_info"] = safe(5)
        meta["date_info"] = safe(6)
        meta["data_interval"] = safe(7)
        meta["independent_variable"] = safe(8)

        return {k: v for k, v in meta.items() if v}

    def read_variable_defs(self) -> List[VariableDef]:
        """
        Best-effort variable definitions, primarily for common FFI=1001 layout:
          - line 10: number of dependent variables
          - line 12+: variable definition lines (often "NAME, UNIT, DESCRIPTION...")

        If layout doesn't match, returns empty list.
        """
        lines = self.read_header_lines()

        # Attempt the common ICARTT/FFI=1001 positions
        # Line 10 (0-index 9) is often number of dependent variables.
        if len(lines) < 11:
            return []

        try:
            n_dep = int(lines[9].strip())
        except Exception:
            return []

        start = 12  # 0-index start of var definition block in common layout
        block = lines[start : start + n_dep]
        out: List[VariableDef] = []

        for ln in block:
            parts = [p.strip() for p in ln.split(",")]
            if not parts:
                continue
            name = parts[0]
            unit = parts[1] if len(parts) > 1 else None
            desc = ",".join(parts[2:]).strip() if len(parts) > 2 else None
            out.append(VariableDef(name=name, unit=unit or None, description=desc or None))

        # Attach per-variable missing if we can infer it (optional)
        miss_map = self._guess_per_variable_missing()
        if miss_map:
            out = [
                VariableDef(v.name, v.unit, v.description, miss_map.get(v.name))
                for v in out
            ]

        return out

    # ----------------------------
    # Missing-value handling
    # ----------------------------
    def _guess_missing_values(self) -> List[Union[str, float, int]]:
        """
        Heuristic: try to extract missing indicators from the header.
        Falls back to common sentinel values.

        Many ICT files have a line describing missing indicators (often around line 12),
        but formats vary. We keep this conservative.
        """
        lines = self.read_header_lines()
        candidates: List[Union[str, float, int]] = []

        # Scan header for something that looks like a missing indicator list: "-9999" etc.
        for ln in lines[: min(len(lines), 200)]:
            # find numeric sentinels that look like -9999, -99999, 9999 etc.
            for tok in ln.replace(",", " ").split():
                if tok.startswith(("-", "+")) and tok[1:].isdigit():
                    val = int(tok)
                    # common missing sentinels are large magnitude
                    if abs(val) >= 999:
                        candidates.append(val)

        # De-duplicate while preserving order
        seen = set()
        ordered = []
        for v in candidates:
            if v not in seen:
                seen.add(v)
                ordered.append(v)

        # Add very common defaults if we found nothing
        if not ordered:
            ordered = [-9999, -99999, -8888, 9999, 99999]

        return ordered

    def _guess_per_variable_missing(self) -> Dict[str, float]:
        """
        Some ICARTT headers specify per-variable missing indicators.
        This is not standardized across all producers; implement only as a best-effort hook.

        Returns {} if nothing reliable is found.
        """
        # For now, keep minimal: many files effectively use a single sentinel across columns.
        # You can extend this if you encounter a known pattern you want to support.
        return {}

    # ----------------------------
    # Exports
    # ----------------------------
    def to_csv(
        self,
        out: Optional[PathLike] = None,
        *,
        na_values: Optional[List[Union[str, float, int]]] = None,
        strip_colnames: bool = True,
    ) -> Path:
        df = self.read_table(na_values=na_values, strip_colnames=strip_colnames)
        out_path = Path(out) if out else self.path.with_suffix(".csv")
        df.to_csv(out_path, index=False)
        return out_path

    def to_parquet(
        self,
        out: Optional[PathLike] = None,
        *,
        na_values: Optional[List[Union[str, float, int]]] = None,
        strip_colnames: bool = True,
    ) -> Path:
        df = self.read_table(na_values=na_values, strip_colnames=strip_colnames)
        out_path = Path(out) if out else self.path.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
        return out_path
