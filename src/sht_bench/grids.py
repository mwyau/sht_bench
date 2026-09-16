from __future__ import annotations

from dataclasses import dataclass

ATMOSPHERIC_GL_GRIDS: dict[int, tuple[int, int, str]] = {
    42: (64, 128, "F32"),
    63: (96, 192, "F48"),
    85: (128, 256, "F64"),
    159: (160, 320, "F80"),
    255: (256, 512, "F128"),
    319: (320, 640, "F160"),
    511: (512, 1024, "F256"),
    639: (640, 1280, "F320"),
    1023: (1024, 2048, "F512"),
    1279: (1280, 2560, "F640"),
}

DEFAULT_GL_LMAX = list(ATMOSPHERIC_GL_GRIDS)
DEFAULT_CC_LMAX = [36, 60, 72, 90, 120, 180, 360, 720]


@dataclass(frozen=True, slots=True)
class SHTCase:
    """One explicit scalar SHT comparison case.

    ``lmax`` is the inclusive mathematical maximum degree used by ``sht_bench``.
    Backend adapters translate it to the convention required by their APIs.
    The case model is deliberately separate from :func:`grid_shape`: the legacy
    matrix keeps its historical CC geometry while focused comparisons can name a
    particular bandwidth on a fixed grid.
    """

    name: str
    grid: str
    nlat: int
    nlon: int
    lmax: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("case name must not be empty")
        if self.grid not in {"gl", "cc"}:
            raise ValueError(f"unsupported SHT case grid: {self.grid!r}")
        for field in ("nlat", "nlon", "lmax"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
            if value < 1:
                raise ValueError(f"{field} must be positive")
        if self.lmax < 0:
            raise ValueError("lmax must be non-negative")
        if self.grid == "cc":
            latitude_limit = self.nlat - 2
            longitude_limit = (self.nlon - 1) // 2
            limit = min(latitude_limit, longitude_limit)
            if self.lmax > limit:
                raise ValueError(
                    "triangular CC case exceeds the recoverable bandwidth: "
                    f"lmax={self.lmax}, limit={limit} for {self.nlat}x{self.nlon}"
                )

    @property
    def mmax(self) -> int:
        """Inclusive triangular order, equal to the case's degree limit."""

        return self.lmax

    @property
    def grid_points(self) -> int:
        return self.nlat * self.nlon


HIGH_BANDWIDTH_CC_CASES: tuple[SHTCase, ...] = (
    SHTCase("cc-73x144-t36", "cc", 73, 144, 36),
    SHTCase("cc-73x144-t70", "cc", 73, 144, 70),
    SHTCase("cc-73x144-t71", "cc", 73, 144, 71),
    SHTCase("cc-129x256-t127", "cc", 129, 256, 127),
    SHTCase("cc-257x512-t255", "cc", 257, 512, 255),
)
HIGH_BANDWIDTH_CC_CASES_BY_NAME = {
    case.name: case for case in HIGH_BANDWIDTH_CC_CASES
}


def high_bandwidth_cc_case(name: str) -> SHTCase:
    """Return a named focused CC case or raise an explicit CLI-friendly error."""

    try:
        return HIGH_BANDWIDTH_CC_CASES_BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(HIGH_BANDWIDTH_CC_CASES_BY_NAME)
        raise ValueError(f"unknown focused CC case {name!r}; choose from {available}") from exc


def validate_cc_bandwidth(nlat: int, nlon: int, lmax: int) -> None:
    """Validate inclusive triangular bandwidth for an explicit CC grid."""

    SHTCase("validation", "cc", nlat, nlon, lmax)


def grid_shape(lmax: int, grid: str = "gl") -> tuple[int, int]:
    """Return the benchmark grid shape for triangular degree *lmax*."""
    if lmax < 2:
        raise ValueError("lmax must be >= 2")
    if grid == "gl":
        case = ATMOSPHERIC_GL_GRIDS.get(lmax)
        if case is not None:
            return case[0], case[1]
        return lmax + 1, 2 * lmax + 1
    if grid == "cc":
        return 2 * lmax + 1, 4 * lmax
    raise ValueError(f"unsupported grid: {grid}")


def is_atmospheric_gl_case(lmax: int) -> bool:
    return lmax in ATMOSPHERIC_GL_GRIDS


def gl_grid_label(lmax: int) -> str:
    case = ATMOSPHERIC_GL_GRIDS.get(lmax)
    return case[2] if case is not None else f"T{lmax}"


def gl_grid_number(lmax: int) -> int | None:
    case = ATMOSPHERIC_GL_GRIDS.get(lmax)
    return case[0] // 2 if case is not None else None


def cc_resolution(lmax: int) -> float:
    return 90.0 / lmax
