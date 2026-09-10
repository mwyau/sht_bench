from __future__ import annotations

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
