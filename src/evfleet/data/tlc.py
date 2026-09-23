"""NYC taxi zones and cleaned yellow-taxi trips for the fleet simulator.

The service area is Manhattan, excluding the three harbour-island zones, which
have no street access. Airports are left out on purpose: airport pickups come in
arrival waves 30-50 minutes from the fleet and need their own staging logic,
which would confound a charging study (see docs/methodology.md). Coordinates are
NAD83 / New York Long Island state plane (EPSG:2263), converted from feet to km.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import shapefile

from evfleet.data.download import TLC_MONTH
from evfleet.io import read_parquet, write_parquet
from evfleet.paths import PROCESSED, RAW, RESULTS, SQL, ensure_dirs

FT_TO_KM = 0.0003048
AIRPORTS = {132: "JFK Airport", 138: "LaGuardia Airport"}
HARBOUR_ISLANDS = {103, 104, 105}  # Governor's / Ellis / Liberty Island
ZONES_ZIP = RAW / "taxi_zones.zip"
TRIPS_RAW = RAW / f"yellow_tripdata_{TLC_MONTH}.parquet"
TRIPS = PROCESSED / "nyc_trips.parquet"
ZONES = PROCESSED / "nyc_zones.parquet"


@dataclass
class ZoneShape:
    location_id: int
    zone: str
    borough: str
    rings: list[np.ndarray]  # each (n, 2) in km


def _ring_area_centroid(ring: np.ndarray) -> tuple[float, float, float]:
    """Signed shoelace area and centroid of a closed ring."""
    x, y = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2.0
    if a == 0:
        return 0.0, float(x.mean()), float(y.mean())
    cx = ((x + x1) * cross).sum() / (6.0 * a)
    cy = ((y + y1) * cross).sum() / (6.0 * a)
    return a, cx, cy


def polygon_area_centroid(rings: list[np.ndarray]) -> tuple[float, float, float]:
    """Area and centroid of a (multi)polygon given as rings.

    Shapefile outer rings are clockwise and holes counter-clockwise, so signed
    areas of holes cancel correctly when summed.
    """
    parts = [_ring_area_centroid(r) for r in rings]
    total = sum(p[0] for p in parts)
    if total == 0:
        pts = np.vstack(rings)
        return 0.0, float(pts[:, 0].mean()), float(pts[:, 1].mean())
    cx = sum(p[0] * p[1] for p in parts) / total
    cy = sum(p[0] * p[2] for p in parts) / total
    return abs(total), cx, cy


def load_zone_shapes(path=ZONES_ZIP) -> list[ZoneShape]:
    z = zipfile.ZipFile(path)
    names = {n.rsplit(".", 1)[-1].lower(): n for n in z.namelist() if not n.endswith("/")}
    reader = shapefile.Reader(
        shp=io.BytesIO(z.read(names["shp"])),
        shx=io.BytesIO(z.read(names["shx"])),
        dbf=io.BytesIO(z.read(names["dbf"])),
    )
    shapes = []
    for sr in reader.iterShapeRecords():
        pts = np.asarray(sr.shape.points, dtype=float) * FT_TO_KM
        bounds = list(sr.shape.parts) + [len(pts)]
        rings = [pts[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
        rec = sr.record.as_dict()
        shapes.append(ZoneShape(int(rec["LocationID"]), rec["zone"], rec["borough"], rings))
    return shapes


def zone_table(shapes: list[ZoneShape]) -> pd.DataFrame:
    rows = []
    for s in shapes:
        area, cx, cy = polygon_area_centroid(s.rings)
        rows.append({"zone_id": s.location_id, "zone": s.zone, "borough": s.borough,
                     "cx_km": cx, "cy_km": cy, "area_km2": area})
    return pd.DataFrame(rows).sort_values("zone_id").reset_index(drop=True)


def service_zone_ids(zones: pd.DataFrame, include_airports: bool = False) -> list[int]:
    manhattan = zones.loc[(zones.borough == "Manhattan") & ~zones.zone_id.isin(HARBOUR_ISLANDS), "zone_id"]
    return sorted(set(manhattan.tolist()) | (set(AIRPORTS) if include_airports else set()))


def build() -> pd.DataFrame:
    ensure_dirs()
    zones = zone_table(load_zone_shapes())
    service = service_zone_ids(zones)
    zones["in_service_area"] = zones.zone_id.isin(service)
    write_parquet(zones, ZONES)

    year, month = map(int, TLC_MONTH.split("-"))
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1)
    text = (SQL / "tlc_trips.sql").read_text()
    for key, value in {
        "raw_parquet": str(TRIPS_RAW),
        "month_start": str(start),
        "month_end": str(end),
        "service_zones": ", ".join(map(str, service)),
    }.items():
        text = text.replace("{" + key + "}", value)

    con = duckdb.connect()
    con.execute(text)
    quality = con.execute(
        """SELECT coalesce(drop_reason, 'kept') AS outcome, count(*) AS rows
           FROM tlc_trips GROUP BY 1 ORDER BY rows DESC"""
    ).df()
    quality["share"] = quality["rows"] / quality["rows"].sum()
    quality.to_csv(RESULTS / "nyc_trips_data_quality.csv", index=False)
    con.execute(
        f"""COPY (SELECT pickup_ts, dropoff_ts, pu, do_zone, distance_km, duration_s
                  FROM tlc_trips WHERE drop_reason IS NULL ORDER BY pickup_ts)
            TO '{TRIPS}' (FORMAT parquet)"""
    )
    con.close()
    return quality


def load_trips(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    where = []
    if start:
        where.append(f"pickup_ts >= TIMESTAMP '{start}'")
    if end:
        where.append(f"pickup_ts < TIMESTAMP '{end}'")
    return read_parquet(TRIPS, " AND ".join(where) if where else None)


def load_zones() -> pd.DataFrame:
    return read_parquet(ZONES)


if __name__ == "__main__":
    print(build().to_string(index=False))
