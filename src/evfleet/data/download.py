"""Download the public raw datasets and record their identity in a manifest.

Raw files are immutable inputs: they are fetched once, hashed (SHA-256) and
listed in ``data/raw/MANIFEST.json`` together with their source URL, licence
and download time. Nothing under ``data/`` is committed to version control.

Usage::

    python -m evfleet.data.download            # everything
    python -m evfleet.data.download palo_alto  # one source
"""

from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evfleet.paths import RAW, ensure_dirs

# The fleet study uses one month of NYC demand; the session study uses the full
# Palo Alto history. Both date ranges are fixed so results are reproducible.
TLC_MONTH = "2025-04"
NYC_LAT, NYC_LON = 40.7812, -73.9665  # Central Park
PA_LAT, PA_LON = 37.4419, -122.1430  # Palo Alto


def _open_meteo(lat: float, lon: float, start: str, end: str, tz: str) -> str:
    q = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m,precipitation",
            "timezone": tz,
        }
    )
    return f"https://archive-api.open-meteo.com/v1/archive?{q}"


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    filename: str
    licence: str
    description: str


SOURCES: dict[str, Source] = {
    s.key: s
    for s in [
        Source(
            "palo_alto",
            "https://data.paloalto.gov/datasets/194693-electric-vehicle-charging-station-usage-july-2011-dec-2020.download/",
            "palo_alto_ev_sessions_2011_2020.csv",
            "Open Data Commons PDDL (City of Palo Alto Open Data)",
            "City of Palo Alto EV charging station usage, Jul 2011 - Dec 2020 (ChargePoint export)",
        ),
        Source(
            "tlc_yellow",
            f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{TLC_MONTH}.parquet",
            f"yellow_tripdata_{TLC_MONTH}.parquet",
            "NYC Taxi & Limousine Commission trip record data (NYC Open Data terms of use)",
            f"NYC yellow taxi trip records, {TLC_MONTH}",
        ),
        Source(
            "tlc_zones",
            "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip",
            "taxi_zones.zip",
            "NYC TLC (NYC Open Data terms of use)",
            "NYC taxi zone polygons (shapefile, EPSG:2263)",
        ),
        Source(
            "tlc_zone_lookup",
            "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
            "taxi_zone_lookup.csv",
            "NYC TLC (NYC Open Data terms of use)",
            "NYC taxi zone id -> borough / zone name",
        ),
        Source(
            "nyiso_dam",
            f"http://mis.nyiso.com/public/csv/damlbmp/{TLC_MONTH.replace('-', '')}01damlbmp_zone_csv.zip",
            f"nyiso_damlbmp_zone_{TLC_MONTH}.zip",
            "NYISO public market data",
            f"NYISO day-ahead zonal LBMP, {TLC_MONTH} (hourly, $/MWh)",
        ),
        Source(
            "weather_nyc",
            _open_meteo(NYC_LAT, NYC_LON, f"{TLC_MONTH}-01", f"{TLC_MONTH}-30", "America/New_York"),
            f"open_meteo_nyc_{TLC_MONTH}.json",
            "Open-Meteo historical weather API, CC BY 4.0",
            "Hourly 2 m temperature and precipitation, Central Park",
        ),
        Source(
            "weather_palo_alto",
            _open_meteo(PA_LAT, PA_LON, "2011-07-01", "2020-12-31", "America/Los_Angeles"),
            "open_meteo_palo_alto_2011_2020.json",
            "Open-Meteo historical weather API, CC BY 4.0",
            "Hourly 2 m temperature and precipitation, Palo Alto",
        ),
    ]
}

MANIFEST = RAW / "MANIFEST.json"


def _ssl_context() -> ssl.SSLContext:
    # Some Python builds (e.g. python.org macOS installers) ship without a CA
    # bundle; certifi's bundle is used when available.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with urllib, falling back to curl.

    The Palo Alto portal answers with a non-standard HTTP status line on its
    redirect, which ``http.client`` rejects but curl tolerates.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "evfleet-research/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as resp, dest.open("wb") as out:
            while block := resp.read(1 << 20):
                out.write(block)
        return
    except Exception as exc:
        curl = shutil.which("curl")
        if curl is None:
            raise
        print(f"  urllib failed ({type(exc).__name__}); retrying with curl")
    subprocess.run([curl, "-fsSL", "--retry", "2", "-o", str(dest), url], check=True)


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def fetch(src: Source, force: bool = False, retries: int = 3) -> dict:
    """Download one source (unless already present) and return its manifest entry."""
    ensure_dirs()
    dest = RAW / src.filename
    manifest = _load_manifest()
    if dest.exists() and not force and src.key in manifest:
        return manifest[src.key]

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            _download(src.url, tmp)
            break
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"download failed for {src.key}: {exc}") from exc
            time.sleep(2 * attempt)
    tmp.replace(dest)

    entry = {
        "url": src.url,
        "file": src.filename,
        "bytes": dest.stat().st_size,
        "sha256": sha256(dest),
        "licence": src.licence,
        "description": src.description,
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest[src.key] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return entry


def main(argv: list[str]) -> None:
    keys = argv or list(SOURCES)
    for key in keys:
        entry = fetch(SOURCES[key])
        print(f"{key:18s} {entry['bytes'] / 1e6:8.2f} MB  sha256={entry['sha256'][:12]}")


if __name__ == "__main__":
    main(sys.argv[1:])
