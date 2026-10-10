"""Download a small, openly available PhysioNet comparator database.

This deliberately does not pretend to download the unpublished Manipal cohort.
Use the PhysioNet website's citation/license terms for any downloaded data.
"""
import argparse
from pathlib import Path
from urllib.request import urlopen

DATABASES = {
    "mitdb": ("https://physionet.org/files/mitdb/1.0.0/", ["100.hea", "100.dat", "100.atr"]),
    "nsrdb": ("https://physionet.org/files/nsrdb/1.0.0/", ["16265.hea", "16265.dat", "16265.atr"]),
    "chfdb": ("https://physionet.org/files/chfdb/1.0.0/", ["chf01.hea", "chf01.dat", "chf01.ecg"]),
    "qtdb": ("https://physionet.org/files/qtdb/1.0.0/", ["sel100.hea", "sel100.dat", "sel100.atr"]),
    "incartdb": ("https://physionet.org/files/incartdb/1.0.0/", ["I01.hea", "I01.dat", "I01.atr"]),
}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--database", choices=sorted(DATABASES), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--all", action="store_true", help="download all files listed by the database index")
    args = p.parse_args(argv)
    base, files = DATABASES[args.database]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.all:
        import re
        from urllib.request import Request
        html = urlopen(Request(base), timeout=60).read().decode("utf-8", "replace")
        files = sorted(set(re.findall(r'href="([^"]+\.(?:hea|dat|atr|ecg))"', html)))
    for name in files:
        target = args.output / name
        print(f"downloading {name} -> {target}")
        with urlopen(base + name, timeout=120) as src, target.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
