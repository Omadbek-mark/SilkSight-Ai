"""Render numbered thumbnail grids per class for manual review (outputs/contact_sheets/).

Usage: python -m scripts.contact_sheets
Each sheet has a matching .txt listing number -> Commons title, for data/exclusions.csv.
"""

from collections import defaultdict

from PIL import Image, ImageDraw

from silksight.dataset import read_manifest
from silksight.settings import CONTACT_SHEETS_DIR, DATA_DIR, MANIFEST_PATH

CELL = 220
COLUMNS = 8


def main() -> int:
    groups = defaultdict(list)
    for row in read_manifest(MANIFEST_PATH):
        groups[row.source_group].append(row)
    CONTACT_SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    for group, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row.title)
        lines = (len(rows) + COLUMNS - 1) // COLUMNS
        sheet = Image.new("RGB", (COLUMNS * CELL, lines * CELL), "white")
        draw = ImageDraw.Draw(sheet)
        index = []
        for number, row in enumerate(rows, start=1):
            with Image.open(DATA_DIR / row.file) as image:
                image.thumbnail((CELL - 8, CELL - 8))
                x, y = ((number - 1) % COLUMNS) * CELL, ((number - 1) // COLUMNS) * CELL
                sheet.paste(image, (x + 4, y + 4))
            draw.rectangle((x + 4, y + 4, x + 40, y + 26), fill="black")
            draw.text((x + 8, y + 8), f"{number}{row.split[0]}", fill="yellow")
            index.append(f"{number}\t{row.split}\t{row.title}")
        sheet.save(CONTACT_SHEETS_DIR / f"{group}.jpg", quality=80)
        (CONTACT_SHEETS_DIR / f"{group}.txt").write_text("\n".join(index) + "\n", "utf-8")
    print(f"Wrote {len(groups)} contact sheets to {CONTACT_SHEETS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
