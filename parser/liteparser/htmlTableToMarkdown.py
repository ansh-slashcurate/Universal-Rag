import re
from bs4 import BeautifulSoup

def html_table_to_markdown(html: str) -> str:
    """
    Convert PaddleStructure HTML table output to markdown.
 
    Handles:
      - colspan / rowspan  → resolved into full grid
      - multi-row headers  → flattened with ' > ' hierarchy
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return html
 
    rows = table.find_all("tr")
    if not rows:
        return ""
 
    # ── Build full grid resolving rowspan/colspan ──
    grid: list[list[str]] = []
    row_spans: dict[int, tuple[int, str]] = {}
 
    for row in rows:
        cells = row.find_all(["td", "th"])
        grid_row: list[str] = []
        col = 0
 
        def drain(col):
            while col in row_spans and row_spans[col][0] > 0:
                remaining, txt = row_spans[col]
                grid_row.append(txt)
                row_spans[col] = (remaining - 1, txt)
                col += 1
            return col
 
        col = drain(col)
 
        for cell in cells:
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            text = re.sub(r"\s+", " ", cell.get_text(separator=" ").strip())
 
            for _ in range(colspan):
                col = drain(col)
                grid_row.append(text)
                if rowspan > 1:
                    row_spans[col] = (rowspan - 1, text)
                col += 1
 
        col = drain(col)
        grid.append(grid_row)
 
    if not grid:
        return ""
 
    num_cols = max(len(r) for r in grid)
 
    # ── Split header rows from data rows ──
    # Header rows = rows with no numeric-looking cells
    header_rows: list[list[str]] = []
    data_rows: list[list[str]] = []
    found_data = False
 
    for row in grid:
        if not found_data and not any(re.match(r"^\d+(\.\d+)?$", c) for c in row):
            header_rows.append(row)
        else:
            found_data = True
            data_rows.append(row)
 
    # Fallback: treat first row as header if everything looked like data
    if not header_rows and data_rows:
        header_rows, data_rows = [data_rows[0]], data_rows[1:]
 
    # ── Flatten multi-row headers with ' > ' ──
    if len(header_rows) == 1:
        flat_header = header_rows[0]
    else:
        flat_header = []
        for c in range(num_cols):
            parts, seen = [], set()
            for hrow in header_rows:
                val = hrow[c] if c < len(hrow) else ""
                if val and val not in seen:
                    parts.append(val)
                    seen.add(val)
            flat_header.append(" > ".join(parts) if parts else "")
 
    def pad(row: list[str]) -> list[str]:
        return row + [""] * (num_cols - len(row))
 
    def md_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"
 
    lines = [
        md_row(pad(flat_header)),
        "| " + " | ".join(["---"] * num_cols) + " |",
        *[md_row(pad(r)) for r in data_rows],
    ]
    return "\n".join(lines)