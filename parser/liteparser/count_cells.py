

def count_cells(md: str) -> int:
    """Count data cells in markdown table (skip header + separator rows)."""
    data_lines = [
        l for l in md.splitlines()
        if l.startswith("|") and "---" not in l
    ][1:]  # skip header row
    return sum(l.count("|") - 1 for l in data_lines)
