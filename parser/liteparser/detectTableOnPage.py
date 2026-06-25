import re

def detect_table_on_page(text: str, min_number_lines: int = 3) -> bool:
    """
    Heuristic: if multiple lines each contain 3+ numbers,
    it's very likely a data table that needs structure analysis.
    """
    lines = text.strip().splitlines()
    number_lines = 0
    for line in lines:
        tokens = line.split()
        numeric = sum(1 for t in tokens if re.match(r"^\d+(\.\d+)?%?$", t))
        if numeric >= 3:
            number_lines += 1
    return number_lines >= min_number_lines