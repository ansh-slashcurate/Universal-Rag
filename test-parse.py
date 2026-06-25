from parser.liteparser.liteparser import parser
import re

liteparser = parser.parse(r"C:\Users\vijaya kumar pappuri\Downloads\parsing-testing1.pdf")

print("lite parser content", liteparser.text)


parsed_pages = liteparser.text if isinstance(liteparser.text, list) else [liteparser.text]
print("="*50)
print("parse pages", parsed_pages[0])


def fix_liteparse_to_markdown(raw_text):
    lines = raw_text.split("\n")
    processed_lines = []
    in_table = False
    
    for line in lines:
        cleaned = line.strip()
        # Catch row markers or casual pipes
        is_row = (len(re.findall(r'\s{3,}', line)) > 2 or "|" in line) and any(char.isdigit() for char in line)
        is_header = "Consultancy Packages" in line or "Type of Structures" in line

        if is_header or is_row:
            columns = [col.strip() for col in re.split(r'\s{2,}(?:\|)?|\|', line) if col.strip()]
            markdown_row = "| " + " | ".join(columns) + " |"
            processed_lines.append(markdown_row)
            
            if is_header and not in_table:
                in_table = True
                separator = "| " + " | ".join(["---"] * len(columns)) + " |"
                processed_lines.append(separator)
        else:
            in_table = False
            processed_lines.append(line)
            
    return "\n".join(processed_lines)


print("=="*50)
mark = fix_liteparse_to_markdown(liteparser.text)
print(mark)


def verify_markdown_features(text):
    """
    Checks if a string contains structural Markdown patterns
    returned by document parsers like LiteParse.
    """
    report = {
        "has_headings": False,
        "has_tables": False,
        "has_lists": False,
        "is_valid_markdown": False
    }
    
    lines = text.split("\n")
    
    for line in lines:
        cleaned = line.strip()
        
        # Check for ATX Headings (e.g., # Heading, ## Subheading)
        if cleaned.startswith("#"):
            report["has_headings"] = True
            
        # Check for GitHub Flavored Markdown Tables (e.g., | cell | cell |)
        if cleaned.startswith("|") and cleaned.endswith("|") and "---" in text:
            report["has_tables"] = True
            
        # Check for Markdown Bullet Lists
        if cleaned.startswith("* ") or cleaned.startswith("- ") or (cleaned[:1].isdigit() and cleaned[1:3] == ". "):
            report["has_lists"] = True

    # If it contains structural layout blocks, confirm it's structured Markdown
    if report["has_headings"] or report["has_tables"] or report["has_lists"]:
        report["is_valid_markdown"] = True
        
    return report

# --- Example Usage ---
status = verify_markdown_features(mark)
print("="*50)
print(status)
# Output: {'has_headings': True, 'has_tables': True, 'has_lists': True, 'is_valid_markdown': True}