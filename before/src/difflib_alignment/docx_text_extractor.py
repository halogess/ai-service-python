"""DOCX text extraction utilities"""


def extract_text_recursive(json_tree):
    """Recursively extract text dari JSON tree"""
    texts = []

    def rec(node):
        if isinstance(node, dict):
            if "rows" in node:
                for row in node.get("rows", []):
                    if isinstance(row, dict):
                        for cell in row.get("cells", []):
                            if isinstance(cell, str):
                                texts.append(cell)
                            elif isinstance(cell, dict):
                                rec(cell)
                return

            if node.get("type") == "text" and "value" in node:
                texts.append(node["value"])
            elif node.get("type") == "math" and "text" in node:
                texts.append(node["text"])

            for k, v in node.items():
                rec(v)

        elif isinstance(node, list):
            for x in node:
                rec(x)

    rec(json_tree)
    result = " ".join(texts)
    
    if not result and isinstance(json_tree, dict) and "content" in json_tree:
        content = json_tree["content"]
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and "value" in item:
                    texts.append(item["value"])
            result = " ".join(texts)
    
    return result
