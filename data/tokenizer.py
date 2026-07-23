import re

TOKEN_PATTERN = re.compile(r"\w+(?:['’-]\w+)*|[^\w\s]", re.UNICODE)

def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)