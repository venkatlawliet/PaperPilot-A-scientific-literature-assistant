#Telling LLM how to generate D2 diagram code and rendering them to SVGs
import os
import re
import subprocess
import tempfile
import textwrap
from typing import Dict
from llm_bridge import answer_with_llama
def extract_d2_block(response_text: str) -> str:
    if not isinstance(response_text, str):
        return ""
    match = re.search(r"```d2(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match2 = re.search(r"```(.*?)```", response_text, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    return response_text.strip()
def render_d2_to_svg(d2_code: str) -> str:
    if not d2_code or not d2_code.strip():
        raise ValueError("Empty D2 code passed to render_d2_to_svg")
    with tempfile.NamedTemporaryFile(suffix=".d2", delete=False, mode="w", encoding="utf-8") as f:
        f.write(d2_code)
        d2_path = f.name
    svg_path = d2_path.replace(".d2", ".svg")
    try:
        subprocess.run(["d2", d2_path, svg_path], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"D2 rendering failed: {e}")
    finally:
        if os.path.exists(d2_path):
            os.unlink(d2_path)
    return svg_path
def _normalize_d2(d2_code: str) -> str:
    if not d2_code:
        return ""
    s = d2_code.strip()
    s = re.sub(r"\s+(?=[A-Za-z_][\w.]*\s*:)", "\n", s)
    s = re.sub(r"\s+(?=[A-Za-z_][\w.]*\s*->\s*)", "\n", s)
    s = s.replace("{", "{\n")
    s = s.replace("}", "\n}")
    s = re.sub(r"\n{3,}", "\n", s)
    return s.strip()
def _remove_stray_trailing_braces(d2_code: str) -> str:
    code = d2_code.strip()
    while code.endswith("}"):
        opens = code.count("{")
        closes = code.count("}")
        if closes > opens:
            code = code.rstrip()[:-1].rstrip()
        else:
            break
    return code
def llm_generate_d2(context_text: str, user_query: str) -> Dict[str, str]:
    prompt = textwrap.dedent(f"""
Your task: Output valid D2 code that will be rendered into a diagram.

Strict formatting rules you MUST follow:
1) Node IDs must use underscores instead of spaces:
   Example: Token_Search, Materials_Property_Matrix
2) Every node must follow:
   ID: "Readable Label"
3) Every edge must follow:
   ID1 -> ID2
   or:
   ID1 -> ID2: "Label"
4) NO trailing text after quotes, EVER.
5) NO prose, NO explanation, ONLY valid D2.

You must construct nodes and relationships ONLY from:
- USER REQUEST intention
- CONTEXT chunks (retrieved evidence)

Output MUST start immediately with D2 code. NO backticks.

USER REQUEST:
{user_query}

CONTEXT:
{context_text}

Now output valid D2 below:
""")
    raw_response = answer_with_llama(
        "",
        prompt,
        model="llama-3.3-70b-versatile",
    )
    # print("user_query:", user_query)
    # print("context_text:", context_text)
    # print("raw_response:", raw_response)
    d2_raw = extract_d2_block(raw_response)
    d2_norm = _normalize_d2(d2_raw)
    d2_clean = _remove_stray_trailing_braces(d2_norm)
    return {
        "raw_response": raw_response,
        "d2_code": d2_clean,
    }