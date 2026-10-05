"""tpuasm 清单的语法高亮：按 vscode-tpuasm 的 TextMate 语法（syntaxes/tpuasm.tmLanguage.json）划分词法单元，输出与 Pandoc 高亮相同的 HTML 结构与 class，沿用网站已有的配色。"""
from html import escape
import re

SLOTS = r"s[01]|dma|va[0-3]|vst|vld[01]?|cld|vx[01]|vr[01]|misc"
TOKEN = re.compile(
    rf"""
      (?P<comment>\#.*$)
    | (?P<directive>(?<![\w.])\.(?:target|encoding|align|empty)\b)
    | (?P<slot>\b(?:{SLOTS})(?=:(?:\s|$)))
    | (?P<predicate>@!?)(?=p[0-9]+)
    | (?P<memory>(?<=\[)[a-z][a-z0-9_]*(?=:))
    | (?P<label>\b[A-Z][A-Za-z_0-9]*\b)
    | (?P<named>\b[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*(?=[ \t]*=))
    | (?P<number>(?<![\w.])[+-]?(?:0x[0-9a-fA-F]+|(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+(?:[eE][+-]?[0-9]+)?|inf)(?![\w.]))
    | (?P<register>\b(?:vm[0-9]+|[psv][0-9]+|(?:gmr|gsfn|gsft|msra|mrf|trf|iar|pcr|gtc)[0-9]+|crf|erf|v2sf|lcclo|lcchi|gtclo|gtchi)\b)
    | (?P<resource>\b(?:imm[0-9]+|vs[0-9]+)\b)
    | (?P<word>\b[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)
    """,
    re.VERBOSE,
)
# TextMate 的 scope 对应到 Pandoc 的高亮 class。
CLASSES = {
    "comment": "co",  # comment.line
    "directive": "pp",  # keyword.control.directive
    "slot": "dt",  # support.type.slot
    "predicate": "op",  # keyword.operator.predicate
    "memory": "dt",  # support.type.memory-space
    "label": "fu",  # storage.type.label
    "named": "at",  # variable.parameter.operand
    "number": "dv",  # constant.numeric
    "register": "va",  # variable.language.register
    "resource": "cn",  # constant.other.resource
    "mnemonic": "kw",  # keyword.control.instruction
}

def highlight_line(line: str) -> str:
    parts: list[str] = []
    position = 0
    # 每条指令的第一个词是助记符：槽名（可带谓词）之后的第一个词；没有槽名的简写行中，行首或分隔符之后的第一个词。
    expect_mnemonic = True
    for match in TOKEN.finditer(line):
        gap = line[position:match.start()]
        if any(separator in gap for separator in ";{、，"):
            expect_mnemonic = True
        parts.append(escape(gap, quote=False))
        kind = match.lastgroup
        text = match.group(0)
        if kind == "slot":
            expect_mnemonic = True
        elif kind == "word":
            kind = "mnemonic" if expect_mnemonic else None
            expect_mnemonic = False
        css = CLASSES.get(kind or "")
        parts.append(f'<span class="{css}">{escape(text, quote=False)}</span>' if css else escape(text, quote=False))
        position = match.end()
    parts.append(escape(line[position:], quote=False))
    return "".join(parts)

def highlight(code: str) -> str:
    """一个 tpuasm 代码块的 HTML，结构与 Pandoc 输出的 div.sourceCode 相同。"""
    lines = "\n".join(highlight_line(line) for line in code.split("\n"))
    return f'<div class="sourceCode"><pre class="sourceCode tpuasm"><code class="sourceCode tpuasm">{lines}</code></pre></div>'
