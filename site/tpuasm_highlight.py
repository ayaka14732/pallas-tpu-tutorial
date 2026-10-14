"""tpuasm 清单的语法高亮：按 vscode-tpuasm 的 TextMate 语法（syntaxes/tpuasm.tmLanguage.json）划分词法单元，输出与 Pandoc 高亮相同的 HTML 结构与 class，沿用网站已有的配色。tpuasm 文档的 Pygments lexer（docs/tpuasm_lexer.py）使用同一套规则。"""
from collections.abc import Iterator
from html import escape
import re

SLOTS = r"s[01]|dma|va[0-3]|vst|vld[01]?|cld|vx[01]?|vr[01]?|misc|stream"
TOKEN = re.compile(
    rf"""
      (?P<comment>\#.*$)
    | (?P<directive>(?<![\w.])\.(?:target|encoding|align|empty)\b)
    | (?P<target>(?<=\.target\ )tpu(?:-[a-z0-9]+)+)
    | (?P<slot>\b(?:{SLOTS})(?=:(?:\s|$)))
    | (?P<predicate>@!?|!)(?=r?p[0-9]+)
    | (?P<memory>(?<=\[)[a-z][a-z0-9_]*(?=:))
    | (?P<function>\b(?:f32bits|const|lo|hi|ones_hi|pair)(?=[ \t]*\())
    | (?P<label>\b[A-Z][A-Za-z_0-9]*\b)
    | (?P<named>\b[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*(?=[ \t]*=))
    | (?P<number>(?<![\w.])[+-]?(?:0x[0-9a-fA-F]+|(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+(?:[eE][+-]?[0-9]+)?|inf)(?![\w.]))
    | (?P<register>\b(?:vm[0-9]+|r?p[0-9]+|[sv][0-9]+|(?:gmr|gsfn|gsft|lmr|msra|msrb|mubr|subr|mrf|trf|iar|pcr|gtc)[0-9]+|crf|drf|erf|sfrf|xrf|v2sf|lr|lcclo|lcchi|gtclo|gtchi)\b)
    | (?P<resource>\b(?:(?:zero_|ones_)?imm[0-9]+(?:_(?:imm[0-9]+|zero))?|vs[0-9]+|pred[0-9]+)\b)
    | (?P<word>\b[a-z_][a-z0-9_]*(?:\.[a-z0-9_]+)*)
    """,
    re.VERBOSE | re.MULTILINE,
)
BRANCH = re.compile(r"s(?:br|call)\.")
# TextMate 的 scope 对应到 Pandoc 的高亮 class。
CLASSES = {
    "comment": "co",  # comment.line
    "directive": "pp",  # keyword.control.directive
    "target": "cn",  # constant.language.target
    "slot": "dt",  # support.type.slot
    "predicate": "op",  # keyword.operator.predicate
    "memory": "dt",  # support.type.memory-space
    "function": "bu",  # support.function.encoding
    "label": "fu",  # storage.type.label
    "named": "at",  # variable.parameter.operand
    "number": "dv",  # constant.numeric
    "register": "va",  # variable.language.register
    "resource": "cn",  # constant.other.resource
    "mnemonic": "kw",  # keyword.control.instruction
}

def tokens(code: str) -> Iterator[tuple[str | None, str]]:
    """把代码块切分为 (词法类别, 文本)；类别为 None 的文本不着色。"""
    position = 0
    depth = 0
    # .encoding 之后的花括号内只有字段约束，没有助记符。
    encoding_depth: int | None = None
    # 每条指令的第一个词是助记符：槽名（可带谓词）之后的第一个词；没有槽名的简写行中，行首或分隔符之后的第一个词。
    expect_mnemonic = True
    branch = False
    for match in TOKEN.finditer(code):
        gap = code[position:match.start()]
        if gap:
            yield None, gap
        depth += gap.count("{") - gap.count("}")
        if encoding_depth is not None and depth <= encoding_depth and "}" in gap:
            encoding_depth = None
        if any(separator in gap for separator in "\n;{、，"):
            expect_mnemonic = True
            branch = False
        kind = match.lastgroup
        text = match.group(0)
        if kind == "directive" and text == ".encoding":
            encoding_depth = depth
        elif kind == "slot":
            expect_mnemonic = True
        elif kind == "word":
            # 花括号外紧跟冒号的词是标号；分支指令的标识符操作数是跳转目标。
            if depth == 0 and code.startswith(":", match.end()):
                kind = "label"
            elif encoding_depth is not None:
                kind = None
            elif expect_mnemonic:
                kind = "mnemonic"
                branch = BRANCH.match(text) is not None
            elif branch:
                kind = "label"
            else:
                kind = None
            expect_mnemonic = False
        yield kind, text
        position = match.end()
    if position < len(code):
        yield None, code[position:]

def highlight(code: str) -> str:
    """一个 tpuasm 代码块的 HTML，结构与 Pandoc 输出的 div.sourceCode 相同。"""
    html = "".join(f'<span class="{CLASSES[kind]}">{escape(text, quote=False)}</span>' if kind else escape(text, quote=False) for kind, text in tokens(code))
    return f'<div class="sourceCode"><pre class="sourceCode tpuasm"><code class="sourceCode tpuasm">{html}</code></pre></div>'
