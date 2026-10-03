"""把教程 README 和实验附件构建成多页静态网站。"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import quote, unquote, urlsplit, urlunsplit

ROOT_DIR = Path(__file__).resolve().parent.parent
SITE_DIR = ROOT_DIR / "site"
OUTPUT_DIR = SITE_DIR / "build"
SITE_TITLE = "Pallas TPU Kernel 开发教程"
GITHUB_URL = "https://github.com/ayaka14732/pallas-tpu-tutorial"
EXTERNAL_REPOSITORIES = {
    "pallas-tpu-readings-dev": "https://github.com/ayaka14732/pallas-tpu-readings-dev",
    "tpu-v4-latency-numbers": "https://github.com/ayaka14732/tpu-v4-latency-numbers",
    "tpuasm": "https://github.com/ayaka14732/tpuasm",
}
ASSET_SUFFIXES = {
    ".csv",
    ".dot",
    ".gif",
    ".jpeg",
    ".jpg",
    ".json",
    ".png",
    ".py",
    ".sh",
    ".svg",
    ".txt",
    ".webp",
}
CHAPTER_PATTERN = re.compile(r"chapter\d{2}")
SECTION_PATTERN = re.compile(r"\d{2}_.+")

@dataclass(frozen=True)
class Page:
    source: Path
    route: str
    title: str
    kind: str
    chapter_route: str = ""

@dataclass(frozen=True)
class Chapter:
    page: Page
    sections: tuple[Page, ...]

@dataclass(frozen=True)
class Heading:
    level: int
    identifier: str
    title: str

class SiteReferenceParser(HTMLParser):
    """收集生成页面中的链接、资源和 ID。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[str] = []
        self.identifiers: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        identifier = attributes.get("id")
        if identifier:
            self.identifiers.add(identifier)
        if tag in {"a", "link"} and attributes.get("href"):
            self.references.append(attributes["href"])
        if tag in {"img", "script"} and attributes.get("src"):
            self.references.append(attributes["src"])
        if attributes.get("data-source-preview"):
            self.references.append(attributes["data-source-preview"])

def run_pandoc(arguments: list[str], input_data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["pandoc", *arguments],
        check=True,
        input=input_data,
        stdout=subprocess.PIPE,
    )
    return result.stdout

def read_title(source: Path) -> str:
    first_line = source.read_text(encoding="utf-8").splitlines()[0]
    if not first_line.startswith("# "):
        raise ValueError(f"{source.relative_to(ROOT_DIR)} 的首行不是一级标题")
    return first_line[2:].strip()

def discover_site() -> tuple[tuple[Page, ...], tuple[Chapter, ...], tuple[str, ...]]:
    root_page = Page(ROOT_DIR / "README.md", "", read_title(ROOT_DIR / "README.md"), "root")
    pages = [root_page]
    chapters = []
    skipped = []
    for chapter_directory in sorted(path for path in ROOT_DIR.iterdir() if path.is_dir() and CHAPTER_PATTERN.fullmatch(path.name)):
        chapter_readme = chapter_directory / "README.md"
        if not chapter_readme.is_file():
            skipped.append(chapter_directory.name)
            continue
        chapter_page = Page(chapter_readme, chapter_directory.name, read_title(chapter_readme), "chapter", chapter_directory.name)
        sections = []
        for section_directory in sorted(path for path in chapter_directory.iterdir() if path.is_dir() and SECTION_PATTERN.fullmatch(path.name)):
            section_readme = section_directory / "README.md"
            if not section_readme.is_file():
                continue
            route = f"{chapter_directory.name}/{section_directory.name}"
            sections.append(Page(section_readme, route, read_title(section_readme), "section", chapter_directory.name))
        chapter = Chapter(chapter_page, tuple(sections))
        chapters.append(chapter)
        pages.append(chapter_page)
        pages.extend(sections)
    return tuple(pages), tuple(chapters), tuple(skipped)

def collect_assets(chapters: tuple[Chapter, ...]) -> set[Path]:
    assets = {Path("LICENSE")}
    assets.update(path.relative_to(ROOT_DIR) for path in (ROOT_DIR / "LICENSES").glob("*.txt"))
    for path in ROOT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in ASSET_SUFFIXES:
            assets.add(path.relative_to(ROOT_DIR))
    for chapter in chapters:
        chapter_directory = ROOT_DIR / chapter.page.route
        for path in chapter_directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in ASSET_SUFFIXES:
                assets.add(path.relative_to(ROOT_DIR))
    return assets

def route_directory(route: str) -> Path:
    return Path(route) if route else Path(".")

def relative_output_href(page: Page, target: Path) -> str:
    relative = os.path.relpath(target, route_directory(page.route))
    return quote(Path(relative).as_posix(), safe="/.")

def relative_page_href(page: Page, target: Page) -> str:
    relative = os.path.relpath(route_directory(target.route), route_directory(page.route))
    href = Path(relative).as_posix()
    return "./" if href == "." else quote(href, safe="/.") + "/"

def append_url_parts(path: str, query: str, fragment: str) -> str:
    return urlunsplit(("", "", path, query, fragment))

def add_attribute(node: dict[str, object], css_class: str | None, name: str, value: str) -> None:
    content = node["c"]
    attributes = content[0]
    classes = attributes[1]
    key_values = attributes[2]
    if css_class and css_class not in classes:
        classes.append(css_class)
    attributes[2] = [item for item in key_values if item[0] != name]
    attributes[2].append([name, value])

def external_repository_href(resolved: Path, query: str, fragment: str) -> str:
    try:
        relative = resolved.relative_to(ROOT_DIR.parent)
    except ValueError as error:
        raise ValueError(f"相对链接越过工作区：{resolved}") from error
    repository_name = relative.parts[0]
    if repository_name not in EXTERNAL_REPOSITORIES:
        raise ValueError(f"没有配置跨仓库链接：{relative}")
    repository_url = EXTERNAL_REPOSITORIES[repository_name]
    remainder = Path(*relative.parts[1:])
    if not remainder.parts or remainder == Path("."):
        href = repository_url
    else:
        href = f"{repository_url}/blob/main/{quote(remainder.as_posix(), safe='/')}"
    if query:
        href += "?" + query
    if fragment:
        href += "#" + fragment
    return href

def rewrite_reference(node: dict[str, object], page: Page, pages_by_source: dict[Path, Page], assets: set[Path]) -> None:
    content = node["c"]
    target = content[2]
    original = target[0]
    parsed = urlsplit(original)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return
    if parsed.path.startswith("/"):
        raise ValueError(f"{page.source.relative_to(ROOT_DIR)} 使用了站点绝对路径：{original}")
    resolved = (page.source.parent / unquote(parsed.path)).resolve()
    try:
        relative = resolved.relative_to(ROOT_DIR)
    except ValueError:
        target[0] = external_repository_href(resolved, parsed.query, parsed.fragment)
        return
    if resolved.is_dir():
        resolved = resolved / "README.md"
        relative = resolved.relative_to(ROOT_DIR)
    if resolved.name == "README.md":
        target_page = pages_by_source.get(resolved)
        if target_page is None:
            raise ValueError(f"{page.source.relative_to(ROOT_DIR)} 链接到未发布页面：{relative}")
        target[0] = append_url_parts(relative_page_href(page, target_page), parsed.query, parsed.fragment)
        return
    if not resolved.is_file():
        raise ValueError(f"{page.source.relative_to(ROOT_DIR)} 链接的文件不存在：{relative}")
    if relative not in assets:
        raise ValueError(f"{page.source.relative_to(ROOT_DIR)} 链接的文件不会发布：{relative}")
    href = relative_output_href(page, relative)
    target[0] = append_url_parts(href, parsed.query, parsed.fragment)
    if node.get("t") == "Link" and (relative.suffix.lower() in {".py", ".txt"} or relative.name == "LICENSE"):
        preview_href = relative_output_href(page, Path(str(relative) + ".html"))
        add_attribute(node, "source-link", "data-source-preview", preview_href)

def transform_references(value: object, page: Page, pages_by_source: dict[Path, Page], assets: set[Path]) -> None:
    if isinstance(value, dict):
        for child in value.values():
            transform_references(child, page, pages_by_source, assets)
        if value.get("t") in {"Image", "Link"}:
            rewrite_reference(value, page, pages_by_source, assets)
        if value.get("t") == "Link":
            parsed = urlsplit(value["c"][2][0])
            if parsed.scheme in {"http", "https"} or parsed.netloc:
                add_attribute(value, None, "target", "_blank")
                add_attribute(value, None, "rel", "noopener")
    elif isinstance(value, list):
        for child in value:
            transform_references(child, page, pages_by_source, assets)

def inline_text(value: object) -> str:
    if isinstance(value, dict):
        node_type = value.get("t")
        content = value.get("c")
        if node_type == "Str" and isinstance(content, str):
            return content
        if node_type in {"Space", "SoftBreak", "LineBreak"}:
            return " "
        if node_type in {"Code", "Math"} and isinstance(content, list):
            return str(content[-1])
        return inline_text(content)
    if isinstance(value, list):
        return "".join(inline_text(item) for item in value)
    return ""

def extract_headings(document: dict[str, object]) -> tuple[Heading, ...]:
    headings = []
    for block in document["blocks"]:
        if block.get("t") != "Header":
            continue
        content = block["c"]
        level = content[0]
        if level not in {2, 3}:
            continue
        identifier = content[1][0]
        headings.append(Heading(level, identifier, inline_text(content[2]).strip()))
    return tuple(headings)

def render_markdown(page: Page, pages_by_source: dict[Path, Page], assets: set[Path]) -> tuple[str, tuple[Heading, ...]]:
    # 管道表格的行超过 --columns 时 Pandoc 会按源码宽度固定列宽，教程中的表格行都很长，所以放宽这个限制。
    document = json.loads(run_pandoc(["--from=markdown+tex_math_single_backslash", "--to=json", "--columns=100000", str(page.source)]))
    transform_references(document, page, pages_by_source, assets)
    headings = extract_headings(document)
    encoded = json.dumps(document, ensure_ascii=False).encode("utf-8")
    rendered = run_pandoc(["--from=json", "--to=html5", "--math-method=mathml", "--syntax-highlighting=pygments", "--wrap=none"], encoded)
    return rendered.decode("utf-8"), headings

def navigation_html(page: Page, root_page: Page, chapters: tuple[Chapter, ...]) -> str:
    root_current = ' aria-current="page"' if page == root_page else ""
    parts = [
        '<nav class="book-nav" aria-label="全书目录">',
        f'<a class="book-nav-home" href="{relative_page_href(page, root_page)}"{root_current}>教程首页</a>',
        '<ol class="book-nav-chapters">',
    ]
    for chapter in chapters:
        current = ' aria-current="page"' if page == chapter.page else ""
        parts.append('<li class="book-nav-chapter">')
        parts.append(f'<a href="{relative_page_href(page, chapter.page)}"{current}>{escape(chapter.page.title)}</a>')
        parts.append('<ol class="book-nav-sections">')
        for section in chapter.sections:
            current = ' aria-current="page"' if page == section else ""
            parts.append(f'<li><a href="{relative_page_href(page, section)}"{current}>{escape(section.title)}</a></li>')
        parts.append("</ol></li>")
    parts.append("</ol></nav>")
    return "\n".join(parts)

def breadcrumbs_html(page: Page, root_page: Page, chapters_by_route: dict[str, Chapter]) -> str:
    if page.kind == "root":
        return '<nav class="breadcrumbs" aria-label="面包屑"><span>教程首页</span></nav>'
    parts = [f'<a href="{relative_page_href(page, root_page)}">教程首页</a>']
    chapter = chapters_by_route[page.chapter_route]
    if page.kind == "chapter":
        parts.append(f'<span aria-current="page">{escape(page.title)}</span>')
    else:
        parts.append(f'<a href="{relative_page_href(page, chapter.page)}">{escape(chapter.page.title)}</a>')
        parts.append(f'<span aria-current="page">{escape(page.title)}</span>')
    return '<nav class="breadcrumbs" aria-label="面包屑">' + '<span class="breadcrumb-separator">/</span>'.join(parts) + "</nav>"

def page_toc_html(headings: tuple[Heading, ...]) -> str:
    if not headings:
        return ""
    parts = ['<nav class="page-toc-nav" aria-label="本页目录">', '<p class="page-toc-title">本页目录</p>', "<ol>"]
    for heading in headings:
        parts.append(f'<li class="toc-level-{heading.level}"><a href="#{quote(heading.identifier)}">{escape(heading.title)}</a></li>')
    parts.append("</ol></nav>")
    return "\n".join(parts)

def pagination_html(page: Page, pages: tuple[Page, ...]) -> str:
    index = pages.index(page)
    previous_page = pages[index - 1] if index > 0 else None
    next_page = pages[index + 1] if index + 1 < len(pages) else None
    parts = ['<nav class="page-pagination" aria-label="相邻页面">']
    if previous_page is None:
        parts.append('<span class="pagination-placeholder"></span>')
    else:
        parts.append(f'<a class="pagination-link pagination-previous" href="{relative_page_href(page, previous_page)}"><span>上一页</span><strong>{escape(previous_page.title)}</strong></a>')
    if next_page is None:
        parts.append('<span class="pagination-placeholder"></span>')
    else:
        parts.append(f'<a class="pagination-link pagination-next" href="{relative_page_href(page, next_page)}"><span>下一页</span><strong>{escape(next_page.title)}</strong></a>')
    parts.append("</nav>")
    return "\n".join(parts)

def html_document(page: Page, content: str, headings: tuple[Heading, ...], pages: tuple[Page, ...], chapters: tuple[Chapter, ...]) -> str:
    root_page = pages[0]
    chapters_by_route = {chapter.page.route: chapter for chapter in chapters}
    stylesheet = relative_output_href(page, Path("style.css"))
    script = relative_output_href(page, Path("site.js"))
    home_href = relative_page_href(page, root_page)
    title = SITE_TITLE if page.kind == "root" else f"{page.title} · {SITE_TITLE}"
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#aa2e45">
<meta name="description" content="系统学习 Pallas TPU kernel 与 TPU v4 用户可编程硬件能力。">
<title>{escape(title)}</title>
<script>try{{const theme=localStorage.getItem("pallas-tpu-theme");if(theme==="light"||theme==="dark")document.documentElement.dataset.theme=theme}}catch(error){{}}</script>
<link rel="stylesheet" href="{stylesheet}">
<script defer src="{script}"></script>
</head>
<body>
<a class="skip-link" href="#main-content">跳到正文</a>
<header class="mobile-header">
<button class="header-button" type="button" data-nav-toggle aria-controls="site-navigation" aria-expanded="false">目录</button>
<a class="mobile-title" href="{home_href}">Pallas TPU Kernel 开发教程</a>
<button class="header-button theme-toggle" type="button" data-theme-toggle><svg class="theme-icon theme-icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg><svg class="theme-icon theme-icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.66 6.34l1.41-1.41"/></svg></button>
</header>
<button class="nav-backdrop" type="button" data-nav-close aria-label="关闭目录" tabindex="-1"></button>
<div class="site-shell">
<aside class="site-sidebar" id="site-navigation">
<div class="site-sidebar-header">
<a class="site-brand" href="{home_href}"><span>Pallas TPU</span><strong>Kernel 开发教程</strong></a>
<button class="sidebar-close" type="button" data-nav-close aria-label="关闭目录">×</button>
</div>
{navigation_html(page, root_page, chapters)}
<div class="site-sidebar-footer">
<a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub ↗</a>
<button class="theme-toggle" type="button" data-theme-toggle><svg class="theme-icon theme-icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg><svg class="theme-icon theme-icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.66 6.34l1.41-1.41"/></svg></button>
</div>
</aside>
<main class="site-main" id="main-content">
<div class="content-column">
{breadcrumbs_html(page, root_page, chapters_by_route)}
<article class="doc-content">
{content}
</article>
{pagination_html(page, pages)}
</div>
</main>
<aside class="page-toc">
{page_toc_html(headings)}
</aside>
</div>
<dialog class="source-viewer" id="source-viewer" aria-labelledby="source-viewer-title">
<header class="source-viewer-header">
<div class="source-viewer-heading"><span class="source-viewer-kind">文件</span><strong id="source-viewer-title"></strong></div>
<div class="source-viewer-actions"><a class="source-viewer-raw" target="_blank" rel="noopener">打开原始文件</a><button type="button" data-source-close aria-label="关闭文件窗口">×</button></div>
</header>
<div class="source-viewer-body"><p class="source-viewer-status" role="status">正在载入文件…</p><div class="source-viewer-content" hidden></div></div>
</dialog>
</body>
</html>
'''

def maximum_backtick_run(source: str) -> int:
    runs = re.findall(r"`+", source)
    return max((len(run) for run in runs), default=0)

def render_source_preview(source_path: Path) -> bytes:
    language = "python" if source_path.suffix.lower() == ".py" else "text"
    source = source_path.read_text(encoding="utf-8")
    fence = "`" * max(3, maximum_backtick_run(source) + 1)
    markdown = f"{fence} {{.{language} .numberLines}}\n{source}\n{fence}\n"
    return run_pandoc(["--from=markdown", "--to=html5", "--syntax-highlighting=pygments", "--wrap=none"], markdown.encode("utf-8"))

def copy_assets(output: Path, assets: set[Path]) -> None:
    for relative in sorted(assets):
        source = ROOT_DIR / relative
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if relative.suffix.lower() in {".py", ".txt"} or relative.name == "LICENSE":
            preview = output / Path(str(relative) + ".html")
            preview.write_bytes(render_source_preview(source))

def parse_html(path: Path) -> SiteReferenceParser:
    parser = SiteReferenceParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser

def validate_site(output: Path) -> None:
    page_files = sorted(output.rglob("index.html"))
    identifiers = {page.resolve(): parse_html(page).identifiers for page in page_files}
    errors = []
    for page in page_files:
        parser = parse_html(page)
        for reference in parser.references:
            parsed = urlsplit(reference)
            if parsed.scheme or parsed.netloc:
                continue
            target = page if not parsed.path else (page.parent / unquote(parsed.path)).resolve()
            try:
                target.relative_to(output.resolve())
            except ValueError:
                errors.append(f"{page.relative_to(output)} 的链接越过构建目录：{reference}")
                continue
            if target.is_dir() or parsed.path.endswith("/"):
                target = target / "index.html"
            if not target.is_file():
                errors.append(f"{page.relative_to(output)} 的链接不存在：{reference}")
                continue
            if parsed.fragment and target.suffix == ".html" and target.name == "index.html":
                target_ids = identifiers.get(target.resolve(), set())
                if unquote(parsed.fragment) not in target_ids:
                    errors.append(f"{page.relative_to(output)} 的锚点不存在：{reference}")
    if errors:
        raise ValueError("生成网站验证失败：\n" + "\n".join(errors))

def build_site() -> tuple[int, tuple[str, ...]]:
    pages, chapters, skipped = discover_site()
    pages_by_source = {page.source.resolve(): page for page in pages}
    assets = collect_assets(chapters)
    with tempfile.TemporaryDirectory(prefix="pallas-tpu-site-") as temporary_directory:
        temporary_output = Path(temporary_directory)
        shutil.copy2(SITE_DIR / "style.css", temporary_output / "style.css")
        shutil.copy2(SITE_DIR / "site.js", temporary_output / "site.js")
        (temporary_output / ".nojekyll").touch()
        copy_assets(temporary_output, assets)
        for page in pages:
            content, headings = render_markdown(page, pages_by_source, assets)
            destination = temporary_output / route_directory(page.route) / "index.html"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(html_document(page, content, headings, pages, chapters), encoding="utf-8")
        validate_site(temporary_output)
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        shutil.copytree(temporary_output, OUTPUT_DIR)
    return len(pages), skipped

def main() -> None:
    page_count, skipped = build_site()
    for chapter in skipped:
        print(f"跳过未发布章节 {chapter}（缺少 README.md）")
    print(f"已生成 {page_count} 个页面：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
