# 教材网站

这里用 Pandoc、Python 标准库和原生 CSS/JavaScript 把教程构建成多页静态网站。根目录、每章（`chapterNN/`）、每个附录（`appendix_*/`）和每节的 `README.md` 分别生成独立页面；只有包含 `README.md` 的章会发布，因此尚未完成章总览的内容不会意外进入网站。

构建器通过 Pandoc AST 转换链接，不直接用正则表达式修改 Markdown。教程内的 README 链接变为目录式网页地址，实验附件保持原路径，跨仓库资料则指向对应 GitHub 仓库。`.py`、`.txt` 和根目录的 `LICENSE` 文件同时生成带行号的预览，普通点击会在当前页面打开文件窗口，修饰键点击或禁用 JavaScript 时仍可直接访问原文件。

机器清单的代码块写成 ```` ```tpuasm ````。Pandoc 没有这种语言的语法定义，[`tpuasm_highlight.py`](tpuasm_highlight.py) 按 [vscode-tpuasm](https://github.com/ayaka14732/vscode-tpuasm) 的 TextMate 语法划分词法单元（槽名、谓词、助记符、寄存器、内存空间、具名操作数、数值、标号、指示、注释），输出与 Pandoc 高亮相同结构和 class 的 HTML，沿用网站已有的配色。它也接受正文中常见的简写：行首的 `[编号]`、`编号:`、`N ×`，以及不带花括号和槽名的指令。

## 本地构建

系统需要提供 Python 3 和 Pandoc 3.12，然后从仓库根目录运行：

```sh
./site/build.sh
```

构建结果位于 `site/build/`。构建过程会检查生成页面引用的本地文件与页内锚点，发现无效链接时直接失败。

本地预览：

```sh
python3 -m http.server --directory site/build 8000
```

然后访问 <http://127.0.0.1:8000/>。文件预览使用 `fetch`，因此不要直接用 `file://` 打开生成的 HTML。
