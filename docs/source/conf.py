# Configuration file for the Sphinx documentation builder.

project = 'Pallas TPU Kernel 开发教程'
copyright = '2025, Pallas TPU Tutorial Contributors'
author = 'Pallas TPU Tutorial Contributors'

extensions = [
    'myst_parser',
    'sphinx_copybutton',
]

myst_enable_extensions = [
    'colon_fence',
    'dollarmath',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
