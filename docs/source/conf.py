# Configuration file for the Sphinx documentation builder.

project = 'Pallas TPU Kernel 开发教程'
copyright = '2025, Pallas TPU Tutorial Contributors'
author = 'Pallas TPU Tutorial Contributors'

extensions = [
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx.ext.mathjax',
    'sphinx_sitemap',
]

myst_enable_extensions = [
    'colon_fence',
    'dollarmath',
    'fieldlist',
    'tasklist',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Furo theme with dark mode support --
html_theme = 'furo'

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#1a73e8",
        "color-brand-content": "#1a73e8",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8ab4f8",
        "color-brand-content": "#8ab4f8",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ayaka14732/pallas-tpu-tutorial",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/ayaka14732/pallas-tpu-tutorial",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- SEO Configuration --

# Site URL for sitemap generation
html_baseurl = 'https://ayaka14732.github.io/pallas-tpu-tutorial/'

# Meta description for search engines
html_meta = {
    "description": "从零开始学习 JAX Pallas TPU Kernel 开发。面向有 JAX 经验但无 kernel 开发经验的开发者，涵盖 TPU 架构、BlockSpec、流水线、MatMul、FlashAttention、Ragged Paged Attention 等核心主题。",
    "keywords": "JAX, Pallas, TPU, kernel, tutorial, FlashAttention, MatMul, Google TPU, machine learning, 教程",
    "author": "Pallas TPU Tutorial Contributors",
    "og:title": "Pallas TPU Kernel 开发教程",
    "og:description": "从零开始学习 JAX Pallas TPU Kernel 开发 — 面向有 JAX 经验但无 kernel 开发经验的开发者。",
    "og:type": "website",
    "og:url": "https://ayaka14732.github.io/pallas-tpu-tutorial/",
    "twitter:card": "summary",
    "twitter:title": "Pallas TPU Kernel 开发教程",
    "twitter:description": "从零开始学习 JAX Pallas TPU Kernel 开发。涵盖 TPU 架构、BlockSpec、流水线、MatMul、FlashAttention 等核心主题。",
    "google-site-verification": "",
}

# Page title separator for better SEO
html_title = "Pallas TPU Kernel 开发教程"

# Sitemap configuration
sitemap_url_scheme = "{link}"

# Custom CSS
def setup(app):
    app.add_css_file('custom.css')
