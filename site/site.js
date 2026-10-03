(() => {
    const root = document.documentElement;
    const body = document.body;
    const darkMode = window.matchMedia("(prefers-color-scheme: dark)");

    function currentTheme() {
        if (root.dataset.theme === "light" || root.dataset.theme === "dark") {
            return root.dataset.theme;
        }
        return darkMode.matches ? "dark" : "light";
    }

    function updateThemeButtons() {
        const nextThemeValue = currentTheme() === "dark" ? "light" : "dark";
        const nextTheme = nextThemeValue === "light" ? "浅色" : "深色";
        for (const button of document.querySelectorAll("[data-theme-toggle]")) {
            button.dataset.nextTheme = nextThemeValue;
            button.setAttribute("aria-label", `切换到${nextTheme}主题`);
            button.setAttribute("title", `切换到${nextTheme}主题`);
        }
    }

    function toggleTheme() {
        const theme = currentTheme() === "dark" ? "light" : "dark";
        root.dataset.theme = theme;
        try {
            localStorage.setItem("pallas-tpu-theme", theme);
        } catch (error) {
            // 浏览器禁用存储时，当前页面仍然可以切换主题。
        }
        updateThemeButtons();
    }

    for (const button of document.querySelectorAll("[data-theme-toggle]")) {
        button.addEventListener("click", toggleTheme);
    }
    darkMode.addEventListener("change", updateThemeButtons);
    updateThemeButtons();

    const navigationToggles = document.querySelectorAll("[data-nav-toggle]");
    const navigationSidebar = document.getElementById("site-navigation");
    const wideLayout = window.matchMedia("(min-width: 801px)");

    function setNavigationOpen(open) {
        body.classList.toggle("nav-open", open);
        navigationSidebar.inert = !wideLayout.matches && !open;
        for (const button of navigationToggles) {
            button.setAttribute("aria-expanded", String(open));
        }
    }

    for (const button of navigationToggles) {
        button.addEventListener("click", () => setNavigationOpen(!body.classList.contains("nav-open")));
    }
    for (const button of document.querySelectorAll("[data-nav-close]")) {
        button.addEventListener("click", () => setNavigationOpen(false));
    }
    wideLayout.addEventListener("change", () => setNavigationOpen(false));
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && body.classList.contains("nav-open")) {
            setNavigationOpen(false);
        }
    });
    setNavigationOpen(false);

    for (const heading of document.querySelectorAll(".doc-content h2[id], .doc-content h3[id]")) {
        const anchor = document.createElement("a");
        anchor.className = "heading-anchor";
        anchor.href = `#${heading.id}`;
        anchor.textContent = "#";
        anchor.setAttribute("aria-label", `链接到“${heading.textContent.trim()}”`);
        heading.append(anchor);
    }

    const tocLinks = Array.from(document.querySelectorAll(".page-toc a[href^='#']"));
    const tocHeadings = tocLinks.map((link) => document.getElementById(decodeURIComponent(link.hash.slice(1)))).filter(Boolean);
    let tocUpdateQueued = false;

    function updateActiveHeading() {
        tocUpdateQueued = false;
        let activeHeading = tocHeadings[0] || null;
        for (const heading of tocHeadings) {
            if (heading.getBoundingClientRect().top <= 160) {
                activeHeading = heading;
            } else {
                break;
            }
        }
        for (const link of tocLinks) {
            if (activeHeading && decodeURIComponent(link.hash.slice(1)) === activeHeading.id) {
                link.setAttribute("aria-current", "true");
            } else {
                link.removeAttribute("aria-current");
            }
        }
    }

    function queueTocUpdate() {
        if (!tocUpdateQueued) {
            tocUpdateQueued = true;
            window.requestAnimationFrame(updateActiveHeading);
        }
    }

    if (tocHeadings.length > 0) {
        window.addEventListener("scroll", queueTocUpdate, {passive: true});
        updateActiveHeading();
    }

    const currentNavigationItem = document.querySelector(".book-nav [aria-current='page']");
    if (currentNavigationItem) {
        currentNavigationItem.scrollIntoView({block: "nearest"});
    }

    const dialog = document.getElementById("source-viewer");
    const dialogTitle = document.getElementById("source-viewer-title");
    const dialogKind = dialog.querySelector(".source-viewer-kind");
    const dialogRawLink = dialog.querySelector(".source-viewer-raw");
    const dialogBody = dialog.querySelector(".source-viewer-body");
    const dialogStatus = dialog.querySelector(".source-viewer-status");
    const dialogContent = dialog.querySelector(".source-viewer-content");
    const sourceCache = new Map();
    let sourceRequest = 0;
    let sourceOpener = null;

    async function fetchSourcePreview(url) {
        if (!sourceCache.has(url)) {
            const request = fetch(url).then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                return response.text();
            }).catch((error) => {
                sourceCache.delete(url);
                throw error;
            });
            sourceCache.set(url, request);
        }
        return sourceCache.get(url);
    }

    function sourceFilename(url) {
        const filename = new URL(url, document.baseURI).pathname.split("/").at(-1);
        try {
            return decodeURIComponent(filename);
        } catch (error) {
            return filename;
        }
    }

    async function openSource(link) {
        sourceRequest += 1;
        const request = sourceRequest;
        sourceOpener = link;
        const rawUrl = link.href;
        const previewUrl = new URL(link.dataset.sourcePreview, document.baseURI).href;
        const filename = sourceFilename(rawUrl);
        dialogTitle.textContent = filename;
        dialogKind.textContent = filename.endsWith(".py") ? "Python" : "Text";
        dialogRawLink.href = rawUrl;
        dialogStatus.textContent = "正在载入文件…";
        dialogStatus.hidden = false;
        dialogStatus.removeAttribute("data-error");
        dialogContent.hidden = true;
        dialogContent.replaceChildren();
        dialogBody.scrollTo(0, 0);
        if (!dialog.open) {
            dialog.showModal();
            root.classList.add("source-viewer-open");
        }
        try {
            const html = await fetchSourcePreview(previewUrl);
            if (request !== sourceRequest) {
                return;
            }
            dialogContent.innerHTML = html;
            dialogStatus.hidden = true;
            dialogContent.hidden = false;
        } catch (error) {
            if (request !== sourceRequest) {
                return;
            }
            dialogStatus.textContent = `无法载入文件：${error.message}`;
            dialogStatus.setAttribute("data-error", "");
        }
    }

    document.addEventListener("click", (event) => {
        const link = event.target.closest("a.source-link[data-source-preview]");
        if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
            return;
        }
        event.preventDefault();
        openSource(link);
    });
    dialog.querySelector("[data-source-close]").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => {
        const bounds = dialog.getBoundingClientRect();
        const outside = event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom;
        if (event.target === dialog && outside) {
            dialog.close();
        }
    });
    dialog.addEventListener("close", () => {
        sourceRequest += 1;
        root.classList.remove("source-viewer-open");
        if (sourceOpener) {
            sourceOpener.focus();
        }
    });
})();
