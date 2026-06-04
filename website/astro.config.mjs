import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import { readdirSync, readFileSync } from "node:fs";
import { basename, extname, join } from "node:path";
import tailwindcss from "@tailwindcss/vite";

import cloudflare from "@astrojs/cloudflare";

const docsDir = new URL("./src/content/docs/", import.meta.url).pathname;

const faviconHead = [
  {
    tag: "link",
    attrs: { rel: "apple-touch-icon", sizes: "57x57", href: "/favicon-57x57.png" },
  },
  {
    tag: "link",
    attrs: { rel: "apple-touch-icon", sizes: "60x60", href: "/favicon-60x60.png" },
  },
  {
    tag: "link",
    attrs: { rel: "apple-touch-icon", sizes: "72x72", href: "/favicon-72x72.png" },
  },
  {
    tag: "link",
    attrs: { rel: "apple-touch-icon", sizes: "76x76", href: "/favicon-76x76.png" },
  },
  {
    tag: "link",
    attrs: { rel: "apple-touch-icon", sizes: "114x114", href: "/favicon-114x114.png" },
  },
  {
    tag: "link",
    attrs: { rel: "apple-touch-icon", sizes: "120x120", href: "/favicon-120x120.png" },
  },
  {
    tag: "link",
    attrs: { rel: "apple-touch-icon", sizes: "144x144", href: "/favicon-144x144.png" },
  },
  {
    tag: "link",
    attrs: { rel: "apple-touch-icon", sizes: "152x152", href: "/favicon-152x152.png" },
  },
  {
    tag: "link",
    attrs: { rel: "apple-touch-icon", sizes: "180x180", href: "/favicon-180x180.png" },
  },
  {
    tag: "link",
    attrs: { rel: "icon", type: "image/png", sizes: "16x16", href: "/favicon-16x16.png" },
  },
  {
    tag: "link",
    attrs: { rel: "icon", type: "image/png", sizes: "32x32", href: "/favicon-32x32.png" },
  },
  {
    tag: "link",
    attrs: { rel: "icon", type: "image/png", sizes: "96x96", href: "/favicon-96x96.png" },
  },
  {
    tag: "link",
    attrs: { rel: "icon", type: "image/png", sizes: "192x192", href: "/favicon-192x192.png" },
  },
  {
    tag: "link",
    attrs: { rel: "shortcut icon", type: "image/x-icon", href: "/favicon.ico" },
  },
  {
    tag: "link",
    attrs: { rel: "icon", type: "image/x-icon", href: "/favicon.ico" },
  },
  {
    tag: "meta",
    attrs: { name: "msapplication-TileColor", content: "#ffffff" },
  },
  {
    tag: "meta",
    attrs: { name: "msapplication-TileImage", content: "/favicon-144x144.png" },
  },
  {
    tag: "meta",
    attrs: { name: "msapplication-config", content: "/browserconfig.xml" },
  },
];

function parseScalar(value) {
  const trimmed = value.trim();

  if (trimmed === "true") return true;
  if (trimmed === "false") return false;
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed);

  return trimmed.replace(/^["']|["']$/g, "");
}

function parseFrontmatter(filePath) {
  const content = readFileSync(filePath, "utf8");
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  const frontmatter = {};
  let section = null;

  if (!match) return frontmatter;

  for (const line of match[1].split(/\r?\n/)) {
    const nestedMatch = line.match(/^  ([A-Za-z0-9_-]+):\s*(.*)$/);
    if (section && nestedMatch) {
      frontmatter[section][nestedMatch[1]] = parseScalar(nestedMatch[2]);
      continue;
    }

    const topLevelMatch = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!topLevelMatch) continue;

    const [, key, value] = topLevelMatch;
    if (value === "") {
      frontmatter[key] = {};
      section = key;
    } else {
      frontmatter[key] = parseScalar(value);
      section = null;
    }
  }

  return frontmatter;
}

function slugifyPathPart(part) {
  return part
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function titleizePathPart(part) {
  return part
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((word) =>
      word.toLowerCase() === "cli" ? "CLI" : word[0].toUpperCase() + word.slice(1)
    )
    .join(" ");
}

function getSidebarOrder(page) {
  return page.frontmatter.sidebar?.order ?? Number.MAX_VALUE;
}

function getPageLabel(page) {
  return page.frontmatter.sidebar?.label ?? page.frontmatter.title ?? titleizePathPart(page.name);
}

function sortSidebarItems(items) {
  return items.sort((a, b) => {
    const orderDelta = a.order - b.order;
    if (orderDelta !== 0) return orderDelta;
    return a.label.localeCompare(b.label);
  });
}

function readDocsDirectory(directory, routeParts = []) {
  const pages = [];
  const groups = [];

  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    if (entry.name.startsWith("_")) continue;

    const entryPath = join(directory, entry.name);

    if (entry.isDirectory()) {
      const group = readDocsDirectory(entryPath, [...routeParts, entry.name]);
      if (group.items.length > 0) groups.push(group);
      continue;
    }

    const extension = extname(entry.name);
    if (extension !== ".md" && extension !== ".mdx") continue;

    const name = basename(entry.name, extension);
    const frontmatter = parseFrontmatter(entryPath);
    if (frontmatter.sidebar?.hidden) continue;

    const pageRouteParts = name === "index" ? routeParts : [...routeParts, name];
    const page = {
      frontmatter,
      label: frontmatter.sidebar?.label ?? frontmatter.title ?? titleizePathPart(name),
      name,
      order: frontmatter.sidebar?.order ?? Number.MAX_VALUE,
      slug: pageRouteParts.map(slugifyPathPart).join("/"),
    };

    pages.push(page);
  }

  const indexPage = pages.find((page) => page.name === "index");
  const childPages = pages.filter((page) => page.name !== "index");
  const orderedChildItems = sortSidebarItems([
    ...childPages.map((page) => ({
      label: getPageLabel(page),
      order: getSidebarOrder(page),
      slug: page.slug,
    })),
    ...groups,
  ]);
  const childItems = orderedChildItems.map(({ label, order, ...item }) => ({
    label,
    ...item,
  }));

  if (routeParts.length === 0) {
    return sortSidebarItems([
      ...pages.map((page) => ({
        label: getPageLabel(page),
        order: getSidebarOrder(page),
        slug: page.slug,
      })),
      ...groups,
    ]).map(({ label, order, ...item }) => ({ label, ...item }));
  }

  const groupLabel = indexPage ? getPageLabel(indexPage) : titleizePathPart(routeParts.at(-1));
  const groupOrder = indexPage
    ? getSidebarOrder(indexPage)
    : Math.min(...orderedChildItems.map((item) => item.order));
  const indexItem = indexPage
    ? [{ label: getPageLabel(indexPage), slug: indexPage.slug }]
    : [];

  return {
    items: [...indexItem, ...childItems],
    label: groupLabel,
    order: groupOrder,
  };
}

export default defineConfig({
  integrations: [
    starlight({
      title: "Morphace",
      description:
        "Generate landmark-based face-morphing videos from two face images.",
      favicon: "/favicon.ico",
      head: faviconHead,
      customCss: ["./src/styles/global.css"],
      sidebar: readDocsDirectory(docsDir),
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 3,
      },
    }),
  ],

  vite: {
    plugins: [tailwindcss()],
  },

  adapter: cloudflare()
});