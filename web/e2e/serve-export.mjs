// Serve the static export (``out/``) the way GitHub Pages does, under ``/lucid``.
//
// ``pnpm dev:fast`` cannot show what a visitor's browser meets: the export has no
// dev overlay and no on-demand compile, and it lives under the ``/lucid``
// basePath.  The exported-build e2e project (playwright.config.ts) opens its
// pages from here.
//
//   node e2e/serve-export.mjs [port]      (default 3100; ``pnpm build`` first)
//
// Kept from Pages: an extensionless path serves ``<path>.html``, a directory
// serves its ``index.html``, anything else is the 404 page with status 404.  No
// dependencies and no caching headers - the e2e only needs the bytes.
// ``EXPORT_DIR`` serves another build (a before/after comparison).
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const BASE = "/lucid";
const ROOT = resolve(process.env.EXPORT_DIR ?? fileURLToPath(new URL("../out", import.meta.url)));
const PORT = Number(process.argv[2] ?? process.env.EXPORT_PORT ?? 3100);

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
  ".webmanifest": "application/manifest+json",
  ".xml": "application/xml",
};

async function isFile(path) {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
}

/** The file a URL path names inside ``out/``, or null. */
async function resolveFile(pathname) {
  if (pathname !== BASE && !pathname.startsWith(`${BASE}/`)) return null;
  const rel = normalize(decodeURIComponent(pathname.slice(BASE.length)));
  const target = resolve(join(ROOT, rel));
  if (target !== ROOT && !target.startsWith(ROOT + sep)) return null;
  for (const candidate of [target, `${target}.html`, join(target, "index.html")]) {
    if (await isFile(candidate)) return candidate;
  }
  return null;
}

const server = createServer(async (req, res) => {
  const { pathname } = new URL(req.url ?? "/", "http://localhost");
  const file = await resolveFile(pathname);
  const target = file ?? join(ROOT, "404.html");
  try {
    const body = await readFile(target);
    res.writeHead(file ? 200 : 404, {
      "content-type": TYPES[extname(target)] ?? "application/octet-stream",
    });
    res.end(body);
  } catch {
    res.writeHead(404, { "content-type": "text/plain" });
    res.end("not found");
  }
});

server.listen(PORT, () => {
  console.log(`serving ${ROOT} at http://localhost:${PORT}${BASE}/`);
});
