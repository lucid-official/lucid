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

/** A malformed request (bad percent-encoding, an unparsable URL) is the client's. */
function isBadRequest(error) {
  return error instanceof URIError || error?.code === "ERR_INVALID_URL";
}

function reply(res, status, type, body) {
  if (res.headersSent) {
    res.destroy();
    return;
  }
  res.writeHead(status, { "content-type": type });
  res.end(body);
}

const server = createServer(async (req, res) => {
  try {
    const { pathname } = new URL(req.url ?? "/", "http://localhost");
    const file = await resolveFile(pathname);
    const target = file ?? join(ROOT, "404.html");
    let body;
    try {
      body = await readFile(target);
    } catch {
      return reply(res, 404, "text/plain", "not found");
    }
    reply(res, file ? 200 : 404, TYPES[extname(target)] ?? "application/octet-stream", body);
  } catch (error) {
    // Whatever one request does, the server stays up for the next.
    if (isBadRequest(error)) reply(res, 400, "text/plain", "bad request");
    else reply(res, 500, "text/plain", "server error");
  }
});

// A missing build would otherwise surface as Playwright's generic 30 s webServer
// timeout (the 404 page it would be serving does not exist either).
if (!(await isFile(join(ROOT, "index.html")))) {
  console.error(`web/out missing — run pnpm build first (no ${join(ROOT, "index.html")})`);
  process.exit(1);
}

// Loopback only: this is a test fixture that serves whatever ``out/`` holds, not
// something to expose on the LAN of a developer machine or a shared runner.
server.listen(PORT, "127.0.0.1", () => {
  console.log(`serving ${ROOT} at http://127.0.0.1:${PORT}${BASE}/`);
});
