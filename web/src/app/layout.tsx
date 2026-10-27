import type { Metadata, Viewport } from "next";
import "./globals.css";
import "katex/dist/katex.min.css";
import { SkipLink } from "@/components/layout/SkipLink";
import { ScrollProgress } from "@/components/layout/ScrollProgress";
import { BackToTop } from "@/components/layout/BackToTop";
import { ReducedMotionProvider } from "@/components/motion/ReducedMotionProvider";
import { ThemeBoot, ThemeProvider } from "@/components/layout/ThemeProvider";
import { RecentPageTracker } from "@/components/layout/RecentPageTracker";
import { InlineCodeCopy } from "@/components/layout/InlineCodeCopy";
import { RouteProgress } from "@/components/layout/RouteProgress";
import { brandColors } from "@/lib/brand-kit";
import { SITE_ORIGIN, brandAsset } from "@/lib/site";

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)",  color: brandColors.night },
    { media: "(prefers-color-scheme: light)", color: brandColors.lavender },
  ],
  colorScheme: "dark light",
};

export const metadata: Metadata = {
  // metadataBase is used by Next to absolutise relative URLs — most
  // importantly the social-preview image — since crawlers skip relative
  // ones.  Origin only: ``brandAsset()`` URLs already start with the base
  // path, and Next would join a base path in metadataBase on top of it.
  metadataBase: new URL(SITE_ORIGIN),
  // The brand kit's icons, linked explicitly rather than as app/favicon.ico
  // + app/icon.svg (see scripts/sync-brand.mjs for the basePath bug that
  // drops the favicon from the home page).  Order as in the kit README:
  // the .ico for browsers without SVG favicons, then the optically
  // corrected SVG and raster fallbacks.
  icons: {
    icon: [
      { url: brandAsset("favicon.ico"), sizes: "16x16 32x32 48x48", type: "image/x-icon" },
      { url: brandAsset("favicon.svg"), type: "image/svg+xml" },
      { url: brandAsset("favicon-16.png"), sizes: "16x16", type: "image/png" },
      { url: brandAsset("favicon-32.png"), sizes: "32x32", type: "image/png" },
      { url: brandAsset("favicon-48.png"), sizes: "48x48", type: "image/png" },
    ],
    apple: [{ url: brandAsset("apple-touch-icon.png"), sizes: "180x180", type: "image/png" }],
  },
  title: {
    default: "Lucid — Apple Silicon ML Framework",
    template: "%s | Lucid",
  },
  description:
    "Production-grade machine learning framework for Apple Silicon. MLX + Accelerate native backend with a familiar nn.Module API.",
  keywords: [
    "machine learning",
    "Apple Silicon",
    "MLX",
    "Accelerate",
    "ML framework",
    "Python",
    "deep learning",
  ],
  authors: [{ name: "Chan Lee" }],
  creator: "Chan Lee",
  openGraph: {
    type: "website",
    locale: "en_US",
    title: "Lucid — Apple Silicon ML Framework",
    description:
      "Production-grade ML framework for Apple Silicon. MLX + Accelerate native backend.",
    siteName: "Lucid Docs",
    // The kit's social card; Next copies it into twitter:image as well.
    images: [
      {
        url: brandAsset("og-image.png"),
        width: 1200,
        height: 630,
        type: "image/png",
        alt: "Lucid — Deep learning framework built from scratch for Apple Silicon",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Lucid — Apple Silicon ML Framework",
    description:
      "Production-grade ML framework for Apple Silicon. MLX + Accelerate native backend.",
  },
  robots: {
    index: true,
    follow: true,
  },
  // No canonical (or og:url) here: every route inherits the root layout's
  // metadata, so one set here would name the home page as the canonical
  // copy of every page.  Each page sets its own through ``canonical()``.
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* ThemeBoot runs SYNCHRONOUSLY before React hydrates so the
            user's persisted theme is applied without a dark/light flash. */}
        <ThemeBoot />
      </head>
      <body>
        <SkipLink />
        <ScrollProgress />
        <RecentPageTracker />
        <InlineCodeCopy />
        <RouteProgress />
        <ThemeProvider>
          <ReducedMotionProvider>{children}</ReducedMotionProvider>
        </ThemeProvider>
        <BackToTop />
      </body>
    </html>
  );
}
