import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Sidebar, type SidebarItem } from "@/components/layout/Sidebar";
import { MobileSidebarProvider } from "@/components/layout/MobileSidebarContext";

const DOCS_SIDEBAR: SidebarItem[] = [
  {
    title: "Getting Started",
    items: [
      { title: "Installation", href: "/docs/installation" },
      { title: "Quickstart",   href: "/docs/quickstart" },
    ],
  },
  {
    title: "Tutorials",
    items: [
      { title: "Train a Model End to End", href: "/docs/tutorials/train-end-to-end" },
      { title: "Mixed Precision Training", href: "/docs/tutorials/mixed-precision" },
      { title: "Speed Up with lucid.compile", href: "/docs/tutorials/compile" },
      { title: "Export to Core ML",        href: "/docs/tutorials/export-coreml" },
    ],
  },
  {
    title: "Guides",
    items: [
      { title: "Training Loop", href: "/docs/training-loop" },
      { title: "Model Zoo",     href: "/docs/model-zoo" },
      { title: "Autograd",      href: "/docs/autograd" },
      { title: "Metal Device",  href: "/docs/metal-device" },
    ],
  },
  {
    title: "Concepts",
    items: [
      { title: "Tensor",     href: "/docs/tensor" },
      { title: "Modules",    href: "/docs/modules" },
      { title: "Optimizers", href: "/docs/optimizers" },
      { title: "C++ Engine", href: "/docs/cpp-engine" },
    ],
  },
  {
    title: "Performance",
    items: [
      { title: "Compile (JIT)",   href: "/docs/compile" },
      { title: "Mixed Precision", href: "/docs/amp" },
      { title: "Profiler",        href: "/docs/profiler" },
    ],
  },
  {
    title: "Numerics",
    items: [
      { title: "Linear Algebra",    href: "/docs/linalg" },
      { title: "FFT",               href: "/docs/fft" },
      { title: "Signal Windows",    href: "/docs/signal" },
      { title: "Special Functions", href: "/docs/special" },
      { title: "Distributions",     href: "/docs/distributions" },
    ],
  },
  {
    title: "Utilities",
    items: [
      { title: "Einops",        href: "/docs/einops" },
      { title: "Serialization", href: "/docs/serialization" },
    ],
  },
  {
    title: "Migration",
    items: [
      { title: "Coming from the Reference Framework", href: "/docs/migration/from-reference-framework" },
    ],
  },
];

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <MobileSidebarProvider items={DOCS_SIDEBAR}>
      <div className="flex min-h-dvh flex-col">
        <Header />
        <div className="mx-auto flex w-full max-w-screen-2xl flex-1 gap-0 px-4 sm:px-6 pt-14">
          <Sidebar items={DOCS_SIDEBAR} className="sticky top-14 h-[calc(100dvh-3.5rem)]" />
          <main
            id="main-content"
            tabIndex={-1}
            className="flex-1 min-w-0 pt-10 pb-12 lg:px-8 focus:outline-none"
          >
            {children}
          </main>
        </div>
        <Footer />
      </div>
    </MobileSidebarProvider>
  );
}
