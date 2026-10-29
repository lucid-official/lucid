import type { Metadata } from "next";
import { canonical } from "@/lib/site";
import Link from "next/link";
import { FadeIn } from "@/components/motion/FadeIn";
import { Badge } from "@/components/ui/badge";
import { ArrowRight } from "lucide-react";

export const metadata: Metadata = {
  title: "Documentation",
  description: "Lucid documentation — guides, concepts, and tutorials.",
  alternates: canonical("/docs"),
};

const QUICK_LINKS = [
  {
    title: "Installation",
    description: "Get Lucid installed on your Apple Silicon Mac.",
    href: "/docs/installation",
    badge: "Start here",
  },
  {
    title: "Quickstart",
    description: "Train your first model in under 5 minutes.",
    href: "/docs/quickstart",
  },
  {
    title: "Train a Model End to End",
    description: "Data, a zoo model, an optimizer, evaluation and checkpoint resume.",
    href: "/docs/tutorials/train-end-to-end",
  },
  {
    title: "Coming from the Reference Framework",
    description: "Import map, the metal device, dtype differences and what is not available.",
    href: "/docs/migration/from-reference-framework",
  },
  {
    title: "Autograd",
    description: "Understand automatic differentiation in Lucid.",
    href: "/docs/autograd",
  },
  {
    title: "Metal Device",
    description: "Leverage the MLX GPU backend on Apple Silicon.",
    href: "/docs/metal-device",
  },
] as const;

export default function DocsPage() {
  return (
    <FadeIn>
      <div className="max-w-3xl">
        <h1 className="text-3xl font-bold text-lucid-text-high mb-3">
          Documentation
        </h1>
        <p className="text-lucid-text-mid mb-10 text-base leading-relaxed">
          Everything you need to get started with Lucid, the production-grade ML
          framework for Apple Silicon.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {QUICK_LINKS.map(({ title, description, href, ...rest }) => {
            const badge = "badge" in rest ? rest.badge : undefined;
            return (
              <Link
                key={href}
                href={href}
                className="group flex items-start justify-between rounded-xl border border-lucid-border bg-lucid-surface p-4 transition-all hover:border-lucid-primary/40 hover:bg-lucid-elevated"
              >
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-lucid-text-high">
                      {title}
                    </span>
                    {badge && <Badge variant="default">{badge}</Badge>}
                  </div>
                  <p className="text-xs text-lucid-text-low leading-relaxed">
                    {description}
                  </p>
                </div>
                <ArrowRight className="h-4 w-4 shrink-0 ml-3 mt-0.5 text-lucid-text-disabled transition-transform group-hover:translate-x-0.5 group-hover:text-lucid-primary" />
              </Link>
            );
          })}
        </div>
      </div>
    </FadeIn>
  );
}
