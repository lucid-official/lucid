"use client";

import * as React from "react";
import { motion, type HTMLMotionProps } from "framer-motion";
import { springs } from "./springs";

interface FadeInProps extends Omit<HTMLMotionProps<"div">, "initial" | "animate" | "transition"> {
  delay?: number;
  y?: number;
  once?: boolean;
  /** Use whileInView (for below-fold content). Default: false (immediate animate) */
  inView?: boolean;
}

export function FadeIn({
  children,
  delay = 0,
  y = 10,
  once = true,
  inView = false,
  className,
  ...props
}: FadeInProps) {
  if (inView) {
    return (
      <motion.div
        initial={{ opacity: 0, y }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once, amount: 0.05 }}
        transition={{ ...springs.smooth, delay }}
        className={className}
        {...props}
      >
        {children}
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ ...springs.smooth, delay }}
      className={className}
      {...props}
    >
      {children}
    </motion.div>
  );
}

interface FadeInStaggerProps {
  children: React.ReactNode;
  staggerDelay?: number;
  className?: string;
  /** Class for the per-child wrapper — e.g. ``h-full`` so cards in a grid row
   *  stretch to equal height. */
  itemClassName?: string;
  /** Use whileInView (for below-fold content). Default: false */
  inView?: boolean;
}

export function FadeInStagger({
  children,
  staggerDelay = 0.07,
  className,
  itemClassName,
  inView = false,
}: FadeInStaggerProps) {
  const viewportProps = inView
    ? { initial: "hidden", whileInView: "visible", viewport: { once: true, amount: 0.05 as const } }
    : { initial: "hidden", animate: "visible" };

  return (
    <motion.div
      variants={{
        hidden: {},
        visible: { transition: { staggerChildren: staggerDelay } },
      }}
      className={className}
      {...viewportProps}
    >
      {React.Children.map(children, (child) => (
        <motion.div
          className={itemClassName}
          variants={{
            hidden: { opacity: 0, y: 12 },
            visible: { opacity: 1, y: 0, transition: springs.smooth },
          }}
        >
          {child}
        </motion.div>
      ))}
    </motion.div>
  );
}
