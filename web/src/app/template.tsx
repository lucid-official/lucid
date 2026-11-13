"use client";

import { motion } from "framer-motion";
import { springs } from "@/components/motion/springs";

/** Route entrance: a fade, never a slide.  This element wraps every page, so a
 *  ``y`` offset here displaces *all* content — including every ``#fragment``
 *  target — for the ~300 ms the spring takes to settle, starting only once the
 *  page has hydrated.  A browser or the Next router scrolls to a heading by its
 *  position at that moment, so a fragment followed during the slide landed off
 *  the ``scroll-mt-24`` line and stayed there (4–11 px measured, with the
 *  reference body's own slide added), and any position read in the tail saw a
 *  page that had not stopped moving.  Opacity moves nothing. */
export default function Template({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={springs.smooth}
    >
      {children}
    </motion.div>
  );
}
