"use client";

import * as React from "react";

/** Land a ``#fragment`` URL on its heading once the page has stopped moving.
 *
 *  The browser follows a fragment once, at load, with a smooth scroll that
 *  fixes its target the moment it starts (``html`` is ``scroll-behavior:
 *  smooth``).  The reference pages are long and the layout above a heading
 *  still changes after that: the web fonts swap in after first paint and the
 *  text re-wraps, moving the heading by hundreds of pixels (−358 / +1 011 px
 *  measured on ``/api/lucid.peft``).  So a copied section link ended short of
 *  its heading — 962 px short at 1x CPU, and ``/api/lucid.nn/Linear#forward`` 62 px
 *  below the ``scroll-mt-24`` line.  This looks again once the fonts are in and
 *  laid out, and puts the heading where the fragment says (LCD-1005).
 *
 *  A scroll the user started wins: any wheel, touch, key, pointer press or click,
 *  a new fragment or a history step ends the wait (seen from hydration on — input
 *  before it is not, and the page then follows the link as asked).  Only a fresh navigation is
 *  followed — a reload or back/forward has the browser's own scroll restoration,
 *  which this must not undo.  A heading that already sits on its line is left
 *  alone.
 *
 *  It cannot work alone: the page content must be in the document when the
 *  browser follows the fragment.  Behind the old root ``loading.tsx`` the export
 *  held the body in a hidden Suspense subtree that an inline script revealed
 *  later (at 4x CPU ~770 ms, after the load event at ~250 ms), so the heading was
 *  not rendered when the browser tried and this had nothing to measure (0 of 6
 *  landed with this effect alone). */
export function FragmentScroll() {
  React.useEffect(() => {
    const id = fragmentId();
    if (!id || navigationType() !== "navigate") return;

    // Any of these is the user (or the history) taking the page somewhere else.
    const takeover = [
      "wheel",
      "touchstart",
      "keydown",
      "pointerdown",
      "click",
      "hashchange",
      "popstate",
    ] as const;
    let live = true;
    // Ends the wait and releases every listener: called on takeover, after the
    // landing, and on unmount.  Removing a listener twice is harmless.
    const stop = () => {
      live = false;
      for (const type of takeover) window.removeEventListener(type, stop, { capture: true });
    };
    for (const type of takeover) window.addEventListener(type, stop, { passive: true, capture: true });

    void layoutSettled().then(() => {
      if (live) landOnHeading(id);
      stop();
    });

    return stop;
  }, []);

  return null;
}

function fragmentId(): string {
  try {
    return decodeURIComponent(window.location.hash.slice(1));
  } catch {
    return "";
  }
}

function navigationType(): string {
  const entry = performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined;
  return entry?.type ?? "navigate";
}

function nextFrame(): Promise<void> {
  return new Promise((done) => requestAnimationFrame(() => done()));
}

/** Resolves once the web fonts are loaded *and* a frame has laid the page out
 *  with them.  ``document.fonts.ready`` resolves at once when no font has started
 *  loading, so look once more after the frame while a load is pending. */
async function layoutSettled(): Promise<void> {
  do {
    await document.fonts.ready;
    await nextFrame();
  } while (document.fonts.status === "loading");
}

function landOnHeading(id: string): void {
  const target = document.getElementById(id);
  if (!target) return;
  const line = parseFloat(getComputedStyle(target).scrollMarginTop) || 0;
  if (Math.abs(target.getBoundingClientRect().top - line) <= 1) return;
  // ``instant``: the page's own ``scroll-behavior: smooth`` would animate over
  // the browser's scroll still in flight, and this call also ends that one.
  target.scrollIntoView({ block: "start", behavior: "instant" });
}
