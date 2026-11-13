/** Route entrance: a fade, never a slide, and one the server renders visible.
 *
 *  This element wraps every page, so a ``y`` offset here displaces *all*
 *  content — including every ``#fragment`` target — for as long as it takes to
 *  settle.  A browser or the Next router scrolls to a heading by its position
 *  at that moment, so a fragment followed during a slide landed off the
 *  ``scroll-mt-24`` line and stayed there (4–11 px measured), and any position
 *  read in the tail saw a page that had not stopped moving.  Opacity moves
 *  nothing (LCD-1001).
 *
 *  The fade is a CSS animation (``.page-enter``), not a framer-motion
 *  ``initial``.  ``initial={{ opacity: 0 }}`` makes the server write
 *  ``style="opacity:0"`` into the exported HTML and keeps the page invisible
 *  until the JavaScript has loaded and hydrated: blank with scripts off, for a
 *  crawler, and for anyone on a slow connection.  A stylesheet animation starts
 *  at first paint and needs no script (LCD-1005).  A new element per route, so
 *  it replays on every navigation. */
export default function Template({
  children,
}: {
  children: React.ReactNode;
}) {
  return <div className="page-enter">{children}</div>;
}
