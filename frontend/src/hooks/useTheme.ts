import { useEffect, useState } from "react";

export type Theme = "dark" | "light";

const STORAGE_KEY = "ryudb.theme.v1";

function load(): Theme {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === "dark" || raw === "light") return raw;
  } catch { /* unavailable */ }
  // Default to the OS preference when no stored choice exists.
  try {
    if (typeof matchMedia !== "undefined" && matchMedia("(prefers-color-scheme: light)").matches) {
      return "light";
    }
  } catch { /* matchMedia unavailable */ }
  return "dark";
}

/** Dark/light theme toggle, persisted to localStorage and reflected on
 * <html data-theme="..."> so the CSS variable palettes switch. Defaults to
 * the OS color-scheme preference on first run. */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(load);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem(STORAGE_KEY, theme); } catch { /* best-effort */ }
  }, [theme]);

  const toggle = () => setTheme((t) => (t === "dark" ? "light" : "dark"));
  return { theme, setTheme, toggle };
}