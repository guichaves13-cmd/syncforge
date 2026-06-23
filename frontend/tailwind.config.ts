import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: { DEFAULT: "#0a0a0c", soft: "#111114", card: "#16161b" },
        line: "#23232a",
        ink: { DEFAULT: "#f2f2f5", muted: "#9aa0aa", subtle: "#5a5f6a" },
        accent: { DEFAULT: "#6366f1", hover: "#7c7ef0", soft: "#1c1d35" },
        ok: "#22c55e",
        warn: "#eab308",
        err: "#ef4444",
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(99,102,241,0.15), 0 8px 32px -8px rgba(99,102,241,0.3)",
      },
    },
  },
  plugins: [],
} satisfies Config;
