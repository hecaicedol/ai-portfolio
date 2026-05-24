import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      colors: {
        ink: {
          900: "#0a0a0c",
          800: "#111114",
          700: "#1a1a20",
          600: "#23232b",
          500: "#3a3a45",
          400: "#5a5a66",
          300: "#8a8a95",
          200: "#c4c4cc",
          100: "#eaeaef",
        },
        accent: {
          400: "#a78bfa",
          500: "#8b5cf6",
          600: "#7c3aed",
        },
        ok: "#22c55e",
        warn: "#f59e0b",
        bad: "#ef4444",
      },
    },
  },
  plugins: [],
};

export default config;
