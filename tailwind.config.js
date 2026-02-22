/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./static/*.html"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Helvetica Neue"', 'Helvetica', 'Arial', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace']
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(56,189,248,.28), 0 18px 38px -18px rgba(56,189,248,.45)'
      }
    }
  },
  plugins: []
};
