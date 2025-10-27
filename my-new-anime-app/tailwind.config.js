/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // --- [ ★★★ 이 부분을 추가하세요 ★★★ ] ---
      colors: {
        'squirtle-light-blue': '#E0F7FA',
        'squirtle-blue': '#7FDBFF',
        'squirtle-dark-blue': '#00A3D9',
        'squirtle-border': '#B2EBF2',
      },
      // ------------------------------------
    },
  },
  plugins: [],
}