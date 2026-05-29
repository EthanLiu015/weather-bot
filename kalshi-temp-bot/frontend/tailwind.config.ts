import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: '#1a1a1a',
        border: '#2a2a2a',
      },
    },
  },
  plugins: [],
}

export default config
