/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Base dark theme colors
        black: '#000000',
        white: '#FFFFFF',
        
        // Zinc scale for UI elements
        zinc: {
          50: '#FAFAFA',
          100: '#F4F4F5',
          200: '#E4E4E7',
          300: '#D4D4D8',
          400: '#A1A1AA',
          500: '#71717A',
          600: '#52525B',
          700: '#3F3F46',
          800: '#27272A',
          900: '#18181B',
          950: '#0A0A0B',
        },
        
        // Shard accent colors
        shard: {
          roampal: '#3B82F6',    // Blue
          dev: '#10B981',        // Green  
          creative: '#F59E0B',   // Amber
          analyst: '#8B5CF6',    // Purple
          coach: '#EC4899',      // Pink
        },
        
        // Semantic colors
        primary: {
          DEFAULT: '#3B82F6',
          dark: '#2563EB',
          light: '#60A5FA',
        },
        success: {
          DEFAULT: '#10B981',
          dark: '#059669',
          light: '#34D399',
        },
        warning: {
          DEFAULT: '#F59E0B',
          dark: '#D97706',
          light: '#FCD34D',
        },
        error: {
          DEFAULT: '#EF4444',
          dark: '#DC2626',
          light: '#F87171',
        },
      },
      
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
      },
      
      fontSize: {
        '2xs': ['11px', '14px'],
        'xs': ['12px', '16px'],
        'sm': ['14px', '20px'],
        'base': ['16px', '24px'],
        'lg': ['18px', '28px'],
        'xl': ['20px', '28px'],
      },
      
      spacing: {
        '18': '4.5rem',
        '72': '18rem',
        '80': '20rem',
        '88': '22rem',
        '96': '24rem',
      },
      
      animation: {
        'fade-in': 'fadeIn 0.2s ease-in',
        'fade-out': 'fadeOut 0.2s ease-out',
        'slide-in': 'slideIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'bounce-soft': 'bounceSoft 1s infinite',
        'pulse-soft': 'pulseSoft 2s infinite',
        'pulse-subtle': 'pulseSubtle 3s ease-in-out infinite',
        'spin-slow': 'spin 2s linear infinite',
        'ping-soft': 'pingSoft 1.5s infinite',
        'glow': 'glow 2s ease-in-out infinite',
        'typing': 'typing 1.4s infinite',
        'blink': 'blink 1s ease-in-out infinite',
        'progress': 'progress 2s ease-in-out infinite',
      },
      
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeOut: {
          '0%': { opacity: '1' },
          '100%': { opacity: '0' },
        },
        slideIn: {
          '0%': { transform: 'translateX(-100%)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        bounceSoft: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-4px)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.8' },
        },
        pingSoft: {
          '0%': { transform: 'scale(1)', opacity: '1' },
          '75%, 100%': { transform: 'scale(1.1)', opacity: '0' },
        },
        glow: {
          '0%, 100%': { boxShadow: '0 0 12px rgba(59, 130, 246, 0.5)' },
          '50%': { boxShadow: '0 0 24px rgba(59, 130, 246, 0.8)' },
        },
        typing: {
          '0%': { opacity: '0' },
          '50%': { opacity: '1' },
          '100%': { opacity: '0' },
        },
        pulseSubtle: {
          '0%, 100%': {
            opacity: '1',
            boxShadow: '0 0 0 0 rgba(59, 130, 246, 0)',
          },
          '50%': {
            opacity: '0.95',
            boxShadow: '0 0 20px 5px rgba(59, 130, 246, 0.1)',
          },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        progress: {
          '0%': { width: '0%' },
          '50%': { width: '70%' },
          '100%': { width: '100%' },
        },
      },
      
      backdropBlur: {
        xs: '2px',
        sm: '4px',
        md: '8px',
        lg: '12px',
        xl: '16px',
      },
      
      borderRadius: {
        'sm': '4px',
        'md': '6px',
        'lg': '8px',
        'xl': '12px',
        '2xl': '16px',
        '3xl': '24px',
      },
      
      boxShadow: {
        'glow-blue': '0 0 24px rgba(59, 130, 246, 0.3)',
        'glow-green': '0 0 24px rgba(16, 185, 129, 0.3)',
        'glow-amber': '0 0 24px rgba(245, 158, 11, 0.3)',
        'glow-purple': '0 0 24px rgba(139, 92, 246, 0.3)',
        'glow-pink': '0 0 24px rgba(236, 72, 153, 0.3)',
        'inner-soft': 'inset 0 2px 4px rgba(0, 0, 0, 0.2)',
        'elevation-1': '0 2px 8px rgba(0, 0, 0, 0.3)',
        'elevation-2': '0 4px 16px rgba(0, 0, 0, 0.4)',
        'elevation-3': '0 8px 24px rgba(0, 0, 0, 0.5)',
      },
      
      transitionTimingFunction: {
        'bounce-in': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
        'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
      },
      
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'mesh-gradient': 'linear-gradient(180deg, #0A0A0B 0%, #131316 50%, #0A0A0B 100%)',
      },
    },
  },
  plugins: [
    // Custom plugin for shard-specific utilities
    function({ addUtilities, theme }) {
      const shardColors = theme('colors.shard');
      const shardUtilities = {};
      
      Object.entries(shardColors).forEach(([name, color]) => {
        // Border utilities
        shardUtilities[`.border-shard-${name}`] = {
          borderColor: color,
        };
        
        // Glow utilities
        shardUtilities[`.glow-shard-${name}`] = {
          boxShadow: `0 0 24px ${color}33`,
        };
        
        // Background utilities
        shardUtilities[`.bg-shard-${name}-soft`] = {
          backgroundColor: `${color}0D`,
        };
        
        // Text utilities
        shardUtilities[`.text-shard-${name}`] = {
          color: color,
        };
      });
      
      addUtilities(shardUtilities);
    },
  ],
}