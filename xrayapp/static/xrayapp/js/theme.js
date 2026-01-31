document.addEventListener('DOMContentLoaded', () => {
  // The template uses a stable id (`theme-toggle`). Keep a class fallback for older markup.
  const themeToggle = document.getElementById('theme-toggle') || document.querySelector('.theme-toggle');
  const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
  
  // Function to set the theme
  const setTheme = (themeName) => {
    document.documentElement.setAttribute('data-bs-theme', themeName);
    localStorage.setItem('theme', themeName);
    updateThemeIcon(themeName);
  };
  
  // Function to toggle the theme
  const toggleTheme = () => {
    const currentTheme = getCurrentTheme();
    if (currentTheme === 'dark') {
      setTheme('light');
    } else {
      setTheme('dark');
    }
  };
  
  // Function to get current theme
  const getCurrentTheme = () => {
    return document.documentElement.getAttribute('data-bs-theme') || 
           (prefersDarkScheme.matches ? 'dark' : 'light');
  };
  
  // Function to update the icon
  const updateThemeIcon = (themeName) => {
    if (!themeToggle) return;
    
    if (themeName === 'dark') {
      themeToggle.innerHTML = '<i class="bi bi-sun-fill"></i>';
      themeToggle.setAttribute('title', gettext('Switch to light mode'));
    } else {
      themeToggle.innerHTML = '<i class="bi bi-moon-fill"></i>';
      themeToggle.setAttribute('title', gettext('Switch to dark mode'));
    }
  };
  
  // Initialize theme icon only (theme already applied by inline script)
  const initializeThemeIcon = () => {
    const theme = getCurrentTheme();
    updateThemeIcon(theme);
  };
  
  // Add event listener to theme toggle button
  if (themeToggle) {
    themeToggle.addEventListener('click', toggleTheme);
  }
  
  // Listen for system theme changes when user preference is 'auto'
  prefersDarkScheme.addEventListener('change', (e) => {
    const userTheme = window.userPreferences?.theme || localStorage.getItem('theme');
    if (!userTheme || userTheme === 'auto') {
      setTheme(e.matches ? 'dark' : 'light');
    }
  });
  
  // Initialize the theme icon (theme already applied by inline script)
  initializeThemeIcon();
}); 