/* Apply the saved theme before the dashboard markup can paint. */
(() => {
    let theme = null;
    try {
        const settings = JSON.parse(
            localStorage.getItem('memoryDashboardSettings') || 'null'
        );
        theme = settings && settings.theme;
    } catch (_error) {
        // Storage is blocked or malformed: app.js follows the system then, so do the same.
    }
    // Same rule as resolveTheme() in app.js: anything but an explicit choice follows the OS.
    const isDark = theme === 'dark' || (
        theme !== 'light'
        && typeof window.matchMedia === 'function'
        && window.matchMedia('(prefers-color-scheme: dark)').matches
    );
    if (isDark) {
        document.body.classList.add('dark-mode');
    }
})();
