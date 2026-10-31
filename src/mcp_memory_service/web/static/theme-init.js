/* Apply the saved theme before the dashboard markup can paint. */
(() => {
    try {
        const settings = JSON.parse(
            localStorage.getItem('memoryDashboardSettings') || 'null'
        );
        if (settings && settings.theme === 'dark') {
            document.body.classList.add('dark-mode');
        }
    } catch (_error) {
        // Keep the default light theme when storage is blocked or malformed.
    }
})();
