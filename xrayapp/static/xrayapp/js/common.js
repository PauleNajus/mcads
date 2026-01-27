// Common JavaScript functions for MCADS

/**
 * Helper to get cookie by name
 * @param {string} name - Cookie name
 * @returns {string|null} - Cookie value
 */
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

/**
 * Fallback for gettext if not available
 */
if (typeof gettext === 'undefined') {
    window.gettext = function(msg) { return msg; };
}

document.addEventListener('DOMContentLoaded', () => {
    // Style card headers based on content
    const headers = document.querySelectorAll('.card-header');
    const headerPatterns = {
        'bg-success': ['Insignificant', 'No Finding'],
        'bg-warning': ['Moderate', 'Nodule', 'Infiltration', 'Atelectasis'],
        'bg-danger': ['Significant', 'Pneumothorax', 'Effusion', 'Cardiomegaly']
    };

    headers.forEach(header => {
        const text = header.textContent;
        // Don't override if it already has a specific background class
        if (header.classList.contains('bg-primary') || 
            header.classList.contains('bg-secondary') ||
            header.classList.contains('bg-info')) {
            return;
        }

        for (const [className, keywords] of Object.entries(headerPatterns)) {
            if (keywords.some(keyword => text.includes(keyword))) {
                header.classList.add(className, 'text-white');
                break;
            }
        }
    });

    // Real-time EEST clock
    function updateEESTTime() {
        const now = new Date();
        const options = {
            timeZone: 'Europe/Vilnius',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        };
        const timeString = new Intl.DateTimeFormat('lt-LT', options).format(now);
        const clockElement = document.getElementById('eest-clock');
        if (clockElement) {
            clockElement.textContent = timeString;
        }
    }

    // Update immediately and then every second
    updateEESTTime();
    setInterval(updateEESTTime, 1000);

    // Register service worker to clean cache
    if ('serviceWorker' in navigator) {
        // Path is relative to the root scope
        navigator.serviceWorker.register("/static/xrayapp/js/service-worker.js?v=1.0.3")
            .catch(() => {
                // Registration failures shouldn't block the app.
                console.debug('Service worker registration failed or blocked');
            });
    }
});
