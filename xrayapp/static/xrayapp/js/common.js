// Common JavaScript functions for MCADS

/**
 * Show a generic message modal (replacement for alert)
 * @param {string} message - Message to display
 * @param {string} title - Optional title (default: "Message")
 * @param {boolean} isError - Optional, if true styles the header as error
 */
window.showModal = function(message, title = null, isError = false) {
    const modalEl = document.getElementById('genericMessageModal');
    if (!modalEl) {
        // Fallback if modal is missing for some reason
        alert(message);
        return;
    }

    const titleEl = document.getElementById('genericMessageModalLabel');
    const bodyEl = document.getElementById('genericMessageModalBody');
    const headerEl = modalEl.querySelector('.modal-header');

    // Set content
    if (bodyEl) bodyEl.textContent = message;
    if (titleEl) titleEl.textContent = title || (isError ? gettext('Error') : gettext('Message'));

    // Style
    if (headerEl) {
        if (isError) {
            headerEl.classList.add('bg-danger', 'text-white');
            headerEl.classList.remove('bg-primary', 'bg-info', 'bg-warning');
             // Also check for close button color if needed
            const closeBtn = headerEl.querySelector('.btn-close');
            if(closeBtn) closeBtn.classList.add('btn-close-white');
        } else {
            headerEl.classList.remove('bg-danger', 'text-white');
            const closeBtn = headerEl.querySelector('.btn-close');
            if(closeBtn) closeBtn.classList.remove('btn-close-white');
        }
    }

    // Show
    // @ts-ignore
    const modal = new bootstrap.Modal(modalEl);
    modal.show();
};

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
    // Opt-out class for headers that must keep the default branded look.
    // Example: interpretability visualization headers contain pathology names (e.g. "Infiltration")
    // and would otherwise get auto-colored (yellow/red) by the keyword matcher below.
    const SKIP_HEADER_AUTOCOLOR_CLASS = 'js-skip-header-autocolor';
    const headerPatterns = {
        'bg-success': ['Insignificant', 'No Finding'],
        'bg-warning': ['Moderate', 'Nodule', 'Infiltration', 'Atelectasis'],
        'bg-danger': ['Significant', 'Pneumothorax', 'Effusion', 'Cardiomegaly']
    };

    headers.forEach(header => {
        if (header.classList.contains(SKIP_HEADER_AUTOCOLOR_CLASS)) {
            return;
        }

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
        const clockElement = document.getElementById('server-time');
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
