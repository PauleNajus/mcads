document.addEventListener('DOMContentLoaded', () => {
  const analysisForm = document.getElementById('analysis-form');
  const formWrapper = document.getElementById('form-wrapper');
  const progressWrapper = document.getElementById('progress-wrapper');
  const progressBar = document.getElementById('analysis-progress-bar');
  const progressPercentage = document.getElementById('progress-percentage');

  // --- Model Selection Persistence ---
  const modelSelect = document.getElementById('model_select');
  if (modelSelect) {
    // Restore last selection
    const lastModel = localStorage.getItem('lastSelectedModel');
    if (lastModel) {
      // Check if the option still exists
      if ([...modelSelect.options].some(o => o.value === lastModel)) {
        modelSelect.value = lastModel;
      }
    }

    // Save selection on change
    modelSelect.addEventListener('change', function() {
      localStorage.setItem('lastSelectedModel', this.value);
    });
  }

  // --- Date Field Handling (Home Page) ---
  // Set today's date as default for X-ray date and ensure proper date format (YYYY-MM-DD)
  const today = new Date();
  const dateField = document.getElementById('id_date_of_xray');
  const birthField = document.getElementById('id_date_of_birth');
  
  // Format today's date in YYYY-MM-DD format
  const formattedDate = today.toISOString().split('T')[0];
  
  // Set default date for X-ray date field
  if (dateField && !dateField.value) {
    dateField.value = formattedDate;
  }
  
  // Force YYYY-MM-DD format for all date fields
  const formatDate = (date) => {
    if (!date) return '';
    const parts = date.split(/[-\/]/);
    
    // If format is MM/DD/YYYY, convert to YYYY-MM-DD
    if (parts.length === 3) {
      if (parts[2].length === 4) { // Likely MM/DD/YYYY format
        return `${parts[2]}-${parts[0].padStart(2, '0')}-${parts[1].padStart(2, '0')}`;
      } else {
        // Already YYYY-MM-DD or similar
        return `${parts[0]}-${parts[1].padStart(2, '0')}-${parts[2].padStart(2, '0')}`;
      }
    }
    return date; // return original if cannot parse
  };
  
  // Add input event listeners to ensure correct format
  [dateField, birthField].forEach(field => {
    if (field) {
      // Format existing value if needed
      if (field.value) {
        field.value = formatDate(field.value);
      }
      
      field.addEventListener('focus', function() {
        this.setAttribute('data-original-value', this.value);
      });
      
      field.addEventListener('blur', function() {
        const value = this.value;
        if (value) {
          this.value = formatDate(value);
          
          // Validate date format
          const isValidDate = /^\d{4}-\d{2}-\d{2}$/.test(this.value);
          if (!isValidDate) {
            // If invalid, restore original value
            this.value = this.getAttribute('data-original-value') || '';
            window.showModal(gettext('Please use the YYYY-MM-DD format for dates'), gettext('Invalid Date Format'), true);
          }
        }
      });
    }
  });

  // --- Form Progress Handling ---

  // Show the progress UI immediately on submit.
  // This prevents a "dead" period for slow server-side work (e.g., DICOM -> PNG conversion).
  const showProgressUI = () => {
    if (formWrapper) formWrapper.style.display = 'none';
    if (progressWrapper) progressWrapper.style.display = 'block';

    // Give the user instant feedback even before we have an upload_id to poll.
    const initialProgress = 1;
    if (progressBar) {
      progressBar.style.width = `${initialProgress}%`;
      progressBar.setAttribute('aria-valuenow', initialProgress);
      if (progressBar.parentElement) {
        progressBar.parentElement.setAttribute('aria-valuenow', initialProgress);
      }
    }
    if (progressPercentage) progressPercentage.textContent = `${initialProgress}% ${gettext('Complete')}`;
  };

  // Restore the form UI if the upload/validation fails.
  const restoreFormUI = () => {
    if (progressWrapper) progressWrapper.style.display = 'none';
    if (formWrapper) formWrapper.style.display = 'block';

    if (progressBar) {
      progressBar.style.width = '0%';
      progressBar.setAttribute('aria-valuenow', 0);
      if (progressBar.parentElement) {
        progressBar.parentElement.setAttribute('aria-valuenow', 0);
      }
    }
    if (progressPercentage) progressPercentage.textContent = `0% ${gettext('Complete')}`;
  };
  
  // Custom file input functionality
  const customFileButton = document.getElementById('custom-file-button');
  const fileInput = document.getElementById('id_image');
  const fileNameDisplay = document.getElementById('file-name-display');
  
  if (customFileButton && fileInput && fileNameDisplay) {
    // Click the custom button to trigger file selection
    customFileButton.addEventListener('click', () => {
      fileInput.click();
    });
    
    // Update display when file is selected
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        fileNameDisplay.value = file.name;
        fileNameDisplay.classList.remove('text-muted');
      } else {
        fileNameDisplay.value = '';
        fileNameDisplay.classList.add('text-muted');
      }
    });

    // Desktop drag & drop: allow dropping a file onto the "Browse" button.
    // This intentionally reuses the existing <input type="file"> + change handler.
    const canUseDesktopDnD = Boolean(
      window.matchMedia &&
        window.matchMedia('(hover: hover) and (pointer: fine)').matches &&
        window.DataTransfer
    );

    if (canUseDesktopDnD) {
      customFileButton.setAttribute('title', gettext('Browse or drop a file'));

      const setDragActive = (isActive) => {
        // Use Bootstrap classes only (no extra CSS).
        customFileButton.classList.toggle('btn-outline-primary', isActive);
        customFileButton.classList.toggle('btn-outline-secondary', !isActive);
      };

      const isFileDrag = (event) => {
        const types = event.dataTransfer?.types;
        return Boolean(types && Array.from(types).includes('Files'));
      };

      customFileButton.addEventListener('dragenter', (event) => {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        setDragActive(true);
      });

      customFileButton.addEventListener('dragover', (event) => {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = 'copy';
        setDragActive(true);
      });

      customFileButton.addEventListener('dragleave', (event) => {
        if (!isFileDrag(event)) return;
        setDragActive(false);
      });

      customFileButton.addEventListener('drop', (event) => {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        setDragActive(false);

        const droppedFile = event.dataTransfer.files?.[0];
        if (!droppedFile) return;

        try {
          const dt = new DataTransfer();
          dt.items.add(droppedFile);
          fileInput.files = dt.files;
          fileInput.dispatchEvent(new Event('change', { bubbles: true }));
        } catch (err) {
          // If the browser forbids programmatic assignment, gracefully fall back.
          console.warn('Drag-and-drop not supported for file input assignment:', err);
        }
      });
    }
  }
  
  // Function to track progress
  const trackProgress = (uploadId) => {
    let currentProgress = 0;
    let oodNotified = false;
    
    // Hide form and show progress bar
    if (formWrapper) formWrapper.style.display = 'none';
    if (progressWrapper) progressWrapper.style.display = 'block';
    
    // Check progress from the server
    const checkProgress = () => {
      fetch(`/progress/${uploadId}/`, { cache: 'no-store' })
        .then(response => response.json())
        .then(data => {
          // Update progress bar
          currentProgress = data.progress;
          if (!Number.isFinite(currentProgress)) currentProgress = 0;
          if (currentProgress < 1) currentProgress = 1; // never show 0%
          if (progressBar) {
            progressBar.style.width = `${currentProgress}%`;
            progressBar.setAttribute('aria-valuenow', currentProgress);
            progressBar.parentElement.setAttribute('aria-valuenow', currentProgress);
          }
          if (progressPercentage) progressPercentage.textContent = `${currentProgress}% ${gettext('Complete')}`;
          
          // Check for OOD status and notify if not already notified
          if (data.requires_expert_review && !oodNotified) {
            oodNotified = true;
            window.showModal(
                gettext('Out-of-Distribution (OOD) image detected. The analysis results may be unreliable and require expert review.'),
                gettext('OOD Detected'),
                true // Show as error/warning
            );
          }

          // Update screen reader announcements
          const statusElement = document.getElementById('analysis-status');
          if (statusElement && currentProgress % 25 === 0) {
            const statusMessages = {
              25: gettext('Image uploaded successfully, analysis 25% complete'),
              50: gettext('AI model processing X-ray data, analysis 50% complete'), 
              75: gettext('Generating predictions, analysis 75% complete'),
              100: gettext('Analysis complete, redirecting to results')
            };
            if (statusMessages[currentProgress]) {
              statusElement.textContent = statusMessages[currentProgress];
            }
          }
          
          // If not complete, check again
          if (currentProgress < 100) {
            setTimeout(() => checkProgress(), 500);
          } else {
            const redirectUrl = `/xray/${data.xray_id}/`;
            // Force redirect to xray results, not the old results URL
            window.location.href = redirectUrl;
          }
        })
        .catch(error => {
          console.error('Error checking progress:', error);
          // Increase progress a bit anyway to give feedback
          currentProgress = Math.min(Math.max(currentProgress, 1) + 5, 95);
          if (progressBar) {
            progressBar.style.width = `${currentProgress}%`;
            progressBar.setAttribute('aria-valuenow', currentProgress);
          }
          if (progressPercentage) progressPercentage.textContent = `${currentProgress}% ${gettext('Complete')}`;
          
          // Try again after a delay
          setTimeout(() => checkProgress(), 1000);
        });
    };
    
    // Start checking progress
    checkProgress();
  };
  
  // Form submission handler
  if (analysisForm) {
    analysisForm.addEventListener('submit', (e) => {
      e.preventDefault();

      // Show progress instantly (even if the server is slow to respond).
      showProgressUI();

      // Let the browser paint the progress UI before starting the upload.
      setTimeout(() => {
        // Create FormData object from the form
        const formData = new FormData(analysisForm);
        
        // Get CSRF token using global helper or fallback to form input
        const csrfToken = getCookie('csrftoken') || document.querySelector('[name=csrfmiddlewaretoken]')?.value;

        // Submit form via AJAX
        fetch('/', {
          method: 'POST',
          body: formData,
          headers: {
            'X-Requested-With': 'XMLHttpRequest',
            'X-CSRFToken': csrfToken,
            'Accept': 'application/json',
          },
          cache: 'no-store',
        })
        .then(async (response) => {
          // Always try to parse JSON (even on non-2xx) so we can show real errors.
          const contentType = response.headers.get('content-type') || '';
          let data = null;
          if (contentType.includes('application/json')) {
            try {
              data = await response.json();
            } catch (err) {
              data = null;
            }
          }

          if (!response.ok) {
            // Prefer server-provided error details when available.
            let errorMessage = (data && data.error) ? data.error : `HTTP error! status: ${response.status}`;

            // If we got field-level errors, surface the most relevant one.
            if (data && data.errors && data.errors.image) {
              const imageErrors = data.errors.image;
              if (Array.isArray(imageErrors) && imageErrors.length) {
                errorMessage = imageErrors[0];
              } else if (typeof imageErrors === 'string') {
                errorMessage = imageErrors;
              }
            }

            // Common reverse-proxy error for large uploads.
            if (response.status === 413) {
              errorMessage = gettext('File too large. Please upload a smaller file.');
            }

            const err = new Error(errorMessage);
            err.status = response.status;
            err.data = data;
            throw err;
          }

          if (!data) {
            throw new Error('Response is not JSON');
          }

          return data;
        })
        .then(data => {
          if (data.upload_id) {
            // Start tracking progress
            trackProgress(data.upload_id);
          } else if (data.error) {
            // Handle validation errors
            console.error('Form validation error:', data.error);
            if (data.errors) {
              console.error('Detailed errors:', data.errors);
            }
            restoreFormUI();
            window.showModal(data.error, gettext('Error'), true);
          } else {
            // Generic error handling
            console.error('Unexpected response format:', data);
            restoreFormUI();
            window.showModal(gettext('Error starting analysis. Please try again.'), gettext('Error'), true);
          }
        })
        .catch(error => {
          console.error('Error submitting form:', error);
          console.error('Error details:', error.message, error.stack);
          restoreFormUI();
          const message = (error && error.message) ? error.message : gettext('Error submitting form. Please try again.');
          window.showModal(message, gettext('Error'), true);
        });
      }, 0);
    });
  }
});