document.addEventListener('DOMContentLoaded', () => {
  const analysisForm = document.getElementById('analysis-form');
  const formWrapper = document.getElementById('form-wrapper');
  const progressWrapper = document.getElementById('progress-wrapper');
  const progressBar = document.getElementById('analysis-progress-bar');
  const progressPercentage = document.getElementById('progress-percentage');
  
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
  }
  
  // Function to simulate/track progress
  const trackProgress = (uploadId) => {
    let currentProgress = 0;
    
    // Hide form and show progress bar
    if (formWrapper) formWrapper.style.display = 'none';
    if (progressWrapper) progressWrapper.style.display = 'block';
    
    // Check progress from the server
    const checkProgress = () => {
      fetch(`/progress/${uploadId}/`)
        .then(response => response.json())
        .then(data => {
          // Update progress bar
          currentProgress = data.progress;
          if (progressBar) {
            progressBar.style.width = `${currentProgress}%`;
            progressBar.setAttribute('aria-valuenow', currentProgress);
            progressBar.parentElement.setAttribute('aria-valuenow', currentProgress);
          }
          if (progressPercentage) progressPercentage.textContent = `${currentProgress}% Complete`;
          
          // Update screen reader announcements
          const statusElement = document.getElementById('analysis-status');
          if (statusElement && currentProgress % 25 === 0) {
            const statusMessages = {
              25: 'Image uploaded successfully, analysis 25% complete',
              50: 'AI model processing X-ray data, analysis 50% complete', 
              75: 'Generating predictions, analysis 75% complete',
              100: 'Analysis complete, redirecting to results'
            };
            if (statusMessages[currentProgress]) {
              statusElement.textContent = statusMessages[currentProgress];
            }
          }
          
          // If not complete, check again
          if (currentProgress < 100) {
            setTimeout(() => checkProgress(), 500);
          } else {
            // Log the redirect URL for debugging
            const redirectUrl = `/xray/${data.xray_id}/`;
            console.log('Redirecting to:', redirectUrl);
            
            // Force redirect to xray results, not the old results URL
            window.location.href = redirectUrl;
          }
        })
        .catch(error => {
          console.error('Error checking progress:', error);
          // Increase progress a bit anyway to give feedback
          currentProgress = Math.min(currentProgress + 5, 95);
          if (progressBar) {
            progressBar.style.width = `${currentProgress}%`;
            progressBar.setAttribute('aria-valuenow', currentProgress);
          }
          if (progressPercentage) progressPercentage.textContent = `${currentProgress}% Complete`;
          
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
      
      // Create FormData object from the form
      const formData = new FormData(analysisForm);
      
      // Submit form via AJAX
      fetch('/', {
        method: 'POST',
        body: formData,
        headers: {
          'X-Requested-With': 'XMLHttpRequest',
          'X-CSRFToken': getCSRFToken(),
        },
      })
      .then(response => response.json())
      .then(data => {
        if (data.upload_id) {
          console.log('Upload successful, tracking progress for ID:', data.upload_id);
          // Start tracking progress
          trackProgress(data.upload_id);
        } else {
          // Error handling
          alert(gettext('Error starting analysis. Please try again.'));
        }
      })
      .catch(error => {
        console.error('Error submitting form:', error);
        alert(gettext('Error submitting form. Please try again.'));
      });
    });
  }
}); 