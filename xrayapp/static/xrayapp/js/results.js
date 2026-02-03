// Results page functionality for MCADS

document.addEventListener('DOMContentLoaded', () => {
    // Helper to format date as YYYY-MM-DD HH:MM:SS
    function formatDateTime(dateString) {
        const d = new Date(dateString);
        return d.getFullYear() + '-' + 
               String(d.getMonth() + 1).padStart(2, '0') + '-' + 
               String(d.getDate()).padStart(2, '0') + ' ' + 
               String(d.getHours()).padStart(2, '0') + ':' + 
               String(d.getMinutes()).padStart(2, '0') + ':' + 
               String(d.getSeconds()).padStart(2, '0');
    }

    // Check if we are on the results page by looking for the X-ray ID script
    const xrayIdScript = document.getElementById('xray-id');
    if (!xrayIdScript) return;

    const XRAY_ID = JSON.parse(xrayIdScript.textContent);

    // Set progress bar widths and aria values from data attributes
    const progressBars = document.querySelectorAll('.progress-bar[data-probability]');
    progressBars.forEach(bar => {
        const percentage = bar.getAttribute('data-probability');
        bar.style.width = percentage + '%';
        bar.setAttribute('aria-valuenow', percentage);
    });
    
    // Pathology explanations data
    const pathologyDataScript = document.getElementById('pathology-data');
    const pathologyExplanations = pathologyDataScript ? JSON.parse(pathologyDataScript.textContent) : {};
    
    // Make existing visualization images clickable for fullscreen view
    const existingVisualizationImages = document.querySelectorAll('.visualization-image');
    makeVisualizationImagesClickable(Array.from(existingVisualizationImages));
    
    // Handle pathology info button clicks
    document.querySelectorAll('.btn-info-pathology').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            const pathology = this.getAttribute('data-pathology');
            const explanation = pathologyExplanations[pathology];
            
            if (explanation) {
                document.getElementById('pathologyName').textContent = pathology;
                document.getElementById('pathologyDescription').textContent = explanation;
                
                // Show the modal
                // @ts-ignore - bootstrap is global
                const modal = new bootstrap.Modal(document.getElementById('pathologyModal'));
                modal.show();
            }
        });
    });
    
    const interpretationButtons = document.querySelectorAll('.interpretation-btn');
    const progressWrapper = document.getElementById('interpretation-progress');
    const progressBar = document.getElementById('interpretation-progress-bar');
    const statusText = document.getElementById('interpretation-status');
    const percentageText = document.getElementById('interpretation-percentage');
    
    // Add click event listeners to interpretation buttons
    interpretationButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Show progress container
            if (progressWrapper) progressWrapper.style.display = 'block';
            
            // Get the URL from the button
            const url = this.getAttribute('href');
            const methodName = this.dataset.method;
            
            // Update status text
            if (statusText) {
                statusText.textContent = gettext('Generating') + ' ' + 
                    (methodName === 'gradcam' ? gettext('Grad-CAM') : gettext('Pixel-Level Interpretability')) + 
                    ' ' + gettext('visualization...');
            }
            
            // Fetch the URL to start the process
            fetch(url, {
                method: 'GET',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'Content-Type': 'application/json',
                }
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'started') {
                        // Start monitoring progress
                        monitorProgress();
                    } else {
                        if (statusText) statusText.textContent = gettext('Error starting interpretation process');
                    }
                })
                .catch(error => {
                    console.error('Error starting interpretation:', error);
                    if (statusText) statusText.textContent = gettext('Error starting interpretation process');
                });
        });
    });
    
    // Add click event listener for segmentation button
    const segmentationBtn = document.getElementById('segmentation-btn');
    if (segmentationBtn) {
        segmentationBtn.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Show progress for segmentation
            if (progressWrapper) progressWrapper.style.display = 'block';
            if (statusText) statusText.textContent = gettext('Starting anatomical segmentation...');
            if (progressBar) {
                progressBar.style.width = '0%';
                progressBar.setAttribute('aria-valuenow', '0');
            }
            if (percentageText) percentageText.textContent = '0% ' + gettext('Complete');
            
            // Start segmentation process
            fetch(this.href, {
                method: 'GET',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'started') {
                        // Start monitoring progress for segmentation
                        monitorSegmentationProgress();
                    } else {
                        if (statusText) statusText.textContent = gettext('Error starting segmentation process');
                    }
                })
                .catch(error => {
                    console.error('Error starting segmentation:', error);
                    if (statusText) statusText.textContent = gettext('Error starting segmentation process');
                });
        });
    }
    
    function monitorProgress() {
        let progress = 5;
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', String(progress));
        }
        if (percentageText) percentageText.textContent = `${progress}% ${gettext('Complete')}`;
        
        // Check progress from the server
        function checkProgress() {
            fetch(`/progress/${XRAY_ID}/`)
                .then(response => {
                    if (!response.ok) {
                        // Handle HTTP errors (404, 500, etc.)
                        return response.json().catch(() => {
                            // If JSON parsing fails, create a generic error object
                            return {
                                error: `HTTP ${response.status}: ${response.statusText}`,
                                progress: 0
                            };
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    // Check if this is an error response
                    if (data.error) {
                        console.error('Server error:', data.error);
                        if (statusText) statusText.textContent = gettext('Error occurred during processing');
                        // Stop checking progress on error
                        return;
                    }
                    progress = data.progress;
                    if (progressBar) {
                        progressBar.style.width = `${progress}%`;
                        progressBar.setAttribute('aria-valuenow', String(progress));
                    }
                    if (percentageText) percentageText.textContent = `${progress}% ${gettext('Complete')}`;
                    
                    // Update status text based on progress
                    if (statusText) {
                        if (progress < 25) {
                            statusText.textContent = gettext('Initializing AI model...');
                        } else if (progress < 50) {
                            statusText.textContent = gettext('Processing X-ray data...');
                        } else if (progress < 75) {
                            statusText.textContent = gettext('Generating heat maps...');
                        } else if (progress < 95) {
                            statusText.textContent = gettext('Finalizing visualization...');
                        } else {
                            statusText.textContent = gettext('Almost complete...');
                        }
                    }
                    
                    // Update screen reader announcements
                    const accessibilityStatus = document.getElementById('interpretation-accessibility-status');
                    if (accessibilityStatus && progress % 25 === 0) {
                        const accessibilityMessages = {
                            25: gettext('Visualization generation 25% complete, initializing AI interpretability models'),
                            50: gettext('Visualization generation 50% complete, processing pixel-level analysis'),
                            75: gettext('Visualization generation 75% complete, generating heat maps and overlays'),
                            100: gettext('Visualization generation complete, displaying results')
                        };
                        if (accessibilityMessages[progress]) {
                            accessibilityStatus.textContent = accessibilityMessages[progress];
                        }
                    }
                    
                    if (progress < 100) {
                        // Check again in 500ms
                        setTimeout(checkProgress, 500);
                    } else {
                        // Complete - display visualizations in-place
                        if (statusText) statusText.textContent = gettext('Visualization complete!');
                        
                        // Hide progress container after a short delay
                        setTimeout(() => {
                            if (progressWrapper) progressWrapper.style.display = 'none';
                            
                            // Display GRAD-CAM visualizations if available
                            if (data.gradcam && data.gradcam.has_gradcam && data.gradcam.visualizations) {
                                displayMultipleGradCAMVisualizations(data.gradcam.visualizations, data.image_url);
                            }
                            
                            // Display PLI visualizations if available
                            if (data.pli && data.pli.has_pli && data.pli.visualizations) {
                                displayMultiplePLIVisualizations(data.pli.visualizations, data.image_url);
                            }
                        }, 1000);
                    }
                })
                .catch(error => {
                    console.error('Error checking progress:', error);
                    if (statusText) statusText.textContent = gettext('Connection error - retrying...');
                    // Keep checking but with longer intervals
                    setTimeout(checkProgress, 2000);
                });
        }
        
        // Start checking
        checkProgress();
    }
    
    function monitorSegmentationProgress() {
        let progress = 5;
        let oodNotified = false;
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', String(progress));
        }
        if (percentageText) percentageText.textContent = `${progress}% ${gettext('Complete')}`;
        
        // Check progress from the server
        function checkProgress() {
            fetch(`/progress/${XRAY_ID}/`)
                .then(response => {
                    if (!response.ok) {
                        return response.json().catch(() => {
                            return {
                                error: `HTTP ${response.status}: ${response.statusText}`,
                                progress: 0
                            };
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.error) {
                        console.error('Server error:', data.error);
                        if (statusText) statusText.textContent = gettext('Error occurred during processing');
                        return;
                    }
                    
                    // Update progress
                    progress = data.progress || 0;
                    if (progressBar) {
                        progressBar.style.width = `${progress}%`;
                        progressBar.setAttribute('aria-valuenow', String(progress));
                    }
                    if (percentageText) percentageText.textContent = `${progress}% ${gettext('Complete')}`;
                    
                    // Check for OOD status and notify if not already notified
                    if (data.requires_expert_review && !oodNotified) {
                        oodNotified = true;
                        window.showModal(
                            gettext('Out-of-Distribution (OOD) image detected during segmentation. The results may be unreliable and require expert review.'),
                            gettext('OOD Detected'),
                            true // Show as error/warning
                        );
                    }
                    
                    // Update status message based on progress
                    if (statusText) {
                        if (progress <= 20) {
                            statusText.textContent = gettext('Loading segmentation model...');
                        } else if (progress <= 60) {
                            statusText.textContent = gettext('Analyzing anatomical structures...');
                        } else if (progress <= 80) {
                            statusText.textContent = gettext('Generating segmentation masks...');
                        } else {
                            statusText.textContent = gettext('Finalizing segmentation results...');
                        }
                    }
                    
                    if (progress < 100) {
                        // Check again in 500ms
                        setTimeout(checkProgress, 500);
                    } else {
                        // Complete - display segmentation results
                        if (statusText) statusText.textContent = gettext('Segmentation complete!');
                        
                        // Hide progress container after a short delay
                        setTimeout(() => {
                            if (progressWrapper) progressWrapper.style.display = 'none';
                            
                            // Display segmentation visualizations if available
                            if (data.segmentation && data.segmentation.has_segmentation && data.segmentation.visualizations) {
                                displaySegmentationVisualizations(data.segmentation.visualizations, data.image_url);
                            }
                        }, 1000);
                    }
                })
                .catch(error => {
                    console.error('Error checking progress:', error);
                    if (statusText) statusText.textContent = gettext('Connection error - retrying...');
                    setTimeout(checkProgress, 2000);
                });
        }
        
        // Start checking
        checkProgress();
    }
    
    function displayMultipleGradCAMVisualizations(visualizations, imageUrl) {
        // Clear any existing multiple visualization containers
        removeExistingMultipleVisualizations('gradcam');
        
        visualizations.forEach((viz, index) => {
            const containerId = `dynamic-gradcam-section-${viz.id}`;
            
            // Create visualization container
            const section = document.createElement('div');
            section.id = `visualization-${viz.id}`;  // Use proper ID for Image Controls
            section.className = 'card mb-4';
            section.setAttribute('data-dynamic-container', containerId);  // Keep track of dynamic ID
            // Responsive Image Controls:
            // - xs/sm: wrap buttons so they fit on mobile
            // - md+: keep a single row with horizontal scroll (labels become visible)
            section.innerHTML = `
                <div class="card-header bg-success text-white">
                    <div class="d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">
                            ${gettext("GRAD-CAM visualization")} - ${viz.target_pathology}
                        </h5>
                        <button class="btn btn-sm btn-danger delete-visualization-btn" 
                                data-viz-id="${viz.id}" 
                                data-viz-type="gradcam"
                                data-pathology="${viz.target_pathology}"
                                title="${gettext('Delete this visualization')}">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="card-body text-center">
                    <div class="row">
                        <div class="col-md-6">
                            <img src="${imageUrl}" alt="${gettext('Original X-ray')}" class="img-fluid visualization-image model-crop">
                            <p class="mt-2 text-center">${gettext("Original X-ray")}</p>
                        </div>
                        <div class="col-md-6">
                            <img src="${viz.overlay_url}?t=${new Date().getTime()}" alt="${gettext('GRAD-CAM Overlay')}" class="img-fluid visualization-image model-crop">
                            <p class="mt-2 text-center">${gettext("GRAD-CAM Overlay")}</p>
                        </div>
                    </div>
                    <!-- Visualization Controls -->
                    <div class="visualization-controls bg-body border rounded-3 p-2 p-md-3 mt-2 mt-md-3 mb-2 mb-md-3">
                        <div class="d-flex flex-wrap align-items-center gap-1 gap-md-2 mb-1 mb-md-2">
                            <small class="text-muted">${gettext("Image Controls")}</small>
                            <button class="btn btn-sm btn-outline-secondary reset-controls-btn d-inline-flex align-items-center gap-1 ms-auto" data-viz-id="${viz.id}">
                                <i class="fas fa-undo"></i> ${gettext("Reset")}
                            </button>
                        </div>
                        <div class="d-flex flex-wrap flex-md-nowrap gap-1 gap-md-2 overflow-auto pb-0 pb-md-1">
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="invert" data-viz-id="${viz.id}" title="${gettext('Invert Colors')}">
                                <i class="fas fa-adjust"></i>
                                <span class="btn-label">${gettext("Invert")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="flip-h" data-viz-id="${viz.id}" title="${gettext('Flip Horizontal')}">
                                <i class="fas fa-arrows-alt-h"></i>
                                <span class="btn-label">${gettext("Flip H")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="flip-v" data-viz-id="${viz.id}" title="${gettext('Flip Vertical')}">
                                <i class="fas fa-arrows-alt-v"></i>
                                <span class="btn-label">${gettext("Flip V")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="rotate" data-viz-id="${viz.id}" title="${gettext('Rotate 90°')}">
                                <i class="fas fa-redo"></i>
                                <span class="btn-label">${gettext("Rotate")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="sharpen" data-viz-id="${viz.id}" title="${gettext('Sharpen')}">
                                <i class="fas fa-search-plus"></i>
                                <span class="btn-label">${gettext("Sharpen")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="brightness-up" data-viz-id="${viz.id}" title="${gettext('Increase Brightness')}">
                                <i class="fas fa-sun"></i>
                                <span class="btn-label">${gettext("Bright +")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="brightness-down" data-viz-id="${viz.id}" title="${gettext('Decrease Brightness')}">
                                <i class="fas fa-moon"></i>
                                <span class="btn-label">${gettext("Bright -")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="contrast-up" data-viz-id="${viz.id}" title="${gettext('Increase Contrast')}">
                                <i class="fas fa-plus-circle"></i>
                                <span class="btn-label">${gettext("Contrast +")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="contrast-down" data-viz-id="${viz.id}" title="${gettext('Decrease Contrast')}">
                                <i class="fas fa-minus-circle"></i>
                                <span class="btn-label">${gettext("Contrast -")}</span>
                            </button>
                        </div>
                    </div>
                    
                    <p class="mt-3 text-muted">
                        ${gettext("GRAD-CAM highlights the regions in the image that strongly influenced the model's prediction for")} ${viz.target_pathology}.
                    </p>
                    <small class="text-muted">
                        ${gettext("Generated on")}: ${formatDateTime(viz.created_at)}
                    </small>
                </div>
            `;
            
            // Insert after the interpretation progress section
            const progressWrapper = document.getElementById('interpretation-progress');
            if (progressWrapper && progressWrapper.parentNode) {
                progressWrapper.parentNode.insertBefore(section, progressWrapper.nextSibling);
            }
            
            // Make images clickable
            const images = section.querySelectorAll('.visualization-image');
            makeVisualizationImagesClickable(Array.from(images));
            
            // Scroll to first visualization
            if (index === 0) {
                setTimeout(() => {
                    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
            }
        });
        
        // Initialize Image Controls for the new GRAD-CAM visualizations
        if (window.initializeVisualizationControls) {
            window.initializeVisualizationControls();
        }
    }
    
    function displayMultiplePLIVisualizations(visualizations, imageUrl) {
        // Clear any existing multiple visualization containers
        removeExistingMultipleVisualizations('pli');
        
        visualizations.forEach((viz, index) => {
            const containerId = `dynamic-pli-section-${viz.id}`;
            
            // Create visualization container
            const section = document.createElement('div');
            section.id = `visualization-${viz.id}`;  // Use proper ID for Image Controls
            section.className = 'card mb-4';
            section.setAttribute('data-dynamic-container', containerId);  // Keep track of dynamic ID
            // Responsive Image Controls:
            // - xs/sm: wrap buttons so they fit on mobile
            // - md+: keep a single row with horizontal scroll (labels become visible)
            section.innerHTML = `
                <div class="card-header bg-info text-white">
                    <div class="d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">
                            ${gettext("Pixel-level interpretability")} - ${viz.target_pathology}
                        </h5>
                        <button class="btn btn-sm btn-danger delete-visualization-btn" 
                                data-viz-id="${viz.id}" 
                                data-viz-type="pli"
                                data-pathology="${viz.target_pathology}"
                                title="${gettext('Delete this visualization')}">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="card-body text-center">
                    <div class="row">
                        <div class="col-md-6">
                            <img src="${imageUrl}" alt="${gettext('Original X-ray')}" class="img-fluid visualization-image">
                            <p class="mt-2 text-center">${gettext("Original X-ray")}</p>
                        </div>
                        <div class="col-md-6">
                            <img src="${viz.overlay_url}?t=${new Date().getTime()}" alt="${gettext('Pixel-Level Overlay')}" class="img-fluid visualization-image">
                            <p class="mt-2 text-center">${gettext("Pixel-level overlay")}</p>
                        </div>
                    </div>
                    <!-- Visualization Controls -->
                    <div class="visualization-controls bg-body border rounded-3 p-2 p-md-3 mt-2 mt-md-3 mb-2 mb-md-3">
                        <div class="d-flex flex-wrap align-items-center gap-1 gap-md-2 mb-1 mb-md-2">
                            <small class="text-muted">${gettext("Image Controls")}</small>
                            <button class="btn btn-sm btn-outline-secondary reset-controls-btn d-inline-flex align-items-center gap-1 ms-auto" data-viz-id="${viz.id}">
                                <i class="fas fa-undo"></i> ${gettext("Reset")}
                            </button>
                        </div>
                        <div class="d-flex flex-wrap flex-md-nowrap gap-1 gap-md-2 overflow-auto pb-0 pb-md-1">
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="invert" data-viz-id="${viz.id}" title="${gettext('Invert Colors')}">
                                <i class="fas fa-adjust"></i>
                                <span class="btn-label">${gettext("Invert")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="flip-h" data-viz-id="${viz.id}" title="${gettext('Flip Horizontal')}">
                                <i class="fas fa-arrows-alt-h"></i>
                                <span class="btn-label">${gettext("Flip H")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="flip-v" data-viz-id="${viz.id}" title="${gettext('Flip Vertical')}">
                                <i class="fas fa-arrows-alt-v"></i>
                                <span class="btn-label">${gettext("Flip V")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="rotate" data-viz-id="${viz.id}" title="${gettext('Rotate 90°')}">
                                <i class="fas fa-redo"></i>
                                <span class="btn-label">${gettext("Rotate")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="sharpen" data-viz-id="${viz.id}" title="${gettext('Sharpen')}">
                                <i class="fas fa-search-plus"></i>
                                <span class="btn-label">${gettext("Sharpen")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="brightness-up" data-viz-id="${viz.id}" title="${gettext('Increase Brightness')}">
                                <i class="fas fa-sun"></i>
                                <span class="btn-label">${gettext("Bright +")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="brightness-down" data-viz-id="${viz.id}" title="${gettext('Decrease Brightness')}">
                                <i class="fas fa-moon"></i>
                                <span class="btn-label">${gettext("Bright -")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="contrast-up" data-viz-id="${viz.id}" title="${gettext('Increase Contrast')}">
                                <i class="fas fa-plus-circle"></i>
                                <span class="btn-label">${gettext("Contrast +")}</span>
                            </button>
                            <button class="btn btn-sm btn-outline-primary control-btn d-inline-flex align-items-center gap-1 flex-shrink-0" data-action="contrast-down" data-viz-id="${viz.id}" title="${gettext('Decrease Contrast')}">
                                <i class="fas fa-minus-circle"></i>
                                <span class="btn-label">${gettext("Contrast -")}</span>
                            </button>
                        </div>
                    </div>
                    
                    <p class="mt-3 text-muted">
                        ${gettext("Pixel-Level Interpretability shows which individual pixels had the most influence on the model's prediction for")} ${viz.target_pathology}.
                    </p>
                    <small class="text-muted">
                        ${gettext("Generated on")}: ${formatDateTime(viz.created_at)}
                        ${viz.threshold ? ` | ${gettext("Threshold")}: ${viz.threshold}` : ''}
                    </small>
                </div>
            `;
            
            // Insert after the interpretation progress section
            const progressWrapper = document.getElementById('interpretation-progress');
            if (progressWrapper && progressWrapper.parentNode) {
                progressWrapper.parentNode.insertBefore(section, progressWrapper.nextSibling);
            }
            
            // Make images clickable
            const images = section.querySelectorAll('.visualization-image');
            makeVisualizationImagesClickable(Array.from(images));
            
            // Scroll to first visualization
            if (index === 0) {
                setTimeout(() => {
                    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
            }
        });
        
        // Initialize Image Controls for the new PLI visualizations
        if (window.initializeVisualizationControls) {
            window.initializeVisualizationControls();
        }
    }
    
    function displaySegmentationVisualizations(visualizations, imageUrl) {
        // Clear any existing segmentation visualization containers
        removeExistingMultipleVisualizations('segmentation');
        
        // Find or create a container for segmentation results
        let segmentationSection = document.getElementById('segmentation-results-section');
        if (!segmentationSection) {
            // Create new section after interpretation section
            const interpretationSection = document.querySelector('.interpretability-card');
            segmentationSection = document.createElement('div');
            segmentationSection.id = 'segmentation-results-section';
            segmentationSection.className = 'mb-5';
            
            if (interpretationSection && interpretationSection.parentNode) {
                interpretationSection.parentNode.insertBefore(segmentationSection, interpretationSection.nextSibling);
            } else {
                // Fallback: insert before the action buttons
                const actionButtons = document.querySelector('.text-center.mb-5');
                if (actionButtons && actionButtons.parentNode) {
                    actionButtons.parentNode.insertBefore(segmentationSection, actionButtons);
                }
            }
        }
        
        // Display combined segmentation visualization
        const combinedViz = visualizations.find(v => v.target_pathology === 'All Structures');
        if (combinedViz) {
            const section = document.createElement('div');
            section.className = 'card mb-4';
            section.setAttribute('data-dynamic-container', 'segmentation-combined');
            section.innerHTML = `
                <div class="card-header bg-info text-white">
                    <h5 class="mb-0">
                        ${gettext("Anatomical Segmentation Results")}
                    </h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <h6>${gettext("Original X-ray")}</h6>
                            <img src="${imageUrl}" alt="${gettext('Original X-ray')}" class="img-fluid mb-3">
                        </div>
                        <div class="col-md-6">
                            <h6>${gettext("Segmentation Overlay")}</h6>
                            <img src="/media/${combinedViz.visualization_path}" alt="${gettext('Segmentation overlay')}" class="img-fluid mb-3">
                        </div>
                    </div>
                    
                    <div class="mt-4">
                        <h6>${gettext("Detected Anatomical Structures")}:</h6>
                        <div class="row">
                            <div class="col-md-6">
                                <ul class="list-unstyled">
                                    <li><span style="color: #0000FF;">●</span> ${gettext("Left Clavicle")}</li>
                                    <li><span style="color: #00FF00;">●</span> ${gettext("Right Clavicle")}</li>
                                    <li><span style="color: #FF0000;">●</span> ${gettext("Left Scapula")}</li>
                                    <li><span style="color: #00FFFF;">●</span> ${gettext("Right Scapula")}</li>
                                    <li><span style="color: #FF00FF;">●</span> ${gettext("Left Lung")}</li>
                                    <li><span style="color: #FFFF00;">●</span> ${gettext("Right Lung")}</li>
                                    <li><span style="color: #8000FF;">●</span> ${gettext("Left Hilus Pulmonis")}</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <ul class="list-unstyled">
                                    <li><span style="color: #FF8000;">●</span> ${gettext("Right Hilus Pulmonis")}</li>
                                    <li><span style="color: #0080FF;">●</span> ${gettext("Heart")}</li>
                                    <li><span style="color: #FF0080;">●</span> ${gettext("Aorta")}</li>
                                    <li><span style="color: #80FF00;">●</span> ${gettext("Facies Diaphragmatica")}</li>
                                    <li><span style="color: #00FF80;">●</span> ${gettext("Mediastinum")}</li>
                                    <li><span style="color: #8080FF;">●</span> ${gettext("Weasand")}</li>
                                    <li><span style="color: #FF8080;">●</span> ${gettext("Spine")}</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            if (segmentationSection) {
                segmentationSection.appendChild(section);
            }
            
            // Make images clickable
            const images = section.querySelectorAll('img');
            makeVisualizationImagesClickable(Array.from(images));
            
            // Scroll to the section
            setTimeout(() => {
                section.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        }
    }
    
    function removeExistingMultipleVisualizations(type) {
        // Remove any existing multiple visualization containers of the same type
        const existingContainers = document.querySelectorAll(`[data-dynamic-container^="dynamic-${type}-section-"]`);
        existingContainers.forEach(container => container.remove());
    }
    
    function makeVisualizationImagesClickable(images) {
        images.forEach(img => {
            img.style.cursor = 'pointer';
            
            // Remove existing listener to avoid duplicates if re-attaching
            // Note: simple addEventListener doesn't support easy removal of anonymous functions
            // but for this case, it's safer to clone the node or check if already attached.
            // Simplified: we assume this is called on new nodes or we accept potential multi-binding
            // if called on existing nodes (though logic above tries to select new ones).
            
            // Add click event listener to each visualization image
            img.onclick = function(e) {
                e.stopPropagation();
                
                // Create overlay elements
                const overlay = document.createElement('div');
                overlay.style.position = 'fixed';
                overlay.style.top = '0';
                overlay.style.left = '0';
                overlay.style.width = '100%';
                overlay.style.height = '100%';
                overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.9)';
                overlay.style.display = 'flex';
                overlay.style.alignItems = 'center';
                overlay.style.justifyContent = 'center';
                overlay.style.zIndex = '9999';
                overlay.style.cursor = 'zoom-out';
                
                // Create expanded image
                const expandedImg = document.createElement('img');
                expandedImg.src = this.src;
                expandedImg.style.maxHeight = '90vh';
                expandedImg.style.maxWidth = '90vw';
                expandedImg.style.objectFit = 'contain';
                
                // Copy the filter style from the original image to preserve effects like invert
                if (this.style.filter) {
                    expandedImg.style.filter = this.style.filter;
                }
                
                // Add close functionality
                overlay.addEventListener('click', function() {
                    document.body.removeChild(overlay);
                });
                
                // Prevent click on image from closing the overlay
                expandedImg.addEventListener('click', function(e) {
                    e.stopPropagation();
                });
                
                // Add keyboard navigation (Escape to close)
                const escapeHandler = function(e) {
                    if (e.key === 'Escape' && document.body.contains(overlay)) {
                        document.body.removeChild(overlay);
                        document.removeEventListener('keydown', escapeHandler);
                    }
                };
                document.addEventListener('keydown', escapeHandler);
                
                // Append elements to the body
                overlay.appendChild(expandedImg);
                document.body.appendChild(overlay);
            };
        });
    }
});

// Visualization Controls Functionality - Global scope for dynamic content
// Store applied effects for each visualization
window.visualizationEffects = window.visualizationEffects || new Map();

// Initialize effects for existing visualizations
window.initializeVisualizationEffects = function() {
    document.querySelectorAll('[id^="visualization-"]').forEach(viz => {
        const vizId = viz.id.replace('visualization-', '');
        if (!window.visualizationEffects.has(vizId)) {
            window.visualizationEffects.set(vizId, {
                invert: false,
                flipH: false,
                flipV: false,
                rotation: 0,
                sharpen: false,
                brightness: 100,
                contrast: 100
            });
        }
    });
};

document.addEventListener('DOMContentLoaded', () => {
    
    // Apply effects to images
    window.applyEffects = function(vizId) {
        const effects = window.visualizationEffects.get(vizId);
        if (!effects) return;
        
        const visualizationElement = document.getElementById(`visualization-${vizId}`);
        if (!visualizationElement) return;
        
        const images = visualizationElement.querySelectorAll('.visualization-image');
        
        // Build CSS filter and transform strings
        let filterStr = '';
        let transformStr = '';
        
        // Calculate final contrast value (sharpen adds to base contrast)
        let finalContrast = effects.contrast;
        if (effects.sharpen) {
            finalContrast = Math.min(200, finalContrast * 1.2); // Apply sharpening multiplier
        }
        
        // Build filter string without duplicates
        if (effects.invert) filterStr += 'invert(1) ';
        filterStr += `brightness(${effects.brightness}%) contrast(${finalContrast}%) `;
        if (effects.sharpen) filterStr += 'saturate(1.3) ';
        
        if (effects.flipH) transformStr += 'scaleX(-1) ';
        if (effects.flipV) transformStr += 'scaleY(-1) ';
        if (effects.rotation !== 0) transformStr += `rotate(${effects.rotation}deg) `;
        
        // Apply to all images in the visualization
        images.forEach(img => {
            img.style.filter = filterStr.trim();
            img.style.transform = transformStr.trim();
            img.style.transition = 'all 0.3s ease';
        });
        
        // Update button states
        window.updateButtonStates(vizId);
    };
    
    // Update button visual states
    window.updateButtonStates = function(vizId) {
        const effects = window.visualizationEffects.get(vizId);
        if (!effects) return;
        
        const visualizationElement = document.getElementById(`visualization-${vizId}`);
        if (!visualizationElement) return;
        
        // Update button active states
        const buttons = visualizationElement.querySelectorAll('.control-btn');
        buttons.forEach(btn => {
            const action = btn.getAttribute('data-action');
            btn.classList.remove('active');
            
            switch(action) {
                case 'invert':
                    if (effects.invert) btn.classList.add('active');
                    break;
                case 'flip-h':
                    if (effects.flipH) btn.classList.add('active');
                    break;
                case 'flip-v':
                    if (effects.flipV) btn.classList.add('active');
                    break;
                case 'sharpen':
                    if (effects.sharpen) btn.classList.add('active');
                    break;
                case 'rotate':
                    if (effects.rotation !== 0) btn.classList.add('active');
                    break;
            }
        });
    };
    
    // Reset all effects for a visualization
    window.resetEffects = function(vizId) {
        window.visualizationEffects.set(vizId, {
            invert: false,
            flipH: false,
            flipV: false,
            rotation: 0,
            sharpen: false,
            brightness: 100,
            contrast: 100
        });
        window.applyEffects(vizId);
    };
    
    // Handle control button clicks
    window.attachControlHandlers = function() {
        document.querySelectorAll('.control-btn').forEach(button => {
            if (!button.hasAttribute('data-control-attached')) {
                button.setAttribute('data-control-attached', 'true');
                button.addEventListener('click', function(e) {
                    e.preventDefault();
                    
                    const vizId = this.getAttribute('data-viz-id');
                    const action = this.getAttribute('data-action');
                    const effects = window.visualizationEffects.get(vizId);
                    
                    if (!effects) return;
                    
                    // Apply the effect based on action
                    switch(action) {
                        case 'invert':
                            effects.invert = !effects.invert;
                            break;
                        case 'flip-h':
                            effects.flipH = !effects.flipH;
                            break;
                        case 'flip-v':
                            effects.flipV = !effects.flipV;
                            break;
                        case 'rotate':
                            effects.rotation = (effects.rotation + 90) % 360;
                            break;
                        case 'sharpen':
                            effects.sharpen = !effects.sharpen;
                            break;
                        case 'brightness-up':
                            effects.brightness = Math.min(200, effects.brightness + 20);
                            break;
                        case 'brightness-down':
                            effects.brightness = Math.max(50, effects.brightness - 20);
                            break;
                        case 'contrast-up':
                            effects.contrast = Math.min(200, effects.contrast + 20);
                            break;
                        case 'contrast-down':
                            effects.contrast = Math.max(50, effects.contrast - 20);
                            break;
                    }
                    
                    window.applyEffects(vizId);
                });
            }
        });
        
        // Handle reset button clicks
        document.querySelectorAll('.reset-controls-btn').forEach(button => {
            if (!button.hasAttribute('data-reset-attached')) {
                button.setAttribute('data-reset-attached', 'true');
                button.addEventListener('click', function(e) {
                    e.preventDefault();
                    
                    const vizId = this.getAttribute('data-viz-id');
                    window.resetEffects(vizId);
                });
            }
        });
    };
    
    // Initialize on page load
    window.initializeVisualizationEffects();
    window.attachControlHandlers();
    
    // Re-attach handlers when new visualizations are added
    const controlObserver = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList') {
                window.initializeVisualizationEffects();
                window.attachControlHandlers();
            }
        });
    });
    
    // Observe the container for changes
    const container = document.querySelector('.col-md-10');
    if (container) {
        controlObserver.observe(container, { childList: true, subtree: true });
    }
    
    // Make functions globally accessible for dynamic content
    window.initializeVisualizationControls = function() {
        window.initializeVisualizationEffects();
        window.attachControlHandlers();
        // Also ensure delete handlers are attached for new visualizations
        if (window.attachDeleteHandlers) {
            window.attachDeleteHandlers();
        }
    };
});

// Delete visualization functionality
document.addEventListener('DOMContentLoaded', () => {
    // Helper function to get CSRF token
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

    // Initialize confirmation button listener
    const confirmBtn = document.getElementById('confirmDeleteVisualizationBtn');
    if (confirmBtn) {
        confirmBtn.addEventListener('click', function() {
            const vizId = this.getAttribute('data-viz-id');
            if (!vizId) return;

            // Show loading state
            this.disabled = true;
            const originalText = this.innerHTML;
            this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> ' + gettext('Deleting...');

            // Make AJAX request to delete visualization
            fetch(`/visualization/${vizId}/delete/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCookie('csrftoken'),
                    'X-Requested-With': 'XMLHttpRequest',
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Hide modal
                    const modalEl = document.getElementById('deleteVisualizationModal');
                    // @ts-ignore
                    const modal = bootstrap.Modal.getInstance(modalEl);
                    if (modal) modal.hide();

                    // Remove the visualization card with animation
                    const card = document.getElementById(`visualization-${vizId}`);
                    if (card) {
                        card.style.transition = 'opacity 0.3s ease-out';
                        card.style.opacity = '0';
                        setTimeout(() => {
                            card.remove();
                        }, 300);
                    }
                } else {
                    window.showModal(data.error || gettext('Failed to delete visualization'), gettext('Error'), true);
                }
            })
            .catch(error => {
                console.error('Error deleting visualization:', error);
                window.showModal(gettext('An error occurred while deleting the visualization'), gettext('Error'), true);
            })
            .finally(() => {
                // Restore button state
                this.disabled = false;
                this.innerHTML = originalText;
            });
        });
    }

    // Function to handle visualization deletion
    function attachDeleteHandlers() {
        document.querySelectorAll('.delete-visualization-btn').forEach(button => {
            if (!button.hasAttribute('data-click-attached')) {
                button.setAttribute('data-click-attached', 'true');
                button.addEventListener('click', function(e) {
                    e.preventDefault();
                    
                    const vizId = this.getAttribute('data-viz-id');
                    const vizType = this.getAttribute('data-viz-type');
                    const pathology = this.getAttribute('data-pathology');
                    
                    // Set data on confirmation button
                    if (confirmBtn) {
                        confirmBtn.setAttribute('data-viz-id', vizId);
                    }

                    // Set message
                    const messageEl = document.getElementById('deleteVisualizationMessage');
                    if (messageEl) {
                        messageEl.textContent = gettext('Are you sure you want to delete the') + ` ${vizType.toUpperCase()} ` + gettext('visualization for') + ` ${pathology}?`;
                    }

                    // Show modal
                    const modalEl = document.getElementById('deleteVisualizationModal');
                    if (modalEl) {
                        // @ts-ignore
                        const modal = new bootstrap.Modal(modalEl);
                        modal.show();
                    }
                });
            }
        });
    }
    
    // Attach handlers to existing delete buttons
    attachDeleteHandlers();
    
    // Re-attach handlers when new visualizations are added dynamically
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList') {
                attachDeleteHandlers();
            }
        });
    });
    
    // Observe the container for changes
    const container = document.querySelector('.col-md-10');
    if (container) {
        observer.observe(container, { childList: true, subtree: true });
    }
    
    // Make function globally accessible for dynamic content
    window.attachDeleteHandlers = attachDeleteHandlers;
});

// Compact print functionality
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('print-to-printer')?.addEventListener('click', e => {
        e.preventDefault();
        window.print();
    });
    
    document.getElementById('print-to-pdf')?.addEventListener('click', e => {
        e.preventDefault();
        const originalTitle = document.title;
        document.title = `MCADS Result - ${new Date().toISOString().split('T')[0]}`;
        window.print();
        setTimeout(() => document.title = originalTitle, 1000);
    });
});
