// Main JavaScript file for sentiment analysis application

document.addEventListener('DOMContentLoaded', function() {
    console.log('Document loaded');
    
    // Debug elements
    console.log('File upload container:', document.getElementById('file-upload-container'));
    console.log('File input:', document.getElementById('file-input'));
    console.log('File label:', document.getElementById('file-label'));
    
    // Elements
    const analyzeForm = document.getElementById('analyze-form');
    const batchAnalyzeForm = document.getElementById('batch-analyze-form');
    const textInput = document.getElementById('text-input');
    const fileInput = document.getElementById('file-input');
    const analyzeButton = document.getElementById('analyze-button');
    const batchAnalyzeButton = document.getElementById('batch-analyze-button');
    const resultContainer = document.getElementById('result-container');
    const chartContainer = document.getElementById('chart-container');
    const loader = document.getElementById('loader');
    const feedbackForm = document.getElementById('feedback-form');
    const fileUploadContainer = document.getElementById('file-upload-container');
    const batchResultsContainer = document.getElementById('batch-results-container');
    const clearButton = document.getElementById('clear-button');
    const exampleButtons = document.querySelectorAll('.example-text');
    const columnSelectionContainer = document.getElementById('column-selection-container');
    
    // Chart instance
    let sentimentChart = null;
    
    // Current analysis results
    let currentAnalysis = null;
    
    // CSV file columns
    let csvColumns = [];
    let selectedTextColumn = '';
    let selectedDateColumn = '';
    
    // Event listeners
    if (analyzeForm) {
        console.log('Analyze form found');
        analyzeForm.addEventListener('submit', handleAnalyze);
    }
    
    if (batchAnalyzeForm) {
        console.log('Batch analyze form found');
        batchAnalyzeForm.addEventListener('submit', handleBatchAnalyze);
    }
    
    if (feedbackForm) {
        feedbackForm.addEventListener('submit', handleFeedback);
    }
    
    // File upload related code
    if (fileUploadContainer && fileInput) {
        console.log('File upload container and input found');
        // This event handler opens the file dialog when clicking on the container
        fileUploadContainer.addEventListener('click', function() {
            console.log('File upload container clicked');
            fileInput.click();
        });

        // This updates the label with the filename once selected and shows column selection
        fileInput.addEventListener('change', function() {
            console.log('File input changed');
            if (fileInput.files.length > 0) {
                const fileLabel = document.getElementById('file-label');
                if (fileLabel) {
                    fileLabel.textContent = fileInput.files[0].name;
                }
                
                // Get CSV columns for selection
                getCsvColumns(fileInput.files[0]);
            } else {
                const fileLabel = document.getElementById('file-label');
                if (fileLabel) {
                    fileLabel.textContent = 'Choose CSV file';
                }
                
                // Hide column selection
                if (columnSelectionContainer) {
                    columnSelectionContainer.style.display = 'none';
                }
            }
        });
    } else {
        console.log('File upload container or input not found');
    }
    
    if (clearButton) {
        clearButton.addEventListener('click', clearResults);
    }
    
    // Add event listeners to example buttons
    if (exampleButtons) {
        exampleButtons.forEach(button => {
            button.addEventListener('click', function() {
                textInput.value = this.getAttribute('data-text');
                analyzeButton.click();
            });
        });
    }
    
    // Get CSV columns for selection
    function getCsvColumns(file) {
        if (!file || !file.name.endsWith('.csv')) {
            return;
        }
        
        // Show loader
        loader.style.display = 'block';
        
        // Send request to get columns
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/get-csv-columns', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loader
            loader.style.display = 'none';
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            console.log('CSV columns:', data);
            
            // Store columns
            csvColumns = data.columns;
            
            // Show column selection
            showColumnSelection(data.columns, data.potential_text_columns, data.potential_date_columns);
        })
        .catch(error => {
            loader.style.display = 'none';
            showError('Error getting CSV columns: ' + error.message);
            console.error('CSV columns error:', error);
        });
    }
    
    // Show column selection UI
    function showColumnSelection(columns, textColumns, dateColumns) {
        if (!columnSelectionContainer) {
            return;
        }
        
        // Create HTML for column selection
        let html = `
            <div class="row mb-3">
                <div class="col-md-6">
                    <label for="text-column-select" class="form-label">Select text column to analyze</label>
                    <select class="form-select" id="text-column-select">
                        ${columns.map(col => {
                            const isRecommended = textColumns.includes(col);
                            return `<option value="${col}" ${isRecommended ? 'selected' : ''}>${col} ${isRecommended ? '(recommended)' : ''}</option>`;
                        }).join('')}
                    </select>
                </div>
                <div class="col-md-6">
                    <label for="date-column-select" class="form-label">Select date column for time trends (optional)</label>
                    <select class="form-select" id="date-column-select">
                        <option value="">None</option>
                        ${columns.map(col => {
                            const isRecommended = dateColumns.includes(col);
                            return `<option value="${col}" ${isRecommended && dateColumns.length === 1 ? 'selected' : ''}>${col} ${isRecommended ? '(recommended)' : ''}</option>`;
                        }).join('')}
                    </select>
                </div>
            </div>
            <div class="alert alert-info" role="alert">
                <i class="fas fa-info-circle me-2"></i>
                Choose which columns to use for analysis. The text column is required, and the date column is optional for time trend analysis.
            </div>
        `;
        
        // Update UI
        columnSelectionContainer.innerHTML = html;
        columnSelectionContainer.style.display = 'block';
        
        // Add event listeners
        const textColumnSelect = document.getElementById('text-column-select');
        const dateColumnSelect = document.getElementById('date-column-select');
        
        if (textColumnSelect) {
            textColumnSelect.addEventListener('change', function() {
                selectedTextColumn = this.value;
            });
            selectedTextColumn = textColumnSelect.value;
        }
        
        if (dateColumnSelect) {
            dateColumnSelect.addEventListener('change', function() {
                selectedDateColumn = this.value;
            });
            selectedDateColumn = dateColumnSelect.value;
        }
    }
    
    // Handle text analysis
    function handleAnalyze(e) {
        e.preventDefault();
        console.log('Analyze form submitted');
        
        const text = textInput.value.trim();
        
        if (!text) {
            showError('Please enter some text to analyze');
            return;
        }
        
        // Show loader
        loader.style.display = 'block';
        resultContainer.innerHTML = '';
        
        // Clear any existing chart
        if (sentimentChart) {
            sentimentChart.destroy();
            sentimentChart = null;
        }
        
        // Send request to server
        const formData = new FormData();
        formData.append('text', text);
        
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loader
            loader.style.display = 'none';
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            console.log('Analysis results received:', data);
            
            // Store current analysis
            currentAnalysis = data;
            
            // Display results
            displayResults(data);
        })
        .catch(error => {
            loader.style.display = 'none';
            showError('Error analyzing text: ' + error.message);
            console.error('Analysis error:', error);
        });
    }
    
    // Handle batch analysis
    function handleBatchAnalyze(e) {
        e.preventDefault();
        console.log('Batch analyze form submitted');
        
        if (!fileInput.files[0]) {
            showError('Please select a CSV file to analyze');
            return;
        }
        
        // Check file type
        const file = fileInput.files[0];
        if (!file.name.endsWith('.csv')) {
            showError('Please upload a CSV file');
            return;
        }
        
        console.log('File selected for batch analysis:', file.name);
        
        // Show loader
        loader.style.display = 'block';
        batchResultsContainer.innerHTML = '';
        
        // Send request to server
        const formData = new FormData();
        formData.append('file', file);
        
        // Add selected columns if available
        if (selectedTextColumn) {
            formData.append('text_column', selectedTextColumn);
        }
        
        if (selectedDateColumn) {
            formData.append('date_column', selectedDateColumn);
        }
        
        fetch('/batch-analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loader
            loader.style.display = 'none';
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            console.log('Batch analysis results received:', data);
            
            // Display batch results
            displayBatchResults(data);
        })
        .catch(error => {
            loader.style.display = 'none';
            showError('Error analyzing file: ' + error.message);
            console.error('Batch analysis error:', error);
        });
    }
    
    // Handle feedback submission
    function handleFeedback(e) {
        e.preventDefault();
        
        if (!currentAnalysis) {
            showError('No analysis to provide feedback for');
            return;
        }
        
        const correctSentiment = document.querySelector('input[name="feedback-sentiment"]:checked');
        
        if (!correctSentiment) {
            showError('Please select the correct sentiment');
            return;
        }
        
        // Send feedback to server
        fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: currentAnalysis.text,
                sentiment: correctSentiment.value
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // Show success message
            const feedbackMessage = document.getElementById('feedback-message');
            feedbackMessage.textContent = data.message;
            feedbackMessage.classList.remove('text-danger');
            feedbackMessage.classList.add('text-success');
            
            // Reset form
            document.querySelectorAll('input[name="feedback-sentiment"]').forEach(input => {
                input.checked = false;
            });
            
            // Hide message after 3 seconds
            setTimeout(() => {
                feedbackMessage.textContent = '';
            }, 3000);
        })
        .catch(error => {
            showError('Error submitting feedback: ' + error.message);
            console.error('Feedback submission error:', error);
        });
    }
    
    // Display analysis results
    function displayResults(data) {
        // Create results HTML
        let resultsHtml = `
            <div class="mb-4">
                <h3>Sentiment Analysis Results</h3>
                <div class="sentiment-label sentiment-${data.sentiment.toLowerCase()}">
                    ${data.sentiment}
                    <span class="confidence-score">${Math.round(data.confidence * 100)}%</span>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <h4>Sentiment Scores</h4>
                    <div id="chart-container" class="chart-container">
                        <canvas id="sentiment-chart"></canvas>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="aspect-container">
                        <h4>Aspect-Based Analysis</h4>
                        <div id="aspects-container">
                            ${data.aspects.length > 0 ? 
                                data.aspects.map(aspect => 
                                    `<div class="aspect-item sentiment-${aspect.sentiment.toLowerCase()}">
                                        ${aspect.aspect}: ${aspect.sentiment}
                                    </div>`
                                ).join('') : 
                                '<p>No specific aspects detected</p>'
                            }
                        </div>
                    </div>
                    
                    <div class="emotion-container mt-4">
                        <h4>Detected Emotions</h4>
                        <div id="emotions-container">
                            ${data.emotions.length > 0 ? 
                                data.emotions.map(emotion => 
                                    `<span class="emotion-badge bg-${getEmotionColor(emotion.emotion)}">
                                        ${emotion.emoji || ''} ${emotion.emotion} (${Math.round(emotion.intensity * 100)}%)
                                    </span>`
                                ).join('') : 
                                '<p>No specific emotions detected</p>'
                            }
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="keyword-cloud mt-4">
                <h4>Key Words</h4>
                <div id="keywords-container">
                    ${data.keywords.length > 0 ? 
                        data.keywords.map(keyword => 
                            `<span class="keyword" style="font-size: ${Math.min(100, keyword.importance * 60 + 80)}%">
                                ${keyword.word}
                            </span>`
                        ).join('') : 
                        '<p>No significant keywords detected</p>'
                    }
                </div>
            </div>
            
            <div class="feedback-form mt-4" id="feedback-container">
                <h4>Was this analysis correct?</h4>
                <form id="feedback-form">
                    <div class="form-check form-check-inline">
                        <input class="form-check-input" type="radio" name="feedback-sentiment" id="feedback-positive" value="Positive">
                        <label class="form-check-label" for="feedback-positive">Positive</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input class="form-check-input" type="radio" name="feedback-sentiment" id="feedback-neutral" value="Neutral">
                        <label class="form-check-label" for="feedback-neutral">Neutral</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input class="form-check-input" type="radio" name="feedback-sentiment" id="feedback-negative" value="Negative">
                        <label class="form-check-label" for="feedback-negative">Negative</label>
                    </div>
                    <button type="submit" class="btn btn-sm btn-outline-info ms-2">Submit Feedback</button>
                    <span id="feedback-message" class="ms-2"></span>
                </form>
            </div>
        `;
        
        // Display results
        resultContainer.innerHTML = resultsHtml;
        
        // Reinitialize feedback form event listener
        document.getElementById('feedback-form').addEventListener('submit', handleFeedback);
        
        // Create chart
        const ctx = document.getElementById('sentiment-chart').getContext('2d');
        sentimentChart = new Chart(ctx, {
            type: 'bar',
            data: data.visualization,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
    
    // Display batch analysis results
    function displayBatchResults(data) {
        console.log('Rendering batch results:', data);
        
        // Create tabs for different visualizations
        let batchHtml = `
            <div class="mb-4">
                <h3>Batch Analysis Results</h3>
                <p>Analyzed ${data.sample_count} texts</p>
            </div>
            
            <!-- Tabs for different visualizations -->
            <ul class="nav nav-tabs mb-3" id="batchTabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" type="button" role="tab" aria-controls="overview" aria-selected="true">
                        Overview
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="keywords-tab" data-bs-toggle="tab" data-bs-target="#keywords" type="button" role="tab" aria-controls="keywords" aria-selected="false">
                        Word Cloud
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="emotions-tab" data-bs-toggle="tab" data-bs-target="#emotions" type="button" role="tab" aria-controls="emotions" aria-selected="false">
                        Emotions
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="distribution-tab" data-bs-toggle="tab" data-bs-target="#distribution" type="button" role="tab" aria-controls="distribution" aria-selected="false">
                        Score Distribution
                    </button>
                </li>
                ${data.time_trend_data ? `
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="time-trends-tab" data-bs-toggle="tab" data-bs-target="#time-trends" type="button" role="tab" aria-controls="time-trends" aria-selected="false">
                        Time Trends
                    </button>
                </li>
                ` : ''}
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="top-reviews-tab" data-bs-toggle="tab" data-bs-target="#top-reviews" type="button" role="tab" aria-controls="top-reviews" aria-selected="false">
                        Top Reviews
                    </button>
                </li>
            </ul>
            
            <!-- Tab content -->
            <div class="tab-content" id="batchTabsContent">
                <!-- Overview tab -->
                <div class="tab-pane fade show active" id="overview" role="tabpanel" aria-labelledby="overview-tab">
                    <div class="row">
                        <div class="col-md-6">
                            <h4>Sentiment Distribution</h4>
                            <div class="chart-container">
                                <canvas id="batch-sentiment-chart"></canvas>
                            </div>
                        </div>
                        
                        <div class="col-md-6">
                            <h4>Summary</h4>
                            <ul class="list-group">
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Positive
                                    <span class="badge bg-success rounded-pill">${data.sentiment_counts.Positive}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Neutral
                                    <span class="badge bg-primary rounded-pill">${data.sentiment_counts.Neutral}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Negative
                                    <span class="badge bg-danger rounded-pill">${data.sentiment_counts.Negative}</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
                
                <!-- Word Cloud tab -->
                <div class="tab-pane fade" id="keywords" role="tabpanel" aria-labelledby="keywords-tab">
                    <div class="row">
                        <div class="col-12">
                            <h4>Key Topics Across All Texts</h4>
                            <div id="word-cloud-container" class="p-4 text-center bg-dark rounded-3" style="min-height: 300px;">
                                ${data.word_cloud_data && data.word_cloud_data.length > 0 ? 
                                    data.word_cloud_data.map(keyword => 
                                        `<span class="keyword" style="font-size: ${Math.min(200, keyword.importance * 30 + 100)}%; 
                                                              margin: 5px;
                                                              display: inline-block;
                                                              opacity: ${Math.min(1, keyword.importance * 0.3 + 0.7)};">
                                            ${keyword.word}
                                        </span>`
                                    ).join('') : 
                                    '<p class="text-muted">No significant keywords found</p>'
                                }
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Emotions tab -->
                <div class="tab-pane fade" id="emotions" role="tabpanel" aria-labelledby="emotions-tab">
                    <div class="row">
                        <div class="col-md-8 mx-auto">
                            <h4>Emotional Breakdown</h4>
                            <div class="chart-container">
                                <canvas id="emotion-radar-chart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Score Distribution tab -->
                <div class="tab-pane fade" id="distribution" role="tabpanel" aria-labelledby="distribution-tab">
                    <div class="row">
                        <div class="col-12">
                            <h4>Sentiment Score Distribution</h4>
                            <div class="chart-container">
                                <canvas id="score-distribution-chart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Time Trends tab -->
                ${data.time_trend_data ? `
                <div class="tab-pane fade" id="time-trends" role="tabpanel" aria-labelledby="time-trends-tab">
                    <div class="row">
                        <div class="col-12">
                            <h4>Sentiment Score Trends Over Time (Grouped by ${data.time_trend_data.score_trends.group_by})</h4>
                            <div class="chart-container">
                                <canvas id="time-trend-chart"></canvas>
                            </div>
                        </div>
                    </div>
                    <div class="row mt-4">
                        <div class="col-12">
                            <h4>Sentiment Percentage Trends Over Time</h4>
                            <div class="chart-container">
                                <canvas id="percentage-trend-chart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                ` : ''}
                
                <!-- Top Reviews tab -->
                <div class="tab-pane fade" id="top-reviews" role="tabpanel" aria-labelledby="top-reviews-tab">
                    <div class="row">
                        <div class="col-md-6">
                            <h4>Most Positive Reviews</h4>
                            <div class="list-group">
                                ${data.top_reviews && data.top_reviews.positive.length > 0 ? 
                                    data.top_reviews.positive.map(review => 
                                        `<div class="list-group-item">
                                            <div class="d-flex w-100 justify-content-between">
                                                <h5 class="mb-1">Positive (${Math.round(review.confidence * 100)}%)</h5>
                                                ${review.date ? `<small class="text-muted">${review.date}</small>` : ''}
                                            </div>
                                            <p class="mb-1">${review.text.length > 150 ? review.text.substring(0, 150) + '...' : review.text}</p>
                                            <small class="text-success">
                                                Keywords: ${review.keywords.slice(0, 3).map(k => k.word).join(', ')}
                                            </small>
                                        </div>`
                                    ).join('') : 
                                    '<p class="text-muted">No positive reviews found</p>'
                                }
                            </div>
                        </div>
                        
                        <div class="col-md-6">
                            <h4>Most Negative Reviews</h4>
                            <div class="list-group">
                                ${data.top_reviews && data.top_reviews.negative.length > 0 ? 
                                    data.top_reviews.negative.map(review => 
                                        `<div class="list-group-item">
                                            <div class="d-flex w-100 justify-content-between">
                                                <h5 class="mb-1">Negative (${Math.round(review.confidence * 100)}%)</h5>
                                                ${review.date ? `<small class="text-muted">${review.date}</small>` : ''}
                                            </div>
                                            <p class="mb-1">${review.text.length > 150 ? review.text.substring(0, 150) + '...' : review.text}</p>
                                            <small class="text-danger">
                                                Keywords: ${review.keywords.slice(0, 3).map(k => k.word).join(', ')}
                                            </small>
                                        </div>`
                                    ).join('') : 
                                    '<p class="text-muted">No negative reviews found</p>'
                                }
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Detailed results section -->
            <div class="detailed-results mt-4">
                <h4>Detailed Results</h4>
                <div class="accordion" id="detailed-results-accordion">
                    ${data.detailed_results.slice(0, 10).map((result, index) => `
                        <div class="accordion-item">
                            <h2 class="accordion-header" id="heading-${index}">
                                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse-${index}" aria-expanded="false" aria-controls="collapse-${index}">
                                    Text ${index + 1}: ${result.sentiment} (${Math.round(result.confidence * 100)}%)
                                </button>
                            </h2>
                            <div id="collapse-${index}" class="accordion-collapse collapse" aria-labelledby="heading-${index}" data-bs-parent="#detailed-results-accordion">
                                <div class="accordion-body">
                                    <p>${result.text}</p>
                                    <div class="sentiment-label sentiment-${result.sentiment.toLowerCase()}">
                                        ${result.sentiment}
                                        <span class="confidence-score">${Math.round(result.confidence * 100)}%</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
                ${data.detailed_results.length > 10 ? 
                    `<p class="text-muted mt-2">Showing first 10 of ${data.detailed_results.length} total results. All ${data.detailed_results.length} entries were analyzed for the visualizations.</p>` : ''}
            </div>
        `;
        
        // Display batch results
        batchResultsContainer.innerHTML = batchHtml;
        
        // Initialize charts
        initBatchCharts(data);
    }
    
    // Initialize all charts for batch analysis
    function initBatchCharts(data) {
        // Sentiment distribution pie chart
        const sentimentCtx = document.getElementById('batch-sentiment-chart').getContext('2d');
        new Chart(sentimentCtx, {
            type: 'pie',
            data: data.aggregate_visualization,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    }
                }
            }
        });
        
        // Emotion radar chart
        if (data.emotion_data && document.getElementById('emotion-radar-chart')) {
            const emotionCtx = document.getElementById('emotion-radar-chart').getContext('2d');
            new Chart(emotionCtx, {
                type: 'radar',
                data: data.emotion_data,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 1
                        }
                    }
                }
            });
        }
        
        // Score distribution histogram
        if (data.score_distribution && document.getElementById('score-distribution-chart')) {
            const distributionCtx = document.getElementById('score-distribution-chart').getContext('2d');
            new Chart(distributionCtx, {
                type: 'bar',
                data: data.score_distribution,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Score Range'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Number of Texts'
                            }
                        }
                    }
                }
            });
        }
        
        // Time trends chart
        if (data.time_trend_data && document.getElementById('time-trend-chart')) {
            const timeCtx = document.getElementById('time-trend-chart').getContext('2d');
            new Chart(timeCtx, {
                type: 'line',
                data: data.time_trend_data.score_trends,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            title: {
                                display: true,
                                text: 'Average Score'
                            }
                        }
                    }
                }
            });
            
            const percentageCtx = document.getElementById('percentage-trend-chart').getContext('2d');
            new Chart(percentageCtx, {
                type: 'line',
                data: data.time_trend_data.percentage_trends,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Percentage (%)'
                            }
                        }
                    }
                }
            });
        }
    }
    
    // Get color for emotion badge
    function getEmotionColor(emotion) {
        const emotionColors = {
            'Happy': 'success',
            'Satisfied': 'info',
            'Neutral': 'secondary',
            'Sad': 'warning',
            'Angry': 'danger'
        };
        
        return emotionColors[emotion] || 'secondary';
    }
    
    // Show error message
    function showError(message) {
        const errorMessage = document.getElementById('error-message');
        if (errorMessage) {
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
            
            // Hide after 5 seconds
            setTimeout(() => {
                errorMessage.style.display = 'none';
            }, 5000);
        }
        console.error('Error:', message);
    }
    
    // Clear results
    function clearResults() {
        // Clear input text
        if (textInput) {
            textInput.value = '';
        }
        
        // Clear results
        if (resultContainer) {
            resultContainer.innerHTML = '';
        }
        
        // Clear any existing chart
        if (sentimentChart) {
            sentimentChart.destroy();
            sentimentChart = null;
        }
    }
});