// Chart.js configuration for sentiment analysis visualizations

// Create a sentiment chart
function createSentimentChart(elementId, data) {
    const ctx = document.getElementById(elementId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels,
            datasets: [{
                label: 'Sentiment Scores',
                data: data.values,
                backgroundColor: [
                    'rgba(255, 99, 132, 0.5)',   // Negative
                    'rgba(54, 162, 235, 0.5)',   // Neutral
                    'rgba(75, 192, 192, 0.5)'    // Positive
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(75, 192, 192, 1)'
                ],
                borderWidth: 1
            }]
        },
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