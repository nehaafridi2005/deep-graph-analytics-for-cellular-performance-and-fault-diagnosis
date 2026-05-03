// Signal Metrics Analysis - Chart Utilities

/**
 * Chart configuration defaults
 */
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            position: 'top',
            labels: {
                usePointStyle: true,
                padding: 20
            }
        },
        tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: '#fff',
            bodyColor: '#fff',
            borderColor: '#fff',
            borderWidth: 1,
            cornerRadius: 8,
            displayColors: true
        }
    },
    scales: {
        x: {
            grid: {
                display: true,
                color: 'rgba(0, 0, 0, 0.1)'
            },
            ticks: {
                font: {
                    size: 12
                }
            }
        },
        y: {
            grid: {
                display: true,
                color: 'rgba(0, 0, 0, 0.1)'
            },
            ticks: {
                font: {
                    size: 12
                }
            }
        }
    }
};

/**
 * Color schemes for charts
 */
const colorSchemes = {
    primary: [
        'rgba(13, 110, 253, 0.8)',
        'rgba(25, 135, 84, 0.8)',
        'rgba(255, 193, 7, 0.8)',
        'rgba(220, 53, 69, 0.8)',
        'rgba(13, 202, 240, 0.8)',
        'rgba(102, 16, 242, 0.8)'
    ],
    primaryBorder: [
        'rgba(13, 110, 253, 1)',
        'rgba(25, 135, 84, 1)',
        'rgba(255, 193, 7, 1)',
        'rgba(220, 53, 69, 1)',
        'rgba(13, 202, 240, 1)',
        'rgba(102, 16, 242, 1)'
    ],
    gradient: [
        'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
        'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
        'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
        'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
        'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)'
    ]
};

/**
 * Create a bar chart with custom configuration
 */
function createBarChart(ctx, data, options = {}) {
    const config = {
        type: 'bar',
        data: data,
        options: {
            ...chartDefaults,
            ...options,
            scales: {
                ...chartDefaults.scales,
                ...options.scales
            }
        }
    };
    
    return new Chart(ctx, config);
}

/**
 * Create a line chart with custom configuration
 */
function createLineChart(ctx, data, options = {}) {
    const config = {
        type: 'line',
        data: data,
        options: {
            ...chartDefaults,
            ...options,
            elements: {
                line: {
                    tension: 0.4
                },
                point: {
                    radius: 6,
                    hoverRadius: 8
                }
            }
        }
    };
    
    return new Chart(ctx, config);
}

/**
 * Create a pie chart with custom configuration
 */
function createPieChart(ctx, data, options = {}) {
    const config = {
        type: 'pie',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            },
            ...options
        }
    };
    
    return new Chart(ctx, config);
}

/**
 * Create a doughnut chart with custom configuration
 */
function createDoughnutChart(ctx, data, options = {}) {
    const config = {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            },
            ...options
        }
    };
    
    return new Chart(ctx, config);
}

/**
 * Create a scatter plot with custom configuration
 */
function createScatterChart(ctx, data, options = {}) {
    const config = {
        type: 'scatter',
        data: data,
        options: {
            ...chartDefaults,
            ...options,
            elements: {
                point: {
                    radius: 4,
                    hoverRadius: 6
                }
            }
        }
    };
    
    return new Chart(ctx, config);
}

/**
 * Create a radar chart with custom configuration
 */
function createRadarChart(ctx, data, options = {}) {
    const config = {
        type: 'radar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            elements: {
                line: {
                    borderWidth: 3
                },
                point: {
                    radius: 4,
                    hoverRadius: 6
                }
            },
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 100
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            ...options
        }
    };
    
    return new Chart(ctx, config);
}

/**
 * Utility function to generate gradient backgrounds
 */
function createGradient(ctx, colorStart, colorEnd) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, colorStart);
    gradient.addColorStop(1, colorEnd);
    return gradient;
}

/**
 * Update chart data dynamically
 */
function updateChartData(chart, newData) {
    chart.data = newData;
    chart.update('active');
}

/**
 * Add data point to existing chart
 */
function addDataPoint(chart, label, data) {
    chart.data.labels.push(label);
    chart.data.datasets.forEach((dataset, index) => {
        dataset.data.push(data[index]);
    });
    chart.update();
}

/**
 * Remove data point from chart
 */
function removeDataPoint(chart, index) {
    chart.data.labels.splice(index, 1);
    chart.data.datasets.forEach(dataset => {
        dataset.data.splice(index, 1);
    });
    chart.update();
}

/**
 * Export chart as image
 */
function exportChart(chart, filename = 'chart.png') {
    const link = document.createElement('a');
    link.download = filename;
    link.href = chart.toBase64Image();
    link.click();
}

/**
 * Resize chart to fit container
 */
function resizeChart(chart) {
    chart.resize();
}

/**
 * Animation configuration
 */
const animationConfig = {
    duration: 1500,
    easing: 'easeInOutQuart',
    delay: (context) => {
        return context.type === 'data' && context.mode === 'default' 
            ? context.dataIndex * 50 
            : 0;
    }
};

/**
 * Create performance comparison chart
 */
function createPerformanceChart(containerId, models, metrics) {
    const ctx = document.getElementById(containerId);
    if (!ctx) return null;
    
    const data = {
        labels: models,
        datasets: Object.keys(metrics[models[0]]).map((metric, index) => ({
            label: metric.charAt(0).toUpperCase() + metric.slice(1),
            data: models.map(model => metrics[model][metric]),
            backgroundColor: colorSchemes.primary[index % colorSchemes.primary.length],
            borderColor: colorSchemes.primaryBorder[index % colorSchemes.primaryBorder.length],
            borderWidth: 2,
            borderRadius: 4,
            borderSkipped: false
        }))
    };
    
    const options = {
        plugins: {
            title: {
                display: true,
                text: 'Model Performance Comparison',
                font: {
                    size: 16,
                    weight: 'bold'
                }
            }
        },
        animation: animationConfig,
        interaction: {
            mode: 'index',
            intersect: false
        }
    };
    
    return createBarChart(ctx.getContext('2d'), data, options);
}

/**
 * Create network type distribution chart
 */
function createNetworkDistributionChart(containerId, networkData) {
    const ctx = document.getElementById(containerId);
    if (!ctx) return null;
    
    const data = {
        labels: Object.keys(networkData),
        datasets: [{
            data: Object.values(networkData),
            backgroundColor: colorSchemes.primary,
            borderColor: colorSchemes.primaryBorder,
            borderWidth: 2
        }]
    };
    
    const options = {
        plugins: {
            title: {
                display: true,
                text: 'Network Type Distribution',
                font: {
                    size: 16,
                    weight: 'bold'
                }
            }
        },
        animation: {
            animateRotate: true,
            animateScale: true,
            duration: 2000
        }
    };
    
    return createDoughnutChart(ctx.getContext('2d'), data, options);
}

/**
 * Create signal strength trend chart
 */
function createSignalTrendChart(containerId, timeData, signalData) {
    const ctx = document.getElementById(containerId);
    if (!ctx) return null;
    
    const data = {
        labels: timeData,
        datasets: [{
            label: 'Signal Strength (dBm)',
            data: signalData,
            borderColor: colorSchemes.primaryBorder[0],
            backgroundColor: colorSchemes.primary[0],
            fill: true,
            tension: 0.4,
            pointBackgroundColor: '#fff',
            pointBorderColor: colorSchemes.primaryBorder[0],
            pointBorderWidth: 2,
            pointRadius: 4
        }]
    };
    
    const options = {
        plugins: {
            title: {
                display: true,
                text: 'Signal Strength Trend',
                font: {
                    size: 16,
                    weight: 'bold'
                }
            }
        },
        scales: {
            y: {
                title: {
                    display: true,
                    text: 'Signal Strength (dBm)'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Time'
                }
            }
        },
        animation: animationConfig
    };
    
    return createLineChart(ctx.getContext('2d'), data, options);
}

/**
 * Initialize all charts on page load
 */
document.addEventListener('DOMContentLoaded', function() {
    // Set Chart.js global defaults
    Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.color = '#495057';
    
    // Initialize responsive behavior
    window.addEventListener('resize', function() {
        Chart.helpers.each(Chart.instances, function(instance) {
            instance.resize();
        });
    });
});

/**
 * Utility to format numbers for display
 */
function formatNumber(value, decimals = 2) {
    return parseFloat(value).toFixed(decimals);
}

/**
 * Utility to format percentage values
 */
function formatPercentage(value, decimals = 1) {
    return `${parseFloat(value).toFixed(decimals)}%`;
}

/**
 * Create loading state for charts
 */
function showChartLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="loading">
                <i class="fas fa-spinner fa-spin fa-2x mb-3"></i>
                <p>Loading chart data...</p>
            </div>
        `;
    }
}

/**
 * Remove loading state and show error
 */
function showChartError(containerId, error) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Error loading chart: ${error}
            </div>
        `;
    }
}

// Export functions for use in other scripts
window.ChartUtils = {
    createBarChart,
    createLineChart,
    createPieChart,
    createDoughnutChart,
    createScatterChart,
    createRadarChart,
    createPerformanceChart,
    createNetworkDistributionChart,
    createSignalTrendChart,
    updateChartData,
    addDataPoint,
    removeDataPoint,
    exportChart,
    resizeChart,
    showChartLoading,
    showChartError,
    formatNumber,
    formatPercentage,
    colorSchemes
};
