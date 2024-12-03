let queriesChartInstance;
let responseTimeChartInstance;
let topicTimelineChartInstance;
let selectedTopics = [];
let isDashboardInitialized = false;

function selectTopic(element) {
    const topic = element.textContent.trim();

    if (selectedTopics.includes(topic)) {
        selectedTopics = selectedTopics.filter(t => t !== topic);
        element.classList.remove('selected');
    } else {
        selectedTopics.push(topic);
        element.classList.add('selected');
    }

    console.log('Selected topics:', selectedTopics);
}

function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();

    if (message) {
        const chatMessages = document.getElementById('chat-messages');

        const userMessage = `<div class="user-message">${message}</div>`;
        chatMessages.innerHTML += userMessage;

        chatMessages.scrollTop = chatMessages.scrollHeight;

        input.value = '';

        getBotResponse(message, selectedTopics);
    }
}

async function getBotResponse(message, topics) {
    const response = await fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message, topics }),
    });
    const data = await response.json();

    const chatMessages = document.getElementById('chat-messages');
    const botMessage = `<div class="bot-message">${data.response}</div>`;
    chatMessages.innerHTML += botMessage;

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function fetchMetrics() {
    try {
        const response = await fetch('http://127.0.0.1:5000/metrics');
        const metrics = await response.json();

        document.getElementById('total-queries').textContent = metrics.total_queries;
        document.getElementById('chitchat-count').textContent = metrics.chitchat_count;
        document.getElementById('most-popular-topic').textContent = metrics.most_popular_topic;

        document.getElementById('min-response-time').textContent = metrics.min_response_time.toFixed(2) + 's';
        document.getElementById('max-response-time').textContent = metrics.max_response_time.toFixed(2) + 's';

        if (!isDashboardInitialized) {
            initializeQueriesChart(metrics.queries_by_topic);
            initializeResponseTimeChart(metrics.avg_response_times);
            initializeTopicTimelineChart(metrics.topic_timeline);
            isDashboardInitialized = true;
        } else {
            updateQueriesChart(metrics.queries_by_topic);
            updateResponseTimeChart(metrics.avg_response_times);
            updateTopicTimelineChart(metrics.topic_timeline);
        }
    } catch (error) {
        console.error("Error fetching metrics:", error);
    }
}

function toggleView() {
    const chatbotContainer = document.querySelector('.chatbot-container');
    const dashboardContainer = document.getElementById('dashboard-container');

    if (chatbotContainer.style.display === 'none') {
        chatbotContainer.style.display = 'flex';
        dashboardContainer.style.display = 'none';
    } else {
        chatbotContainer.style.display = 'none';
        dashboardContainer.style.display = 'block';
        fetchMetrics();
    }
}

// Initialize Queries Chart
function initializeQueriesChart(queriesByTopic) {
    const ctx = document.getElementById('queriesChart').getContext('2d');
    queriesChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(queriesByTopic),
            datasets: [{
                label: 'Number of Queries',
                data: Object.values(queriesByTopic),
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                },
            },
        },
    });
}

// Update Queries Chart
function updateQueriesChart(queriesByTopic) {
    if (queriesChartInstance) {
        queriesChartInstance.data.labels = Object.keys(queriesByTopic);
        queriesChartInstance.data.datasets[0].data = Object.values(queriesByTopic);
        queriesChartInstance.update();
    }
}

// Initialize Response Time Chart
function initializeResponseTimeChart(avgResponseTimes) {
    const ctx = document.getElementById('responseTimeChart').getContext('2d');
    responseTimeChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Object.keys(avgResponseTimes),
            datasets: [{
                label: 'Average Response Time (s)',
                data: Object.values(avgResponseTimes),
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 2,
                fill: false,
            }],
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                },
            },
        },
    });
}

// Update Response Time Chart
function updateResponseTimeChart(avgResponseTimes) {
    if (responseTimeChartInstance) {
        responseTimeChartInstance.data.labels = Object.keys(avgResponseTimes);
        responseTimeChartInstance.data.datasets[0].data = Object.values(avgResponseTimes);
        responseTimeChartInstance.update();
    }
}

// Initialize Topic Timeline Chart
function initializeTopicTimelineChart(topicTimeline) {
    const ctx = document.getElementById('topicTimelineChart').getContext('2d');
    const data = prepareTopicTimelineData(topicTimeline);

    topicTimelineChartInstance = new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            scales: {
                x: { stacked: true },
                y: { stacked: true, beginAtZero: true },
            },
        },
    });
}

// Update Topic Timeline Chart
function updateTopicTimelineChart(topicTimeline) {
    if (topicTimelineChartInstance) {
        const data = prepareTopicTimelineData(topicTimeline);
        topicTimelineChartInstance.data = data;
        topicTimelineChartInstance.update();
    }
}

// Prepare Topic Timeline Data
function prepareTopicTimelineData(topicTimeline) {
    const timeStamps = [];
    const topicFrequency = {};

    topicTimeline.forEach(item => {
        const timestamp = new Date(item.timestamp * 1000).toLocaleTimeString();
        if (!timeStamps.includes(timestamp)) {
            timeStamps.push(timestamp);
        }
        if (!topicFrequency[timestamp]) {
            topicFrequency[timestamp] = {};
        }
        topicFrequency[timestamp][item.topic] = (topicFrequency[timestamp][item.topic] || 0) + 1;
    });

    const labels = timeStamps;
    const datasets = [];
    const allTopics = [...new Set(topicTimeline.map(item => item.topic))];

    allTopics.forEach(topic => {
        const topicData = labels.map(time => topicFrequency[time]?.[topic] || 0);
        datasets.push({
            label: topic,
            data: topicData,
            backgroundColor: getRandomColor(),
            borderWidth: 1,
            stack: 'stack1',
        });
    });

    return { labels, datasets };
}

// Helper Function to Generate Random Colors
function getRandomColor() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}

