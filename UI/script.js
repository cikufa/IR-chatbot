let selectedTopic = null;  // Variable to store the selected topic

function selectTopic(element) {
    // Remove 'selected' class from all topics
    const topics = document.querySelectorAll('#topics-list li');
    topics.forEach(topic => topic.classList.remove('selected'));

    // Add 'selected' class to the clicked topic
    element.classList.add('selected');

    // Store the selected topic
    selectedTopic = element.textContent.trim();
    console.log('Selected topic:', selectedTopic);
}

function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();

    if (message) {
        if (!selectedTopic) {
            alert('Please select a topic before sending your message.');
            return;
        }

        // Display user's message
        const chatMessages = document.getElementById('chat-messages');
        const userMessage = `<div class="user-message">${message}</div>`;
        chatMessages.innerHTML += userMessage;

        // Scroll to the bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Clear input field
        input.value = '';

        // Send message and selected topic to the bot
        getBotResponse(message, selectedTopic);
    }
}

async function getBotResponse(message, topic) {
    const dataToSend = { message };
    if (topic) {
        dataToSend.topic = topic;
    }

    const response = await fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(dataToSend),
    });

    const data = await response.json();

    const chatMessages = document.getElementById('chat-messages');
    const botMessage = `<div class="bot-message">${data.response}</div>`;
    chatMessages.innerHTML += botMessage;

    // Scroll to the bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
