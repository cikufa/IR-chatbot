// script.js

let selectedTopics = [];

function selectTopic(element) {
    const topic = element.textContent.trim();

    if (selectedTopics.includes(topic)) {
        // Deselect the topic
        selectedTopics = selectedTopics.filter(t => t !== topic);
        element.classList.remove('selected');
    } else {
        // Select the topic
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

        // Display user's message
        const userMessage = `<div class="user-message">${message}</div>`;
        chatMessages.innerHTML += userMessage;

        // Scroll to the bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Clear input field
        input.value = '';

        // Send message and selected topics to the server
        getBotResponse(message, selectedTopics);
    }
}
async function getBotResponse(message, topics) {
    const response = await fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message, topics }), // Send the user's message and selected topics as JSON
    });
    const data = await response.json();

    const chatMessages = document.getElementById('chat-messages');
    const botMessage = `<div class="bot-message">${data.response}</div>`;
    chatMessages.innerHTML += botMessage;

    // Scroll to the bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}