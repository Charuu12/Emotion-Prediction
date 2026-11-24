// ==========================================
// EMOTIVE — Emotion Intelligence
// Frontend JavaScript
// ==========================================


// DOM Elements
const userText = document.getElementById("userText");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");

const charCount = document.getElementById("charCount");
const healthDot = document.getElementById("healthDot");
const healthText = document.getElementById("healthText");

const emptyState = document.getElementById("emptyState");
const resultContent = document.getElementById("resultContent");

const emotionEmoji = document.getElementById("emotionEmoji");
const emotionName = document.getElementById("emotionName");
const confidenceText = document.getElementById("confidenceText");

const bars = document.getElementById("bars");
const insightText = document.getElementById("insightText");
const errorBox = document.getElementById("errorBox");


// API Endpoints
const API_URL = "";


// Emotion Configuration
const emotionConfig = {
    sadness: {
        emoji: "😢",
        color: "#8fa8ff"
    },
    joy: {
        emoji: "😊",
        color: "#79d6c5"
    },
    love: {
        emoji: "❤️",
        color: "#ff91b5"
    },
    anger: {
        emoji: "😠",
        color: "#ff8585"
    },
    fear: {
        emoji: "😨",
        color: "#c0a5ff"
    },
    surprise: {
        emoji: "😲",
        color: "#f4c779"
    }
};


// ==========================================
// 1. CHECK BACKEND HEALTH
// ==========================================

async function checkHealth() {

    try {

        const response = await fetch(`${API_URL}/health`);

        if (!response.ok) {
            throw new Error("Backend unavailable");
        }

        const data = await response.json();

        if (data.model_loaded) {

            healthDot.classList.add("online");
            healthText.textContent = "Model online";

        } else {

            healthDot.classList.remove("online");
            healthText.textContent = "Model not loaded";

        }

    } catch (error) {

        healthDot.classList.remove("online");
        healthText.textContent = "API unavailable";

    }

}


// ==========================================
// 2. CHARACTER COUNTER
// ==========================================

userText.addEventListener("input", () => {

    const length = userText.value.length;

    charCount.textContent = `${length} / 2000`;

});


// ==========================================
// 3. SAMPLE TEXT BUTTONS
// ==========================================

const sampleButtons = document.querySelectorAll(".chip");

sampleButtons.forEach(button => {

    button.addEventListener("click", () => {

        userText.value = button.dataset.text;

        charCount.textContent =
            `${userText.value.length} / 2000`;

        hideError();

        userText.focus();

    });

});


// ==========================================
// 4. CLEAR BUTTON
// ==========================================

clearBtn.addEventListener("click", () => {

    userText.value = "";

    charCount.textContent = "0 / 2000";

    emptyState.style.display = "block";

    resultContent.style.display = "none";

    bars.innerHTML = "";

    hideError();

    userText.focus();

});


// ==========================================
// 5. ERROR HANDLING
// ==========================================

function showError(message) {

    errorBox.textContent = message;

    errorBox.style.display = "block";

}


function hideError() {

    errorBox.textContent = "";

    errorBox.style.display = "none";

}


// ==========================================
// 6. LOADING STATE
// ==========================================

function setLoading(isLoading) {

    analyzeBtn.disabled = isLoading;

    if (isLoading) {

        analyzeBtn.innerHTML = `
            Analysing...
            <span aria-hidden="true">⏳</span>
        `;

    } else {

        analyzeBtn.innerHTML = `
            Analyse emotion
            <span aria-hidden="true">↗</span>
        `;

    }

}


// ==========================================
// 7. ANALYSE EMOTION
// ==========================================

async function analyzeEmotion() {

    const text = userText.value.trim();

    hideError();


    // Validate input

    if (!text) {

        showError("Please enter some text before analysing.");

        userText.focus();

        return;

    }


    if (text.length > 2000) {

        showError("Text cannot exceed 2000 characters.");

        return;

    }


    setLoading(true);


    try {

        // Send request to FastAPI

        const response = await fetch(`${API_URL}/predict`, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: text
            })

        });


        // Handle unsuccessful response

        if (!response.ok) {

            const errorData = await response.json().catch(() => ({}));

            throw new Error(
                errorData.detail || "Unable to analyse the text."
            );

        }


        // Convert response into JSON

        const data = await response.json();


        // Extract prediction results

        const predictedEmotion = data.predicted_emotion;

        const confidence = data.confidence;


        // Support both spellings in case backend key changes

        const probabilities =
            data.all_probabilities || data.all_probabilites;


        if (!predictedEmotion || confidence === undefined || !probabilities) {

            throw new Error("Invalid response received from the server.");

        }


        // Display results

        displayResults(
            predictedEmotion,
            confidence,
            probabilities
        );


    } catch (error) {

        console.error("Prediction error:", error);

        showError(
            error.message || "Something went wrong. Please try again."
        );

    } finally {

        setLoading(false);

    }

}


// ==========================================
// 8. DISPLAY RESULTS
// ==========================================

function displayResults(predictedEmotion, confidence, probabilities) {


    // Hide empty state

    emptyState.style.display = "none";

    resultContent.style.display = "block";


    // Get emotion configuration

    const config = emotionConfig[predictedEmotion.toLowerCase()] || {
        emoji: "✨",
        color: "#a99bff"
    };


    // Display dominant emotion

    emotionEmoji.textContent = config.emoji;

    emotionName.textContent = predictedEmotion;

    confidenceText.textContent =
        `${(Number(confidence) * 100).toFixed(2)}% confidence`;


    // Clear previous bars

    bars.innerHTML = "";


    // Create probability bars

    const sortedEmotions = Object.entries(probabilities)
        .sort((a, b) => Number(b[1]) - Number(a[1]));


    sortedEmotions.forEach(([emotion, probability], index) => {

        const percentage = Number(probability) * 100;

        const emotionData = emotionConfig[emotion.toLowerCase()] || {
            color: "#a99bff"
        };


        // Create bar container

        const barRow = document.createElement("div");

        barRow.className = "bar-row";


        barRow.innerHTML = `

            <div class="bar-info">

                <span>${emotion}</span>

                <b>${percentage.toFixed(2)}%</b>

            </div>

            <div class="track">

                <div class="fill"
                     style="background: linear-gradient(90deg, ${emotionData.color}, #79d6c5);">
                </div>

            </div>

        `;


        bars.appendChild(barRow);


        // Animate progress bar

        const fill = barRow.querySelector(".fill");

        setTimeout(() => {

            fill.style.width = `${Math.min(100, Math.max(0, percentage))}%`;

        }, 100 + index * 100);

    });


    // Update insight text

    insightText.innerHTML = `
        <strong>Reading:</strong>
        The model identified <strong>${predictedEmotion}</strong>
        as the dominant emotion. These probabilities represent
        the model's predictions, not a definitive measure of
        how someone feels.
    `;


}


// ==========================================
// 9. ANALYSE BUTTON EVENT
// ==========================================

analyzeBtn.addEventListener("click", analyzeEmotion);


// ==========================================
// 10. KEYBOARD SHORTCUT
// Ctrl + Enter / Cmd + Enter
// ==========================================

userText.addEventListener("keydown", (event) => {

    if (
        event.key === "Enter" &&
        (event.ctrlKey || event.metaKey)
    ) {

        event.preventDefault();

        analyzeEmotion();

    }

});


// ==========================================
// 11. INITIALIZE APPLICATION
// ==========================================

document.addEventListener("DOMContentLoaded", () => {

    checkHealth();

    // Check backend status periodically

    setInterval(checkHealth, 30000);

});