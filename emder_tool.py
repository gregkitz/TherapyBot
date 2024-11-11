from fasthtml.common import *
import os


def emdr_tool():
    js_code = """
    let dotX, dotY, dotRadius, canvasWidth, canvasHeight, direction, dotSpeed;
    let clickRightSound, clickLeftSound;
    let isRunning = false;
    let startButton;
    let timerSelect, timerDisplay;
    let timerDuration = 0;
    let timeRemaining = 0;
    let timerInterval;

    function setup() {
        const canvas = document.getElementById('emdrCanvas');
        canvasWidth = canvas.width = window.innerWidth;
        canvasHeight = canvas.height = window.innerHeight;
        dotRadius = 60;
        dotX = canvasWidth / 2;
        dotY = canvasHeight / 2;
        dotSpeed = 40;
        direction = 1;

        clickRightSound = new Audio('/static/click_right.wav');
        clickLeftSound = new Audio('/static/click_left.wav');

        // Preload sounds
        clickRightSound.load();
        clickLeftSound.load();

        window.addEventListener('resize', resizeCanvas);
        window.addEventListener('keydown', handleKeyPress);

        // Draw initial state
        drawStaticState();

        // Add start button
        startButton = document.createElement('button');
        startButton.textContent = 'Start EMDR';
        startButton.style.position = 'absolute';
        startButton.style.top = '20px';
        startButton.style.left = '20px';
        startButton.style.transition = 'opacity 0.3s ease-in-out';
        document.body.appendChild(startButton);

        startButton.addEventListener('click', toggleEMDR);

        // Add timer select dropdown
        timerSelect = document.createElement('select');
        timerSelect.style.position = 'absolute';
        timerSelect.style.top = '20px';
        timerSelect.style.left = '150px';
        timerSelect.style.fontSize = '16px';
        timerSelect.style.padding = '5px';
        timerSelect.style.borderRadius = '5px';
        timerSelect.style.border = '1px solid #ccc';

        // Populate dropdown options from 5 to 45 minutes in 5-minute increments
        for (let i = 5; i <= 45; i += 5) {
            let option = document.createElement('option');
            option.value = i;
            option.textContent = i + ' min';
            timerSelect.appendChild(option);
        }
        timerSelect.value = '15'; // Default to 15 minutes
        document.body.appendChild(timerSelect);

        // Add timer display
        timerDisplay = document.createElement('span');
        timerDisplay.style.position = 'absolute';
        timerDisplay.style.top = '20px';
        timerDisplay.style.left = '250px';
        timerDisplay.style.fontSize = '18px';
        timerDisplay.style.color = 'white';
        document.body.appendChild(timerDisplay);

        // Add event listeners for button visibility
        document.addEventListener('mousemove', showButton);
        startButton.addEventListener('mouseout', hideButtonIfRunning);
    }

    function toggleEMDR() {
        if (!isRunning) {
            isRunning = true;
            startButton.textContent = 'Stop EMDR';

            // Get the timer duration from dropdown
            timerDuration = parseInt(timerSelect.value) * 60; // Convert minutes to seconds
            timeRemaining = timerDuration;
            if (timeRemaining > 0) {
                // Start the timer
                timerInterval = setInterval(updateTimer, 1000); // Update every second
                updateTimerDisplay();
            } else {
                // If no timer is set, clear the display
                timerDisplay.textContent = '';
            }

            requestAnimationFrame(draw);
            hideButtonIfRunning();
        } else {
            stopEMDR();
        }
    }

    function stopEMDR() {
        isRunning = false;
        startButton.textContent = 'Start EMDR';
        drawStaticState();
        showButton();

        // Clear the timer if it's running
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }

        // Clear the timer display
        timerDisplay.textContent = '';
    }

    function updateTimer() {
        if (timeRemaining > 0) {
            timeRemaining--;
            updateTimerDisplay();
            if (timeRemaining <= 0) {
                // Time's up, stop EMDR
                stopEMDR();
            }
        }
    }

    function updateTimerDisplay() {
        let minutes = Math.floor(timeRemaining / 60);
        let seconds = timeRemaining % 60;
        timerDisplay.textContent = minutes + ':' + seconds.toString().padStart(2, '0');
    }

    function showButton() {
        startButton.style.opacity = '1';
        startButton.style.pointerEvents = 'auto';
    }

    function hideButtonIfRunning() {
        if (isRunning) {
            startButton.style.opacity = '0';
            startButton.style.pointerEvents = 'none';
        }
    }

    function resizeCanvas() {
        const canvas = document.getElementById('emdrCanvas');
        canvasWidth = canvas.width = window.innerWidth;
        canvasHeight = canvas.height = window.innerHeight;
        dotY = canvasHeight / 2;
        if (!isRunning) {
            drawStaticState();
        }
    }

    function handleKeyPress(event) {
        if (event.key === 'ArrowUp') {
            dotSpeed += 1;
        } else if (event.key === 'ArrowDown') {
            dotSpeed = Math.max(1, dotSpeed - 1);
        }
    }

    function drawStaticState() {
        const canvas = document.getElementById('emdrCanvas');
        const ctx = canvas.getContext('2d');

        ctx.fillStyle = '#3B5998';  // Dark blue background
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);

        ctx.fillStyle = '#FF0000';  // Red dot
        ctx.beginPath();
        ctx.arc(canvasWidth / 2, canvasHeight / 2, dotRadius, 0, Math.PI * 2);
        ctx.fill();
    }

    function draw() {
        if (!isRunning) return;

        const canvas = document.getElementById('emdrCanvas');
        const ctx = canvas.getContext('2d');

        ctx.fillStyle = '#3B5998';  // Dark blue background
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);

        dotX += dotSpeed * direction;

        const edgeThreshold = 20;
        if (direction === 1 && dotX >= canvasWidth - dotRadius - edgeThreshold) {
            clickRightSound.play().catch(e => console.error("Error playing sound:", e));
        } else if (direction === -1 && dotX <= dotRadius + edgeThreshold) {
            clickLeftSound.play().catch(e => console.error("Error playing sound:", e));
        }

        if (dotX <= dotRadius || dotX >= canvasWidth - dotRadius) {
            direction *= -1;
        }

        ctx.fillStyle = '#FF0000';  // Red dot
        ctx.beginPath();
        ctx.arc(dotX, dotY, dotRadius, 0, Math.PI * 2);
        ctx.fill();

        requestAnimationFrame(draw);
    }

    document.addEventListener('DOMContentLoaded', setup);
    """

    html_content = Html(
        Head(
            Title("EMDR Tool"),
            Script(js_code),
            Style("""
                body { margin: 0; overflow: hidden; }
                canvas { display: block; }
                button { 
                    font-size: 16px; 
                    padding: 10px 20px; 
                    cursor: pointer;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    transition: opacity 0.3s ease-in-out, background-color 0.3s ease;
                }
                button:hover {
                    background-color: #45a049;
                }
                select {
                    font-size: 16px;
                    padding: 5px;
                    border-radius: 5px;
                    border: 1px solid #ccc;
                }
                span {
                    font-size: 18px;
                    color: white;
                }
            """)
        ),
        Body(
            Canvas(id="emdrCanvas"),
        )
    )

    return html_content


app, rt = fast_app()


@rt('/')
def get():
    return emdr_tool()


@rt('/static/{filename}')
def get(filename: str):
    file_path = os.path.join('resources', filename)
    if not os.path.exists(file_path):
        raise NotFoundError(f"File {filename} not found")
    return FileResponse(file_path)


if __name__ == '__main__':
    serve(port=5005)