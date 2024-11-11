def get_audio_recording_js():
    script = '''
    let targetMediaRecorder;
    let targetAudioChunks = [];
    let isTargetRecording = false;

    function toggleTargetRecording() {
        if (!isTargetRecording) {
            startTargetRecording();
        } else {
            stopTargetRecording();
        }
    }

    function startTargetRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                targetMediaRecorder = new MediaRecorder(stream);
                targetMediaRecorder.ondataavailable = e => {
                    targetAudioChunks.push(e.data);
                };
                targetMediaRecorder.onstop = sendTargetAudioToServer;
                targetMediaRecorder.start();
                isTargetRecording = true;
                document.getElementById('target-record-button').textContent = '⏹ Stop Recording';
                document.getElementById('target-recording-status').textContent = 'Recording...';
            })
            .catch(err => {
                console.error('Error accessing microphone', err);
                alert('Could not access your microphone.');
            });
    }

    function stopTargetRecording() {
        if (targetMediaRecorder) {
            targetMediaRecorder.stop();
            isTargetRecording = false;
            document.getElementById('target-record-button').textContent = '🎤 Start Recording';
            document.getElementById('target-recording-status').textContent = 'Processing...';
        }
    }

    function sendTargetAudioToServer() {
        const audioBlob = new Blob(targetAudioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio_data', audioBlob, 'target_recording.wav');

        fetch('/emdr/transcribe-target', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('target_description').value = data.transcription;
            document.getElementById('target-recording-status').textContent = '';
            targetAudioChunks = [];
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('target-recording-status').textContent = 'Error processing audio';
        });
    }
    '''
    return script
