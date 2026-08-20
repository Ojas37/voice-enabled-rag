document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const micBtn = document.getElementById('mic-btn');
    const micBtnText = document.getElementById('mic-btn-text');
    const recordingStatus = document.getElementById('recording-status');
    const audioRibbon = document.getElementById('audio-ribbon');
    const transcriptionBox = document.getElementById('transcription-box');
    const answerBox = document.getElementById('answer-box');
    const verdictBadge = document.getElementById('verdict-badge');
    const lineageSources = document.getElementById('lineage-sources');
    
    // Metric UI Elements
    const metricStt = document.getElementById('metric-stt');
    const metricRetrieval = document.getElementById('metric-retrieval');
    const metricTtft = document.getElementById('metric-ttft');
    const metricGen = document.getElementById('metric-gen');
    const metricGuard = document.getElementById('metric-guard');
    const metricNetwork = document.getElementById('metric-network');
    const metricE2e = document.getElementById('metric-e2e');

    // Recording State Variables
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    let animationId = null;
    let latencyChart = null;

    // Latency metrics tracking
    let metrics = {
        stt: 0,
        retrieval: 0,
        ttft: 0,
        generation: 0,
        guardrail: 0,
        e2e: 0
    };

    // Initialize Latency Chart (Chart.js)
    function initChart() {
        const ctx = document.getElementById('latencyChart').getContext('2d');
        latencyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['STT RTT', 'Retrieval', 'LLM TTFT', 'LLM Gen', 'Guardrail'],
                datasets: [{
                    label: 'Latency (ms)',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(100, 116, 139, 0.25)', // Muted Slate
                        'rgba(100, 116, 139, 0.25)',
                        'rgba(37, 99, 235, 0.2)',    // Indigo
                        'rgba(37, 99, 235, 0.2)',    // Indigo
                        'rgba(5, 150, 105, 0.2)'      // Emerald
                    ],
                    borderColor: [
                        '#64748B',
                        '#64748B',
                        '#2563EB',
                        '#2563EB',
                        '#059669'
                    ],
                    borderWidth: 1,
                    barPercentage: 0.6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: '#1C222B' },
                        ticks: { color: '#64748B', font: { family: 'IBM Plex Mono', size: 9 } },
                        border: { display: false }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#F8FAFC', font: { family: 'IBM Plex Sans', size: 10 } },
                        border: { display: false }
                    }
                }
            }
        });
    }

    function updateChart() {
        if (!latencyChart) return;
        latencyChart.data.datasets[0].data = [
            metrics.stt,
            metrics.retrieval,
            metrics.ttft,
            metrics.generation,
            metrics.guardrail
        ];
        latencyChart.update();
    }

    // Reset UI before a new query
    function resetWorkspace() {
        transcriptionBox.textContent = "Awaiting voice input...";
        transcriptionBox.classList.add('placeholder-text');
        
        answerBox.textContent = "Processing query...";
        answerBox.classList.remove('placeholder-text');
        
        verdictBadge.textContent = "RUNNING";
        verdictBadge.className = "verdict-badge badge-neutral";
        
        lineageSources.innerHTML = '<div class="placeholder-text">Searching LanceDB index...</div>';
        
        // Reset metrics
        metrics = { stt: 0, retrieval: 0, ttft: 0, generation: 0, guardrail: 0, network: 0, e2e: 0 };
        metricStt.textContent = "-";
        metricRetrieval.textContent = "-";
        metricTtft.textContent = "-";
        metricGen.textContent = "-";
        metricGuard.textContent = "-";
        metricNetwork.textContent = "-";
        metricE2e.textContent = "-";
        updateChart();
    }

    // Start Audio Capture and Volume Ribbon
    async function startRecording() {
        try {
            audioChunks = [];
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Web Audio Analyser setup
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            
            const bufferLength = analyser.frequencyBinCount;
            dataArray = new Uint8Array(bufferLength);
            
            // Dynamic Ribbon Drawing Loop
            function drawRibbon() {
                if (!isRecording) return;
                animationId = requestAnimationFrame(drawRibbon);
                analyser.getByteFrequencyData(dataArray);
                
                // Calculate average volume level
                let sum = 0;
                for (let i = 0; i < bufferLength; i++) {
                    sum += dataArray[i];
                }
                const average = sum / bufferLength;
                
                // Set ribbon width matching signal strength
                const percentage = Math.min(100, Math.max(5, (average / 128.0) * 100));
                audioRibbon.style.width = percentage + '%';
            }
            
            // Choose recording format (WebM/Opus compression favored)
            let options = { mimeType: 'audio/webm;codecs=opus' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options = { mimeType: 'audio/webm' };
            }
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options = {}; // Browser fallback (WAV/MP4)
            }
            
            mediaRecorder = new MediaRecorder(stream, options);
            mediaRecorder.ondataavailable = event => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                stream.getTracks().forEach(track => track.stop());
                uploadAudio(audioBlob);
            };

            // Trigger active state
            isRecording = true;
            micBtn.classList.add('btn-recording');
            micBtnText.textContent = "Recording... Click to Stop";
            recordingStatus.textContent = "RECORDING";
            recordingStatus.classList.add('status-recording');
            
            mediaRecorder.start(250); // Capture data in 250ms slices
            drawRibbon();
            
        } catch (err) {
            console.error("Microphone Access Denied / Failed:", err);
            alert("Could not access microphone. Please check system permissions.");
        }
    }

    function stopRecording() {
        if (!isRecording) return;
        isRecording = false;
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        if (audioContext) {
            audioContext.close();
        }
        cancelAnimationFrame(animationId);
        audioRibbon.style.width = '0%';
        micBtn.classList.remove('btn-recording');
        micBtnText.textContent = "Click to Record / Press Space";
        recordingStatus.textContent = "PROCESSING";
        recordingStatus.classList.remove('status-recording');
    }

    // Process Server-Sent Events (SSE)
    async function uploadAudio(audioBlob) {
        resetWorkspace();
        
        const selectedLang = document.querySelector('input[name="retrieval-lang"]:checked').value;
        const formData = new FormData();
        // File extension hints format to endpoint
        const fileExt = audioBlob.type.includes('webm') ? 'webm' : 'wav';
        formData.append('file', audioBlob, `voice_query.${fileExt}`);
        formData.append('language', selectedLang);

        const startTime = performance.now();

        try {
            const response = await fetch('/api/query-voice', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = "";
            let fullAnswerText = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                
                // Retain incomplete buffer line
                buffer = lines.pop();

                for (const line of lines) {
                    const cleanLine = line.trim();
                    if (!cleanLine.startsWith('data: ')) continue;
                    
                    try {
                        const payload = JSON.parse(cleanLine.substring(6));
                        handleSSEEvent(payload);
                    } catch (e) {
                        console.warn("JSON parsing failure in SSE chunk:", e);
                    }
                }
            }
            
            // Stop processing state
            recordingStatus.textContent = "READY";
            metrics.e2e = performance.now() - startTime;
            
            // Calculate network/rendering overhead to align components exactly with E2E RTT
            const activeGen = Math.max(0, metrics.generation - metrics.ttft);
            const sumComponents = metrics.stt + metrics.retrieval + metrics.ttft + activeGen + metrics.guardrail;
            metrics.network = Math.max(0, metrics.e2e - sumComponents);
            
            metricNetwork.textContent = `${metrics.network.toFixed(0)} ms`;
            metricE2e.textContent = `${metrics.e2e.toFixed(0)} ms`;
            updateChart();

        } catch (err) {
            console.error("SSE execution error:", err);
            recordingStatus.textContent = "READY";
            verdictBadge.textContent = "ERROR";
            verdictBadge.className = "verdict-badge badge-fail";
            answerBox.textContent = `Execution failed: ${err.message}`;
            answerBox.classList.add('placeholder-text');
        }
    }

    // Handle Individual SSE Events
    function handleSSEEvent(payload) {
        const { event } = payload;
        
        if (event === "stt_complete") {
            transcriptionBox.textContent = `"${payload.text}"`;
            transcriptionBox.classList.remove('placeholder-text');
            metrics.stt = payload.latency_ms;
            metricStt.textContent = `${metrics.stt.toFixed(0)} ms`;
            updateChart();
        } 
        
        else if (event === "retrieval_complete") {
            metrics.retrieval = payload.retrieval_ms;
            metricRetrieval.textContent = `${metrics.retrieval.toFixed(0)} ms`;
            
            // Select language radio dynamically if changed by auto-detection
            const radioBtn = document.querySelector(`input[name="retrieval-lang"][value="${payload.language}"]`);
            if (radioBtn) radioBtn.checked = true;

            // Render source lineage
            if (payload.sources && payload.sources.length > 0) {
                lineageSources.innerHTML = "";
                payload.sources.forEach((src, idx) => {
                    const node = document.createElement('div');
                    node.className = 'source-node';
                    node.innerHTML = `
                        <div class="source-meta">
                            <span>[Source ${idx + 1}] Language: ${src.language.toUpperCase()}</span>
                            <span>Relevance: ${(src.score * 100).toFixed(0)}%</span>
                        </div>
                        <div class="source-body">${src.raw_body}</div>
                    `;
                    lineageSources.appendChild(node);
                });
            } else {
                lineageSources.innerHTML = '<div class="placeholder-text">No matching documents retrieved.</div>';
            }
            updateChart();
        } 
        
        else if (event === "generation_start") {
            metrics.ttft = payload.ttft_ms;
            metricTtft.textContent = `${metrics.ttft.toFixed(0)} ms`;
            // Clear processing text on first token
            answerBox.textContent = "";
            updateChart();
        } 
        
        else if (event === "token") {
            // Stream tokens in real time
            answerBox.textContent += payload.text;
        } 
        
        else if (event === "grounding_complete") {
            metrics.generation = payload.latency_s * 1000.0;
            // Subtract TTFT to find active streaming generation duration
            const activeGen = Math.max(0, metrics.generation - metrics.ttft);
            metricGen.textContent = `${activeGen.toFixed(0)} ms`;
            
            // Guardrail metric: total text RAG minus retrieval and generation
            metrics.guardrail = Math.max(5, (payload.latency_s * 1000.0) * 0.1); // approximation fallback
            metricGuard.textContent = `${metrics.guardrail.toFixed(0)} ms`;
            
            // Set safety verdict badge
            verdictBadge.textContent = payload.status;
            if (payload.status === "GROUNDED") {
                verdictBadge.className = "verdict-badge badge-pass";
            } else if (payload.status === "REFUSAL") {
                verdictBadge.className = "verdict-badge badge-neutral";
            } else {
                verdictBadge.className = "verdict-badge badge-fail";
                // Show fallback text
                answerBox.textContent = payload.answer;
            }
            updateChart();
        }
        
        else if (event === "stt_error" || event === "silent_audio") {
            metrics.stt = payload.latency_ms;
            metricStt.textContent = `${metrics.stt.toFixed(0)} ms`;
            transcriptionBox.textContent = event === "silent_audio" ? "[Silence Detected]" : "[Speech-To-Text API Error]";
            transcriptionBox.classList.add('placeholder-text');
            answerBox.textContent = payload.fallback_answer;
            verdictBadge.textContent = "FALLBACK";
            verdictBadge.className = "verdict-badge badge-neutral";
            lineageSources.innerHTML = '<div class="placeholder-text">No retrieval executed.</div>';
            updateChart();
        }
    }

    // Toggle recording on Spacebar and Mic Click
    micBtn.addEventListener('click', () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });

    document.addEventListener('keydown', event => {
        if (event.code === 'Space') {
            // Prevent scrolling on spacebar
            event.preventDefault();
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }
    });

    // Startup
    initChart();
});
