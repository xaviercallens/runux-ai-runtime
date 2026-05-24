// Simulates a live data feed from the AI research agent
document.addEventListener('DOMContentLoaded', () => {
    const tpuFlopsEl = document.getElementById('tpu-flops');
    const k1TpsEl = document.getElementById('k1-tps');
    const agentScoreEl = document.getElementById('agent-score');
    const terminal = document.getElementById('terminal-logs');

    let currentTflops = 150.0;
    let currentTps = 45.2;
    let currentScore = 602.0;

    const logMessages = [
        "LLM proposed tuning RVV vector register grouping...",
        "Compiling runux-report benchmark...",
        "Simulating TPU v5e mesh network...",
        "Validating power profile limits...",
        "{SUCCESS} New metric achieved: Score +1.2%",
        "Committing changes to branch...",
        "LLM analyzing cache hit rates on K3...",
        "{WARNING} Gatekeeper: Memory alignment constraint failed.",
        "Reverting to last known good configuration.",
        "Proposing novel Sparse Attention pattern..."
    ];

    function addLog(msg) {
        const line = document.createElement('div');
        line.className = 'log-line';
        if (msg.includes('{SUCCESS}')) {
            line.classList.add('success');
            msg = msg.replace('{SUCCESS}', '');
        } else if (msg.includes('{WARNING}')) {
            line.classList.add('warning');
            msg = msg.replace('{WARNING}', '');
        }
        line.textContent = msg;
        terminal.appendChild(line);
        terminal.scrollTop = terminal.scrollHeight;

        // Keep terminal from growing indefinitely
        if (terminal.children.length > 20) {
            terminal.removeChild(terminal.firstChild);
        }
    }

    function updateMetrics() {
        // Simulate incremental optimizations discovered by the agent
        if (Math.random() > 0.6) {
            currentTflops += Math.random() * 0.5;
            currentTps += Math.random() * 0.2;
            currentScore = (currentTps * 10.0) + currentTflops;
            
            tpuFlopsEl.innerHTML = `${currentTflops.toFixed(1)} <span class="unit">TFLOPS</span>`;
            k1TpsEl.innerHTML = `${currentTps.toFixed(1)} <span class="unit">Tok/s</span>`;
            agentScoreEl.textContent = currentScore.toFixed(1);
        }

        const randomLog = logMessages[Math.floor(Math.random() * logMessages.length)];
        addLog(randomLog);
    }

    // Simulate agent loop ticks
    setInterval(updateMetrics, 3000);
});
