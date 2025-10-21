class ScriptRunner {
    constructor() {
        this.runBtn = document.getElementById('run-btn');
        this.outputDiv = document.getElementById('output');
        this.preprocessBtn = document.getElementById('preprocess-btn');
        this.preprocessOutputDiv = document.getElementById('preprocess-output');
    }

    init() {
        if (this.runBtn) {
            this.runBtn.classList.add('button');
            this.runBtn.addEventListener('click', async () => {
                this.outputDiv.innerText = 'Running prediction...';
                try {
                    const result = await window.electronAPI.runScript('LSTM/code.py');
                    this.outputDiv.innerText = result;
                } catch (error) {
                    this.outputDiv.innerText = `Error: ${error}`;
                }
            });
        }

        if (this.preprocessBtn) {
            this.preprocessBtn.classList.add('button');
            this.preprocessBtn.addEventListener('click', async () => {
                this.preprocessOutputDiv.innerText = 'Running preprocessing script...';
                try {
                    const result = await window.electronAPI.runScript('processed/get_preprocessed.py');
                    this.preprocessOutputDiv.innerText = result;
                } catch (error) {
                    this.preprocessOutputDiv.innerText = `Error: ${error}`;
                }
            });
        }
    }
}

export default ScriptRunner;
