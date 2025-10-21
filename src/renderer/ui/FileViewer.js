class FileViewer {
    constructor() {
        this.financialReportsContainer = document.getElementById('financial-reports-list');
        this.financialIndexesContainer = document.getElementById('financial-indexes-list');
        this.financialReportsContent = document.getElementById('financial-reports-content');
        this.financialIndexesContent = document.getElementById('financial-indexes-content');
    }

    async populateFiles(dir, element, contentElement, extensions) {
        try {
            const files = await window.electronAPI.getFiles(dir);
            const filteredFiles = files.filter(file => extensions.some(ext => file.endsWith(ext)));

            for (const file of filteredFiles) {
                const button = document.createElement('button');
                button.className = 'file-list__button';
                button.innerText = file;
                button.addEventListener('click', async () => {
                    try {
                        const content = await window.electronAPI.readFile(`${dir}/${file}`);
                        contentElement.innerText = content;
                    } catch (error) {
                        console.error(`Error reading file ${dir}/${file}:`, error);
                        contentElement.innerText = 'Error reading file.';
                    }
                });
                element.appendChild(button);
            }
        } catch (error) {
            console.error(`Error populating files from ${dir}:`, error);
        }
    }

    init() {
        this.populateFiles('Financial_Research', this.financialReportsContainer, this.financialReportsContent, ['.csv', '.txt']);
        this.populateFiles('Report', this.financialReportsContainer, this.financialReportsContent, ['.csv', '.txt']);
        this.populateFiles('Financial_Index', this.financialIndexesContainer, this.financialIndexesContent, ['.csv', '.txt']);
    }
}

export default FileViewer;
