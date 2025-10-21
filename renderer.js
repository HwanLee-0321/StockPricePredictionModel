import UIManager from './src/renderer/ui/UIManager.js';
import FileViewer from './src/renderer/ui/FileViewer.js';
import ScriptRunner from './src/renderer/core/ScriptRunner.js';

document.addEventListener('DOMContentLoaded', () => {
    new UIManager();
    const fileViewer = new FileViewer();
    fileViewer.init();
    const scriptRunner = new ScriptRunner();
    scriptRunner.init();
});
