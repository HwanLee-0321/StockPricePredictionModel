const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  runScript: (scriptPath) => ipcRenderer.invoke('run-script', scriptPath),
  getFiles: (dir) => ipcRenderer.invoke('get-files', dir),
  readFile: (filePath) => ipcRenderer.invoke('read-file', filePath)
});
