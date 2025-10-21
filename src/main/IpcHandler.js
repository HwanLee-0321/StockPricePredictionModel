const { ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

class IpcHandler {
    constructor(app) {
        this.app = app;
    }

    register() {
        ipcMain.handle('run-script', async (event, scriptPath) => {
            return new Promise((resolve, reject) => {
                const python = spawn('python', [scriptPath]);
                let output = '';
                let error = '';

                python.stdout.on('data', (data) => {
                    output += data.toString();
                });

                python.stderr.on('data', (data) => {
                    error += data.toString();
                });

                python.on('close', (code) => {
                    if (code !== 0) {
                        reject(error);
                    } else {
                        resolve(output);
                    }
                });
            });
        });

        ipcMain.handle('get-files', async (event, dir) => {
            return new Promise((resolve, reject) => {
                const directoryPath = path.join(this.app.getAppPath(), dir);
                fs.readdir(directoryPath, (err, files) => {
                    if (err) {
                        reject(err);
                    } else {
                        resolve(files);
                    }
                });
            });
        });

        ipcMain.handle('read-file', async (event, filePath) => {
            return new Promise((resolve, reject) => {
                const fullPath = path.join(this.app.getAppPath(), filePath);
                fs.readFile(fullPath, 'utf-8', (err, data) => {
                    if (err) {
                        reject(err);
                    } else {
                        resolve(data);
                    }
                });
            });
        });
    }
}

module.exports = IpcHandler;
