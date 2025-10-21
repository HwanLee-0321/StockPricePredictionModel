
const { app, BrowserWindow } = require('electron');

const path = require('path');

const IpcHandler = require('./src/main/IpcHandler');



class MainApplication {

    constructor() {

        this.mainWindow = null;

        this.app = app;

    }



    createWindow() {

        this.mainWindow = new BrowserWindow({

            width: 800,

            height: 600,

            webPreferences: {

                preload: path.join(__dirname, 'src/main/preload.js'),

                contextIsolation: true,

                nodeIntegration: false

            }

        });



        this.mainWindow.loadFile('index.html');

    }



    start() {

        this.app.on('ready', () => {

            this.createWindow();

            const ipcHandler = new IpcHandler(this.app);

            ipcHandler.register();

        });



        this.app.on('window-all-closed', () => {

            if (process.platform !== 'darwin') {

                this.app.quit();

            }

        });



        this.app.on('activate', () => {

            if (BrowserWindow.getAllWindows().length === 0) {

                this.createWindow();

            }

        });

    }

}



const mainApp = new MainApplication();

mainApp.start();
