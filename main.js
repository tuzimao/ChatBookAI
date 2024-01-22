import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import isDev from 'electron-is-dev'
import { server, getPort } from './expressApp.js';
import syncing from './src/syncing.js';
import settings from 'electron-settings';


const PORT = getPort();

let mainWindow;

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  //Start Setting Page
  mainWindow.loadFile('src/settings/index.html');

  //Start Chat Book
  ipcMain.on('start-chat-book', async (event, data) => {
    const ChatBookSetting = await settings.get('chat-book');
    console.log("ChatBookSetting main.js", ChatBookSetting)
    syncing.initChatBookDb(ChatBookSetting);
    mainWindow.loadURL('http://localhost:' + PORT);
    setTimeout(intervalTaskShortTime, 5 * 1000);
    setTimeout(intervalTaskLongTime, 30 * 1000);
  });

  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
    app.quit();
  });
}

function openNewURL(url) {
  const newWindow = new BrowserWindow({ width: 800, height: 600 });
  newWindow.loadURL(url);
}

async function intervalTaskShortTime() {
  try {
    console.log('Executing intervalTaskShortTime tasks...');
    const startTime = Date.now();
    await Promise.all([
      syncing.parseFiles()
    ]);
    const executionTime = Date.now() - startTime;
    console.log(`All syncing tasks completed in ${executionTime} ms. Waiting for next interval...`);
    console.log('Resuming interval tasks.');
    const nextInterval = 10 * 1000;
    setTimeout(intervalTaskShortTime, nextInterval);
  } catch (error) {
    console.error('Error in intervalTaskShortTime:', error);
  }
}

async function intervalTaskLongTime() {
  try {
    console.log('Executing intervalTaskLongTime tasks...');
    const startTime = Date.now();
    await Promise.all([
      syncing.deleteLog()
    ]);
    const executionTime = Date.now() - startTime;
    console.log(`All syncing tasks completed in ${executionTime} ms. Waiting for next interval...`);
    console.log('Resuming interval tasks.');
    const nextInterval = 1800 * 1000;
    setTimeout(intervalTaskLongTime, nextInterval);
  } catch (error) {
    console.error('Error in intervalTaskLongTime:', error);
  }
}

app.whenReady().then(()=>{
  createMainWindow();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createMainWindow();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    server.close();
    app.quit();
  }
});

ipcMain.on('open-folder-dialog', async (event) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
  });
  if (!result.canceled && result.filePaths.length > 0) {
    event.reply('selected-folder', result.filePaths[0]);
  }
});

ipcMain.on('save-chat-book', async (event, data) => {
  await settings.set('chat-book', data);
  console.log("save-chat-book", data);
  //mainWindow.webContents.send('data-chat-book', data);
});

ipcMain.on('get-chat-book', async (event) => {
  const data = await settings.get('chat-book');
  console.log("get-chat-book", data);
  event.reply('data-chat-book', data);
});
