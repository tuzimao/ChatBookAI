  import axios from 'axios'
  import fs from 'fs'
  import multer from 'multer'
  import path from 'path'
  import { fileURLToPath } from 'url'
  import { dirname, join } from 'path'
  import * as crypto from 'crypto'
  import { OpenAI, ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
  import { PromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";
  import { LLMChain } from "langchain/chains";
  import { Calculator } from "langchain/tools/calculator";
  import { BufferMemory } from "langchain/memory";
  import { ConversationChain } from "langchain/chains";
  import { HumanMessage, AIMessage } from "@langchain/core/messages";
  import { ChatMessageHistory } from "langchain/stores/message/in_memory";

  import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
  import { PineconeStore } from '@langchain/community/vectorstores/pinecone';
  import { Pinecone } from '@pinecone-database/pinecone';
  import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
  import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
  import { TextLoader } from 'langchain/document_loaders/fs/text';
  import { DocxLoader } from 'langchain/document_loaders/fs/docx';
  import { JSONLoader } from 'langchain/document_loaders/fs/json';
  import { JSONLinesLoader } from 'langchain/document_loaders/fs/json';
  import { CSVLoader } from 'langchain/document_loaders/fs/csv';
  import { UnstructuredLoader } from 'langchain/document_loaders/fs/unstructured';

  import { Document } from '@langchain/core/documents';
  import { RunnableSequence } from '@langchain/core/runnables';
  import { StringOutputParser } from '@langchain/core/output_parsers';

  import sqlite3 from 'sqlite3';
  const sqlite3Verbose = sqlite3.verbose();

  //.ENV
  import dotenv from 'dotenv';
  import { exit } from 'process';
  dotenv.config();

  const CONDENSE_TEMPLATE_INIT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

  <chat_history>
    {chat_history}
  </chat_history>

  Follow Up Input: {question}
  Standalone question:`;

  const QA_TEMPLATE_INIT = `You are an expert researcher. Use the following pieces of context to answer the question at the end.
  If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
  If the question is not related to the context or chat history, politely respond that you are tuned to only answer questions that are related to the context.

  <context>
    {context}
  </context>

  <chat_history>
    {chat_history}
  </chat_history>

  Question: {question}
  Helpful answer in markdown:`;

  const __filename = fileURLToPath(import.meta.url);
  const __dirname = dirname(__filename);

  let DataDir = null;
  let db = null;
  let userId = 1;
  let knowledgeId = 0;

  //Only for npm run express
  await initChatBookDb({"NodeStorageDirectory": process.env.NodeStorageDirectory});
  
  async function initChatBookDb(ChatBookSetting) {
    DataDir = ChatBookSetting && ChatBookSetting.NodeStorageDirectory ? ChatBookSetting.NodeStorageDirectory : "D:\\";
    enableDir(DataDir);
    db = new sqlite3Verbose.Database(DataDir + '/ChatBook.db', { 
      encoding: 'utf8' 
    });
    db.serialize(() => {
        db.run(`
            CREATE TABLE IF NOT EXISTS setting (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT KEY not null,
                content TEXT not null,
                type TEXT not null,
                knowledgeId TEXT not null,
                userId INTEGER not null,
                UNIQUE(name, userId, knowledgeId)
            );
        `);
        db.run(`insert or ignore into setting (name, content, type, knowledgeId, userId) values('OPENAI_API_BASE','','openaisetting',1,1);`);
        db.run(`insert or ignore into setting (name, content, type, knowledgeId, userId) values('OPENAI_API_KEY','','openaisetting',1,1);`);
        db.run(`insert or ignore into setting (name, content, type, knowledgeId, userId) values('Temperature','0.1','openaisetting',1,1);`);
        db.run(`insert or ignore into setting (name, content, type, knowledgeId, userId) values('ModelName','gpt-3.5-turbo','openaisetting',1,1);`);
        db.run(`insert or ignore into setting (name, content, type, knowledgeId, userId) values('CONDENSE_TEMPLATE',?,'TEMPLATE_1',1,1);`, [CONDENSE_TEMPLATE_INIT]);
        db.run(`insert or ignore into setting (name, content, type, knowledgeId, userId) values('QA_TEMPLATE',?,'TEMPLATE_1',1,1);`, [QA_TEMPLATE_INIT]);
        db.run(`
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledgeId INTEGER not null,
                suffixName TEXT not null,
                newName TEXT UNIQUE not null,
                originalName TEXT not null,
                hash TEXT not null,
                status INTEGER not null default 0,
                summary TEXT not null default '',
                timestamp INTEGER not null default 0,
                userId INTEGER not null
            );
        `);
        db.run(`
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TEXT not null,
                content TEXT not null,
                knowledgeId INTEGER not null,
                userId INTEGER not null
            );
        `);
        db.run(`
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT not null,
                summary TEXT not null,
                timestamp INTEGER not null default 0,
                userId INTEGER not null,
                UNIQUE(name, userId)
            );
        `);
        db.run(`insert or ignore into knowledge (id, name, summary, timestamp, userId) values(1, 'Default','Default','`+Date.now()+`', 1);`);
        db.run(`
            CREATE TABLE IF NOT EXISTS chatlog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledgeId INTEGER not null default 0,
                send TEXT  not null,
                received TEXT not null,
                userId INTEGER not null default 0,
                timestamp INTEGER not null default 0,
                source TEXT not null,
                history TEXT not null
            );
        `);
    });
    enableDir(DataDir + '/uploadfiles/');
    enableDir(DataDir + '/parsedfiles/');
  }

  async function getLLMSSetting(knowledgeId) {
    const knowledgeIdFilter = filterString(knowledgeId)
    const userIdFilter = Number(userId)
    const SettingRS = await new Promise((resolve, reject) => {
            db.all("SELECT name,content from setting where type='openaisetting' and knowledgeId='"+knowledgeIdFilter+"' and userId='"+userIdFilter+"'", (err, result) => {
              if (err) {
                reject(err);
              } else {
                resolve(result ? result : null);
              }
            });
          });
    const OpenAISetting = {}
    if(SettingRS)  {
      SettingRS.map((Item)=>{
        OpenAISetting[Item.name] = Item.content
      })
    }
    return OpenAISetting
  }

  async function setOpenAISetting(Params) {
    const knowledgeIdFilter = filterString(Params.knowledgeId)
    const userIdFilter = Number(userId)
    try {
      const insertSetting = db.prepare('INSERT OR REPLACE INTO setting (name, content, type, knowledgeId, userId) VALUES (?, ?, ?, ?, ?)');
      insertSetting.run('OPENAI_API_BASE', Params.OPENAI_API_BASE, 'openaisetting', knowledgeIdFilter, userIdFilter);
      insertSetting.run('OPENAI_API_KEY', Params.OPENAI_API_KEY, 'openaisetting', knowledgeIdFilter, userIdFilter);
      insertSetting.run('Temperature', Params.Temperature, 'openaisetting', knowledgeIdFilter, userIdFilter);
      insertSetting.run('ModelName', Params.ModelName, 'openaisetting', knowledgeIdFilter, userIdFilter);
      insertSetting.finalize();
    }
    catch (error) {
      log('Error setOpenAISetting:', error.message);
    }
    return {"status":"ok", "msg":"Updated Success"}
  }

  async function getTemplate(knowledgeId) {
    const knowledgeIdFilter = filterString(knowledgeId)
    const userIdFilter = Number(userId)
    const SettingRS = await new Promise((resolve, reject) => {
            const Templatename = "TEMPLATE"
            db.all("SELECT name,content from setting where type='"+Templatename+"' and knowledgeId='"+knowledgeIdFilter+"' and userId='"+userIdFilter+"'", (err, result) => {
              if (err) {
                reject(err);
              } else {
                resolve(result ? result : null);
              }
            });
          });
    const Template = {}
    if(SettingRS)  {
      SettingRS.map((Item)=>{
        Template[Item.name.replace("_" + String(knowledgeIdFilter),"")] = Item.content
      })
    }
    return Template
  }

  async function setTemplate(Params) {
    try{
      const knowledgeIdFilter = Number(Params.knowledgeId)
      const userIdFilter = Number(userId)
      const Templatename = "TEMPLATE"
      const insertSetting = db.prepare('INSERT OR REPLACE INTO setting (name, content, type, knowledgeId, userId) VALUES (?, ?, ?, ?, ?)');
      insertSetting.run('CONDENSE_TEMPLATE', Params.CONDENSE_TEMPLATE, Templatename, knowledgeIdFilter, userIdFilter);
      insertSetting.run('QA_TEMPLATE', Params.QA_TEMPLATE, Templatename, knowledgeIdFilter, userIdFilter);
      insertSetting.finalize();
    }
    catch (error) {
      log('Error setOpenAISetting:', error.message);
    }
    return {"status":"ok", "msg":"Updated Success"}
  }

  async function addKnowledge(Params) {
    try{
      const userIdFilter = Number(userId)
      Params.name = filterString(Params.name)
      Params.summary = filterString(Params.summary)
      const RecordId = await new Promise((resolve, reject) => {
        db.get("SELECT id from knowledge where name = '"+filterString(Params.name)+"' and userId = '"+userIdFilter+"'", (err, result) => {
          if (err) {
            reject(err);
          } else {
            resolve(result ? result.id : null);
          }
        });
      });
      console.log("RecordId", RecordId, userId)
      if(RecordId && RecordId > 0) {
        Params.id = RecordId
        setKnowledge(Params)
      }
      else {
        const insertSetting = db.prepare('INSERT OR REPLACE INTO knowledge (name, summary, timestamp, userId) VALUES (?, ?, ?, ?)');
        insertSetting.run(Params.name, Params.summary, Date.now(), userIdFilter);
        insertSetting.finalize();
      }
    }
    catch (error) {
      log('Error setOpenAISetting:', error.message);
    }
    return {"status":"ok", "msg":"Updated Success"}
  }

  async function setKnowledge(Params) {
    try{
      Params.id = Number(Params.id)
      Params.name = filterString(Params.name)
      Params.summary = filterString(Params.summary)
      const updateSetting = db.prepare('update knowledge set name = ?, summary = ?, timestamp = ? where id = ?');
      updateSetting.run(Params.name, Params.summary, Date.now(), Params.id);
      updateSetting.finalize();
    }
    catch (error) {
      log('Error setOpenAISetting:', error.message);
    }
    return {"status":"ok", "msg":"Updated Success"}
  }

  function uploadfiles() {
    const storage = multer.diskStorage({
      destination: (req, file, cb) => {
        cb(null, DataDir + '/uploadfiles/'); // 设置上传文件保存的目录
      },
      filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
        const FileNameNew = file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname).toLowerCase();
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname).toLowerCase());
        //const absolutePath = path.resolve('uploadfiles', FileNameNew);
        //const calculateFileHash = calculateFileHashSync(absolutePath);
        //log("calculateFileHash", absolutePath)
      },
    });
    const upload = multer({ storage: storage });
    return upload
  }

  async function uploadfilesInsertIntoDb(files, knowledgeId) {
    //const originalName = Buffer.from(files[0].originalname, 'hex').toString('utf8');
    //log("originalName", files[0].originalname)
    const filesInfo = files.map(file => {
      const filePath = path.join(DataDir, 'uploadfiles', file.filename);
      const fileHash = calculateFileHashSync(filePath);
      return {
        originalName: file.originalname,
        newName: file.filename,
        hash: fileHash,
      };
    });
    const insertFiles = db.prepare('INSERT OR IGNORE INTO files (knowledgeId, suffixName, newName, originalName, hash, timestamp, userId) VALUES (?,?,?,?,?,?,?)');
    filesInfo.map((Item)=>{
      const suffixName = path.extname(Item.originalName).toLowerCase();
      insertFiles.run(knowledgeId, suffixName, Item.newName, Item.originalName, Item.hash, Date.now(), Number(userId));
      // Move Files To KnowledgeId Dir
      enableDir(DataDir + '/uploadfiles/' + String(userId) )
      enableDir(DataDir + '/uploadfiles/' + String(userId) + '/' + String(knowledgeId))
      fs.rename(DataDir + '/uploadfiles/' + Item.newName, DataDir + '/uploadfiles/'  + String(userId) + '/' + String(knowledgeId) + '/' + Item.newName, (err) => {
        if (err) {
          log('Error moving file:', err, Item);
        } else {
          log('File moved successfully.', Item);
        }
      });
    })
    insertFiles.finalize();
  }

  async function getFilesPage(pageid, pagesize) {
    const pageidFiler = Number(pageid) < 0 ? 0 : Number(pageid);
    const pagesizeFiler = Number(pagesize) < 5 ? 5 : Number(pagesize);
    const From = pageidFiler * pagesizeFiler;
    const RecordsTotal = await new Promise((resolve, reject) => {
      db.get("SELECT COUNT(*) AS NUM from files", (err, result) => {
        if (err) {
          reject(err);
        } else {
          resolve(result ? result.NUM : null);
        }
      });
    });
    const RecordsAll = await new Promise((resolve, reject) => {
                            db.all("SELECT * from files where 1=1 order by status desc, timestamp desc limit "+ Number(pagesize) +" offset "+ From +"", (err, result) => {
                              if (err) {
                                reject(err);
                              } else {
                                resolve(result ? result : null);
                              }
                            });
                          });
    let RSDATA = []
    if(RecordsAll != undefined) {
      RSDATA = await Promise.all(
        RecordsAll.map(async (Item)=>{
            let ItemStatus = "Ready To Parse"
            switch(Item.status) {
              case 1:
                ItemStatus = 'Finished'
                break;
              case -1:
                ItemStatus = 'File Not Exist'
                break;
            }
            return {...Item, status:ItemStatus, timestamp: formatDateFromTimestamp(Item.timestamp)}
          })
      );
      log("getFilesPage", RSDATA)
    }
    const RS = {};
    RS['allpages'] = Math.ceil(RecordsTotal/pagesizeFiler);
    RS['data'] = RSDATA.filter(element => element !== null && element !== undefined && element !== '');
    RS['from'] = From;
    RS['pageid'] = pageidFiler;
    RS['pagesize'] = pagesizeFiler;
    RS['total'] = RecordsTotal;
    return RS;
  }

  async function getFilesKnowledgeId(KnowledgeId, pageid, pagesize) {
    const KnowledgeIdFiler = Number(KnowledgeId) < 0 ? 0 : Number(KnowledgeId);
    const pageidFiler = Number(pageid) < 0 ? 0 : Number(pageid);
    const pagesizeFiler = Number(pagesize) < 5 ? 5 : Number(pagesize);
    const From = pageidFiler * pagesizeFiler;
    const RecordsTotal = await new Promise((resolve, reject) => {
      db.get("SELECT COUNT(*) AS NUM from files where knowledgeId = '"+KnowledgeIdFiler+"'", (err, result) => {
        if (err) {
          reject(err);
        } else {
          resolve(result ? result.NUM : null);
        }
      });
    });
    const RecordsAll = await new Promise((resolve, reject) => {
                            db.all("SELECT * from files where knowledgeId = '"+KnowledgeIdFiler+"' order by status desc, timestamp desc limit "+ Number(pagesize) +" offset "+ From +"", (err, result) => {
                              if (err) {
                                reject(err);
                              } else {
                                resolve(result ? result : null);
                              }
                            });
                          });
    let RSDATA = []
    if(RecordsAll != undefined) {
      RSDATA = await Promise.all(
        RecordsAll.map(async (Item)=>{
            let ItemStatus = "Ready To Parse"
            switch(Item.status) {
              case 1:
                ItemStatus = 'Finished'
                break;
              case -1:
                ItemStatus = 'File Not Exist'
                break;
            }
            return {...Item, status:ItemStatus, timestamp: formatDateFromTimestamp(Item.timestamp)}
          })
      );
      //log("getFilesKnowledgeId", RSDATA)
    }
    const RS = {};
    RS['allpages'] = Math.ceil(RecordsTotal/pagesizeFiler);
    RS['data'] = RSDATA.filter(element => element !== null && element !== undefined && element !== '');
    RS['from'] = From;
    RS['pageid'] = pageidFiler;
    RS['pagesize'] = pagesizeFiler;
    RS['total'] = RecordsTotal;
    return RS;
  }

  async function getChatLogByKnowledgeIdAndUserId(KnowledgeId, userId, pageid, pagesize) {
    const KnowledgeIdFiler = Number(KnowledgeId) < 0 ? 0 : Number(KnowledgeId);
    const userIdFiler = Number(userId) < 0 ? 0 : Number(userId);
    const pageidFiler = Number(pageid) < 0 ? 0 : Number(pageid);
    const pagesizeFiler = Number(pagesize) < 5 ? 5 : Number(pagesize);
    const From = pageidFiler * pagesizeFiler;
    const RecordsTotal = await new Promise((resolve, reject) => {
      db.get("SELECT COUNT(*) AS NUM from chatlog where knowledgeId = '"+KnowledgeIdFiler+"' and userId = '"+userIdFiler+"'", (err, result) => {
        if (err) {
          reject(err);
        } else {
          resolve(result ? result.NUM : null);
        }
      });
    });
    const RecordsAll = await new Promise((resolve, reject) => {
                            db.all("SELECT * from chatlog where knowledgeId = '"+KnowledgeIdFiler+"' and userId = '"+userIdFiler+"' order by timestamp desc limit "+ Number(pagesize) +" offset "+ From +"", (err, result) => {
                              if (err) {
                                reject(err);
                              } else {
                                resolve(result ? result : null);
                              }
                            });
                          });
    let RSDATA = []
    if(RecordsAll != undefined) {
      RSDATA = await Promise.all(
        RecordsAll.map(async (Item)=>{
            return Item
          })
      );
      log("getChatLogByKnowledgeIdAndUserId", RSDATA)
    }
    const RS = {};
    RS['allpages'] = Math.ceil(RecordsTotal/pagesizeFiler);
    RS['data'] = RSDATA.filter(element => element !== null && element !== undefined && element !== '');
    RS['from'] = From;
    RS['pageid'] = pageidFiler;
    RS['pagesize'] = pagesizeFiler;
    RS['total'] = RecordsTotal;
    return RS;
  }
  
  async function getLogsPage(pageid, pagesize) {
    const pageidFiler = Number(pageid) < 0 ? 0 : Number(pageid);
    const pagesizeFiler = Number(pagesize) < 5 ? 5 : Number(pagesize);
    const From = pageidFiler * pagesizeFiler;
    const RecordsTotal = await new Promise((resolve, reject) => {
      db.get("SELECT COUNT(*) AS NUM from logs", (err, result) => {
        if (err) {
          reject(err);
        } else {
          resolve(result ? result.NUM : null);
        }
      });
    });
    const RecordsAll = await new Promise((resolve, reject) => {
                            db.all("SELECT * from logs where 1=1 order by id desc limit "+ Number(pagesize) +" offset "+ From +"", (err, result) => {
                              if (err) {
                                reject(err);
                              } else {
                                resolve(result ? result : null);
                              }
                            });
                          });
    let RSDATA = []
    if(RecordsAll != undefined) {
      RSDATA = await Promise.all(
        RecordsAll.map(async (Item)=>{
            return Item
          })
      );
      //log("getLogsPage", RSDATA)
    }
    const RS = {};
    RS['allpages'] = Math.ceil(RecordsTotal/pagesizeFiler);
    RS['data'] = RSDATA.filter(element => element !== null && element !== undefined && element !== '');
    RS['from'] = From;
    RS['pageid'] = pageidFiler;
    RS['pagesize'] = pagesizeFiler;
    RS['total'] = RecordsTotal;
    return RS;
  }

  async function getKnowledgePage(pageid, pagesize) {
    const userIdFiler = Number(userId) < 0 ? 0 : Number(userId);
    const pageidFiler = Number(pageid) < 0 ? 0 : Number(pageid);
    const pagesizeFiler = Number(pagesize) < 5 ? 5 : Number(pagesize);
    const From = pageidFiler * pagesizeFiler;
    const RecordsTotal = await new Promise((resolve, reject) => {
      db.get("SELECT COUNT(*) AS NUM from knowledge where userId='"+userIdFiler+"'", (err, result) => {
        if (err) {
          reject(err);
        } else {
          resolve(result ? result.NUM : null);
        }
      });
    });
    const RecordsAll = await new Promise((resolve, reject) => {
                            db.all("SELECT * from knowledge where userId='"+userIdFiler+"' order by id desc limit "+ Number(pagesize) +" offset "+ From +"", (err, result) => {
                              if (err) {
                                reject(err);
                              } else {
                                resolve(result ? result : null);
                              }
                            });
                          });
    let RSDATA = []
    if(RecordsAll != undefined) {
      RSDATA = await Promise.all(
        RecordsAll.map(async (Item)=>{
            return Item
          })
      );
      //log("getKnowledgePage", RSDATA)
    }
    const RS = {};
    RS['allpages'] = Math.ceil(RecordsTotal/pagesizeFiler);
    RS['data'] = RSDATA.filter(element => element !== null && element !== undefined && element !== '');
    RS['from'] = From;
    RS['pageid'] = pageidFiler;
    RS['pagesize'] = pagesizeFiler;
    RS['total'] = RecordsTotal;
    return RS;
  }

  function enableDir(directoryPath) {
    try {
        fs.accessSync(directoryPath, fs.constants.F_OK);
    } 
    catch (err) {
        try {
            fs.mkdirSync(directoryPath, { recursive: true });
        } catch (err) {
            log(`Error creating directory ${directoryPath}: ${err.message}`);
            throw err;
        }
    }
  }

  function timestampToDate(timestamp) {
    const date = new Date(Number(timestamp) * 1000);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
  
    return `${year}-${month}-${day}`;
  }
  
  async function log(Action1, Action2='', Action3='', Action4='', Action5='', Action6='', Action7='', Action8='', Action9='', Action10='') {
    const currentDate = new Date();
    const currentDateTime = currentDate.toLocaleString();
    const content = JSON.stringify(Action1) +" "+ JSON.stringify(Action2) +" "+ JSON.stringify(Action3) +" "+ JSON.stringify(Action4) +" "+ JSON.stringify(Action5) +" "+ JSON.stringify(Action6) +" "+ JSON.stringify(Action7) +" "+ JSON.stringify(Action8) +" "+ JSON.stringify(Action9) +" "+ JSON.stringify(Action10);
    const insertStat = db.prepare('INSERT OR REPLACE INTO logs (datetime, content, knowledgeId, userId) VALUES (? ,? ,? ,?)');
    insertStat.run(currentDateTime, content, knowledgeId, userId);
    insertStat.finalize();
    console.log(Action1, Action2, Action3, Action4, Action5, Action6, Action7, Action8, Action9, Action10)
  }

  async function deleteLog() {
    const MaxId = await new Promise((resolve, reject) => {
      db.get("SELECT MAX(id) AS NUM FROM logs", (err, result) => {
        if (err) {
          reject(err);
        } else {
          resolve(result ? result.NUM : null);
        }
      });
    });
    const DeleteId = MaxId - 1000;
    const DeleteLog = db.prepare("delete from logs where id < ?");
    DeleteLog.run(DeleteId);
    DeleteLog.finalize();
  }

  async function GetSetting(Name, KnowledgeId, userId) {
    return await new Promise((resolve, reject) => {
      db.get("SELECT content FROM setting where name='"+Name+"' and KnowledgeId='"+KnowledgeId+"' and userId='"+userId+"'", (err, result) => {
        if (err) {
          reject(err);
        } else {
          resolve(result ? result.content : null);
        }
      });
    });
  }

  function calculateFileHashSync(filePath) {
    try {
      const fileContent = fs.readFileSync(filePath);
      const hash = crypto.createHash('sha256');
      hash.update(fileContent);
      return hash.digest('hex');
    } 
    catch (error) {
      throw error;
    }
  }

  function readFile(Dir, FileName, Mark, OpenFormat) {
    const filePath = DataDir + '/' + Dir + '/' + FileName;
    if(isFile(filePath)) {
      log("filePath", filePath)
      const data = fs.readFileSync(filePath, OpenFormat);
      return data;
    }
    else {
      log("[" + Mark + "] Error read file:", filePath);
      return null;
    }
  }

  function writeFile(Dir, FileName, FileContent, Mark) {
    const directoryPath = DataDir + '/' + Dir;
    enableDir(directoryPath)
    const TxFilePath = directoryPath + "/" + FileName
    try {
      fs.writeFileSync(TxFilePath, FileContent);
      return true;
    } 
    catch (err) {
      log("[" + Mark + "] Error writing to file:", err);
      return false;
    }
  }

  function mkdirForData() {
    fs.mkdir(DataDir + '/pdf', { recursive: true }, (err) => {});
    fs.mkdir(DataDir + '/doc', { recursive: true }, (err) => {});
    fs.mkdir(DataDir + '/xls', { recursive: true }, (err) => {});
    fs.mkdir(DataDir + '/ppt', { recursive: true }, (err) => {});
    fs.mkdir(DataDir + '/txt', { recursive: true }, (err) => {});
    fs.mkdir(DataDir + '/html', { recursive: true }, (err) => {});
    fs.mkdir(DataDir + '/url', { recursive: true }, (err) => {});
    fs.mkdir(DataDir + '/audio', { recursive: true }, (err) => {});
    fs.mkdir(DataDir + '/video', { recursive: true }, (err) => {});
  }

  function filterString(input) {
    log("filterString input:", input)
    if (typeof value === 'number') {

      return input;
    } 
    else if (typeof value === 'string') {

      return input;
    } else {

      return input;
    }
  }

  function copyFileSync(source, destination) {
    try {
      const content = fs.readFileSync(source);
      fs.writeFileSync(destination, content);
      log('File copied successfully!');
      return true;
    } catch (error) {
      log('Error copying file:', error);
      return false;
    }
  }

  function isFile(filePath) {
    try {
      const stats = fs.statSync(filePath);
      if (stats.isFile() && stats.size > 0) {
        return true;
      } else {
        return false;
      }
    } catch (err) {
      return false;
    }
  }

  function formatDateFromTimestamp(timestamp) {
    const date = new Date(timestamp);
  
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
  
    const formattedDate = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
  
    return formattedDate;
  }

  const restrictToLocalhost = (req, res, next) => {
    // Check if the request is coming from localhost (127.0.0.1 or ::1)
    const ipAddress = req.ip || req.connection.remoteAddress;
    if (ipAddress === '127.0.0.1' || ipAddress === '::1') {
      // Allow the request to proceed
      next();
    } else {
      // Respond with a 403 Forbidden status for requests from other IP addresses
      res.status(403).send('Forbidden: Access allowed only from localhost.');
    }
  };

  export default {
    db,
    initChatBookDb,
    deleteLog,
    isFile,
    readFile,
    writeFile,
    filterString,
    mkdirForData,
    copyFileSync,
    enableDir,
    timestampToDate,
    restrictToLocalhost,
    getLLMSSetting,
    setOpenAISetting,
    getTemplate,
    setTemplate,
    addKnowledge,
    setKnowledge,
    uploadfiles,
    calculateFileHashSync,
    uploadfilesInsertIntoDb,
    getFilesPage,
    getFilesKnowledgeId,
    getChatLogByKnowledgeIdAndUserId,
    getLogsPage,
    getKnowledgePage,
    GetSetting
  };