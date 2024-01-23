// expressApp.js
import express from 'express';
import syncing from './src/syncing.js';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import cors from 'cors';
import bodyParser from 'body-parser';
import agentRoutes from './src/router/agent.js';
import cron from 'node-cron';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const PORT = process.env.PORT || 1988;

const expressApp = express();

expressApp.use(cors());
expressApp.use(bodyParser.json());

cron.schedule('*/3 * * * *', () => {
  console.log('Task Begin !');
  syncing.parseFiles();
  console.log('Task End !');
});

expressApp.get('/debug', async (req, res) => {
  await syncing.debug(res);
  res.end(); 
});

expressApp.get('/chat', async (req, res) => {
  //const { question, history } = req.params;
  const KnowledgeId = 36
  const userId = 1
  const question = "what is Bitcoin?"
  const history = []
  const ChatMsg = await syncing.chat(KnowledgeId, userId, question, history);
  res.json(ChatMsg).end(); 
});

expressApp.post('/chat/chat', async (req, res) => {
  const { question, history } = req.body;
  const userId = 1;
  console.log("question", question)
  await syncing.chatChat(res, 0, Number(userId), question, history);
  res.end(); 
});

expressApp.post('/chat/knowledge', async (req, res) => {
  const { KnowledgeId, question, history } = req.body;
  const userId = 1;
  console.log("question", question)
  const ChatMsg = await syncing.chatKnowledge(res, Number(KnowledgeId), Number(userId), question, history);
  console.log("ChatMsg", ChatMsg)
  //res.json(ChatMsg).end(); 
});

expressApp.post('/setopenai', async (req, res) => {
  console.log("req.body", req.body)
  const OpenAISetting = await syncing.setOpenAISetting(req.body);
  res.json(OpenAISetting).end(); 
});

expressApp.get('/getopenai/:knowledgeId', async (req, res) => {
  const { knowledgeId} = req.params;
  const OpenAISetting = await syncing.getOpenAISetting(knowledgeId);
  res.json(OpenAISetting).end(); 
});

expressApp.post('/addtemplate', async (req, res) => {
  console.log("req.body", req.body)
  const Template = await syncing.addTemplate(req.body);
  res.json(Template).end(); 
});

expressApp.post('/settemplate', async (req, res) => {
  console.log("req.body", req.body)
  const Template = await syncing.setTemplate(req.body);
  res.json(Template).end(); 
});

expressApp.get('/gettemplate/:knowledgeId', async (req, res) => {
  const { knowledgeId} = req.params;
  const Template = await syncing.getTemplate(knowledgeId);
  res.json(Template).end(); 
});

expressApp.post('/addknowledge', async (req, res) => {
  console.log("req.body", req.body)
  const Template = await syncing.addKnowledge(req.body);
  res.json(Template).end(); 
});

expressApp.post('/setknowledge', async (req, res) => {
  console.log("req.body", req.body)
  const Template = await syncing.setKnowledge(req.body);
  res.json(Template).end(); 
});

expressApp.post('/uploadfiles', syncing.uploadfiles().array('files', 10), async (req, res) => {
  syncing.uploadfilesInsertIntoDb(req.files, req.body.knowledgeId);
  res.json({"status":"ok", "msg":"Uploaded Success"}).end(); 
});

expressApp.get('/parseFolderFiles', async (req, res) => {
  await syncing.parseFolderFiles("D:/GitHub/tu/gpt4-pdf-chatbot-langchain/docs");
  res.json({}).end(); 
});

expressApp.get('/parseFiles', async (req, res) => {
  await syncing.parseFiles();
  res.json({}).end(); 
});

expressApp.get('/files/:pageid/:pagesize', async (req, res) => {
  const { pageid, pagesize } = req.params;
  const getFilesPage = await syncing.getFilesPage(pageid, pagesize);
  //console.log("getFilesPage", getFilesPage)
  res.status(200).json(getFilesPage).end(); 
});

expressApp.get('/files/:knowledgeId/:pageid/:pagesize', async (req, res) => {
  const { knowledgeId, pageid, pagesize } = req.params;
  const getFilesPage = await syncing.getFilesKnowledgeId(knowledgeId, pageid, pagesize);
  //console.log("getFilesPage", getFilesPage)
  res.status(200).json(getFilesPage).end(); 
});

expressApp.get('/chatlog/:knowledgeId/:userId/:pageid/:pagesize', async (req, res) => {
  const { knowledgeId, userId, pageid, pagesize } = req.params;
  const getChatLogByKnowledgeIdAndUserId = await syncing.getChatLogByKnowledgeIdAndUserId(knowledgeId, userId, pageid, pagesize);
  //console.log("getChatLogByKnowledgeIdAndUserId", getChatLogByKnowledgeIdAndUserId)
  res.status(200).json(getChatLogByKnowledgeIdAndUserId).end(); 
});

expressApp.get('/logs/:pageid/:pagesize', async (req, res) => {
  const { pageid, pagesize } = req.params;
  const getLogsPage = await syncing.getLogsPage(pageid, pagesize);
  //console.log("getLogsPage", getLogsPage)
  res.status(200).json(getLogsPage).end(); 
});

expressApp.get('/knowledge/:pageid/:pagesize', async (req, res) => {
  const { pageid, pagesize } = req.params;
  const getKnowledgePage = await syncing.getKnowledgePage(pageid, pagesize);
  //console.log("getKnowledgePage", getKnowledgePage)
  res.status(200).json(getKnowledgePage).end(); 
});




expressApp.use('/', agentRoutes);

// Middleware to conditionally serve static files based on the request's IP
const serveStaticLocally = (req, res, next) => {
  // Check if the request is coming from localhost (127.0.0.1 or ::1)
  const ipAddress = req.ip || req.connection.remoteAddress;
  if (ipAddress === '127.0.0.1' || ipAddress === '::1') {
    // Serve static files only for requests from localhost
    express.static(join(__dirname, 'html'))(req, res, next);
  } else {
    // Proceed to the next middleware for requests from other IP addresses
    next();
  }
};
expressApp.use(serveStaticLocally);

expressApp.get('/', syncing.restrictToLocalhost, (req, res) => {
  res.sendFile(join(__dirname, 'html', 'index.html'));
});

expressApp.get('*', syncing.restrictToLocalhost, (req, res) => {
  res.sendFile(join(__dirname, 'html', 'index.html'));
});

const startServer = (port) => {
  return expressApp.listen(port, () => {
    console.log(`Express server is running on port ${port}`);
  });
};

const getPort = () => {
  return PORT;
};

const server = startServer(PORT);

export { expressApp, server, getPort };
