import axios from 'axios'
import fs from 'fs'
import multer from 'multer'
import path from 'path'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import * as crypto from 'crypto'
import { PromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
import { Calculator } from "langchain/tools/calculator";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";

import { ChatBaiduWenxin } from "@langchain/community/chat_models/baiduwenxin";

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

import syncing from '../syncing.js';

//.ENV
import dotenv from 'dotenv';
import { exit } from 'process';
dotenv.config();

/*
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_Temperature = 0.9
*/
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_ENVIRONMENT = process.env.PINECONE_ENVIRONMENT;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;
const PINECONE_NAME_SPACE = process.env.PINECONE_NAME_SPACE;

let DataDir = null;
let model = null;
let ChatBaiduWenxinModel = null
let pinecone = null
let getLLMSSettingData = null
let knowledgeId = 0
let userId = 1


  async function initChatBookBaiduWenxin(knowledgeId) {
    await initChatBookBaiduWenxinStream(knowledgeId)
  }

  async function initChatBookBaiduWenxinStream(knowledgeId) {
    getLLMSSettingData = await syncing.getLLMSSetting(knowledgeId);
    console.log("BaiduWenxin getLLMSSettingData", getLLMSSettingData, knowledgeId)
    const OPENAI_API_BASE = getLLMSSettingData.OPENAI_API_BASE;
    const BAIDU_API_KEY = getLLMSSettingData.BAIDU_API_KEY ?? "1AWXpm1Cd8lbxmAaFoPR0dNx";
    const BAIDU_SECRET_KEY = getLLMSSettingData.BAIDU_SECRET_KEY ?? "TQy5sT9Mz4xKn0tR8h7W6LxPWIUNnXqq";
    const OPENAI_Temperature = getLLMSSettingData.Temperature ?? 1;
    if(BAIDU_API_KEY && PINECONE_API_KEY && PINECONE_ENVIRONMENT) {
      process.env.BAIDU_API_KEY = BAIDU_API_KEY
      process.env.BAIDU_SECRET_KEY = BAIDU_SECRET_KEY
      try {
        ChatBaiduWenxinModel = new ChatBaiduWenxin({
          modelName: getLLMSSettingData.ModelName ?? "ERNIE-Bot-4", // Available models: ERNIE-Bot, ERNIE-Bot-turbo, ERNIE-Bot-4
          temperature: OPENAI_Temperature,
          baiduApiKey: process.env.BAIDU_API_KEY, // In Node.js defaults to process.env.BAIDU_API_KEY
          baiduSecretKey: process.env.BAIDU_SECRET_KEY, // In Node.js defaults to process.env.BAIDU_SECRET_KEY
        });
      }
      catch(error) {
        console.log("initChatBookBaiduWenxinStream ChatBaiduWenxinModel:", error)
      }
      pinecone = new Pinecone({environment: PINECONE_ENVIRONMENT, apiKey: PINECONE_API_KEY,});
    }
  }

  async function chatChatBaiduWenxin(res, knowledgeId, userId, question, history) {
    await initChatBookBaiduWenxinStream(knowledgeId)
    const input2 = [new HumanMessage(question)];
    const response = await ChatBaiduWenxinModel.call(input2);
    console.log("response", response.content)
    const insertChatLog = syncing.db.prepare('INSERT OR REPLACE INTO chatlog (knowledgeId, send, Received, userId, timestamp, source, history) VALUES (?,?,?,?,?,?,?)');
    insertChatLog.run(knowledgeId, question, response.content, userId, Date.now(), JSON.stringify([]), JSON.stringify(history));
    insertChatLog.finalize();
    res.end();
  }

  async function chatKnowledgeBaiduWenxin(res, knowledgeId, userId, question, history) {
    await initChatBookBaiduWenxinStream(knowledgeId)
    // create chain
    const CONDENSE_TEMPLATE = await syncing.GetSetting("CONDENSE_TEMPLATE", knowledgeId, userId);
    const QA_TEMPLATE       = await syncing.GetSetting("QA_TEMPLATE", knowledgeId, userId);

    log("Chat knowledgeId", knowledgeId)
    log("Chat CONDENSE_TEMPLATE", CONDENSE_TEMPLATE)
    log("Chat QA_TEMPLATE", QA_TEMPLATE)
    log("Chat PINECONE_INDEX_NAME", PINECONE_INDEX_NAME)
    
    if (!question) {
      return { message: 'No question in the request' };
    }
  
    // OpenAI recommends replacing newlines with spaces for best results
    const sanitizedQuestion = question.trim().replaceAll('\n', ' ');
  
    try {
      
      const index = pinecone.Index(PINECONE_INDEX_NAME);
  
      /* create vectorstore */

      const PINECONE_NAME_SPACE_USE = PINECONE_NAME_SPACE + '_' + String(knowledgeId)
      log("Chat PINECONE_NAME_SPACE_USE", PINECONE_NAME_SPACE_USE)

      const embeddings = new OpenAIEmbeddings({openAIApiKey:getLLMSSettingData.OPENAI_API_KEY});
      
      const vectorStore = await PineconeStore.fromExistingIndex(
        embeddings,
        {
          pineconeIndex: index,
          textKey: 'text',
          namespace: PINECONE_NAME_SPACE_USE,
        },
      );
      
      // Use a callback to get intermediate sources from the middle of the chain
      let resolveWithDocuments;
      const documentPromise = new Promise((resolve) => {
        resolveWithDocuments = resolve;
      });

      const retriever = vectorStore.asRetriever({
        callbacks: [
          {
            handleRetrieverEnd(documents) {
              resolveWithDocuments(documents);
            },
          },
        ],
      });

      const chain = makeChain(retriever, CONDENSE_TEMPLATE, QA_TEMPLATE);

      const pastMessages = history.map((message) => {
                                    return [`Human: ${message[0]}`, `Assistant: ${message[1]}`].join('\n');
                                  }).join('\n');
  
      // Ask a question using chat history
      const response = await chain.invoke({
        question: sanitizedQuestion,
        chat_history: pastMessages,
      });
  
      const sourceDocuments = await documentPromise;

      const insertChatLog = syncing.db.prepare('INSERT OR REPLACE INTO chatlog (knowledgeId, send, Received, userId, timestamp, source, history) VALUES (?,?,?,?,?,?,?)');
      insertChatLog.run(Number(knowledgeId), question, response, userId, Date.now(), JSON.stringify(sourceDocuments), JSON.stringify(history));
      insertChatLog.finalize();
      res.end();
      return { text: response, sourceDocuments };
    } 
    catch (error) {
      log('Error Chat:', error);
      return { error: error.message || 'Something went wrong' };
    }
  }

  function makeChain(retriever, CONDENSE_TEMPLATE, QA_TEMPLATE) {
    const condenseQuestionPrompt = ChatPromptTemplate.fromTemplate(CONDENSE_TEMPLATE);
    const answerPrompt = ChatPromptTemplate.fromTemplate(QA_TEMPLATE);

    // Rephrase the initial question into a dereferenced standalone question based on
    // the chat history to allow effective vectorstore querying.
    const standaloneQuestionChain = RunnableSequence.from([
      condenseQuestionPrompt,
      ChatBaiduWenxinModel,
      new StringOutputParser(),
    ]);

    // Retrieve documents based on a query, then format them.
    const retrievalChain = retriever.pipe(combineDocumentsFn);

    // Generate an answer to the standalone question based on the chat history
    // and retrieved documents. Additionally, we return the source documents directly.
    const answerChain = RunnableSequence.from([
      {
        context: RunnableSequence.from([
          (input) => input.question,
          retrievalChain,
        ]),
        chat_history: (input) => input.chat_history,
        question: (input) => input.question,
      },
      answerPrompt,
      ChatBaiduWenxinModel,
      new StringOutputParser(),
    ]);

    // First generate a standalone question, then answer it based on
    // chat history and retrieved context documents.
    const conversationalRetrievalQAChain = RunnableSequence.from([
      {
        question: standaloneQuestionChain,
        chat_history: (input) => input.chat_history,
      },
      answerChain,
    ]);

    return conversationalRetrievalQAChain;
  }

  function combineDocumentsFn(docs, separator = '\n\n') {
    const serializedDocs = docs.map((doc) => doc.pageContent);
    return serializedDocs.join(separator);
  }
  
  async function debugBaiduWenxin(res) {
    chatChatBaiduWenxin(res, "BaiduWenxin", 1, 'what is bitcoin?', [])
  }

  async function log(Action1, Action2='', Action3='', Action4='', Action5='', Action6='', Action7='', Action8='', Action9='', Action10='') {
    const currentDate = new Date();
    const currentDateTime = currentDate.toLocaleString();
    const content = JSON.stringify(Action1) +" "+ JSON.stringify(Action2) +" "+ JSON.stringify(Action3) +" "+ JSON.stringify(Action4) +" "+ JSON.stringify(Action5) +" "+ JSON.stringify(Action6) +" "+ JSON.stringify(Action7) +" "+ JSON.stringify(Action8) +" "+ JSON.stringify(Action9) +" "+ JSON.stringify(Action10);
    const insertStat = syncing.db.prepare('INSERT OR REPLACE INTO logs (datetime, content, knowledgeId, userId) VALUES (? ,? ,? ,?)');
    insertStat.run(currentDateTime, content, knowledgeId, userId);
    insertStat.finalize();
    console.log(Action1, Action2, Action3, Action4, Action5, Action6, Action7, Action8, Action9, Action10)
  }

  export default {
    debugBaiduWenxin,
    chatChatBaiduWenxin,
    chatKnowledgeBaiduWenxin
  };

