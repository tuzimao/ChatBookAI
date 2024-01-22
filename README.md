# ChatBook

主要面向企业/政府/学校/其它组织,提供组织内部知识库管理和服务,使用企业私有数据进行投喂进行训练数据,然后提供给用户使用.
使用场景:
    1 行业性垂直网站: 如房产领域,可以提供房产法规,交易,政策性咨询等服务.
    2 客户服务: 智能化的客户服务,可以帮企业节约客户服务成本.
    3 数据分析: 企业内部数据分析

## 安装包安装
    直接使用Windows exe安装包进行安装,仅支持64位系统.
    1 下载地址: https://github.com/chatbookai/ChatBook/releases
    2 安装以后,需要设置上传文件的存储路径
    3 点击开始就可以看到软件界面
    4 设置OPENAI的API KEY,和PINECONE的API KEY
        OPENAI_API_KEY: 在openai官网获取
        ModelName: 模型名称,如: gpt-3.5-turbo
        Temperature: 如果不知道具体作用,填写0即可
        PINECONE_API_KEY: 向量数据库, 在https://www.pinecone.io/上面注册一个用户,申请即可,免费.
        PINECONE_ENVIRONMENT: 在pinecone.io上面获取
        PINECONE_INDEX_NAME: 在pinecone.io上面获取
        PINECONE_NAME_SPACE: 任何字母,非空即可,主要用于区分不同的知识库
    5 设置知识库
    6 上传文件,等待2分钟左右,就会完成文件解析
    7 在Chat模块进行对话,就可以看到你的个人知识库
    8 你可以根据自己需要设置自己的模板,来适应不同的需求
    9 使用的是1988端口，如果需要在其它电脑访问，请使用http://IP:1988的形式访问
    10 可以在日志模块查询到系统运行的日志信息

## 编译安装
```
npm install
```

## 启动桌面客户端程序
```
npm start
```

## 桌面客户端文件打包
```
npm run build
```
打包前请先把前端文件放入html目录,html文件是通过前端项目使用npm run export生成.
打包好的文件是在dist目录.

## 单独启动后端程序

打开文件src/syncing.js, 找到以下这一行,并且去掉注释,修改NodeStorageDirectory的值为一个可用的文件夹,也可以在.env中设置NodeStorageDirectory的值,你上传的文件会保存到这个目录下面.
```
await initChatBookDb({"NodeStorageDirectory": process.env.NodeStorageDirectory});
```
然后执行
```
npm run express
```
修改以后express会自动重新启动,立即生效.
### 注意:
    1 当你需要打包为二进制文件的时候,需要注释这一行,因为文件路径会在二进制文件运行的时候进行配置,不需要在代码中配置.
    2 当你启动express以后,前端编译好的HTML,可以放到任何地方进行运行,实现前后端分离,或是直接给其它应用提供后端服务.
    3 启动以后,可以在控制台看到系统运行日志.
    4 如果安装有问题,请把NODE版本调为: 18.17.1

## 前端项目
    https://github.com/chatbookai/ChatBookUI

## 技术架构
    1 LLM:  Langchain, Pinecone, OPENAI, 后续会持续集成其它模型
    2 后端: Electron, Express
    3 前端: React, NEXT, MUI
