  // blockRoutes.js

  import express from 'express';
  import syncing from '../syncing.js';

  const router = express.Router();

  router.get('/profile/:id', async (req, res) => {
    const { id } = req.params;
    const getWalletAddressProfile = await syncing.getWalletAddressProfile(id);
    console.log("getWalletAddressProfile", getWalletAddressProfile)
    res.status(200).json(getWalletAddressProfile).end();  
  });

  router.get('/address/agent/:pageid/:pagesize', async (req, res) => {
    const { pageid, pagesize } = req.params;
    const getAgentList = await syncing.getAgentList(pageid, pagesize);
    console.log("getAgentList", getAgentList)
    res.status(200).json(getAgentList).end();  
  });

  router.get('/address/referee/:address/:pageid/:pagesize', async (req, res) => {
    const { address, pageid, pagesize } = req.params;
    const getAgentList = await syncing.getAddressReferee(address, pageid, pagesize);
    console.log("getAgentList", getAgentList)
    res.status(200).json(getAgentList).end();  
  });

  export default router;
