mvcnn=torch.load('data_hierarchy_tree/mvcnn/mvcnn.net'):float()
mvcnn_cnn1=mvcnn.modules[1]
AllData=mvcnn_cnn1:forward(AllData) -- little trick for speed training
--torch.save('Alldata',AllData)
--AllData=torch.load('Alldata')
mvcnn_cnn2=mvcnn.modules[4]
viewFeatureNet = nn.Sequential()
viewFeatureNet:add(ViewSelect(AllData,opt.view_num,viewsLoc,false))
--viewFeatureNet:add(mvcnn_cnn1)   

recurrent1 = nn.Linear(opt.hiddenSize, opt.hiddenSize)

-- recurrent neural network
mergeModule=nn.Sequential()
--          :add(nn.ParallelTable():add(nn.Reshape(opt.hiddenSize,1,true)):add(nn.Reshape(opt.hiddenSize,1,true)))
          :add(nn.JoinTable(1,1))
          :add(nn.Reshape(2,opt.hiddenSize))
          :add(nn.Max(1,2))
            
rnn1=nn.Recurrent(nn.Identity(), viewFeatureNet, nn.Identity(), nn.Identity(), 99999,mergeModule)

-- actions (nbvRegNet)
nbvRegNet = nn.Sequential()
nbvRegNet:add(nn.Linear(opt.hiddenSize, 2))
nbvRegNet:add(nn.HardTanh(-1.11,1.11)) -- bounds sample between -1 and 1
nbvRegNet:add(nn.ReinforceNormal(opt.locatorStd, opt.stochastic)) -- sample from normal, uses REINFORCE learning rule
nbvRegNet:add(nn.HardTanh(-1.11,1.11)) -- bounds sample between -1 and 1


locationFeatureNet = nn.Sequential()
locationFeatureNet:add(nn.Linear(2, opt.locationFeatureSize))
locationFeatureNet:add(nn[opt.transfer]())


viewglimpse = nn.Sequential()
              :add(nn.ParallelTable():add(nn.Identity()):add(locationFeatureNet))
              :add(nn.CMulTable())

--rnn2: action contain rnn2
recurrent2 = nn.Linear(opt.hiddenSize, opt.hiddenSize)  

action = nn.Sequential()
              :add(nn.Recurrent(opt.hiddenSize, viewglimpse, recurrent2,  nn[opt.transfer](), 99999))
              :add(nbvRegNet)


attention = RecurrentAttention(rnn1, action, opt.rho, {opt.hiddenSize}, opt.view_num,viewsLoc)

-- model is a reinforcement learning agent
agent = nn.Sequential()
agent:add(nn.Convert())
agent:add(attention)

-- classifier :
paraDealer=nn.ParallelTable()
for i=1,opt.rho do
  local classifier=mvcnn_cnn2:clone('weight','bias','gradWeight','gradBias')
  paraDealer:add(classifier)
end

agent:add(paraDealer)
--classifier=nn.Sequential()
--          :add(mvcnn_cnn2)
--classifier=nn.Sequential()
--          :add(nn.Linear(opt.hiddenSize, #ds:classes()))
--          :add(nn.LogSoftMax())
--agent:add(classifier)

-- add the baseline reward predictor
baselineNet = nn.Sequential()
--baselineNet:add(nn.Linear(opt.hiddenSize,1))
baselineNet:add(nn.Constant(1,1))
baselineNet:add(nn.Add(1))

--concat = nn.ConcatTable():add(nn.Identity()):add(baselineNet)

-- output will be : {classpred, {classpred, basereward}}
--agent:add(concat)

