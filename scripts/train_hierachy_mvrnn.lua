--require ('mobdebug').start()
require 'dp'
require 'rnn'
require 'optim'
require 'ViewSelect'
require 'RewardCriterion'
require 'RecurrentAttention_ex'
require 'cutorch'
require 'cunn'
require 'paths'
local matio = require 'matio'
torch.setdefaulttensortype('torch.FloatTensor')

version = 12
--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a 3d hierachy attention Model')
cmd:text('Options:')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')

cmd:option('--maxEpoch', 200, 'maximum number of epochs to run')
cmd:option('--maxTries', 100, 'maximum number of epochs to try to find a better local minima for early-stopping')


cmd:option('--progress', false, 'print progress bar')
--[[ reinforce ]]--


cmd:option('--stochastic', false, 'Reinforce modules forward inputs stochastically during evaluation')


cmd:option('-seed',123,'torch manual random number generator seed')
--[[ data ]]--
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation')
cmd:option('--noTest', false, 'dont propagate through the test set')
----------------------------------------------
cmd:option('--hiddenSize', 256, 'number of hidden units used in Simple RNN.')
cmd:option('--viewFeatureSize', 256, 'number of hidden units used in view hidden layer')
cmd:option('--locationFeatureSize', 256, 'number of hidden units used in locator hidden layer')
cmd:option('--batchSize',128, 'number of examples per batch')
cmd:option('--learningRate', 0.01, 'learningrate at t=0')
cmd:option('--locatorStd', 0.22, 'stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')
cmd:option('--imageSize', 32, '')
cmd:option('--view_num', 21, 'num of all view')
cmd:option('--rho', 5, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--transfer', 'ReLU', 'activatin function')
cmd:option('--rewardScale',1, "scale of positive reward (negative is 0)")
cmd:option('--minLR', 0.000001, 'minimum learning rate')
cmd:option('--saturateEpoch', 800, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--save_dir', 'cur_model', 'model save path')
cmd:option('--dataset', '', 'train data')
cmd:text()

opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end
torch.manualSeed(opt.seed)
if opt.save_dir ~= '' then
  dp.SAVE_DIR=opt.save_dir
end
print(opt.dataset)
--
require('viewsLoc')
viewsLoc= getAllViewsLoc()

--load data
function loadDataset(fileName, maxLoadnum)
  
   local f = matio.load(fileName)
   local data = f.images.data:type(torch.getdefaulttensortype())
   data=data:permute(4,3,1,2)
   local nExample = data:size(1)
   local labels = f.images.labels:long():resize(nExample,1)

   local maxLoad= maxLoadnum or nExample
   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<exdata> loading only ' .. nExample .. ' examples')
   end
   data=data:narrow(1,1,maxLoad)
   labels = labels:narrow(1,1,maxLoad)


   local dataset = {}
   dataset.data = data
   dataset.labels = labels

   function dataset:putLabelInData(start_num)
      local start_num=start_num or 0
      data_size=data:size(1)
      for i=start_num+1,start_num+data_size do
          data[i-start_num][1][1][1]=i
      end
   end
     
   
   function dataset:size()
      return nExample
   end
   
   function dataset:shuffleData(view_num)
     local shuffle = torch.randperm(data:size(1)/view_num)
     local shuffle_data=torch.Tensor():resizeAs(data)
     local shuffle_lables=torch.LongTensor():resizeAs(labels)
     for i=1,data:size(1)/view_num do
        local target_num=shuffle[i]
        for j=1,view_num do
          shuffle_data[(i-1)*view_num+j]:copy(data[(target_num-1)*view_num+j])
          shuffle_lables[(i-1)*view_num+j]:copy(labels[(target_num-1)*view_num+j])
        end
     end
     data=shuffle_data
     labels=shuffle_lables
   end

   setmetatable(dataset, {__index = function(self, index)
			     local input = self.data[index]
			     local class = self.labels[index]
			     local example = {input, class}
                        return example
   end})

   return dataset
end

function dataCreation()
   -- 1. load images into input and target Tensors
   local trainData=loadDataset(opt.dataset)
   local testData=loadDataset(opt.dataset)
   local pureTrainData=trainData.data:clone()
   local pureTestData=testData.data:clone()
   trainData:putLabelInData(0)
   train_size=trainData:size()
   testData:putLabelInData(train_size)

   -- 2.wrap data into dp.Views
   local trainInput = dp.ImageView('bchw', trainData.data)
   local trainTarget = dp.ClassView('b', trainData.labels:view(-1))
   local validInput = dp.ImageView('bchw', testData.data)
   local validTarget = dp.ClassView('b', testData.labels:view(-1))
   local testInput = dp.ImageView('bchw', testData.data)
   local testTarget = dp.ClassView('b', testData.labels:view(-1))

   trainTarget:setClasses({'1', '2', '3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'})
   validTarget:setClasses({'1', '2', '3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'})
   testTarget:setClasses({'1', '2', '3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'})
   -- 3. wrap views into datasets

   local train = dp.DataSet{inputs=trainInput,targets=trainTarget,which_set='train'}
   local valid = dp.DataSet{inputs=validInput,targets=validTarget,which_set='valid'}
   local test = dp.DataSet{inputs=testInput,targets=testTarget,which_set='test'}

   -- 4. wrap datasets into datasource

   local ds = dp.DataSource{train_set=train,valid_set=valid,test_set=test}
   ds:classes{'1', '2', '3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}

   return ds,pureTrainData,pureTestData
end

ds,pureTrainData,pureTestData=dataCreation()
-------------
AllData=torch.cat(pureTrainData,pureTestData,1)
if pureTrainData:size(1)< opt.view_num*20 then
  opt.batchSize=32
end
--[[Model]]--
dofile 'model.lua'
print(agent)
parameters=agent:parameters()
opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
train = dp.Optimizer{
  loss =nn.ModuleCriterion(RewardCriterion(agent, opt.rewardScale,baselineNet,viewsLoc), nil, nn.Convert()) -- REINFORCE
  ,
  epoch_callback = function(model, report) -- called every epoch
    if report.epoch > 0 then
      opt.learningRate = opt.learningRate + opt.decayFactor
      opt.learningRate = math.max(opt.minLR, opt.learningRate)
      if not opt.silent then
        print("learningRate", opt.learningRate)
      end
    end
  end,
  callback = function(model, report)
    collectgarbage()
    if opt.cutoffNorm > 0 then
      local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
      opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
      if opt.lastEpoch < report.epoch and not opt.silent then
        print("mean gradParam norm", opt.meanNorm)
      end
    end
    model:updateGradParameters(opt.momentum) -- affects gradParams
    model:updateParameters(opt.learningRate) -- affects params
    model:maxParamNorm(opt.maxOutNorm) -- affects params
    model:zeroGradParameters() -- affects gradParams 
  end,
  feedback = dp.Confusion{output_module=nn.SelectTable(-1)},
  sampler = dp.ShuffleSampler{
  epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
  },
  progress = opt.progress
}
valid = dp.Evaluator{
  feedback = dp.Confusion{output_module=nn.SelectTable(-1)},
  sampler = dp.Sampler{epoch_size = opt.validEpochSize, batch_size = opt.batchSize},
  progress = opt.progress
}
if not opt.noTest then
  tester = dp.Evaluator{
  feedback = dp.Confusion{output_module=nn.SelectTable(-1)},
  sampler = dp.Sampler{batch_size = opt.batchSize}
}
end
--[[Experiment]]--
xp = dp.Experiment{
  model = agent,
  optimizer = train,
  validator = valid,
  tester = tester,
  observer = {
    ad,
    dp.FileLogger(),
    dp.EarlyStopper{
      max_epochs = opt.maxTries,
      error_report={'validator','feedback','confusion','accuracy'},
      maximize = true
    }
  },
  random_seed = os.time(),
  max_epoch = opt.maxEpoch
}
--[[GPU or CPU]]--
if opt.cuda then
  cutorch.setDevice(opt.useDevice)
  xp:cuda()
end
xp:verbose(not opt.silent)
if not opt.silent then
  print"Agent :"
  print(agent)
end
xp.opt = opt
xp:run(ds)
