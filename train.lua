--require ('mobdebug').start()
require 'dp'
require 'rnn'
require 'optim'
require 'util/data_create'
require 'ViewSelect'
require 'RewardCriterion'
require 'RecurrentAttention_ex'
require 'cutorch'
require 'cunn'
require 'paths'
version = 12
--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a 3d attention Model')
cmd:text('Options:')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')

cmd:option('--maxEpoch', 2000, 'maximum number of epochs to run')
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
cmd:option('--mvcnn_dir','data_hierarchy_tree/mvcnn','mvcnn_net save path')
cmd:text()
opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end
torch.manualSeed(opt.seed)
if opt.save_dir ~= '' then
  dp.SAVE_DIR=opt.save_dir
end
--
require('util/viewsLoc')
viewsLoc= getAllViewsLoc()

--load data
ds,pureTrainData,pureTestData=dataCreation()
AllData=torch.cat(pureTrainData,pureTestData,1)

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
