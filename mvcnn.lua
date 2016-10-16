--require ('mobdebug').start()
require 'nn'
require 'dp' 
require 'optim'
require 'image'
require 'paths'
require 'util/data_loader'
----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('Mvcnn Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', 'mvcnn', 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-learningRate', 0.05, 'learning rate at t=0')
cmd:option('-batchSize', 5, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-view_num', 21, 'nb of views to use')
cmd:option('-cuda',true,'')
cmd:option('-useDevice', 1, 'sets the device (GPU) to use')
cmd:option('-threads', 2, 'nb of threads to use')
cmd:text()
opt = cmd:parse(arg)
-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. opt.threads)
--
classes={'chair', 'display', 'flowerpot','guitar','table'}

   model = nn.Sequential()
      ------------------------------------------------------------
      -- convolutional network
      ------------------------------------------------------------
      -- stage 1 : mean+std normalization -> filter bank -> squashing -> max pooling
      model_cnn1=nn.Sequential()
      model_cnn1:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
      model_cnn1:add(nn.Tanh())
      model_cnn1:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1))
      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
      model_cnn1:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
      model_cnn1:add(nn.Tanh())
      model_cnn1:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 3 : standard 2-layer MLP:
      model_cnn1:add(nn.Reshape(64*3*3))
      model_cnn1:add(nn.Linear(64*3*3, 256))
      model_cnn1:add(nn.Tanh())



   model:add(model_cnn1)
        :add(nn.Reshape(opt.batchSize,opt.view_num,256,false))
        :add(nn.Max(1,2))
   model_cnn2=nn.Sequential()
   model_cnn2:add(nn.Linear(256, #classes))
   model_cnn2:add(nn.LogSoftMax())
   model:add(model_cnn2)
  ------------------------------------------------------------




-- verbose
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--

criterion = nn.ClassNLLCriterion()

if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   model=model:cuda()
   criterion=criterion:cuda()
end
----------------------------------------------------------------------
-- get/create dataset
--
-- load dataset
trainData =exdata.loadTrainSet()


testData=exdata.loadTestSet()




----------------------------------------------------------------------
-- define training and testing functions
--
-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
accLogger = optim.Logger(paths.concat(opt.save, 'accuracy.log'))
errLogger = optim.Logger(paths.concat(opt.save, 'error.log'   ))


-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1
  
   -- local vars
   local time = sys.clock()
   local trainError = 0
   dataset:shuffleData(opt.view_num)
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize*opt.view_num do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local k=1
      local tn=1
      local tmp_sample_num=math.min(t+opt.batchSize*opt.view_num-1,dataset:size())-t+1
      local inputs = torch.Tensor(tmp_sample_num,1,32,32)
      local targets=torch.LongTensor(tmp_sample_num/opt.view_num):fill(0)
      
      for i = t,math.min(t+opt.batchSize*opt.view_num-1,dataset:size()) do
         local input = dataset.data[i]
         inputs[k]:copy(input)
         if (k-1)%opt.view_num==0 then
          local target = dataset.labels[i]
          targets[tn]=target[1]
          tn=tn+1
         end
         k=k+1
      end

      local feval = function(x)
         -- get new parameters
	     collectgarbage()
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

        if opt.cuda then
          inputs=inputs:cuda()
          targets=targets:cuda()
        end

         local output = model:forward(inputs)
         local loss = criterion:forward(output, targets)
         -- estimate df/dW
         local df_do = criterion:backward(output, targets)
         model:backward(inputs, df_do)

         -- update confusion
         for i=1,targets:size(1) do
          confusion:add(output[i], targets[i])
         end
         -- print (confusion)
         trainError = trainError + loss
 
         return loss,gradParameters
      end

         config = config or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = 2e-5}
         optim.sgd(feval, parameters, config)

   end

   -- train error
   trainError = trainError / math.floor(dataset:size()/(opt.batchSize*opt.view_num))

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   local trainAccuracy = confusion.totalValid * 100
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'mvcnn.net')
   os.execute('mkdir -p ' .. paths.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)
   collectgarbage()
   -- next epoch
   epoch = epoch + 1

   return trainAccuracy, trainError
end


function test(dataset)
    collectgarbage()
   -- epoch tracker
   epoch = epoch or 1
  
   -- local vars
   local time = sys.clock()
   local testError = 0

   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize*opt.view_num do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize*opt.view_num,1,32,32)
      local targets=torch.LongTensor(opt.batchSize):fill(0)
      local k=1
      local tn=1
      for i = t,math.min(t+opt.batchSize*opt.view_num-1,dataset:size()) do
         local input = dataset.data[i]
         inputs[k]:copy(input)
         if (k-1)%opt.view_num==0 then
          local target = dataset.labels[i]
          targets[tn]=target[1]
          tn=tn+1
         end
         k=k+1
      end
      if opt.cuda then
        inputs=inputs:cuda()
        targets=targets:cuda()
      end
      local output = model:forward(inputs)
      local err = criterion:forward(output, targets)
      testError = testError + err
      for i=1,targets:size(1) do
        confusion:add(output[i], targets[i])
      end

   end

   -- train error
   testError = testError / math.floor(dataset:size()/(opt.batchSize*opt.view_num))

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   local testAccuracy = confusion.totalValid * 100
   confusion:zero()

   return testAccuracy, testError
end

----------------------------------------------------------------------
-- and train!
--
local epochTimes=1
while epochTimes<1000 do
   -- train/test
   trainAcc, trainErr = train(trainData)
   testAcc,  testErr  = test (testData)
   -- update logger
   accLogger:add{['% train accuracy'] = trainAcc, ['% test accuracy'] = testAcc}
   errLogger:add{['% train error']    = trainErr, ['% test error']    = testErr}

   -- plot logger
   accLogger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
   errLogger:style{['% train error']    = '-', ['% test error']    = '-'}
   accLogger:plot()
   errLogger:plot()
   epochTimes=epochTimes+1
end