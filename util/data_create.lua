require 'dp'
require './data_loader'

function dataCreation()
   -- 1. load images into input and target Tensors
   local trainData=exdata.loadTrainSet()
   local testData=exdata.loadTestSet()
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

   trainTarget:setClasses({'chair', 'display', 'flowerpot','guitar','table'})
   validTarget:setClasses({'chair', 'display', 'flowerpot','guitar','table'})
   testTarget:setClasses({'chair', 'display', 'flowerpot','guitar','table'})
   -- 3. wrap views into datasets

   local train = dp.DataSet{inputs=trainInput,targets=trainTarget,which_set='train'}
   local valid = dp.DataSet{inputs=validInput,targets=validTarget,which_set='valid'}
   local test = dp.DataSet{inputs=testInput,targets=testTarget,which_set='test'}

   -- 4. wrap datasets into datasource

   local ds = dp.DataSource{train_set=train,valid_set=valid,test_set=test}
   ds:classes{'chair', 'display', 'flowerpot','guitar','table'}

   return ds,pureTrainData,pureTestData
end
