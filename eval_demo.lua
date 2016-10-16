--require ('mobdebug').start()
require 'dp'
require 'rnn'
require 'optim'
require 'util/data_create'
require 'ViewSelect'
require 'RecurrentAttention_ex'
require 'RewardCriterion'

local matio = require 'matio'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a 3d attention Model')
cmd:text('Options:')
cmd:option('--cuda', true, 'model was saved with cuda')
cmd:option('--xpPath', 'data_hierarchy_tree/cur_model/model.dat', 'path to a previously saved model')
cmd:option('--stochastic', false, 'evaluate the model stochatically. Generate glimpses stochastically')
cmd:option('--view_num', 21, 'num of  all view')
cmd:option('--rho', 5, 'time-steps')
cmd:option('--eval_dir', 'evaluation', 'save path')
cmd:text()
local opt = cmd:parse(arg or {})
if opt.cuda then
  require 'cutorch'
  require 'cunn'
end
if not paths.dirp(opt.eval_dir) then
    os.execute('mkdir ' .. opt.eval_dir)
end


require'util/viewsLoc'

viewsLoc= getAllViewsLoc()

classes={'chair', 'display', 'flowerpot','guitar','table'}
function getNextViewId(location)
  local viewId
  _,viewId=(viewsLoc-location:resize(1,2):expandAs(viewsLoc)):norm(2,2):min(1)
  return viewId
end
function getNextView(location,input)
   local index_input=torch.round(input[1][1][1])
   local viewId=index_input%opt.view_num
   if viewId==0 then
      viewId=opt.view_num
   end
   local objId = (index_input-viewId)/opt.view_num
   local nextViewId=getNextViewId(location)[1][1]
   local output=AllData[objId*opt.view_num+nextViewId]
   local result=output:clone()
   return result
end

xp = torch.load(opt.xpPath)
model = xp:model().module:float() 
tester = xp:tester() or xp:validator() -- dp.Evaluator
tester:sampler()._epoch_size = nil
conf = tester:feedback() -- dp.Confusion
cm = conf._cm -- optim.ConfusionMatrix
print("Last evaluation of "..(xp:tester() and 'test' or 'valid').." set :")
print(cm)

ds,pureTrainData,pureTestData=dataCreation()
AllData=torch.cat(pureTrainData,pureTestData,1)

ra = model:findModules('RecurrentAttention')[1]

-- stochastic or deterministic
for i=1,#ra.actions do
   local rn = ra.action:getStepModule(i):findModules('nn.ReinforceNormal')[1]
   rn.stochastic = opt.stochastic
end

model:training() -- otherwise the rnn doesn't save intermediate time-step states

if not opt.stochastic then
   for i=1,#ra.actions do
      local rn = ra.action:getStepModule(i):findModules('nn.ReinforceNormal')[1]
      rn.stdev = 0 -- deterministic
   end
end
--
-- save sequences
inputs = ds:get('test','inputs')
targets = ds:get('test','targets', 'b')
output = model:forward(inputs)
locations = ra.actions
sequences = torch.IntTensor(inputs:size(1),opt.rho)
sequences_locations = torch.Tensor(inputs:size(1),opt.rho*2)

for i=1,inputs:size(1) do
  for j=1,opt.rho do
     sequences[i][j]=getNextViewId(locations[j][i])
     sequences_locations[i][2*(j-1)+1]=locations[j][i][1]
     sequences_locations[i][2*(j-1)+2]=locations[j][i][2]
  end
end

matio.save(opt.eval_dir .. '/eval_sequences.mat',sequences)
matio.save(opt.eval_dir .. '/eval_sequences_locations.mat',sequences_locations)
--
input = inputs:narrow(1,1,30)
target= targets:narrow(1,1,30)
confusion = optim.ConfusionMatrix(classes)
test_inputs = ds:get('test','inputs')
test_targets = ds:get('test','targets', 'b')


output = model:forward(input)

for i = 1,target:size(1) do
   confusion:add(output[opt.rho][i], target[i])
end
print(confusion)

locations = ra.actions

views_seqs = {}


for i=1,input:size(1) do
   local img = input[i]
   for j,location in ipairs(locations) do
      local views = views_seqs[j] or {}
      views_seqs[j] = views
      
      local xy = location[i]
      views[i] =getNextView(xy,input[i])
      
      collectgarbage()
   end
end
paths.mkdir(opt.eval_dir .. '/views_seqs')
for j,views in ipairs(views_seqs) do
   local g = image.toDisplayTensor{input=views,nrow=10,padding=3}
   image.save(opt.eval_dir .. "/views_seqs/view_"..j..".png", g)
end


