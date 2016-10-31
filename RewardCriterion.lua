
local RewardCriterion, parent = torch.class("RewardCriterion", "nn.Criterion")

function RewardCriterion:__init(module, scale,baselineNet,viewsLoc,entropyParam,mvCostParam)
   parent.__init(self)
   self.module = module -- so it can call module:reinforce(reward)
   self.scale = scale or 1 -- scale of reward
   self.MSECriterion =nn.MSECriterion() -- baseline criterion
   self.ClassNLLCriterion =nn.ClassNLLCriterion() -- loss criterion
   self.sizeAverage = true
   self.gradInput = {}
   self.allReward={}
   self.entropyParam= entropyParam or 5
   self.mvCostParam = mvConstParam or 0.1
   self.baselineNet=baselineNet
   self.viewsLoc=viewsLoc
end

function RewardCriterion:getViewId(location)
  local viewId,value
  value,results=(self.viewsLoc-location:clone():resize(1,2):repeatTensor(self.viewsLoc:size(1),1)):norm(2,2):min(1)
  viewId=results[1][1]
  return viewId
end

function RewardCriterion:getShannoEntropy(classPred)
   prob=torch.exp(classPred)
   local entropy=-torch.dot(prob,classPred)
   return entropy
end


function RewardCriterion:computeMvCost(loc_k1,loc_k2)
   local delt_loc=loc_k2-loc_k1
   local dist=delt_loc:norm()/2
   return dist
end
function RewardCriterion:updateOutput(input, target)
   assert(torch.type(input) == 'table')
   self.reward = self.reward or input[1].new()
   self.reward:resize(target:size(1)):fill(0)
   local ra = self.module:findModules('RecurrentAttention')[1]
   local locations = ra.actions
   local rho=#locations
   local ram_softmax=input
   for i=1,target:size(1) do
      local tmp_target=target[i]
      local tmp_entropy=torch.Tensor(rho):zero()
      local tmp_maxId=torch.LongTensor(rho):zero()
      local tmp_maxPro=torch.Tensor(rho):zero()
      for j=1,rho do
          local value,maxId= torch.max(ram_softmax[j][i],1)
          tmp_entropy[j]=self:getShannoEntropy(ram_softmax[j][i])
          tmp_maxId[j]=maxId[1]
          tmp_maxPro[j]=value[1]
      end  
      if tmp_maxId[1]==tmp_target then
          self.reward[i]=self.reward[i]+1
      end
      local flag=false
      for k=2,rho do
        if tmp_maxId[k]==tmp_target then
           self.reward[i]=self.reward[i]+1
           local deltEntropy=tmp_entropy[k-1]-tmp_entropy[k]
           self.reward[i]=self.reward[i]+deltEntropy*self.entropyParam-self.mvCostParam*self:computeMvCost(locations[k-1][i],locations[k][i])
        end
        if location[k][i]:clone():abs():max()>1 then
            flag=true
        end
      end
      if flag then
          self.reward[i]=0
      end
      
      local view_num=self.viewsLoc:size(1)
      local tmp_viewIdsCount=torch.LongTensor(view_num):fill(0) 
      for k=1,rho do
        local tmp_viewId=self:getViewId(locations[k][i])
        tmp_viewIdsCount[tmp_viewId]=tmp_viewIdsCount[tmp_viewId]+1
      end
      if tmp_viewIdsCount:gt(1):sum()>0 then
          self.reward[i]=0
      end 
      
   end
   self.reward:div(rho)
   self.output = -self.reward:sum()
   if self.sizeAverage then
      self.output = self.output/input[1]:size(1)
   end

   return self.output
end

function RewardCriterion:updateGradInput(input, target)
   local rho=#input
   local baseline=self.baselineNet:forward(input[rho])
   -- reduce variance of reward using baseline
   self.vrReward = self.vrReward or self.reward.new()
   self.vrReward:resizeAs(self.reward):copy(self.reward)
  -- self.vrReward:add(-1, baseline)
   if self.sizeAverage then
      self.vrReward:div(input[1]:size(1))
   end
   -- broadcast reward to modules
   self.vrReward:mul(self.scale)
   self.module:reinforce(self.vrReward)    
   for i=1,#input do
  --   self.gradInput[i]=self.ClassNLLCriterion:backward(input[i], target)
      self.gradInput[i]=self.gradInput[i] or input[i].new()
      self.gradInput[i]:resizeAs(input[i]):zero()
   end
   -- learn the baseline reward
   self.baselineNet:zeroGradParameters()
   local gradInput_baseline = self.MSECriterion:backward(baseline, self.reward)
   self.baselineNet:backward(input[rho],gradInput_baseline)
   -- 
   return self.gradInput
end

function RewardCriterion:type(type)
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end
