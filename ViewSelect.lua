local ViewSelect, parent = torch.class('ViewSelect', 'nn.Module')

function ViewSelect:__init(AllData,view_num,viewsLoc,evaluation)
  parent.__init(self)
  self.AllData=AllData
  self.viewsLoc=viewsLoc
  self.gradInput = {torch.Tensor(), torch.Tensor()}
  self.view_num= view_num
  self.evaluation= evaluation or false
end

function ViewSelect:getNextViewId(location,n_sample)
  local n_sample=n_sample
  local nextViewIds=torch.LongTensor(n_sample):fill(0)
  for k=1,n_sample do
    local mindist=9999.0
    for i=1,self.view_num do
      local dist=torch.abs(self.viewsLoc[i]-location[k]):norm()
      if dist < mindist then
        nextViewIds[k]=i
        mindist=dist
      end
    end
    if location[k]:clone():abs():max()>1.1 then -- a little bigger than 1
        nextViewIds[k]=0
    end
  end
  return nextViewIds
end

function ViewSelect:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table')
   assert(#inputTable >= 2)
   local input, location = unpack(inputTable)
   self.evaluation = true
   if self.evaluation == false then
     local n_sample = input:size(1)
     local objIds={}
     local index_input=input[{{},1,1,1}]:clone():float():round()
     for i=1,n_sample do
       local view_temp=index_input[i]%self.view_num
       if view_temp==0 then
         view_temp=self.view_num
       end
       objIds[i]=(index_input[i]-view_temp)/self.view_num
     end
     local nextViewIds=self:getNextViewId(location,n_sample)
     --local output=torch.rand(input:size())
     local output=torch.rand(input:size(1),256)
     if torch.type(input)=='torch.CudaTensor' then output=output:cuda() end
     for i=1,n_sample do
       if nextViewIds[i] ~=0 then
         assert(objIds[i]*self.view_num+nextViewIds[i]<=self.AllData:size(1))
         output[i]:copy(self.AllData[objIds[i]*self.view_num+nextViewIds[i]])
       end  
     end
     self.output=output  
   else
     self.output=input
   end
   return self.output
end

function ViewSelect:updateGradInput(inputTable, gradOutput)
  local input, location = unpack(inputTable)
  local gradInput, gradLocation = unpack(self.gradInput)
  gradInput:resizeAs(input):zero()
  gradLocation:resizeAs(location):zero() -- no backprop through location
  --gradInput:resizeAs(gradOutput)
 -- gradInput:copy(gradOutput)
  self.gradInput[1] = gradInput
  self.gradInput[2] = gradLocation
  return self.gradInput
end