local RecurrentAttention, parent = torch.class("RecurrentAttention", "nn.AbstractSequencer")

function RecurrentAttention:__init(rnn, action, nStep, hiddenSize,view_num,viewsLoc)
   parent.__init(self)
   assert(torch.isTypeOf(action, 'nn.Module'))
   assert(torch.type(nStep) == 'number')
   assert(torch.type(hiddenSize) == 'table')
   assert(torch.type(hiddenSize[1]) == 'number', "Does not support table hidden layers" )
   
   self.rnn = rnn
   -- we can decorate the module with a Recursor to make it AbstractRecurrent
   self.rnn = (not torch.isTypeOf(rnn, 'nn.AbstractRecurrent')) and nn.Recursor(rnn) or rnn
   -- backprop through time (BPTT) will be done online (in reverse order of forward)
   self.rnn:backwardOnline()
   for i,modula in ipairs(self.rnn:listModules()) do
      if torch.isTypeOf(modula, "nn.AbstractRecurrent") then
         modula.copyInputs = false
         modula.copyGradOutputs = false
      end
   end
   
   -- samples an x,y actions for each example
   self.action =  (not torch.isTypeOf(action, 'nn.AbstractRecurrent')) and nn.Recursor(action) or action 
   self.action:backwardOnline()
   self.hiddenSize = hiddenSize
   self.nStep = nStep
   
   self.modules = {self.rnn, self.action}
   self.output = {} -- rnn output
   self.actions = {} -- action output
   self.viewsLoc=viewsLoc
   self.view_num=view_num 
   self.forwardActions = false
   
   self.gradHidden = {}
end

function RecurrentAttention:updateOutput(input)
   self.rnn:forget()
   self.action:forget()
   local n_sample = input:size(1)
   local viewIds={}
   local initial_actions=input.new():resize(n_sample,2)
   local index_input=input[{{},1,1,1}]:clone():float():round()
   for i=1,n_sample do
     local view_temp=index_input[i]%self.view_num
     if view_temp==0 then
       view_temp=self.view_num
     end
     viewIds[i]=view_temp
     initial_actions[i]:copy(self.viewsLoc[view_temp])
   end
   for step=1,self.nStep do
      if step == 1 then
         self._initInput = self._initInput or input.new()
         self._initInput:resize(input:size(1),table.unpack(self.hiddenSize)):zero()
         self.actions[1] = self.action:updateOutput{self._initInput,initial_actions}
         self.actions[1]:copy(initial_actions)
      else
         self.actions[step] = self.action:updateOutput{self.output[step-1],self.actions[step-1]}
      end
      
      local output = self.rnn:updateOutput{input, self.actions[step]}    
      self.output[step] = self.forwardActions and {output, self.actions[step]} or output
      
   end
   
   return self.output
end


function RecurrentAttention:updateGradInput(input, gradOutput)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   
   for step=self.nStep,1,-1 do
     -- Note : gradOutput is ignored by REINFORCE modules so we give a zero Tensor instead
     -- 1. backward through the action layer
     self._gradAction = self._gradAction or self.action.output.new()
     if not self._gradAction:isSameSizeAs(self.action.output) then
        self._gradAction:resizeAs(self.action.output):zero()
     end
     gradAction_ = self._gradAction
     if step > 1 then
        gradAction = self.action:updateGradInput({self.output[step-1],self.actions[step-1]}, gradAction_)[2]
     end
     -- 2. backward through the rnn layer
     local gradOutput_ = gradOutput[step]:fill(0)
     local gradInput = self.rnn:updateGradInput({input, self.actions[step]}, gradOutput_)[1]
     if step == self.nStep then
        self.gradInput:resizeAs(gradInput):copy(gradInput)
     else
        self.gradInput:add(gradInput)
     end
   end
   return self.gradInput
end

function RecurrentAttention:accGradParameters(input, gradOutput, scale)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
   
   -- back-propagate through time (BPTT)
   for step=self.nStep,1,-1 do
      -- 1.backward through the action layer
      local gradAction_ = self._gradAction
            
      if step > 1 then
         self.action:accGradParameters({self.output[step-1],self.actions[step-1]},gradAction_, scale)
      end
            -- 2. backward through the rnn layer
      self.rnn:accGradParameters({input, self.actions[step]},gradOutput[step], scale)
   end
end

function RecurrentAttention:accUpdateGradParameters(input, gradOutput, lr)
   assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")
    
   -- backward through the action layers
   for step=self.nStep,1,-1 do
      -- backward through the action layer
      local gradAction_ =  self._gradAction
      
      if step > 1 then
         -- Note : gradOutput is ignored by REINFORCE modules so we give action.output as a dummy variable
         self.action:accUpdateGradParameters({self.output[step-1],self.actions[step-1]}, gradAction_, lr)
      end
      
   end
end



function RecurrentAttention:type(type)
   self._input = nil
   self._actions = nil
   self._crop = nil
   self._pad = nil
   self._byte = nil
   return parent.type(self, type)
end

function RecurrentAttention:__tostring__()
   local tab = '  '
   local line = '\n'
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {'
   str = str .. line .. tab .. 'action : ' .. tostring(self.action):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'rnn     : ' .. tostring(self.rnn):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end
