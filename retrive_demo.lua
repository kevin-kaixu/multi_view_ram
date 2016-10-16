--require ('mobdebug').start()
require 'dp'
require 'rnn'
require 'optim'
require 'ViewSelect'
require 'RecurrentAttention_ex'
require 'RewardCriterion'
require 'cutorch'
require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')

local matio = require 'matio'

cmd = torch.CmdLine()
cmd:text()
cmd:text('retrive for a object')
cmd:text('Options:')
cmd:option('--cuda', false, 'model was saved with cuda')
cmd:option('--stochastic', false, 'evaluate the model stochatically. Generate glimpses stochastically')
cmd:option('--view_num', 21, 'num of  all view')
cmd:option('--rho', 5, 'time-steps')
cmd:option('--hiddenSize', 256, 'number of hidden units used in Simple RNN')
cmd:text()
local opt = cmd:parse(arg or {})


function getViewId(location,viewsLoc)
  local viewId
  _,viewId=(viewsLoc-location:resize(1,2):expandAs(viewsLoc)):norm(2,2):min(1)
  viewId=viewId[1][1]
  return viewId
end
 
----prepare test data--
main_dir='data_hierarchy_tree'
local threshold=0.98
require('util/viewsLoc')
viewsLoc= getAllViewsLoc()
--local f=torch.load('') -- 1 to 21 views 
local f = matio.load('data_hierarchy_tree/5classes_data_test.mat')
local data = f.images.data:type(torch.getdefaulttensortype())
data=data:permute(4,3,1,2):narrow(1,1,50*opt.view_num)
for obj=1,data:size(1),opt.view_num do
  cur_data=data:narrow(1,obj,opt.view_num)
  local input_view_id=2

  local step=1
  local input=cur_data[input_view_id]
  rnn_results={}
  view_seqs={}
  locations={}
  node_record={}
  locations[1]=viewsLoc[input_view_id]
  table.insert(view_seqs,input_view_id)
  ----first load root model  --- root torch model start
  local cur_ram_model_path=paths.concat(main_dir,'cur_model','model.dat')
  local cur_mvcnn_model_path=paths.concat(main_dir,'mvcnn','mvcnn.net')
  local cur_train_data_path ---=paths.concat(main_dir,'5classes_data_test.mat')

  cur_xp =torch.load(cur_ram_model_path)
  cur_model = cur_xp:model().module:float() 
  cur_mvcnn =torch.load(cur_mvcnn_model_path):float()
  cur_fes_extra=cur_mvcnn.modules[1]
  cur_ram= cur_model:findModules('RecurrentAttention')[1]
  cur_rnn= cur_ram.rnn    --here the rnn just for pooling, so no need change
  cur_action=cur_ram.action
  cur_rnn:forget()    
  cur_action:forget()
  ----action init
  if not opt.stochastic then
     for i=1,opt.rho do
        local rn = cur_action:getStepModule(i):findModules('nn.ReinforceNormal')[1]
        rn.stdev = 0 -- deterministic
        
     end
  end
  for i=1,opt.rho do
      local viewSelect=cur_rnn:getStepModule(i):findModules('ViewSelect')[1]
      viewSelect.evaluation=true
  end

  initInput = input.new()
  initInput:resize(1,opt.hiddenSize):zero()
  cur_action:updateOutput{initInput,viewsLoc[input_view_id]}
  ---first action (location) is given

  ------------------

  cur_classifier=cur_mvcnn.modules[4]
  -- compute -----------------------------
  -- forward rnn ---
  cur_input=cur_fes_extra:updateOutput(input):view(-1)
  local cur_rnn_output=cur_rnn:updateOutput{cur_input, locations[step]}
  rnn_results[step]=cur_rnn_output
  ---forward classifier --
  local cur_result= cur_classifier:updateOutput(cur_rnn_output)
  local maxPro,tar_label
  maxPro,tar_label=torch.exp(cur_result):max(1)
  local first_tar_label=tar_label[1]
  --- if go on --
  cur_main_dir=main_dir
  local node_count=1
  local isreach_leaf_node=false
  while not isreach_leaf_node  do 
    if maxPro[1] >threshold then 
      if node_count==1 and tar_label[1] ~= 1 then
          --check for chair
          goto continue 
      end
    --- go to sub node ---
      node_record[node_count]=tar_label[1]
      node_count=node_count+1
      step=step+1
      local sub_dir_name='subclass' .. tar_label[1]
      local sub_node_dir=paths.concat(cur_main_dir,sub_dir_name)
      if not paths.dirp(sub_node_dir) then
          sub_dir_name='subclass' .. tar_label[1] .. '_oneShot'
          isreach_leaf_node=true
          sub_node_dir=paths.concat(cur_main_dir,sub_dir_name)
      end
      ----------------------------
      ---- update path -----------
      cur_main_dir=sub_node_dir

      cur_ram_model_path = paths.concat(cur_main_dir,'cur_model','model.dat')
      cur_mvcnn_model_path = paths.concat(cur_main_dir,'mvcnn','mvcnn.net')
      cur_train_data_path = paths.concat(cur_main_dir,'imdAllData3.mat')
      if isreach_leaf_node then
          cur_train_data_path = paths.concat(cur_main_dir,'oneShot_data.mat')
      end
      ---- compute -------------------
      cur_xp =torch.load(cur_ram_model_path)
      cur_model = cur_xp:model().module:float() 
      cur_mvcnn =torch.load(cur_mvcnn_model_path):float()
      cur_fes_extra=cur_mvcnn.modules[1]
      cur_ram= cur_model:findModules('RecurrentAttention')[1]
      cur_action=cur_ram.action
      cur_action:forget()
      if not opt.stochastic then
         for i=1,opt.rho do
            local rn = cur_action:getStepModule(i):findModules('nn.ReinforceNormal')[1]
            rn.stdev = 0 -- deterministic
         end
      end
      ----action init
      if step <3 then
            initInput = input.new()
            initInput:resize(1,opt.hiddenSize):zero()
            cur_action:updateOutput{initInput,viewsLoc[input_view_id]}
      else
        for i=1,step-2 do
          if i == 1 then
            initInput = input.new()
            initInput:resize(1,opt.hiddenSize):zero()
            cur_action:updateOutput{initInput,viewsLoc[input_view_id]}
          else
            cur_action:updateOutput{rnn_results[i-1],locations[i-1]}
          end
        end
      end


      cur_classifier=cur_mvcnn.modules[4]
      
      pred_action=cur_action:updateOutput{rnn_results[step-1],locations[step-1]} 
      locations[step]=pred_action
      next_view_id = getViewId(pred_action,viewsLoc)
      view_seqs[step]=next_view_id
      cur_input=cur_fes_extra:updateOutput(cur_data[next_view_id])
      cur_rnn_output=cur_rnn:updateOutput{cur_input, locations[step]}
      rnn_results[step]=cur_rnn_output
      ---forward classifier --
      cur_result= cur_classifier:updateOutput(cur_rnn_output)
      maxPro,tar_label=torch.exp(cur_result):max(1)
    else
      -- keep on current node ------
      step=step+1
      -- go to next view point, forward action net --
      pred_action=cur_action:updateOutput{rnn_results[step-1],locations[step-1]} 
      locations[step]=pred_action
      next_view_id = getViewId(pred_action,viewsLoc)
      view_seqs[step]=next_view_id
      cur_input=cur_fes_extra:updateOutput(cur_data[next_view_id])
      cur_rnn_output=cur_rnn:updateOutput{cur_input, locations[step]}
      rnn_results[step]=cur_rnn_output
      ---forward classifier --
      cur_result= cur_classifier:updateOutput(cur_rnn_output)
      
      maxPro,tar_label=torch.exp(cur_result):max(1)
      if node_count==1 then
          first_tar_label=tar_label[1]
      end
    end
    
    if step == opt.rho then
      if first_tar_label ~= 1 then
          --check for chair
          goto continue 
      end
      -- go to leaf 
      while not isreach_leaf_node  do 
        node_record[node_count]=tar_label[1]
        node_count=node_count+1
        local sub_dir_name='subclass' .. tar_label[1]
        local sub_node_dir=paths.concat(cur_main_dir,sub_dir_name)
        if not paths.dirp(sub_node_dir) then
            sub_dir_name='subclass' .. tar_label[1] .. '_oneShot'
            isreach_leaf_node=true
            sub_node_dir=paths.concat(cur_main_dir,sub_dir_name)
        end
        ----------------------------
        ---- update path -----------
        cur_main_dir=sub_node_dir

        cur_ram_model_path = paths.concat(cur_main_dir,'cur_model','model.dat')
        cur_mvcnn_model_path = paths.concat(cur_main_dir,'mvcnn','mvcnn.net')
        cur_train_data_path = paths.concat(cur_main_dir,'imdAllData3.mat')
        if isreach_leaf_node then
            cur_train_data_path = paths.concat(cur_main_dir,'oneShot_data.mat')
        end
        ---- compute -------------------
        cur_xp =torch.load(cur_ram_model_path)
        cur_model = cur_xp:model().module:float() 
        cur_mvcnn =torch.load(cur_mvcnn_model_path):float()
        cur_fes_extra=cur_mvcnn.modules[1]
        cur_ram= cur_model:findModules('RecurrentAttention')[1]
        cur_action=cur_ram.action
        cur_action:forget()
        if not opt.stochastic then
           for i=1,opt.rho do
              local rn = cur_action:getStepModule(i):findModules('nn.ReinforceNormal')[1]
              rn.stdev = 0 -- deterministic
           end
        end
        ----action init
        if step <3 then
              initInput = input.new()
              initInput:resize(1,opt.hiddenSize):zero()
              cur_action:updateOutput{initInput,viewsLoc[input_view_id]}
        else
          for i=1,step-2 do
            if i == 1 then
              initInput = input.new()
              initInput:resize(1,opt.hiddenSize):zero()
              cur_action:updateOutput{initInput,viewsLoc[input_view_id]}
            else
              cur_action:updateOutput{rnn_results[i-1],locations[i-1]}
            end
          end
        end


        cur_classifier=cur_mvcnn.modules[4]
        
        pred_action=cur_action:updateOutput{rnn_results[step-1],locations[step-1]} 
        locations[step]=pred_action
        next_view_id = getViewId(pred_action,viewsLoc)
        view_seqs[step]=next_view_id
        cur_input=cur_fes_extra:updateOutput(data[next_view_id])
        cur_rnn_output=cur_rnn:updateOutput{cur_input, locations[step]}
        rnn_results[step]=cur_rnn_output
        ---forward classifier --
        cur_result= cur_classifier:updateOutput(cur_rnn_output)
        maxPro,tar_label=torch.exp(cur_result):max(1)
      end
    end
    if isreach_leaf_node then
        node_record[node_count]=tar_label[1]
      --  retrive_data=torch.load(cur_train_data_path)
        local f = matio.load(cur_train_data_path)
        local cur_retrive_data = f.images.data:type(torch.getdefaulttensortype())
        cur_retrive_data=cur_retrive_data:permute(4,3,1,2)
        retrive_data=cur_retrive_data:narrow(1,opt.view_num*(tar_label[1]-1)+1,opt.view_num)
    end
    
  end

  viewGlimpses = {}

  for i=1,#view_seqs do
    viewGlimpses[i]=cur_data[view_seqs[i]]
  end
  if not paths.dirp('retrive_res/view_seqs') then
      paths.mkdir('retrive_res/view_seqs')
  end
  local g = image.toDisplayTensor{input=viewGlimpses,nrow=10,padding=3}
  local cur_obj_id=(obj-1)/opt.view_num+1
  image.save("retrive_res/view_seqs/obj_".. cur_obj_id .. "th_view_seqs.png", g)

  compare_retrive={}
  for i=1,10 do
    compare_retrive[i]=cur_data[i]
    compare_retrive[i+10]=retrive_data[i]
  end
  if not paths.dirp('retrive_res/compare') then
    paths.mkdir('retrive_res/compare')
  end
  local cop = image.toDisplayTensor{input=compare_retrive,nrow=10,padding=3}
  image.save("retrive_res/compare/compare_" .. cur_obj_id .. ".png", cop)
  collectgarbage()
  ::continue::
end
----


