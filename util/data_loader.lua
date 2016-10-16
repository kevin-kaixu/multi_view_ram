require 'torch'
require 'paths'
local matio = require 'matio'
torch.setdefaulttensortype('torch.FloatTensor')


exdata = {}
exdata.path_dataset = 'data_hierarchy_tree'
exdata.path_trainset = paths.concat(exdata.path_dataset, '5classes_data_train.mat')
exdata.path_testset = paths.concat(exdata.path_dataset, '5classes_data_test.mat')

function exdata.loadTrainSet(maxLoad)
   return exdata.loadDataset(exdata.path_trainset, maxLoad)
end

function exdata.loadTestSet(maxLoad)
   return exdata.loadDataset(exdata.path_testset, maxLoad)
end

function exdata.loadDataset(fileName, maxLoadnum)
  
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
   
   function dataset:putLabelOnSelectData(selectviews,allView_num,start_num)
      assert(torch.type(selectviews)=='table')
      assert(torch.type(selectviews)=='number')
      assert(#selectviews<allView_num)
      local start_num=start_num or 0
      local newData,newLabels
      for i=1,AllView_num do
          local tmp_data=data:narrow(1,(i-1)*allView_num+1,allView_num):index(1,torch.LongTensor(selectviews))
          local tmp_label=labels:narrow(1,(i-1)*allView_num+1,allView_num):index(1,torch.LongTensor(selectviews))
          for j=1,tmp_data:size(1) do
              tmp_data[j][1][1][1]=start_num+(i-1)*allView_num+selectviews[j]
          end
          if i==1 then 
            newData=tmp_data
            newLabels=tmp_label
          else
            newData=torch.cat(newData,tmp_data,1)
            newLabels=torch.cat(newLabels,tmp_label,1)
          end     
      end
      data=newData
      labels=newLabels
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
