function mapTheta2Coord(r,theta)
  local x=r*torch.cos(theta)
  local y=r*torch.sin(theta)
  return torch.Tensor{x,y}
end

function getAllViewsLoc()
  local map_num={1,4,6,10}
  local r_size={0,1/3,2/3,1}
  local viewsLoc=torch.zeros(21,2)
  local count_num=1
  for i=#r_size,1,-1 do
    for j=0,map_num[i]-1 do
      local theta=math.pi*2/map_num[i]*j
      viewsLoc[count_num]:copy(mapTheta2Coord(r_size[i],theta))
      count_num=count_num+1
    end
  end
  return viewsLoc
end



