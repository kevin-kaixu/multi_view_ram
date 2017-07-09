#! /usr/bin/env bash
main_dir=`pwd`
function train_mvrnn_model(){
    current_node=`ls *.mat 2>/dev/null`
    if [ -n $current_node ]; then
	echo $current_node
	cp $main_dir/ViewSelect.lua .
	cp $main_dir/RewardCriterion.lua .
	cp $main_dir/RecurrentAttention_ex.lua .
	cp $main_dir/scripts/train_hierachy_mvrnn.lua .
	cp $main_dir/util/viewsLoc.lua .
	cp $main_dir/model.lua .
	output=`th train_hierachy_mvrnn.lua --dataset $current_node`
	rm ViewSelect.lua
	rm RewardCriterion.lua
	rm RecurrentAttention_ex.lua
	rm train_hierachy_mvrnn.lua
	rm viewsLoc.lua
	rm model.lua
    fi
    for sub_node in `ls`; do
	if [ -d $sub_node ]; then
	    if [ \( ! $sub_node == "mvcnn" \) -a \( ! $sub_node == "cur_model" \)  ] ; then
		    cd $sub_node
		    train_mvrnn_model
		    cd ..
            fi
	fi
    done
}
cd data_hierarchy_tree
train_mvrnn_model
