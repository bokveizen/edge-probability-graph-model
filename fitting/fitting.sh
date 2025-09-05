graph_name_list=(
    "facebook"
    "hamsterster"
    "web-spam"
    "polblogs"
    "bio-CE-PG"
    "bio-SC-HT"
)

option=$1

for graph_name in ${graph_name_list[@]}; do
    if [ "$option" -eq 1 ] || [ "$option" = "all" ]; then
        python CL_iid.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 0.0 --wn 0.0 --float64 --name t1
        # python CL_iid.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 1.0 --wn 0.0 --float64 --name t1w1;
        # python CL_iid.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 0.0 --wn 1.0 --float64 --name t1n1;
        # python CL_iid.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 1.0 --wn 1.0 --float64 --name t1w1n1;
    elif [ "$option" -eq 2 ] || [ "$option" = "all" ]; then
        python CL_iter.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 0.0 --float64 --name t1
        # python CL_iter.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 1.0 --float64 --name t1w1
    elif [ "$option" -eq 3 ] || [ "$option" = "all" ]; then
        python KR_iid.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 0.0 --wn 0.0 --float64 --name t1
        # python KR_iid.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 1.0 --wn 0.0 --float64 --name t1w1;
        # python KR_iid.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 0.0 --wn 1.0 --float64 --name t1n1;
        # python KR_iid.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 1.0 --wn 1.0 --float64 --name t1w1n1;
    elif [ "$option" -eq 4 ] || [ "$option" = "all" ]; then
        python KR_iid_joint.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 0.0 --wn 0.0 --float64 --name t1
        # python KR_iid_joint.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 1.0 --wn 0.0 --float64 --name t1w1;
        # python KR_iid_joint.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 0.0 --wn 1.0 --float64 --name t1n1;
        # python KR_iid_joint.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 1.0 --wn 1.0 --float64 --name t1w1n1;
    elif [ "$option" -eq 5 ] || [ "$option" = "all" ]; then
        python KR_iter.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 0.0 --float64 --name t1
        # python KR_iter.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 1.0 --float64 --name t1w1
    elif [ "$option" -eq 6 ] || [ "$option" = "all" ]; then
        python SBM_iid.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 0.0 --wn 0.0 --float64 --name t1
        # python SBM_iid.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 1.0 --wn 0.0 --float64 --name t1w1;
        # python SBM_iid.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 0.0 --wn 1.0 --float64 --name t1n1;
        # python SBM_iid.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 1.0 --wn 1.0 --float64 --name t1w1n1;
    elif [ "$option" -eq 7 ] || [ "$option" = "all" ]; then
        python SBM_iter.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 0.0 --float64 --name t1
        # python SBM_iter.py --gpu 0 --dataset ${graph_name} --lr 0.01 --ep 10000 --wt 1.0 --ww 1.0 --float64 --name t1w1
    fi
done
