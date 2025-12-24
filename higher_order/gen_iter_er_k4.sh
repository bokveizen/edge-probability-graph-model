graph_name_list=(
    "facebook"
    "hamsterster"
    "web-spam"
    "polblogs"
    "bio-CE-PG"
    "bio-SC-HT"
)

# input
p_fitting_res="../fit_res_k4/ER_iter"

# ouput
p_generation_res="../gen_res_k4/ER_iter"
mkdir -p ${p_generation_res}

n_graphs=100

for graph_name in "${graph_name_list[@]}"; do
    mkdir -p "${p_generation_res}/${graph_name}"
    ./gen_iter_er "${p_fitting_res}/${graph_name}.txt" "${p_generation_res}/${graph_name}/res" "${n_graphs}" 0
done
