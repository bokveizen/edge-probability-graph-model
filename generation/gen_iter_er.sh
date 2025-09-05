graph_name_list=(
    "facebook"
    "hamsterster"
    "web-spam"
    "polblogs"
    "bio-CE-PG"
    "bio-SC-HT"
)

# input
p_fitting_res="../fit_res/ER_iter"

res_dir_list=(
    "triangle"    
)

# ouput
p_generation_res="../gen_res/ER_iter"
mkdir -p ${p_generation_res}

n_graphs=100

for graph_name in "${graph_name_list[@]}"; do
    for res_dir in "${res_dir_list[@]}"; do
        mkdir -p "${p_generation_res}/${res_dir}"
        ./gen_iter_er "${p_fitting_res}/${res_dir}/${graph_name}.txt" "${p_generation_res}/${res_dir}/${graph_name}" "${n_graphs}" 0
    done
done
