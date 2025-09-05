# for each "model"
# for each "*.txt" in the subfolder "../fit_res/from_scratch/{model}"

n_graphs=100

n="1024"
p="0.01"
g_list=(
    # "0p0"
    "0p01"
    "0p02"
    "0p03"
    "0p04"
    "0p05"
    "0p06"
    "0p07"
    "0p08"
    "0p09"
    "0p1"
)

alpha_list=(
    "-0p5"
    "0"
    "0p5"
)

R="100000"

# SBM
p_fitting_res="../fit_res/from_scratch/SBM"

# SBM-iid
p_generation_res="../gen_res/from_scratch/SBM_iid"
mkdir -p ${p_generation_res}

for g in "${g_list[@]}"; do
    for alpha in "${alpha_list[@]}"; do
        graph_name="n${n}_p${p}_g${g}_alpha${alpha}_R${R}.txt"
        p_save="${p_generation_res}/${g}/${alpha}"
        mkdir -p ${p_save}
        ./gen_iid_sbm "${p_fitting_res}/${graph_name}" "${p_save}/res" "${n_graphs}"
    done
done
