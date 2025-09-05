# for each "model"
# for each "*.txt" in the subfolder "../fit_res/from_scratch/{model}"

n_graphs=100

n="1024"
p="0p01"
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
R="100000"

# ER
p_fitting_res="../fit_res/from_scratch/ER_iid"

# ER-iid
p_generation_res="../gen_res/from_scratch/ER_iid"
mkdir -p ${p_generation_res}

for g in "${g_list[@]}"; do
    graph_name="n${n}_p${p}_g${g}_R${R}.txt"
    p_save="${p_generation_res}/${g}"
    mkdir -p ${p_save}
    ./gen_iid_er "${p_fitting_res}/${graph_name}" "${p_save}/res" "${n_graphs}"
done
