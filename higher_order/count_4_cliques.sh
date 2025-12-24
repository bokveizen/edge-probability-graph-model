g++ -O3 -fopenmp count_4_cliques.cpp -o count_4_cliques;

for dataset in facebook hamsterster polblogs web-spam bio-CE-PG bio-SC-HT; do
    echo "Counting 4-cliques for ${dataset}...";
    ./count_4_cliques ../data/${dataset}.txt;
done