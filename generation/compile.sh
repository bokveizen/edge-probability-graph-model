compile_program=$1
if [ -z "$compile_program" ]; then
    compile_program="ALL"
fi

if [ "$compile_program" = "ALL" ] || [ "$compile_program" = "iid_er" ]; then
    g++ gen_iid_er.cpp -o gen_iid_er -O2 -fopenmp;
fi

if [ "$compile_program" = "ALL" ] || [ "$compile_program" = "iter_er" ]; then
    g++ gen_iter_er.cpp -o gen_iter_er -O2 -fopenmp;
fi

if [ "$compile_program" = "ALL" ] || [ "$compile_program" = "iid_cl" ]; then
    g++ gen_iid_cl.cpp -o gen_iid_cl -O2 -fopenmp;
fi

if [ "$compile_program" = "ALL" ] || [ "$compile_program" = "iter_cl" ]; then
    g++ gen_iter_cl.cpp -o gen_iter_cl -O2 -fopenmp;
fi

if [ "$compile_program" = "ALL" ] || [ "$compile_program" = "iid_kr" ]; then
    g++ gen_iid_kr.cpp -o gen_iid_kr -O2 -fopenmp;
fi

if [ "$compile_program" = "ALL" ] || [ "$compile_program" = "iter_kr" ]; then
    g++ gen_iter_kr.cpp -o gen_iter_kr -O2 -fopenmp;
fi

if [ "$compile_program" = "ALL" ] || [ "$compile_program" = "iid_sbm" ]; then
    g++ gen_iid_sbm.cpp -o gen_iid_sbm -O2 -fopenmp;
fi

if [ "$compile_program" = "ALL" ] || [ "$compile_program" = "iter_sbm" ]; then
    g++ gen_iter_sbm.cpp -o gen_iter_sbm -O2 -fopenmp;
fi

if [ "$compile_program" = "ALL" ] || [ "$compile_program" = "graph_analy" ]; then
    g++ graph_analy.cpp -o graph_analy -O2 -fopenmp;
fi
