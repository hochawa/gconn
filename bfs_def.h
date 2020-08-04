#ifndef __BFS_DEF_DEFINED
#define __BFS_DEF_DEFINED

// seed: the index of a source vertex

#define BFS_EXEC(xxx,www,yyy,ddd)\
        bfs_init(d_input);\
         T_START\
        bfs_run<0>(seed, *d_input);\
        d_input->max_c = seed;\
        sampling_phase3_all<xxx<yyy,ddd>,www<yyy,ddd>,0, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
        if(d_input->is_sym == 0) {\
                comp_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
		d_input->sym_off = d_input->E;\
                sampling_inv_phase3<xxx<yyy,ddd>,www<yyy,ddd>,0, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
        }\
        CC_GPU(yyy);\
        T_END\
        break;

#define RAND_BFS_EXEC(xxx,www,yyy,ddd)\
				bfs_init(d_input);\
					T_START\
                                        rand_gen<<<CEIL(d_input->E,32*tb_size), tb_size>>>(*d_input, d_input->_Efront);\
		        bfs_run<1>(seed, *d_input);\
		        d_input->max_c = seed;\
                                sampling_phase3_all<xxx<yyy,ddd>,www<yyy,ddd>,1, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
                                if(d_input->is_sym == 0) {\
                                        compL_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
					d_input->sym_off = d_input->E;\
                                        sampling_inv_phase3<xxx<yyy,ddd>,www<yyy,ddd>,1, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
                                }                                                               \
                                CC_GPU(yyy);\
                                T_END\
                                break;

#define SIM_BFS_EXEC(fff,vvv,eee,aaa,rrr,STR) \
        bfs_init(d_input);\
         T_START\
        bfs_run<0>(seed, *d_input);\
        d_input->max_c = seed;\
        union_find_tarjan<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>,sampling_inv_phase3<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>, vvv,eee,aaa,rrr, 1>(d_input->V, *d_input);\
	CC_GPU(find_compress);\
        T_END\
        break;  

#define STER_BFS_EXEC(fff,vvv,eee,aaa,rrr,STR) \
        bfs_init(d_input);\
         T_START\
        bfs_run<0>(seed, *d_input);\
        d_input->max_c = seed;\
        union_find_ster<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>,sampling_inv_phase3<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>, vvv,eee,aaa,rrr, 1>(d_input->V, *d_input);\
	CC_GPU(find_compress);\
        T_END\
        break;  

#define SV_BFS_EXEC(fff,vvv,eee,aaa,rrr,STR) \
        bfs_init(d_input);\
         T_START\
        bfs_run<0>(seed, *d_input);\
        d_input->max_c = seed;\
        union_find_sv<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>,sampling_inv_phase3<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>, vvv,eee,aaa,rrr, 1>(d_input->V, *d_input);\
	CC_GPU(find_compress);\
        T_END\
        break;  

#define LP_BFS_EXEC(fff,vvv,eee,aaa,rrr,STR) \
        bfs_init(d_input);\
         T_START\
        bfs_run<0>(seed, *d_input);\
        d_input->max_c = seed;\
        union_find_lp<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>,sampling_inv_phase3<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>, vvv,eee,aaa,rrr, 1>(d_input->V, *d_input);\
	CC_GPU(find_compress);\
        T_END\
        break;  

#endif
