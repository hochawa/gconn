#ifndef __COO_SAMPLE_DEF_DEFINED
#define __COO_SAMPLE_DEF_DEFINED

#define COO_SAMPLE_EXEC(xxx,www,yyy,ddd) if(d_input->is_sym) { \
			T_START\
			for(uintT gran = 0; gran < d_input->E; gran += d_input->coo_sample_size) {\
				d_input->offset = gran;\
				d_input->size = MIN(d_input->E, gran + d_input->coo_sample_size) - gran;\
				if(gran) edge_relabeling<0><<<CEIL(d_input->size, tb_size), tb_size>>>(d_input->E, *d_input);\
				union_find_gpu_COO_SAMPLE<xxx<yyy,ddd>,www<yyy,ddd>, 0, 0><<<CEIL(d_input->size, tb_size), tb_size>>>(d_input->E, *d_input);\
			}\
			CC_GPU(yyy);\
			T_END\
			} else {\
			T_START\
			for(uintT gran = 0; gran < d_input->E; gran += d_input->coo_sample_size) {\
				d_input->offset = gran;\
				d_input->size = MIN(d_input->E, gran + d_input->coo_sample_size) - gran;\
				if(gran) edge_relabeling<0><<<CEIL(d_input->size, tb_size), tb_size>>>(d_input->E, *d_input);\
				union_find_gpu_COO_SAMPLE<xxx<yyy,ddd>,www<yyy,ddd>, 1, 0><<<CEIL(d_input->size, tb_size), tb_size>>>(d_input->E, *d_input);\
			}\
			}\
			CC_GPU(yyy);\
			T_END\
			break;


#define SIM_COO_SAMPLE_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	union_find_tarjan_COO_SAMPLE<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,vvv,eee,aaa,rrr, 0>(d_input->E, *d_input);\
	CC_GPU(find_compress);\
	T_END\
	break;	


#define STER_COO_SAMPLE_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	union_find_ster_COO_SAMPLE<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,vvv,eee,aaa,rrr, 0>(d_input->E, *d_input);\
	CC_GPU(find_compress);\
	T_END\
	break;	

#define SV_COO_SAMPLE_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	union_find_sv_COO_SAMPLE<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,vvv,eee,aaa,rrr, 0>(d_input->E, *d_input);\
	CC_GPU(find_compress);\
	T_END\
	break;	



#define LP_COO_SAMPLE_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	union_find_lp_COO_SAMPLE<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,vvv,eee,aaa,rrr, 0>(d_input->E, *d_input);\
	CC_GPU(find_compress);\
	T_END\
	break;	


#define RAND_COO_SAMPLE_EXEC(xxx,www,yyy,ddd) if(d_input->is_sym) { \
			T_START\
			rand_gen<<<CEIL(d_input->E,32*tb_size), tb_size>>>(*d_input, d_input->_Efront);\
			for(uintT gran = 0; gran < d_input->E; gran += d_input->coo_sample_size) {\
				d_input->offset = gran;\
				d_input->size = MIN(d_input->E, gran + d_input->coo_sample_size) - gran;\
				if(gran) edge_relabeling<1><<<CEIL(d_input->size, tb_size), tb_size>>>(d_input->E, *d_input);\
				union_find_gpu_COO_SAMPLE<xxx<yyy,ddd>,www<yyy,ddd>, 0, 0><<<CEIL(d_input->size, tb_size), tb_size>>>(d_input->E, *d_input);\
			}\
			CC_GPU(yyy);\
			T_END\
			} else {\
			T_START\
			rand_gen<<<CEIL(grid_size_union,32), tb_size>>>(*d_input, d_input->_Efront);\
				for(uintT gran = 0; gran < d_input->E; gran += d_input->coo_sample_size) {\
				d_input->offset = gran;\
				d_input->size = MIN(d_input->E, gran + d_input->coo_sample_size) - gran;\
				if(gran) edge_relabeling<1><<<CEIL(d_input->size, tb_size), tb_size>>>(d_input->E, *d_input);\
				union_find_gpu_COO_SAMPLE<xxx<yyy,ddd>,www<yyy,ddd>, 1, 0><<<CEIL(d_input->size, tb_size), tb_size>>>(d_input->E, *d_input);\
			}\
			CC_GPU(yyy);\
			T_END\
			}\
			break;

#endif
