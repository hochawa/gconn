#ifndef __COO_MACRO_DEFINED
#define __COO_MACRO_DEFINED

	#ifdef CHUNK_STREAMING

#define COO_EXEC(xxx,www,yyy,ddd) if(d_input->is_sym) { \
			T_START\
			for(uintT gran = 0; gran < d_input->E; gran += d_input->chunk_size) {\
				d_input->offset = gran;\
				d_input->size = MIN(d_input->E, gran + d_input->chunk_size) - gran;\
				union_find_gpu_COO_SAMPLE<xxx<yyy,ddd>,www<yyy,ddd>, 0, 0><<<CEIL(d_input->size, tb_size), tb_size>>>(d_input->E, *d_input);\
			}\
			CC_GPU(yyy);\
			T_END\
			} else {\
			T_START\
			for(uintT gran = 0; gran < d_input->E; gran += d_input->chunk_size) {\
				d_input->offset = gran;\
				d_input->size = MIN(d_input->E, gran + d_input->chunk_size) - gran;\
				union_find_gpu_COO_SAMPLE<xxx<yyy,ddd>,www<yyy,ddd>, 1, 0><<<CEIL(d_input->size, tb_size), tb_size>>>(d_input->E, *d_input);\
			}\
				CC_GPU(yyy);\
			}\
			T_END\
			break;

#define SIM_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	for(uintT gran = 0; gran < d_input->E; gran += d_input->chunk_size) {\
				d_input->offset = gran;\
				d_input->size = MIN(d_input->E, gran + d_input->chunk_size) - gran;\
				union_find_tarjan_CHUNK<union_find_gpu_COO_SAMPLE<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, union_find_gpu_COO_SAMPLE<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,vvv,eee,aaa,rrr, 0>(d_input->size, *d_input);\
	}\
	CC_GPU(find_compress);\
	T_END\
	break;	

#define STER_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	for(uintT gran = 0; gran < d_input->E; gran += d_input->chunk_size) {\
				d_input->offset = gran;\
				d_input->size = MIN(d_input->E, gran + d_input->chunk_size) - gran;\
				union_find_ster_CHUNK<union_find_gpu_COO_SAMPLE<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, union_find_gpu_COO_SAMPLE<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,vvv,eee,aaa,rrr, 0>(d_input->size, *d_input);\
	}\
	CC_GPU(find_compress);\
	T_END\
	break;	

#define SV_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	for(uintT gran = 0; gran < d_input->E; gran += d_input->chunk_size) {\
				d_input->offset = gran;\
				d_input->size = MIN(d_input->E, gran + d_input->chunk_size) - gran;\
				union_find_sv_CHUNK<union_find_gpu_COO_SAMPLE<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, union_find_gpu_COO_SAMPLE<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,vvv,eee,aaa,rrr, 0>(d_input->size, *d_input);\
	}\
	CC_GPU(find_compress);\
	T_END\
	break;	

#define LP_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	for(uintT gran = 0; gran < d_input->E; gran += d_input->chunk_size) {\
				d_input->size = MIN(d_input->E, gran + d_input->chunk_size);\
				union_find_lp_CHUNK<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,vvv,eee,aaa,rrr, 0>(d_input->size, *d_input);\
	}\
	CC_GPU(find_compress);\
	T_END\
	break;	

#define RAND_EXEC(xxx,www,yyy,ddd) if(d_input->is_sym) { \
			T_START\
			rand_gen<<<CEIL(d_input->E,32*tb_size), tb_size>>>(*d_input, d_input->_Efront);\
			for(uintT gran = 0; gran < d_input->E; gran += d_input->chunk_size) {\
				d_input->offset = gran;\
				d_input->size = MIN(d_input->E, gran + d_input->chunk_size) - gran;\
				union_find_gpu_COO_SAMPLE<xxx<yyy,ddd>,www<yyy,ddd>, 0, 0><<<CEIL(d_input->size, tb_size), tb_size>>>(d_input->E, *d_input);\
			}\
			CC_GPU(yyy);\
			T_END\
			} else {\
			T_START\
			rand_gen<<<CEIL(grid_size_union,32), tb_size>>>(*d_input, d_input->_Efront);\
				for(uintT gran = 0; gran < d_input->E; gran += d_input->chunk_size) {\
				d_input->offset = gran;\
				d_input->size = MIN(d_input->E, gran + d_input->chunk_size) - gran;\
				union_find_gpu_COO_SAMPLE<xxx<yyy,ddd>,www<yyy,ddd>, 1, 0><<<CEIL(d_input->size, tb_size), tb_size>>>(d_input->E, *d_input);\
			}\
			CC_GPU(yyy);\
			T_END\
			}\
			break;

	#else

#define COO_EXEC(xxx,www,yyy,ddd) if(d_input->is_sym) { \
			T_START\
			union_find_gpu_COO<xxx<yyy,ddd>,www<yyy,ddd>, 0, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				CC_GPU(yyy);\
				T_END\
			} else {\
			T_START\
			union_find_gpu_COO<xxx<yyy,ddd>,www<yyy,ddd>, 1, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				CC_GPU(yyy);\
				T_END\
			}\
				break;


#define SIM_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	union_find_tarjan<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,vvv,eee,aaa,rrr, 0>(d_input->E, *d_input);\
	CC_GPU(find_compress);\
	T_END\
	break;	

#define STER_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	union_find_ster<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,vvv,eee,aaa,rrr, 0>(d_input->E, *d_input);\
	CC_GPU(find_compress);\
	T_END\
	break;	

#define SV_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	union_find_sv<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,vvv,eee,aaa,rrr, 0>(d_input->E, *d_input);\
	CC_GPU(find_compress);\
	T_END\
	break;	

#define LP_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	union_find_lp<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,vvv,eee,aaa,rrr, 0>(d_input->E, *d_input);\
	CC_GPU(find_compress);\
	T_END\
	break;	

#define RAND_EXEC(xxx,www,yyy,ddd) if(d_input->is_sym) { \
			T_START\
			rand_gen<<<CEIL(d_input->E,32*tb_size), tb_size>>>(*d_input, d_input->_Efront);\
			union_find_gpu_COO<xxx<yyy,ddd>,www<yyy,ddd>, 0, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				CC_GPU(yyy);\
				T_END\
			} else {\
			T_START\
			rand_gen<<<CEIL(grid_size_union,32), tb_size>>>(*d_input, d_input->_Efront);\
			union_find_gpu_COO<xxx<yyy,ddd>,www<yyy,ddd>, 1, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				CC_GPU(yyy);\
				T_END\
			}\
				break;

	#endif

#endif
