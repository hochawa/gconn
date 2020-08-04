#ifndef __CSR_DEF_DEFINED
#define __CSR_DEF_DEFINED


#define CSR_EXEC(xxx,www,yyy,ddd) if(d_input->is_sym) {\
			 T_START\
			union_find_gpu_CSR<xxx<yyy,ddd>,www<yyy,ddd>, 0, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				CC_GPU(yyy);\
				T_END\
				} else {\
			 T_START\
			union_find_gpu_CSR<xxx<yyy,ddd>,www<yyy,ddd>, 1, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				CC_GPU(yyy);\
				T_END\
			}\
				break;

#define SIM_CSR_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	union_find_tarjan<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, vvv,eee,aaa,rrr,0>(d_input->V, *d_input);\
	CC_GPU(find_compress);\
	T_END\
	break;	

#define STER_CSR_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	union_find_ster<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, vvv,eee,aaa,rrr,0>(d_input->V, *d_input);\
	CC_GPU(find_compress);\
	T_END\
	break;	

#define SV_CSR_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	union_find_sv<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, vvv,eee,aaa,rrr,0>(d_input->V, *d_input);\
	CC_GPU(find_compress);\
	T_END\
	break;	

#define LP_CSR_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
	union_find_lp<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>,STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 1, 1>, vvv,eee,aaa,rrr,0>(d_input->V, *d_input);\
	CC_GPU(find_compress);\
	T_END\
	break;	


#define RAND_CSR_EXEC(xxx,www,yyy,ddd) if(d_input->is_sym) {\
			 T_START\
			rand_gen<<<CEIL(d_input->E,32*tb_size), tb_size>>>(*d_input, d_input->_Efront);\
			union_find_gpu_CSR<xxx<yyy,ddd>,www<yyy,ddd>, 0, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				CC_GPU(yyy);\
				T_END\
				} else {\
			 T_START\
			rand_gen<<<CEIL(d_input->E,32*tb_size), tb_size>>>(*d_input, d_input->_Efront);\
			union_find_gpu_CSR<xxx<yyy,ddd>,www<yyy,ddd>, 1, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				CC_GPU(yyy);\
				T_END\
			}\
				break;

#endif
