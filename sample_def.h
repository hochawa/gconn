#ifndef __SAMPLE_DEF_DEFINED
#define __SAMPLE_DEF_DEFINED

#ifdef FUS_SAMPLE


#define SAMPLE_EXEC(xxx,www,yyy,ddd)    T_START\
                                        cudaMemset(qp, 0, sizeof(uintT)*4);\
                                        sampling_phase1_fusionX<xxx<yyy,ddd>,www<yyy,ddd>><<<grid_size_final, tb_size>>>(tot_elt, *d_input);\
                                        comp_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
                                 d_input->max_c = find_largest_component<0>(d_input->V, d_input->sample_size, d_input->parent, d_input->lparent, sample, _sample);\
                                gen_ff<<<grid_size_final, tb_size>>>(d_input->V, *d_input, q, qp);\
                                cudaMemcpy(&hqp, qp, sizeof(uintT), cudaMemcpyDeviceToHost);\
                                sampling_phase33<xxx<yyy,ddd>,www<yyy,ddd>,0, 0><<<CEIL(hqp*4, tb_size), tb_size>>>(tot_elt, q, hqp, *d_input);\
                                if(d_input->is_sym == 0) {\
                                        comp_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
                                        d_input->sym_off = d_input->E;\
                                        sampling_inv_phase3<xxx<yyy,ddd>,www<yyy,ddd>,0, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
                                }                                                               \
                                CC_GPU(yyy);\
                                T_END\
                                break;

#define SAMPLE_EXEC_SUBOPT(xxx,www,yyy,ddd)   	T_START\
					sampling_phase1_fusionX<xxx<yyy,ddd>,www<yyy,ddd>><<<grid_size_final, tb_size>>>(tot_elt, *d_input);\
					comp_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
				 d_input->max_c = find_largest_component<0>(d_input->V, d_input->sample_size, d_input->parent, d_input->lparent, sample, _sample);\
				sampling_phase3<xxx<yyy,ddd>,www<yyy,ddd>,0, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				if(d_input->is_sym == 0) {\
					comp_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
					d_input->sym_off = d_input->E;\
					sampling_inv_phase3<xxx<yyy,ddd>,www<yyy,ddd>,0, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				}								\
				CC_GPU(yyy);\
				T_END\
				break;



#define SAMPLE_EXEC0(xxx,www,yyy,ddd) T_START\
                                for(int i=0;i<d_input->sample_k;i++) {\
                                        sampling_phase1<xxx<yyy,ddd>,www<yyy,ddd>><<<grid_size_final, tb_size>>>(tot_elt, *d_input, i);\
                                        comp_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
                                }\
				 d_input->max_c = find_largest_component<0>(d_input->V, d_input->sample_size, d_input->parent, d_input->lparent, sample, _sample);\
				sampling_phase3<xxx<yyy,ddd>,www<yyy,ddd>,0, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				if(d_input->is_sym == 0) {\
					comp_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
					d_input->sym_off = d_input->E;\
					sampling_inv_phase3<xxx<yyy,ddd>,www<yyy,ddd>,0, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				}								\
				CC_GPU(yyy);\
				T_END\
				break;



#define RAND_SAMPLE_EXEC(xxx,www,yyy,ddd) T_START\
					rand_gen<<<CEIL(d_input->E,32*tb_size), tb_size>>>(*d_input, d_input->_Efront);\
					sampling_phase1_fusion<xxx<yyy,ddd>,www<yyy,ddd>><<<grid_size_final*2, tb_size>>>(tot_elt, *d_input);\
					compL_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
				d_input->max_c = find_largest_component<1>(d_input->V, d_input->sample_size, d_input->parent, d_input->lparent, sample, _sample);\
				sampling_phase3_all<xxx<yyy,ddd>,www<yyy,ddd>,1, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				if(d_input->is_sym == 0) {\
					compL_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
					d_input->sym_off = d_input->E;\
					sampling_inv_phase3<xxx<yyy,ddd>,www<yyy,ddd>,1, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				}								\
				CC_GPU(yyy);\
				T_END\
				break;

#define SIM_SAMPLE_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
				sampling_phase1_fusion<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>><<<grid_size_final*2, tb_size>>>(tot_elt, *d_input);\
				comp_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
				 d_input->max_c = find_largest_component<0>(d_input->V, d_input->sample_size, d_input->parent, d_input->lparent, sample, _sample);\
        union_find_tarjan<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>,sampling_inv_phase3<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>, vvv,eee,aaa,rrr, 1>(d_input->V, *d_input);\
	CC_GPU(find_compress);\
        T_END\
        break;  


#define STER_SAMPLE_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
                		cudaMemcpy(&d_input->parent[d_input->V], &d_input->parent[0], sizeof(uintT) * d_input->V, cudaMemcpyDeviceToDevice);\
				sampling_phase1_fusion<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>><<<grid_size_final*2, tb_size>>>(tot_elt, *d_input);\
				comp_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
				 d_input->max_c = find_largest_component<0>(d_input->V, d_input->sample_size, d_input->parent, d_input->lparent, sample, _sample);\
        union_find_ster<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>,sampling_inv_phase3<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>, vvv,eee,aaa,rrr, 1>(d_input->V, *d_input);\
	CC_GPU(find_compress);\
        T_END\
        break;  



#define SV_SAMPLE_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
				sampling_phase1_fusion<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>><<<grid_size_final*2, tb_size>>>(tot_elt, *d_input);\
				comp_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
				 d_input->max_c = find_largest_component<0>(d_input->V, d_input->sample_size, d_input->parent, d_input->lparent, sample, _sample);\
        union_find_sv<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>,sampling_inv_phase3<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>, vvv,eee,aaa,rrr, 1>(d_input->V, *d_input);\
	CC_GPU(find_compress);\
        T_END\
        break;  

#define LP_SAMPLE_EXEC(fff,vvv,eee,aaa,rrr,STR) T_START\
				sampling_phase1_fusion<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>><<<grid_size_final*2, tb_size>>>(tot_elt, *d_input);\
				comp_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
				 d_input->max_c = find_largest_component<0>(d_input->V, d_input->sample_size, d_input->parent, d_input->lparent, sample, _sample);\
        union_find_lp<STR<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>,sampling_inv_phase3<blank_fun<find_naive,blank_fun2>,fff<find_naive,blank_fun2>, 0, 1>, vvv,eee,aaa,rrr, 1>(d_input->V, *d_input);\
	CC_GPU(find_compress);\
        T_END\
        break;  


#endif


#ifdef NOT_FUS_SAMPLE
#define SAMPLE_EXEC(xxx,www,yyy,ddd) T_START\
                                for(int i=0;i<d_input->sample_k;i++) {\
                                        sampling_phase1<xxx<yyy,ddd>,www<yyy,ddd>><<<grid_size_final, tb_size>>>(tot_elt, *d_input, i);\
                                        comp_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
                                }\
				 d_input->max_c = find_largest_component<0>(d_input->V, d_input->sample_size, d_input->parent, d_input->lparent, sample, _sample);\
				sampling_phase3<xxx<yyy,ddd>,www<yyy,ddd>,0, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				if(d_input->is_sym == 0) {\
					comp_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
					sampling_inv_phase3<xxx<yyy,ddd>,www<yyy,ddd>,0, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				}								\
				CC_GPU(yyy);\
				T_END\
				break;

#define RAND_SAMPLE_EXEC(xxx,www,yyy,ddd) T_START\
					rand_gen<<<CEIL(d_input->E,32*tb_size), tb_size>>>(*d_input, d_input->_Efront);\
                                for(int i=0;i<d_input->sample_k;i++) {\
                                        sampling_phase1<xxx<yyy,ddd>,www<yyy,ddd>><<<grid_size_final, tb_size>>>(tot_elt, *d_input, i);\
                                        compL_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
                                }\
				 d_input->max_c = find_largest_component<1>(d_input->V, d_input->sample_size, d_input->parent, d_input->lparent, sample, _sample);\
				sampling_phase3<xxx<yyy,ddd>,www<yyy,ddd>,1, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				if(d_input->is_sym == 0) {\
					compL_tree<<<grid_size_final, tb_size>>>(d_input->V, *d_input);\
					sampling_inv_phase3<xxx<yyy,ddd>,www<yyy,ddd>,1, 0><<<grid_size_union, tb_size>>>(tot_elt, *d_input);\
				}\
				CC_GPU(yyy);\
				T_END\
				break;
#endif


#endif
