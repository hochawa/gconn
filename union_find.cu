#ifndef __UNION_DEFINED
#define __UNION_DEFINED


#include "tarjan.h"
#include "LP.h"
#include "SV.h"
#include "stergiou.h"
#include "kernel.h"
#include "sample.h"
#include "bfs.h"
#include "coo_def.h"
#include "coo_sample_def.h"
#include "csr_def.h"

__global__ void get_next_vertex(uintT prev_value, uintT V, uintT *parent, uintT *start)
{
        uintT local_start = UNINIT;
        __shared__ uintT sm_start;
        if(threadIdx.x == 0) 
                sm_start = UNINIT;
        __syncthreads();
        for(uintT i=prev_value+blockIdx.x*blockDim.x + threadIdx.x; i<V; i+=gridDim.x * blockDim.x) {
                if(local_start == UNINIT && parent[i] == i) {local_start = i;}
        }
	__syncwarp();
	uintT tmp_local = __shfl_down_sync(-1, local_start, 16);
	if(tmp_local < local_start) local_start = tmp_local;
	__syncwarp();
 
	tmp_local = __shfl_down_sync(-1, local_start, 8);
	if(tmp_local < local_start) local_start = tmp_local;
	__syncwarp();
  
	tmp_local = __shfl_down_sync(-1, local_start, 4);
	if(tmp_local < local_start) local_start = tmp_local;
	__syncwarp();

	tmp_local = __shfl_down_sync(-1, local_start, 2);
	if(tmp_local < local_start) local_start = tmp_local;
	__syncwarp();

	tmp_local = __shfl_down_sync(-1, local_start, 1);
	if(tmp_local < local_start) local_start = tmp_local;
	__syncwarp();
 
	if(threadIdx.x % 32 == 0) {
        atomicMin(&sm_start, local_start);
	}
        __syncthreads();
	if(threadIdx.x == 0)
        atomicMin(start, sm_start);
}

template <int is_long>
__global__ void init_parent_arr(struct graph_data d_input)
{
	uintT idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < d_input.V) {
		if(is_long == 0) d_input.parent[idx] = idx;
		else d_input.lparent[idx] = (ULONG_T_MAX | (ulongT) idx);
	}
}

template <int is_long>
__global__ void init_parent_arr2(struct graph_data d_input)
{
	uintT idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < d_input.V) {
		if(is_long == 0) {
			uintT v = idx;
			if (d_input.dst_idx[d_input.csr_ptr[idx]] < v && d_input.csr_ptr[idx] < d_input.csr_ptr[idx+1]) {
				v = d_input.dst_idx[d_input.csr_ptr[idx]];
				#if defined(SP_TREE)
                                d_input.hook[idx] = d_input.csr_ptr[idx] + d_input.sym_off;
				#endif
			}
			else { 
				uintT ptr = atomicAggInc(d_input.queuep);
				d_input.queue[ptr] = idx;
			}
			d_input.parent[idx] = v;
		}
		else d_input.lparent[idx] = (ULONG_T_MAX | (ulongT) idx);
	}
}


void init_gpu(struct graph_data *d_input)
{
        if(d_input->algo == RAND_NAIVE || d_input->algo == RAND_SPLIT_2) {
		#if defined(SCC) || defined(SP_TREE)
		init_parent_arr<1><<<CEIL(d_input->V, SBSIZE),SBSIZE>>>(*d_input);
		#endif
		#if defined(STREAMING_SIMPLE)
		cudaMemset(d_input->lparent, -1, sizeof(ulongT)*(d_input->V));
		#endif

	} else {
		#if defined(SCC) || defined(SP_TREE)
		init_parent_arr<0><<<CEIL(d_input->V, SBSIZE),SBSIZE>>>(*d_input);
		#endif
		#if defined(STREAMING_SIMPLE)
		cudaMemset(d_input->parent, -1, sizeof(uintT)*(d_input->V));
		#endif
		if(d_input->two_parent) 
			cudaMemset(&d_input->parent[d_input->V], 0, sizeof(uintT)*(d_input->V));
	}
	
	#if defined(STREAMING) || defined(STREAMING_SYNC) || defined(STREAMING_SIMPLE)
	cudaMemset(d_input->bin_p1, -1, sizeof(uintT)*32);
	cudaMemset(d_input->bin_p2, -1, sizeof(uintT)*32);
	#endif

}

void init_gpu2(struct graph_data *d_input)
{
        if(d_input->algo == RAND_NAIVE || d_input->algo == RAND_SPLIT_2) {
		#if defined(SCC) || defined(SP_TREE)
		init_parent_arr<1><<<CEIL(d_input->V, SBSIZE),SBSIZE>>>(*d_input);
		#endif
		#if defined(STREAMING_SIMPLE)
		cudaMemset(d_input->lparent, -1, sizeof(ulongT)*(d_input->V));
		#endif

	} else {
		#if defined(SCC) || defined(SP_TREE)
		init_parent_arr2<0><<<CEIL(d_input->V, SBSIZE),SBSIZE>>>(*d_input);
		#endif
		#if defined(STREAMING_SIMPLE)
		cudaMemset(d_input->parent, -1, sizeof(uintT)*(d_input->V));
		#endif
		//need to be eliminated
		cudaMemset(&d_input->parent[d_input->V], 0, sizeof(uintT)*(d_input->V));
	}
	
	#if defined(STREAMING) || defined(STREAMING_SYNC) || defined(STREAMING_SIMPLE)
	cudaMemset(d_input->bin_p1, -1, sizeof(uintT)*32);
	cudaMemset(d_input->bin_p2, -1, sizeof(uintT)*32);
	#endif

}



void copy_cpu_to_gpu(struct graph_data *h_input, struct graph_data *d_input)
{
	d_input->V = h_input->V;
	d_input->E = h_input->E;
	d_input->format = h_input->format;
	d_input->algo = h_input->algo;
	d_input->is_sym = h_input->is_sym;
	d_input->sample_k = h_input->sample_k;
	d_input->sample_size = h_input->sample_size;
	d_input->size = d_input->E;

	if(d_input->algo == STERGIOS || d_input->algo == SVA) {
		d_input->two_parent = 1;
	} else {
		d_input->two_parent = 0;
	}

        cudaMalloc((void **)&(d_input->dst_idx), sizeof(uintT)*(h_input->E));
 	cudaMalloc((void **)&(d_input->label), sizeof(uintT)*(h_input->V));
	cudaMalloc((void **)&(d_input->hook), sizeof(uintT)*(h_input->V));

	cudaMemcpy(d_input->dst_idx, h_input->dst_idx, sizeof(uintT)*(h_input->E), cudaMemcpyHostToDevice);
	cudaMemcpy(d_input->label, h_input->label, sizeof(uintT)*(h_input->V), cudaMemcpyHostToDevice);
	cudaMemcpy(d_input->hook, h_input->hook, sizeof(uintT)*(h_input->V), cudaMemcpyHostToDevice);

        if(d_input->algo == RAND_NAIVE || d_input->algo == RAND_SPLIT_2) {
		cudaMalloc((void **)&(d_input->lparent), sizeof(ulongT)*(h_input->V));

	} else {
		cudaMalloc((void **)&(d_input->parent), sizeof(uintT)*(h_input->V)*2);
	}
	
	if(h_input->format == COO || h_input->format == COO_SAMPLE) {	
		cudaMalloc((void **)&(d_input->src_idx), sizeof(uintT)*(h_input->E));
		cudaMemcpy(d_input->src_idx, h_input->src_idx, sizeof(uintT)*(h_input->E), cudaMemcpyHostToDevice);
	}
	if(h_input->format == CSR || h_input->format == SAMPLE || h_input->format == BFS || h_input->format == IHOOK) {
		cudaMalloc((void **)&(d_input->csr_ptr), sizeof(uintT)*((h_input->V)+1));
        	cudaMemcpy(d_input->csr_ptr, h_input->csr_ptr, sizeof(uintT)*((h_input->V)+1), cudaMemcpyHostToDevice);
		if((h_input->format == SAMPLE || h_input->format == BFS || h_input->format == IHOOK) && h_input->is_sym == 0) {
			cudaMalloc((void **)&(d_input->csr_inv_ptr), sizeof(uintT)*((h_input->V)+1));
		        cudaMalloc((void **)&(d_input->dst_inv_idx), sizeof(uintT)*(h_input->E));
       		 	cudaMemcpy(d_input->csr_inv_ptr, h_input->csr_inv_ptr, sizeof(uintT)*((h_input->V)+1), cudaMemcpyHostToDevice);
			cudaMemcpy(d_input->dst_inv_idx, h_input->dst_inv_idx, sizeof(uintT)*(h_input->E), cudaMemcpyHostToDevice);
		}
	}
	d_input->max_c = 0;

	#if defined(STREAMING) || defined(STREAMING_SYNC) || defined(STREAMING_SIMPLE)
	cudaMalloc((void **)&d_input->bin_p1, sizeof(uintT)*32);
	cudaMalloc((void **)&d_input->bin_p2, sizeof(uintT)*32);
	#endif

	#if defined(PATH_LENGTH)
	cudaMalloc((void **)&d_input->path_length, sizeof(uintT)*PATH_SIZE);
	cudaMalloc((void **)&d_input->path_max, sizeof(uintT)*PATH_SIZE);
	cudaMalloc((void **)&d_input->length, sizeof(uintT)*PATH_SIZE);
	cudaMemset(d_input->path_length, 0, sizeof(uintT)*PATH_SIZE);
	cudaMemset(d_input->path_max, 0, sizeof(uintT)*PATH_SIZE);
	cudaMemset(d_input->length, 0, sizeof(uintT)*PATH_SIZE);
	#endif

}

void copy_gpu_to_cpu(struct graph_data *h_input, struct graph_data *d_input)
{
	memset(h_input->label, -1, sizeof(uintT)*(h_input->V));
	cudaMemcpy(h_input->label, d_input->label, sizeof(uintT)*(h_input->V), cudaMemcpyDeviceToHost);

	cudaMemcpy(h_input->hook, d_input->hook, sizeof(uintT)*(h_input->V), cudaMemcpyDeviceToHost);
 
	if(d_input->algo == RAND_NAIVE || d_input->algo == RAND_SPLIT_2) {
		cudaMemcpy(h_input->lparent, d_input->lparent, sizeof(ulongT)*(h_input->V), cudaMemcpyDeviceToHost);
	} else {
		cudaMemcpy(h_input->parent, d_input->parent, sizeof(uintT)*(h_input->V), cudaMemcpyDeviceToHost);
	}



	int cnt=0;
	int largest_cc=0;
	for(int i=0;i<h_input->V;i++) {
		if(h_input->label[i] == i) {
			cnt++;
		}
//#define MEASURE_LARGEST_CC
#ifdef MEASURE_LARGEST_CC
		if(h_input->label[i] == d_input->max_c) {
			largest_cc++;
		}
#endif
	}
#ifdef MEASURE_LARGEST_CC
		printf("%d", largest_cc);
		exit(0);
#endif


	h_input->cc_cnt = cnt;
#ifdef PRINT
	fprintf(stderr,"%d,\n", cnt);
#endif

	// free arrays on GPUs

	cudaFree(d_input->dst_idx);
	cudaFree(d_input->label);
	cudaFree(d_input->hook);
	
        if(d_input->algo == RAND_NAIVE || d_input->algo == RAND_SPLIT_2) {
		cudaFree(d_input->lparent);

        } else {
		cudaFree(d_input->parent);
        }

        if(h_input->format == COO || h_input->format == COO_SAMPLE) {
		cudaFree(d_input->src_idx);
        }
        if(h_input->format == CSR || h_input->format == SAMPLE || h_input->format == BFS || h_input->format == IHOOK) {
		cudaFree(d_input->csr_ptr);
                if((h_input->format == SAMPLE || h_input->format == BFS || h_input->format == IHOOK) && h_input->is_sym == 0) {
			cudaFree(d_input->csr_inv_ptr);
			cudaFree(d_input->dst_inv_idx);
                }
        }

        #if defined(STREAMING) || defined(STREAMING_SYNC) || defined(STREAMING_SIMPLE)
	cudaFree(d_input->bin_p1);
	cudaFree(d_input->bin_p2);
        #endif

	if(h_input->format == BFS) {
		cudaFree(d_input->bflag);
		cudaFree(d_input->bflag2);
		cudaFree(d_input->curr_f);
		cudaFree(d_input->_fsize);
		cudaFree(d_input->sum_deg);
	}
}

double process(struct graph_data *d_input)
{
	#if defined(STREAMING_SYNC)
	d_input->iter = 1;
	#endif

	int tb_size = SBSIZE;
	int grid_size_union;
	int grid_size_final = CEIL(d_input->V, tb_size);
	uintT mc;

        float tot_ms;
        cudaEvent_t event1, event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);

	cudaMalloc((void **)&d_input->_Efront, sizeof(uintT)*CEIL(d_input->E,32)+1024);

	long *txtx; uintT *pat; // temp arrays
	d_input->sym_off = 0;

        int dev;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        int mp = deviceProp.multiProcessorCount*(1024/SBSIZE); // occ = 0.5
        //int mp = deviceProp.multiProcessorCount*(512/SBSIZE); // occ = 0.25

	void * args[9];
        args[0] = (void *)&(d_input->E);
        args[1] = (void *)&(*d_input);

	if(d_input->format == COO) {	

		grid_size_union = CEIL(d_input->E, tb_size);
		uintT tot_elt = d_input->E;		
		cudaMalloc((void **)&d_input->_Vflag, sizeof(uintT));
		cudaMalloc((void **)&d_input->_Eflag, sizeof(uintT));
		cudaMalloc((void **)&d_input->_Vfront, sizeof(uintT)*CEIL(d_input->V,32));
		cudaMalloc((void **)&d_input->_Rfront, sizeof(uintT)*CEIL(d_input->V,32));

		switch(d_input->algo) {
			case ASYNC_NAIVE:
				COO_EXEC(blank_fun, union_async, find_naive, blank_fun2);

			case ASYNC_COMPRESS:
				COO_EXEC(blank_fun, union_async, find_compress, blank_fun2);

			case ASYNC_HALVE:
				COO_EXEC(blank_fun, union_async,find_halve, blank_fun2);

			case ASYNC_SPLIT:
				COO_EXEC(blank_fun, union_async, find_split, blank_fun2);

			case ASYNC_A_HALVE:
				COO_EXEC(blank_fun, union_async,find_a_halve, blank_fun2);

			case ASYNC_A_SPLIT:
				COO_EXEC(blank_fun, union_async,find_a_split, blank_fun2);


			case STOPT_NAIVE:
				COO_EXEC(blank_fun, union_stopt, find_naive, blank_fun2);

			case STOPT_COMPRESS:
				COO_EXEC(blank_fun, union_stopt, find_compress, blank_fun2);

			case STOPT_HALVE:
				COO_EXEC(blank_fun, union_stopt,find_halve, blank_fun2);

			case STOPT_SPLIT:
				COO_EXEC(blank_fun, union_stopt, find_split, blank_fun2);

			case STOPT_A_HALVE:
				COO_EXEC(blank_fun, union_stopt,find_a_halve, blank_fun2);

			case STOPT_A_SPLIT:
				COO_EXEC(blank_fun, union_stopt,find_a_split, blank_fun2);


			case EARLY_NAIVE:
				COO_EXEC(blank_fun, union_early, find_naive, blank_fun2);
			case EARLY_COMPRESS:
				COO_EXEC(blank_fun, union_early, find_compress, blank_fun2);
			case EARLY_A_HALVE:
				COO_EXEC(blank_fun, union_early, find_a_halve, blank_fun2);
			case EARLY_A_SPLIT:
				COO_EXEC(blank_fun, union_early, find_a_split, blank_fun2);

			case ECL_NAIVE:
				COO_EXEC(ecl_fun, union_ecl, find_naive, blank_fun2);

			case ECL_COMPRESS:
				COO_EXEC(ecl_fun, union_ecl,find_compress, blank_fun2);

			case ECL_HALVE:
				COO_EXEC(ecl_fun, union_ecl,find_halve, blank_fun2);

			case ECL_SPLIT:
				COO_EXEC(ecl_fun, union_ecl, find_split, blank_fun2);

			case ECL_A_HALVE:
				COO_EXEC(ecl_fun, union_ecl,find_a_halve, blank_fun2);

			case ECL_A_SPLIT:
				COO_EXEC(ecl_fun, union_ecl,find_a_split, blank_fun2);

			case AFFOREST_NAIVE:
				COO_EXEC(blank_fun, union_afforest, find_naive, blank_fun2);

			case AFFOREST_COMPRESS:
				COO_EXEC(blank_fun, union_afforest,find_compress, blank_fun2);

			case AFFOREST_HALVE:
				COO_EXEC(blank_fun, union_afforest,find_halve, blank_fun2);

			case AFFOREST_SPLIT:
				COO_EXEC(blank_fun, union_afforest, find_split, blank_fun2);

			case AFFOREST_A_HALVE:
				COO_EXEC(blank_fun, union_afforest,find_a_halve, blank_fun2);

			case AFFOREST_A_SPLIT:
				COO_EXEC(blank_fun, union_afforest,find_a_split, blank_fun2);

			case SIM_C_U_S_A:
				SIM_EXEC(sim_c_u_a, 0, 1, 1, 0, union_find_gpu_COO);
			case SIM_C_R_S_A:
				SIM_EXEC(sim_c_r_a, 0, 1, 1, 1, union_find_gpu_COO);
			case SIM_P_U_S_A:
				SIM_EXEC(sim_p_u_a, 0, 1, 1, 0, union_find_gpu_COO);
			case SIM_P_R_S_A:
				SIM_EXEC(sim_p_r_a, 0, 1, 1, 1, union_find_gpu_COO);

			case SIM_P_U_S:
				SIM_EXEC(sim_p_u, 0, 0, 0, 0, union_find_gpu_COO);
			case SIM_P_R_S:
				SIM_EXEC(sim_p_r, 0, 0, 0, 1, union_find_gpu_COO);
			case SIM_E_U_S_A:
				SIM_EXEC(sim_e_u_a, 0, 1, 1, 0, union_find_gpu_COO);
			case SIM_E_U_S:
				SIM_EXEC(sim_e_u, 0, 0, 0, 0, union_find_gpu_COO);

			case SIM_C_U_SS_A:
				SIM_EXEC(sim_c_u_a, 1, 1, 1, 1, union_find_gpu_COO);
			case SIM_C_R_SS_A:
				SIM_EXEC(sim_c_r_a, 1, 1, 1, 1, union_find_gpu_COO);
			case SIM_P_U_SS_A:
				SIM_EXEC(sim_p_u_a, 1, 1, 1, 0, union_find_gpu_COO);
			case SIM_P_R_SS_A:
				SIM_EXEC(sim_p_r_a, 1, 1, 1, 1, union_find_gpu_COO);

			case SIM_P_U_SS:
				SIM_EXEC(sim_p_u, 1, 0, 0, 0, union_find_gpu_COO);
			case SIM_P_R_SS:
				SIM_EXEC(sim_p_r, 1, 1, 0, 1, union_find_gpu_COO);
			case SIM_E_U_SS_A: 
				SIM_EXEC(sim_e_u_a, 1, 1, 1, 0, union_find_gpu_COO);
			case SIM_E_U_SS:
				SIM_EXEC(sim_e_u, 1, 0, 0, 0, union_find_gpu_COO);

			case STERGIOS:
				STER_EXEC(ster, 0, 0, 0, 0, union_find_gpu_COO);
			case SVA:
				SV_EXEC(sv, 0, 0, 0, 0, union_find_gpu_COO);
			case LPA:
				LP_EXEC(lp, 0, 0, 0, 0, union_find_gpu_COO);

			case RAND_NAIVE:
				RAND_EXEC(blank_fun, union_rand, find_r_naive, blank_fun2);

			case RAND_SPLIT_2:
				RAND_EXEC(blank_fun, union_rand, find_r_split_2, blank_fun2);


			case REMCAS_NAIVE_SPLICE_CAS:
				COO_EXEC(blank_fun, union_rem_fence, find_naive, splice_CAS);

			case REMCAS_COMPRESS_SPLICE_CAS:
				COO_EXEC(blank_fun, union_rem_fence, find_compress, splice_CAS);

			case REMCAS_HALVE_SPLICE_CAS:
				COO_EXEC(blank_fun, union_rem_fence, find_halve, splice_CAS);

			case REMCAS_SPLIT_SPLICE_CAS:
				COO_EXEC(blank_fun, union_rem_fence, find_split, splice_CAS);

			case REMCAS_A_HALVE_SPLICE_CAS:
				COO_EXEC(blank_fun, union_rem_fence, find_a_halve, splice_CAS);

			case REMCAS_A_SPLIT_SPLICE_CAS:
				COO_EXEC(blank_fun, union_rem_fence, find_a_split, splice_CAS);


			case REMCAS_NAIVE_SPLIT_ONE:
				COO_EXEC(blank_fun, union_rem, find_naive, split_one);

			case REMCAS_COMPRESS_SPLIT_ONE:
				COO_EXEC(blank_fun, union_rem, find_compress, split_one);

			case REMCAS_HALVE_SPLIT_ONE:
				COO_EXEC(blank_fun, union_rem, find_halve, split_one);

			case REMCAS_SPLIT_SPLIT_ONE:
				COO_EXEC(blank_fun, union_rem, find_split, split_one);

			case REMCAS_A_HALVE_SPLIT_ONE:
				COO_EXEC(blank_fun, union_rem, find_a_halve, split_one);

			case REMCAS_A_SPLIT_SPLIT_ONE:
				COO_EXEC(blank_fun, union_rem, find_a_split, split_one);


			case REMCAS_NAIVE_HALVE_ONE:
				COO_EXEC(blank_fun, union_rem, find_naive, halve_one);

			case REMCAS_COMPRESS_HALVE_ONE:
				COO_EXEC(blank_fun, union_rem, find_compress, halve_one);

			case REMCAS_HALVE_HALVE_ONE:
				COO_EXEC(blank_fun, union_rem, find_halve, halve_one);

			case REMCAS_SPLIT_HALVE_ONE:
				COO_EXEC(blank_fun, union_rem, find_split, halve_one);

			case REMCAS_A_HALVE_HALVE_ONE:
				COO_EXEC(blank_fun, union_rem, find_a_halve, halve_one);

			case REMCAS_A_SPLIT_HALVE_ONE:
				COO_EXEC(blank_fun, union_rem, find_a_split, halve_one);


			case REMLOCK_NAIVE_SPLICE_CAS:
				COO_EXEC(blank_fun, union_rem_lock_fence, find_naive, splice_CAS);

			case REMLOCK_COMPRESS_SPLICE_CAS:
				COO_EXEC(blank_fun, union_rem_lock_fence, find_compress, splice_CAS);

			case REMLOCK_HALVE_SPLICE_CAS:
				COO_EXEC(blank_fun, union_rem_lock_fence, find_halve, splice_CAS);

			case REMLOCK_SPLIT_SPLICE_CAS:
				COO_EXEC(blank_fun, union_rem_lock_fence, find_split, splice_CAS);

			case REMLOCK_A_HALVE_SPLICE_CAS:
				COO_EXEC(blank_fun, union_rem_lock_fence, find_a_halve, splice_CAS);

			case REMLOCK_A_SPLIT_SPLICE_CAS:
				COO_EXEC(blank_fun, union_rem_lock_fence, find_a_split, splice_CAS);


			case REMLOCK_NAIVE_SPLIT_ONE:
				COO_EXEC(blank_fun, union_rem_lock, find_naive, split_one);

			case REMLOCK_COMPRESS_SPLIT_ONE:
				COO_EXEC(blank_fun, union_rem_lock, find_compress, split_one);

			case REMLOCK_HALVE_SPLIT_ONE:
				COO_EXEC(blank_fun, union_rem_lock, find_halve, split_one);

			case REMLOCK_SPLIT_SPLIT_ONE:
				COO_EXEC(blank_fun, union_rem_lock, find_split, split_one);

			case REMLOCK_A_HALVE_SPLIT_ONE:
				COO_EXEC(blank_fun, union_rem_lock, find_a_halve, split_one);

			case REMLOCK_A_SPLIT_SPLIT_ONE:
				COO_EXEC(blank_fun, union_rem_lock, find_a_split, split_one);


			case REMLOCK_NAIVE_HALVE_ONE:
				COO_EXEC(blank_fun, union_rem_lock, find_naive, halve_one);

			case REMLOCK_COMPRESS_HALVE_ONE:
				COO_EXEC(blank_fun, union_rem_lock, find_compress, halve_one);

			case REMLOCK_HALVE_HALVE_ONE:
				COO_EXEC(blank_fun, union_rem_lock, find_halve, halve_one);

			case REMLOCK_SPLIT_HALVE_ONE:
				COO_EXEC(blank_fun, union_rem_lock, find_split, halve_one);

			case REMLOCK_A_HALVE_HALVE_ONE:
				COO_EXEC(blank_fun, union_rem_lock, find_a_halve, halve_one);

			case REMLOCK_A_SPLIT_HALVE_ONE:
				COO_EXEC(blank_fun, union_rem_lock, find_a_split, halve_one);



		};
		cudaFree(d_input->_Vflag); cudaFree(d_input->_Eflag); cudaFree(d_input->_Vfront); cudaFree(d_input->_Rfront);

	}


	if(d_input->format == COO_SAMPLE) {	

		grid_size_union = CEIL(d_input->E, tb_size);
		uintT tot_elt = d_input->E;		
		cudaMalloc((void **)&d_input->_Vflag, sizeof(uintT));
		cudaMalloc((void **)&d_input->_Eflag, sizeof(uintT));
		cudaMalloc((void **)&d_input->_Vfront, sizeof(uintT)*CEIL(d_input->V,32));
		cudaMalloc((void **)&d_input->_Rfront, sizeof(uintT)*CEIL(d_input->V,32));

		d_input->coo_sample_size = (d_input->V / 2);	
		int p_is_sym = d_input->is_sym;
		d_input->is_sym = 0;

		switch(d_input->algo) {
			case ASYNC_NAIVE:
				COO_SAMPLE_EXEC(blank_fun, union_async, find_naive, blank_fun2);

			case ASYNC_COMPRESS:
				COO_SAMPLE_EXEC(blank_fun, union_async, find_compress, blank_fun2);

			case ASYNC_HALVE:
				COO_SAMPLE_EXEC(blank_fun, union_async,find_halve, blank_fun2);

			case ASYNC_SPLIT:
				COO_SAMPLE_EXEC(blank_fun, union_async, find_split, blank_fun2);

			case ASYNC_A_HALVE:
				COO_SAMPLE_EXEC(blank_fun, union_async,find_a_halve, blank_fun2);

			case ASYNC_A_SPLIT:
				COO_SAMPLE_EXEC(blank_fun, union_async,find_a_split, blank_fun2);



			case STOPT_NAIVE:
				COO_SAMPLE_EXEC(blank_fun, union_stopt, find_naive, blank_fun2);

			case STOPT_COMPRESS:
				COO_SAMPLE_EXEC(blank_fun, union_stopt, find_compress, blank_fun2);

			case STOPT_HALVE:
				COO_SAMPLE_EXEC(blank_fun, union_stopt,find_halve, blank_fun2);

			case STOPT_SPLIT:
				COO_SAMPLE_EXEC(blank_fun, union_stopt, find_split, blank_fun2);

			case STOPT_A_HALVE:
				COO_SAMPLE_EXEC(blank_fun, union_stopt,find_a_halve, blank_fun2);

			case STOPT_A_SPLIT:
				COO_SAMPLE_EXEC(blank_fun, union_stopt,find_a_split, blank_fun2);



			case EARLY_NAIVE:
				COO_SAMPLE_EXEC(blank_fun, union_early, find_naive, blank_fun2);
			case EARLY_COMPRESS:
				COO_SAMPLE_EXEC(blank_fun, union_early, find_compress, blank_fun2);
			case EARLY_A_HALVE:
				COO_SAMPLE_EXEC(blank_fun, union_early, find_a_halve, blank_fun2);
			case EARLY_A_SPLIT:
				COO_SAMPLE_EXEC(blank_fun, union_early, find_a_split, blank_fun2);



			case ECL_NAIVE:
				COO_SAMPLE_EXEC(ecl_fun, union_ecl, find_naive, blank_fun2);

			case ECL_COMPRESS:
				COO_SAMPLE_EXEC(ecl_fun, union_ecl,find_compress, blank_fun2);

			case ECL_HALVE:
				COO_SAMPLE_EXEC(ecl_fun, union_ecl,find_halve, blank_fun2);

			case ECL_SPLIT:
				COO_SAMPLE_EXEC(ecl_fun, union_ecl, find_split, blank_fun2);

			case ECL_A_HALVE:
				COO_SAMPLE_EXEC(ecl_fun, union_ecl,find_a_halve, blank_fun2);

			case ECL_A_SPLIT:
				COO_SAMPLE_EXEC(ecl_fun, union_ecl,find_a_split, blank_fun2);

			case AFFOREST_NAIVE:
				COO_SAMPLE_EXEC(blank_fun, union_afforest, find_naive, blank_fun2);

			case AFFOREST_COMPRESS:
				COO_SAMPLE_EXEC(blank_fun, union_afforest,find_compress, blank_fun2);

			case AFFOREST_HALVE:
				COO_SAMPLE_EXEC(blank_fun, union_afforest,find_halve, blank_fun2);

			case AFFOREST_SPLIT:
				COO_SAMPLE_EXEC(blank_fun, union_afforest, find_split, blank_fun2);

			case AFFOREST_A_HALVE:
				COO_SAMPLE_EXEC(blank_fun, union_afforest,find_a_halve, blank_fun2);

			case AFFOREST_A_SPLIT:
				COO_SAMPLE_EXEC(blank_fun, union_afforest,find_a_split, blank_fun2);

			case SIM_C_U_S_A:
				SIM_COO_SAMPLE_EXEC(sim_c_u_a, 0, 1, 1, 0, union_find_gpu_COO_SAMPLE);
			case SIM_C_R_S_A:
				SIM_COO_SAMPLE_EXEC(sim_c_r_a, 0, 1, 1, 1, union_find_gpu_COO_SAMPLE);
			case SIM_P_U_S_A:
				SIM_COO_SAMPLE_EXEC(sim_p_u_a, 0, 1, 1, 0, union_find_gpu_COO_SAMPLE);
			case SIM_P_R_S_A:
				SIM_COO_SAMPLE_EXEC(sim_p_r_a, 0, 1, 1, 1, union_find_gpu_COO_SAMPLE);

			case SIM_P_U_S:
				SIM_COO_SAMPLE_EXEC(sim_p_u, 0, 0, 0, 0, union_find_gpu_COO_SAMPLE);
			case SIM_P_R_S:
				SIM_COO_SAMPLE_EXEC(sim_p_r, 0, 0, 0, 1, union_find_gpu_COO_SAMPLE);
			case SIM_E_U_S_A:
				SIM_COO_SAMPLE_EXEC(sim_e_u_a, 0, 1, 1, 0, union_find_gpu_COO_SAMPLE);
			case SIM_E_U_S:
				SIM_COO_SAMPLE_EXEC(sim_e_u, 0, 0, 0, 0, union_find_gpu_COO_SAMPLE);

			case SIM_C_U_SS_A:
				SIM_COO_SAMPLE_EXEC(sim_c_u_a, 1, 1, 1, 1, union_find_gpu_COO_SAMPLE);
			case SIM_C_R_SS_A:
				SIM_COO_SAMPLE_EXEC(sim_c_r_a, 1, 1, 1, 1, union_find_gpu_COO_SAMPLE);
			case SIM_P_U_SS_A:
				SIM_COO_SAMPLE_EXEC(sim_p_u_a, 1, 1, 1, 0, union_find_gpu_COO_SAMPLE);
			case SIM_P_R_SS_A:
				SIM_COO_SAMPLE_EXEC(sim_p_r_a, 1, 1, 1, 1, union_find_gpu_COO_SAMPLE);

			case SIM_P_U_SS:
				SIM_COO_SAMPLE_EXEC(sim_p_u, 1, 0, 0, 0, union_find_gpu_COO_SAMPLE);
			case SIM_P_R_SS:
				SIM_COO_SAMPLE_EXEC(sim_p_r, 1, 1, 0, 1, union_find_gpu_COO_SAMPLE);
			case SIM_E_U_SS_A: 
				SIM_COO_SAMPLE_EXEC(sim_e_u_a, 1, 1, 1, 0, union_find_gpu_COO_SAMPLE);
			case SIM_E_U_SS:
				SIM_COO_SAMPLE_EXEC(sim_e_u, 1, 0, 0, 0, union_find_gpu_COO_SAMPLE);

			case STERGIOS:
				STER_COO_SAMPLE_EXEC(ster, 0, 0, 0, 0, union_find_gpu_COO_SAMPLE);
			case SVA:
				SV_COO_SAMPLE_EXEC(sv, 0, 0, 0, 0, union_find_gpu_COO_SAMPLE);
			case LPA:
				LP_COO_SAMPLE_EXEC(lp, 0, 0, 0, 0, union_find_gpu_COO_SAMPLE);

			case RAND_NAIVE:
				RAND_COO_SAMPLE_EXEC(blank_fun, union_rand, find_r_naive, blank_fun2);

			case RAND_SPLIT_2:
				RAND_COO_SAMPLE_EXEC(blank_fun, union_rand, find_r_split_2, blank_fun2);


			case REMCAS_NAIVE_SPLICE_CAS:
				COO_SAMPLE_EXEC(blank_fun, union_rem_fence, find_naive, splice_CAS);

			case REMCAS_COMPRESS_SPLICE_CAS:
				COO_SAMPLE_EXEC(blank_fun, union_rem_fence, find_compress, splice_CAS);

			case REMCAS_HALVE_SPLICE_CAS:
				COO_SAMPLE_EXEC(blank_fun, union_rem_fence, find_halve, splice_CAS);

			case REMCAS_SPLIT_SPLICE_CAS:
				COO_SAMPLE_EXEC(blank_fun, union_rem_fence, find_split, splice_CAS);

			case REMCAS_A_HALVE_SPLICE_CAS:
				COO_SAMPLE_EXEC(blank_fun, union_rem_fence, find_a_halve, splice_CAS);

			case REMCAS_A_SPLIT_SPLICE_CAS:
				COO_SAMPLE_EXEC(blank_fun, union_rem_fence, find_a_split, splice_CAS);


			case REMCAS_NAIVE_SPLIT_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem, find_naive, split_one);

			case REMCAS_COMPRESS_SPLIT_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem, find_compress, split_one);

			case REMCAS_HALVE_SPLIT_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem, find_halve, split_one);

			case REMCAS_SPLIT_SPLIT_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem, find_split, split_one);

			case REMCAS_A_HALVE_SPLIT_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem, find_a_halve, split_one);

			case REMCAS_A_SPLIT_SPLIT_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem, find_a_split, split_one);


			case REMCAS_NAIVE_HALVE_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem, find_naive, halve_one);

			case REMCAS_COMPRESS_HALVE_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem, find_compress, halve_one);

			case REMCAS_HALVE_HALVE_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem, find_halve, halve_one);

			case REMCAS_SPLIT_HALVE_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem, find_split, halve_one);

			case REMCAS_A_HALVE_HALVE_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem, find_a_halve, halve_one);

			case REMCAS_A_SPLIT_HALVE_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem, find_a_split, halve_one);


			case REMLOCK_NAIVE_SPLICE_CAS:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock_fence, find_naive, splice_CAS);

			case REMLOCK_COMPRESS_SPLICE_CAS:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock_fence, find_compress, splice_CAS);

			case REMLOCK_HALVE_SPLICE_CAS:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock_fence, find_halve, splice_CAS);

			case REMLOCK_SPLIT_SPLICE_CAS:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock_fence, find_split, splice_CAS);

			case REMLOCK_A_HALVE_SPLICE_CAS:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock_fence, find_a_halve, splice_CAS);

			case REMLOCK_A_SPLIT_SPLICE_CAS:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock_fence, find_a_split, splice_CAS);


			case REMLOCK_NAIVE_SPLIT_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock_fence, find_naive, split_one);

			case REMLOCK_COMPRESS_SPLIT_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock, find_compress, split_one);

			case REMLOCK_HALVE_SPLIT_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock, find_halve, split_one);

			case REMLOCK_SPLIT_SPLIT_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock, find_split, split_one);

			case REMLOCK_A_HALVE_SPLIT_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock, find_a_halve, split_one);

			case REMLOCK_A_SPLIT_SPLIT_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock, find_a_split, split_one);


			case REMLOCK_NAIVE_HALVE_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock, find_naive, halve_one);

			case REMLOCK_COMPRESS_HALVE_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock, find_compress, halve_one);

			case REMLOCK_HALVE_HALVE_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock, find_halve, halve_one);

			case REMLOCK_SPLIT_HALVE_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock, find_split, halve_one);

			case REMLOCK_A_HALVE_HALVE_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock, find_a_halve, halve_one);

			case REMLOCK_A_SPLIT_HALVE_ONE:
				COO_SAMPLE_EXEC(blank_fun, union_rem_lock, find_a_split, halve_one);



		};
		cudaFree(d_input->_Vflag); cudaFree(d_input->_Eflag); cudaFree(d_input->_Vfront); cudaFree(d_input->_Rfront);


		d_input->is_sym = p_is_sym;

	}

	
	if(d_input->format == CSR) {

		grid_size_union = CEIL(d_input->V, tb_size/4);
		uintT tot_elt = d_input->V;		
		cudaMalloc((void **)&d_input->_Vflag, sizeof(uintT));
		cudaMalloc((void **)&d_input->_Eflag, sizeof(uintT));
		cudaMalloc((void **)&d_input->_Vfront, sizeof(uintT)*CEIL(d_input->V,32));
		cudaMalloc((void **)&d_input->_Rfront, sizeof(uintT)*CEIL(d_input->V,32));

		switch(d_input->algo) {
			case ASYNC_NAIVE:
				CSR_EXEC(blank_fun, union_async, find_naive, blank_fun2);

			case ASYNC_COMPRESS:
				CSR_EXEC(blank_fun, union_async,find_compress, blank_fun2);

			case ASYNC_HALVE:
				CSR_EXEC(blank_fun, union_async,find_halve, blank_fun2);

			case ASYNC_SPLIT:
				CSR_EXEC(blank_fun, union_async, find_split, blank_fun2);

			case ASYNC_A_HALVE:
				CSR_EXEC(blank_fun, union_async,find_a_halve, blank_fun2);

			case ASYNC_A_SPLIT:
				CSR_EXEC(blank_fun, union_async,find_a_split, blank_fun2);



			case STOPT_NAIVE:
				CSR_EXEC(blank_fun, union_stopt, find_naive, blank_fun2);

			case STOPT_COMPRESS:
				CSR_EXEC(blank_fun, union_stopt, find_compress, blank_fun2);

			case STOPT_HALVE:
				CSR_EXEC(blank_fun, union_stopt,find_halve, blank_fun2);

			case STOPT_SPLIT:
				CSR_EXEC(blank_fun, union_stopt, find_split, blank_fun2);

			case STOPT_A_HALVE:
				CSR_EXEC(blank_fun, union_stopt,find_a_halve, blank_fun2);

			case STOPT_A_SPLIT:
				CSR_EXEC(blank_fun, union_stopt,find_a_split, blank_fun2);


			case EARLY_NAIVE:
				CSR_EXEC(blank_fun, union_early, find_naive, blank_fun2);
			case EARLY_COMPRESS:
				CSR_EXEC(blank_fun, union_early, find_compress, blank_fun2);
			case EARLY_A_HALVE:
				CSR_EXEC(blank_fun, union_early, find_a_halve, blank_fun2);
			case EARLY_A_SPLIT:
				CSR_EXEC(blank_fun, union_early, find_a_split, blank_fun2);



			case ECL_NAIVE:
				CSR_EXEC(ecl_fun, union_ecl, find_naive, blank_fun2);

			case ECL_COMPRESS:
				CSR_EXEC(ecl_fun, union_ecl,find_compress, blank_fun2);

			case ECL_HALVE:
				CSR_EXEC(ecl_fun, union_ecl,find_halve, blank_fun2);

			case ECL_SPLIT:
				CSR_EXEC(ecl_fun, union_ecl, find_split, blank_fun2);

			case ECL_A_HALVE:
				CSR_EXEC(ecl_fun, union_ecl,find_a_halve, blank_fun2);

			case ECL_A_SPLIT:
				CSR_EXEC(ecl_fun, union_ecl,find_a_split, blank_fun2);

			case AFFOREST_NAIVE:
				CSR_EXEC(blank_fun, union_afforest, find_naive, blank_fun2);

			case AFFOREST_COMPRESS:
				CSR_EXEC(blank_fun, union_afforest,find_compress, blank_fun2);

			case AFFOREST_HALVE:
				CSR_EXEC(blank_fun, union_afforest,find_halve, blank_fun2);

			case AFFOREST_SPLIT:
				CSR_EXEC(blank_fun, union_afforest, find_split, blank_fun2);

			case AFFOREST_A_HALVE:
				CSR_EXEC(blank_fun, union_afforest,find_a_halve, blank_fun2);

			case AFFOREST_A_SPLIT:
				CSR_EXEC(blank_fun, union_afforest,find_a_split, blank_fun2);

			case RAND_NAIVE:
				RAND_CSR_EXEC(blank_fun, union_rand, find_r_naive, blank_fun2);

			case RAND_SPLIT_2:
				RAND_CSR_EXEC(blank_fun, union_rand, find_r_split_2, blank_fun2);

			case SIM_P_U_S:
				SIM_CSR_EXEC(sim_p_u, 0, 0, 0, 0, union_find_gpu_CSR);
			case SIM_P_R_S:
				SIM_CSR_EXEC(sim_p_r, 0, 0, 0, 1, union_find_gpu_CSR);
			case SIM_E_U_S:
				SIM_CSR_EXEC(sim_e_u, 0, 0, 0, 0, union_find_gpu_CSR);

			case SIM_P_U_SS:
				SIM_CSR_EXEC(sim_p_u, 1, 0, 0, 0, union_find_gpu_CSR);
			case SIM_P_R_SS:
				SIM_CSR_EXEC(sim_p_r, 1, 1, 0, 1, union_find_gpu_CSR);
			case SIM_E_U_SS:
				SIM_CSR_EXEC(sim_e_u, 1, 0, 0, 0, union_find_gpu_CSR);

			case STERGIOS:
				STER_CSR_EXEC(ster, 1, 0, 0, 0, union_find_gpu_CSR);

			case SVA:
				SV_CSR_EXEC(sv, 1, 0, 0, 0, union_find_gpu_CSR);
			case LPA:
				LP_CSR_EXEC(lp, 1, 0, 0, 0, union_find_gpu_CSR);


			case REMCAS_NAIVE_SPLICE_CAS:
				CSR_EXEC(blank_fun, union_rem_fence, find_naive, splice_CAS);

			case REMCAS_COMPRESS_SPLICE_CAS:
				CSR_EXEC(blank_fun, union_rem_fence, find_compress, splice_CAS);

			case REMCAS_HALVE_SPLICE_CAS:
				CSR_EXEC(blank_fun, union_rem_fence, find_halve, splice_CAS);

			case REMCAS_SPLIT_SPLICE_CAS:
				CSR_EXEC(blank_fun, union_rem_fence, find_split, splice_CAS);

			case REMCAS_A_HALVE_SPLICE_CAS:
				CSR_EXEC(blank_fun, union_rem_fence, find_a_halve, splice_CAS);

			case REMCAS_A_SPLIT_SPLICE_CAS:
				CSR_EXEC(blank_fun, union_rem_fence, find_a_split, splice_CAS);


			case REMCAS_NAIVE_SPLIT_ONE:
				CSR_EXEC(blank_fun, union_rem, find_naive, split_one);

			case REMCAS_COMPRESS_SPLIT_ONE:
				CSR_EXEC(blank_fun, union_rem, find_compress, split_one);

			case REMCAS_HALVE_SPLIT_ONE:
				CSR_EXEC(blank_fun, union_rem, find_halve, split_one);

			case REMCAS_SPLIT_SPLIT_ONE:
				CSR_EXEC(blank_fun, union_rem, find_split, split_one);

			case REMCAS_A_HALVE_SPLIT_ONE:
				CSR_EXEC(blank_fun, union_rem, find_a_halve, split_one);

			case REMCAS_A_SPLIT_SPLIT_ONE:
				CSR_EXEC(blank_fun, union_rem, find_a_split, split_one);


			case REMCAS_NAIVE_HALVE_ONE:
				CSR_EXEC(blank_fun, union_rem, find_naive, halve_one);

			case REMCAS_COMPRESS_HALVE_ONE:
				CSR_EXEC(blank_fun, union_rem, find_compress, halve_one);

			case REMCAS_HALVE_HALVE_ONE:
				CSR_EXEC(blank_fun, union_rem, find_halve, halve_one);

			case REMCAS_SPLIT_HALVE_ONE:
				CSR_EXEC(blank_fun, union_rem, find_split, halve_one);

			case REMCAS_A_HALVE_HALVE_ONE:
				CSR_EXEC(blank_fun, union_rem, find_a_halve, halve_one);

			case REMCAS_A_SPLIT_HALVE_ONE:
				CSR_EXEC(blank_fun, union_rem, find_a_split, halve_one);


			case REMLOCK_NAIVE_SPLICE_CAS:
				CSR_EXEC(blank_fun, union_rem_lock_fence, find_naive, splice_CAS);

			case REMLOCK_COMPRESS_SPLICE_CAS:
				CSR_EXEC(blank_fun, union_rem_lock_fence, find_compress, splice_CAS);

			case REMLOCK_HALVE_SPLICE_CAS:
				CSR_EXEC(blank_fun, union_rem_lock_fence, find_halve, splice_CAS);

			case REMLOCK_SPLIT_SPLICE_CAS:
				CSR_EXEC(blank_fun, union_rem_lock_fence, find_split, splice_CAS);

			case REMLOCK_A_HALVE_SPLICE_CAS:
				CSR_EXEC(blank_fun, union_rem_lock_fence, find_a_halve, splice_CAS);

			case REMLOCK_A_SPLIT_SPLICE_CAS:
				CSR_EXEC(blank_fun, union_rem_lock_fence, find_a_split, splice_CAS);


			case REMLOCK_NAIVE_SPLIT_ONE:
				CSR_EXEC(blank_fun, union_rem_lock, find_naive, split_one);

			case REMLOCK_COMPRESS_SPLIT_ONE:
				CSR_EXEC(blank_fun, union_rem_lock, find_compress, split_one);

			case REMLOCK_HALVE_SPLIT_ONE:
				CSR_EXEC(blank_fun, union_rem_lock, find_halve, split_one);

			case REMLOCK_SPLIT_SPLIT_ONE:
				CSR_EXEC(blank_fun, union_rem_lock, find_split, split_one);

			case REMLOCK_A_HALVE_SPLIT_ONE:
				CSR_EXEC(blank_fun, union_rem_lock, find_a_halve, split_one);

			case REMLOCK_A_SPLIT_SPLIT_ONE:
				CSR_EXEC(blank_fun, union_rem_lock, find_a_split, split_one);


			case REMLOCK_NAIVE_HALVE_ONE:
				CSR_EXEC(blank_fun, union_rem_lock, find_naive, halve_one);

			case REMLOCK_COMPRESS_HALVE_ONE:
				CSR_EXEC(blank_fun, union_rem_lock, find_compress, halve_one);

			case REMLOCK_HALVE_HALVE_ONE:
				CSR_EXEC(blank_fun, union_rem_lock, find_halve, halve_one);

			case REMLOCK_SPLIT_HALVE_ONE:
				CSR_EXEC(blank_fun, union_rem_lock, find_split, halve_one);

			case REMLOCK_A_HALVE_HALVE_ONE:
				CSR_EXEC(blank_fun, union_rem_lock, find_a_halve, halve_one);

			case REMLOCK_A_SPLIT_HALVE_ONE:
				CSR_EXEC(blank_fun, union_rem_lock, find_a_split, halve_one);

		};

		cudaFree(d_input->_Vflag); cudaFree(d_input->_Eflag); cudaFree(d_input->_Vfront); cudaFree(d_input->_Rfront);

	}

	if(d_input->format == SAMPLE) {


		uintT *q, *qp;
		uintT hqp;
		cudaMalloc((void **)&q, sizeof(uintT)*(d_input->V));
		cudaMalloc((void **)&qp, sizeof(uintT)*4);

	
		grid_size_union = CEIL(d_input->V, tb_size/4);
		uintT tot_elt = d_input->V;		

		d_input->sample_k = 2;
		d_input->sample_size = 1024;

		cudaMalloc((void **)&d_input->_Vflag, sizeof(uintT));
		cudaMalloc((void **)&d_input->_Eflag, sizeof(uintT));
		cudaMalloc((void **)&d_input->_Vfront, sizeof(uintT)*CEIL(d_input->V,32));
		cudaMalloc((void **)&d_input->_Rfront, sizeof(uintT)*CEIL(d_input->V,32));

		uintT *sample = (uintT *)malloc(sizeof(uintT)*d_input->sample_size);
		uintT *_sample;
		cudaMalloc((void **)&_sample, sizeof(uintT)*(d_input->sample_size)); 

		cudaMalloc((void **)&d_input->queue, sizeof(uintT)*(d_input->V));
		cudaMalloc((void **)&d_input->queuep, sizeof(uintT));
		cudaMemset(d_input->queuep, 0, sizeof(uintT));

		switch(d_input->algo) {
			case ASYNC_NAIVE:
				SAMPLE_EXEC(blank_fun, union_async, find_naive, blank_fun2);

			case ASYNC_COMPRESS:
				SAMPLE_EXEC(blank_fun, union_async,find_compress, blank_fun2);

			case ASYNC_HALVE:
				SAMPLE_EXEC(blank_fun, union_async,find_halve, blank_fun2);

			case ASYNC_SPLIT:
				SAMPLE_EXEC(blank_fun, union_async, find_split, blank_fun2);

			case ASYNC_A_HALVE:
				SAMPLE_EXEC(blank_fun, union_async,find_a_halve, blank_fun2);

			case ASYNC_A_SPLIT:
				SAMPLE_EXEC(blank_fun, union_async,find_a_split, blank_fun2);


			case STOPT_NAIVE:
				SAMPLE_EXEC(blank_fun, union_stopt, find_naive, blank_fun2);

			case STOPT_COMPRESS:
				SAMPLE_EXEC(blank_fun, union_stopt, find_compress, blank_fun2);

			case STOPT_HALVE:
				SAMPLE_EXEC(blank_fun, union_stopt,find_halve, blank_fun2);

			case STOPT_SPLIT:
				SAMPLE_EXEC(blank_fun, union_stopt, find_split, blank_fun2);

			case STOPT_A_HALVE:
				SAMPLE_EXEC(blank_fun, union_stopt,find_a_halve, blank_fun2);

			case STOPT_A_SPLIT:
				SAMPLE_EXEC(blank_fun, union_stopt,find_a_split, blank_fun2);


			case EARLY_NAIVE:
				SAMPLE_EXEC(blank_fun, union_early, find_naive, blank_fun2);
			case EARLY_COMPRESS:
				SAMPLE_EXEC(blank_fun, union_early, find_compress, blank_fun2);
			case EARLY_A_HALVE:
				SAMPLE_EXEC(blank_fun, union_early, find_a_halve, blank_fun2);
			case EARLY_A_SPLIT:
				SAMPLE_EXEC(blank_fun, union_early, find_a_split, blank_fun2);
	

			case ECL_NAIVE:
				SAMPLE_EXEC(ecl_fun, union_ecl, find_naive, blank_fun2);

			case ECL_COMPRESS:
				SAMPLE_EXEC(ecl_fun, union_ecl,find_compress, blank_fun2);

			case ECL_HALVE:
				SAMPLE_EXEC(ecl_fun, union_ecl,find_halve, blank_fun2);

			case ECL_SPLIT:
				SAMPLE_EXEC(ecl_fun, union_ecl, find_split, blank_fun2);

			case ECL_A_HALVE:
				SAMPLE_EXEC(ecl_fun, union_ecl,find_a_halve, blank_fun2);

			case ECL_A_SPLIT:
				SAMPLE_EXEC(ecl_fun, union_ecl,find_a_split, blank_fun2);

			case AFFOREST_NAIVE:
				SAMPLE_EXEC0(blank_fun, union_afforest, find_naive, blank_fun2);

			case AFFOREST_COMPRESS:
				SAMPLE_EXEC0(blank_fun, union_afforest,find_compress, blank_fun2);

			case AFFOREST_HALVE:
				SAMPLE_EXEC0(blank_fun, union_afforest,find_halve, blank_fun2);

			case AFFOREST_SPLIT:
				SAMPLE_EXEC0(blank_fun, union_afforest, find_split, blank_fun2);

			case AFFOREST_A_HALVE:
				SAMPLE_EXEC0(blank_fun, union_afforest,find_a_halve, blank_fun2);

			case AFFOREST_A_SPLIT:
				SAMPLE_EXEC0(blank_fun, union_afforest,find_a_split, blank_fun2);


			case RAND_NAIVE:
				RAND_SAMPLE_EXEC(blank_fun, union_rand, find_r_naive, blank_fun2);

			case RAND_SPLIT_2:
				RAND_SAMPLE_EXEC(blank_fun, union_rand, find_r_split_2, blank_fun2);



			case REMCAS_NAIVE_SPLICE_CAS:
				SAMPLE_EXEC(blank_fun, union_rem_fence, find_naive, splice_CAS);

			case REMCAS_COMPRESS_SPLICE_CAS:
				SAMPLE_EXEC(blank_fun, union_rem_fence, find_compress, splice_CAS);

			case REMCAS_HALVE_SPLICE_CAS:
				SAMPLE_EXEC(blank_fun, union_rem_fence, find_halve, splice_CAS);

			case REMCAS_SPLIT_SPLICE_CAS:
				SAMPLE_EXEC(blank_fun, union_rem_fence, find_split, splice_CAS);

			case REMCAS_A_HALVE_SPLICE_CAS:
				SAMPLE_EXEC(blank_fun, union_rem_fence, find_a_halve, splice_CAS);

			case REMCAS_A_SPLIT_SPLICE_CAS:
				SAMPLE_EXEC(blank_fun, union_rem_fence, find_a_split, splice_CAS);


			case REMCAS_NAIVE_SPLIT_ONE:
				SAMPLE_EXEC(blank_fun, union_rem, find_naive, split_one);

			case REMCAS_COMPRESS_SPLIT_ONE:
				SAMPLE_EXEC(blank_fun, union_rem, find_compress, split_one);

			case REMCAS_HALVE_SPLIT_ONE:
				SAMPLE_EXEC(blank_fun, union_rem, find_halve, split_one);

			case REMCAS_SPLIT_SPLIT_ONE:
				SAMPLE_EXEC(blank_fun, union_rem, find_split, split_one);

			case REMCAS_A_HALVE_SPLIT_ONE:
				SAMPLE_EXEC(blank_fun, union_rem, find_a_halve, split_one);

			case REMCAS_A_SPLIT_SPLIT_ONE:
				SAMPLE_EXEC(blank_fun, union_rem, find_a_split, split_one);


			case REMCAS_NAIVE_HALVE_ONE:
				SAMPLE_EXEC(blank_fun, union_rem, find_naive, halve_one);

			case REMCAS_COMPRESS_HALVE_ONE:
				SAMPLE_EXEC(blank_fun, union_rem, find_compress, halve_one);

			case REMCAS_HALVE_HALVE_ONE:
				SAMPLE_EXEC(blank_fun, union_rem, find_halve, halve_one);

			case REMCAS_SPLIT_HALVE_ONE:
				SAMPLE_EXEC(blank_fun, union_rem, find_split, halve_one);

			case REMCAS_A_HALVE_HALVE_ONE:
				SAMPLE_EXEC(blank_fun, union_rem, find_a_halve, halve_one);

			case REMCAS_A_SPLIT_HALVE_ONE:
				SAMPLE_EXEC(blank_fun, union_rem, find_a_split, halve_one);


			case REMLOCK_NAIVE_SPLICE_CAS:
				SAMPLE_EXEC(blank_fun, union_rem_lock_fence, find_naive, splice_CAS);

			case REMLOCK_COMPRESS_SPLICE_CAS:
				SAMPLE_EXEC(blank_fun, union_rem_lock_fence, find_compress, splice_CAS);

			case REMLOCK_HALVE_SPLICE_CAS:
				SAMPLE_EXEC(blank_fun, union_rem_lock_fence, find_halve, splice_CAS);

			case REMLOCK_SPLIT_SPLICE_CAS:
				SAMPLE_EXEC(blank_fun, union_rem_lock_fence, find_split, splice_CAS);

			case REMLOCK_A_HALVE_SPLICE_CAS:
				SAMPLE_EXEC(blank_fun, union_rem_lock_fence, find_a_halve, splice_CAS);

			case REMLOCK_A_SPLIT_SPLICE_CAS:
				SAMPLE_EXEC(blank_fun, union_rem_lock_fence, find_a_split, splice_CAS);


			case REMLOCK_NAIVE_SPLIT_ONE:
				SAMPLE_EXEC(blank_fun, union_rem_lock, find_naive, split_one);

			case REMLOCK_COMPRESS_SPLIT_ONE:
				SAMPLE_EXEC(blank_fun, union_rem_lock, find_compress, split_one);

			case REMLOCK_HALVE_SPLIT_ONE:
				SAMPLE_EXEC(blank_fun, union_rem_lock, find_halve, split_one);

			case REMLOCK_SPLIT_SPLIT_ONE:
				SAMPLE_EXEC(blank_fun, union_rem_lock, find_split, split_one);

			case REMLOCK_A_HALVE_SPLIT_ONE:
				SAMPLE_EXEC(blank_fun, union_rem_lock, find_a_halve, split_one);

			case REMLOCK_A_SPLIT_SPLIT_ONE:
				SAMPLE_EXEC(blank_fun, union_rem_lock, find_a_split, split_one);


			case REMLOCK_NAIVE_HALVE_ONE:
				SAMPLE_EXEC(blank_fun, union_rem_lock, find_naive, halve_one);

			case REMLOCK_COMPRESS_HALVE_ONE:
				SAMPLE_EXEC(blank_fun, union_rem_lock, find_compress, halve_one);

			case REMLOCK_HALVE_HALVE_ONE:
				SAMPLE_EXEC(blank_fun, union_rem_lock, find_halve, halve_one);

			case REMLOCK_SPLIT_HALVE_ONE:
				SAMPLE_EXEC(blank_fun, union_rem_lock, find_split, halve_one);

			case REMLOCK_A_HALVE_HALVE_ONE:
				SAMPLE_EXEC(blank_fun, union_rem_lock, find_a_halve, halve_one);

			case REMLOCK_A_SPLIT_HALVE_ONE:
				SAMPLE_EXEC(blank_fun, union_rem_lock, find_a_split, halve_one);



			case SIM_P_U_S:
				SIM_SAMPLE_EXEC(sim_p_u, 0, 0, 0, 0, sampling_phase3_all);
			case SIM_P_R_S:
				SIM_SAMPLE_EXEC(sim_p_r, 0, 0, 0, 1, sampling_phase3_all);
			case SIM_E_U_S:
				SIM_SAMPLE_EXEC(sim_e_u, 0, 0, 0, 0, sampling_phase3_all);

			case SIM_P_U_SS:
				SIM_SAMPLE_EXEC(sim_p_u, 1, 0, 0, 0, sampling_phase3_all);
			case SIM_P_R_SS:
				SIM_SAMPLE_EXEC(sim_p_r, 1, 1, 0, 1, sampling_phase3_all);
			case SIM_E_U_SS:
				SIM_SAMPLE_EXEC(sim_e_u, 1, 0, 0, 0, sampling_phase3_all);

			case STERGIOS:
				STER_SAMPLE_EXEC(ster, 1, 0, 0, 0, sampling_phase3_all);
			case SVA:
				SV_SAMPLE_EXEC(sv, 1, 0, 0, 0, sampling_phase3_all);
			case LPA:
				LP_SAMPLE_EXEC(lp, 1, 0, 0, 0, sampling_phase3_all);



		}
		cudaFree(_sample);
		free(sample);	

		cudaFree(d_input->_Vflag); cudaFree(d_input->_Eflag); cudaFree(d_input->_Vfront); cudaFree(d_input->_Rfront);

	} 

	if(d_input->format == IHOOK) {

		uintT *q, *qp;
		uintT hqp;
		uintT hh;
		cudaMalloc((void **)&q, sizeof(uintT)*(d_input->V));
		cudaMalloc((void **)&qp, sizeof(uintT)*4);

	
		grid_size_union = CEIL(d_input->V, tb_size/4);
		uintT tot_elt = d_input->V;		

		d_input->sample_k = 2;
		d_input->sample_size = 1024;

		cudaMalloc((void **)&d_input->_Vflag, sizeof(uintT));
		cudaMalloc((void **)&d_input->_Eflag, sizeof(uintT));
		cudaMalloc((void **)&d_input->_Vfront, sizeof(uintT)*CEIL(d_input->V,32));
		cudaMalloc((void **)&d_input->_Rfront, sizeof(uintT)*CEIL(d_input->V,32));

		uintT *sample = (uintT *)malloc(sizeof(uintT)*d_input->sample_size);
		uintT *_sample;
		cudaMalloc((void **)&_sample, sizeof(uintT)*(d_input->sample_size)); 

		cudaMalloc((void **)&d_input->queue, sizeof(uintT)*(d_input->V));
		cudaMalloc((void **)&d_input->queuep, sizeof(uintT));
		cudaMemset(d_input->queuep, 0, sizeof(uintT));

		switch(d_input->algo) {
			case ASYNC_NAIVE:
				IHOOK_EXEC(blank_fun, union_async, find_naive, blank_fun2);

			case ASYNC_COMPRESS:
				IHOOK_EXEC(blank_fun, union_async,find_compress, blank_fun2);

			case ASYNC_HALVE:
				IHOOK_EXEC(blank_fun, union_async,find_halve, blank_fun2);

			case ASYNC_SPLIT:
				IHOOK_EXEC(blank_fun, union_async, find_split, blank_fun2);

			case ASYNC_A_HALVE:
				IHOOK_EXEC(blank_fun, union_async,find_a_halve, blank_fun2);

			case ASYNC_A_SPLIT:
				IHOOK_EXEC(blank_fun, union_async,find_a_split, blank_fun2);


			case STOPT_NAIVE:
				IHOOK_EXEC(blank_fun, union_stopt, find_naive, blank_fun2);

			case STOPT_COMPRESS:
				IHOOK_EXEC(blank_fun, union_stopt, find_compress, blank_fun2);

			case STOPT_HALVE:
				IHOOK_EXEC(blank_fun, union_stopt,find_halve, blank_fun2);

			case STOPT_SPLIT:
				IHOOK_EXEC(blank_fun, union_stopt, find_split, blank_fun2);

			case STOPT_A_HALVE:
				IHOOK_EXEC(blank_fun, union_stopt,find_a_halve, blank_fun2);

			case STOPT_A_SPLIT:
				IHOOK_EXEC(blank_fun, union_stopt,find_a_split, blank_fun2);


			case EARLY_NAIVE:
				IHOOK_EXEC(blank_fun, union_early, find_naive, blank_fun2);
			case EARLY_COMPRESS:
				IHOOK_EXEC(blank_fun, union_early, find_compress, blank_fun2);
			case EARLY_A_HALVE:
				IHOOK_EXEC(blank_fun, union_early, find_a_halve, blank_fun2);
			case EARLY_A_SPLIT:
				IHOOK_EXEC(blank_fun, union_early, find_a_split, blank_fun2);
	

			case ECL_NAIVE:
				IHOOK_EXEC(ecl_fun, union_ecl, find_naive, blank_fun2);

			case ECL_COMPRESS:
				IHOOK_EXEC(ecl_fun, union_ecl,find_compress, blank_fun2);

			case ECL_HALVE:
				IHOOK_EXEC(ecl_fun, union_ecl,find_halve, blank_fun2);

			case ECL_SPLIT:
				IHOOK_EXEC(ecl_fun, union_ecl, find_split, blank_fun2);

			case ECL_A_HALVE:
				IHOOK_EXEC(ecl_fun, union_ecl,find_a_halve, blank_fun2);

			case ECL_A_SPLIT:
				IHOOK_EXEC(ecl_fun, union_ecl,find_a_split, blank_fun2);

			case AFFOREST_NAIVE:
				IHOOK_EXEC(blank_fun, union_afforest, find_naive, blank_fun2);

			case AFFOREST_COMPRESS:
				IHOOK_EXEC(blank_fun, union_afforest,find_compress, blank_fun2);

			case AFFOREST_HALVE:
				IHOOK_EXEC(blank_fun, union_afforest,find_halve, blank_fun2);

			case AFFOREST_SPLIT:
				IHOOK_EXEC(blank_fun, union_afforest, find_split, blank_fun2);

			case AFFOREST_A_HALVE:
				IHOOK_EXEC(blank_fun, union_afforest,find_a_halve, blank_fun2);

			case AFFOREST_A_SPLIT:
				IHOOK_EXEC(blank_fun, union_afforest,find_a_split, blank_fun2);

			case RAND_NAIVE:
				RAND_IHOOK_EXEC(blank_fun, union_rand, find_r_naive, blank_fun2);

			case RAND_SPLIT_2:
				RAND_IHOOK_EXEC(blank_fun, union_rand, find_r_split_2, blank_fun2);



			case REMCAS_NAIVE_SPLICE_CAS:
				IHOOK_EXEC(blank_fun, union_rem_fence, find_naive, splice_CAS);

			case REMCAS_COMPRESS_SPLICE_CAS:
				IHOOK_EXEC(blank_fun, union_rem_fence, find_compress, splice_CAS);

			case REMCAS_HALVE_SPLICE_CAS:
				IHOOK_EXEC(blank_fun, union_rem_fence, find_halve, splice_CAS);

			case REMCAS_SPLIT_SPLICE_CAS:
				IHOOK_EXEC(blank_fun, union_rem_fence, find_split, splice_CAS);

			case REMCAS_A_HALVE_SPLICE_CAS:
				IHOOK_EXEC(blank_fun, union_rem_fence, find_a_halve, splice_CAS);

			case REMCAS_A_SPLIT_SPLICE_CAS:
				IHOOK_EXEC(blank_fun, union_rem_fence, find_a_split, splice_CAS);


			case REMCAS_NAIVE_SPLIT_ONE:
				IHOOK_EXEC(blank_fun, union_rem, find_naive, split_one);

			case REMCAS_COMPRESS_SPLIT_ONE:
				IHOOK_EXEC(blank_fun, union_rem, find_compress, split_one);

			case REMCAS_HALVE_SPLIT_ONE:
				IHOOK_EXEC(blank_fun, union_rem, find_halve, split_one);

			case REMCAS_SPLIT_SPLIT_ONE:
				IHOOK_EXEC(blank_fun, union_rem, find_split, split_one);

			case REMCAS_A_HALVE_SPLIT_ONE:
				IHOOK_EXEC(blank_fun, union_rem, find_a_halve, split_one);

			case REMCAS_A_SPLIT_SPLIT_ONE:
				IHOOK_EXEC(blank_fun, union_rem, find_a_split, split_one);


			case REMCAS_NAIVE_HALVE_ONE:
				IHOOK_EXEC(blank_fun, union_rem, find_naive, halve_one);

			case REMCAS_COMPRESS_HALVE_ONE:
				IHOOK_EXEC(blank_fun, union_rem, find_compress, halve_one);

			case REMCAS_HALVE_HALVE_ONE:
				IHOOK_EXEC(blank_fun, union_rem, find_halve, halve_one);

			case REMCAS_SPLIT_HALVE_ONE:
				IHOOK_EXEC(blank_fun, union_rem, find_split, halve_one);

			case REMCAS_A_HALVE_HALVE_ONE:
				IHOOK_EXEC(blank_fun, union_rem, find_a_halve, halve_one);

			case REMCAS_A_SPLIT_HALVE_ONE:
				IHOOK_EXEC(blank_fun, union_rem, find_a_split, halve_one);


			case REMLOCK_NAIVE_SPLICE_CAS:
				IHOOK_EXEC(blank_fun, union_rem_lock_fence, find_naive, splice_CAS);

			case REMLOCK_COMPRESS_SPLICE_CAS:
				IHOOK_EXEC(blank_fun, union_rem_lock_fence, find_compress, splice_CAS);

			case REMLOCK_HALVE_SPLICE_CAS:
				IHOOK_EXEC(blank_fun, union_rem_lock_fence, find_halve, splice_CAS);

			case REMLOCK_SPLIT_SPLICE_CAS:
				IHOOK_EXEC(blank_fun, union_rem_lock_fence, find_split, splice_CAS);

			case REMLOCK_A_HALVE_SPLICE_CAS:
				IHOOK_EXEC(blank_fun, union_rem_lock_fence, find_a_halve, splice_CAS);

			case REMLOCK_A_SPLIT_SPLICE_CAS:
				IHOOK_EXEC(blank_fun, union_rem_lock_fence, find_a_split, splice_CAS);


			case REMLOCK_NAIVE_SPLIT_ONE:
				IHOOK_EXEC(blank_fun, union_rem_lock, find_naive, split_one);

			case REMLOCK_COMPRESS_SPLIT_ONE:
				IHOOK_EXEC(blank_fun, union_rem_lock, find_compress, split_one);

			case REMLOCK_HALVE_SPLIT_ONE:
				IHOOK_EXEC(blank_fun, union_rem_lock, find_halve, split_one);

			case REMLOCK_SPLIT_SPLIT_ONE:
				IHOOK_EXEC(blank_fun, union_rem_lock, find_split, split_one);

			case REMLOCK_A_HALVE_SPLIT_ONE:
				IHOOK_EXEC(blank_fun, union_rem_lock, find_a_halve, split_one);

			case REMLOCK_A_SPLIT_SPLIT_ONE:
				IHOOK_EXEC(blank_fun, union_rem_lock, find_a_split, split_one);


			case REMLOCK_NAIVE_HALVE_ONE:
				IHOOK_EXEC(blank_fun, union_rem_lock, find_naive, halve_one);

			case REMLOCK_COMPRESS_HALVE_ONE:
				IHOOK_EXEC(blank_fun, union_rem_lock, find_compress, halve_one);

			case REMLOCK_HALVE_HALVE_ONE:
				IHOOK_EXEC(blank_fun, union_rem_lock, find_halve, halve_one);

			case REMLOCK_SPLIT_HALVE_ONE:
				IHOOK_EXEC(blank_fun, union_rem_lock, find_split, halve_one);

			case REMLOCK_A_HALVE_HALVE_ONE:
				IHOOK_EXEC(blank_fun, union_rem_lock, find_a_halve, halve_one);

			case REMLOCK_A_SPLIT_HALVE_ONE:
				IHOOK_EXEC(blank_fun, union_rem_lock, find_a_split, halve_one);



			case SIM_P_U_S:
				SIM_IHOOK_EXEC(sim_p_u, 0, 0, 0, 0, sampling_phase3_all);
			case SIM_P_R_S:
				SIM_IHOOK_EXEC(sim_p_r, 0, 0, 0, 1, sampling_phase3_all);
			case SIM_E_U_S:
				SIM_IHOOK_EXEC(sim_e_u, 0, 0, 0, 0, sampling_phase3_all);

			case SIM_P_U_SS:
				SIM_IHOOK_EXEC(sim_p_u, 1, 0, 0, 0, sampling_phase3_all);
			case SIM_P_R_SS:
				SIM_IHOOK_EXEC(sim_p_r, 1, 1, 0, 1, sampling_phase3_all);
			case SIM_E_U_SS:
				SIM_IHOOK_EXEC(sim_e_u, 1, 0, 0, 0, sampling_phase3_all);

			case STERGIOS:
				STER_IHOOK_EXEC(ster, 1, 0, 0, 0, sampling_phase3_all);
			case SVA:
				SV_IHOOK_EXEC(sv, 1, 0, 0, 0, sampling_phase3_all);
			case LPA:
				LP_IHOOK_EXEC(lp, 1, 0, 0, 0, sampling_phase3_all);



		}
		cudaFree(_sample);
		free(sample);	

		cudaFree(d_input->_Vflag); cudaFree(d_input->_Eflag); cudaFree(d_input->_Vfront); cudaFree(d_input->_Rfront);

	} 




	
	if(d_input->format == BFS) {

			
		uintT tot_elt = d_input->V;		
		grid_size_union = CEIL(d_input->V, tb_size/4);
		uintT seed = 0;

		d_input->sample_k = 0; // invalidate
		d_input->sample_size = 1024;

		cudaMalloc((void **)&d_input->_Vflag, sizeof(uintT));
		cudaMalloc((void **)&d_input->_Eflag, sizeof(uintT));
		cudaMalloc((void **)&d_input->_Vfront, sizeof(uintT)*CEIL(d_input->V,32));
		cudaMalloc((void **)&d_input->_Rfront, sizeof(uintT)*CEIL(d_input->V,32));

cudaDeviceSynchronize();

	switch(d_input->algo) {
			case ASYNC_NAIVE:
				BFS_EXEC(blank_fun, union_async, find_naive, blank_fun2);

			case ASYNC_COMPRESS:
				BFS_EXEC(blank_fun, union_async,find_compress, blank_fun2);

			case ASYNC_HALVE:
				BFS_EXEC(blank_fun, union_async,find_halve, blank_fun2);

			case ASYNC_SPLIT:
				BFS_EXEC(blank_fun, union_async, find_split, blank_fun2);

			case ASYNC_A_HALVE:
				BFS_EXEC(blank_fun, union_async,find_a_halve, blank_fun2);

			case ASYNC_A_SPLIT:
				BFS_EXEC(blank_fun, union_async,find_a_split, blank_fun2);



			case STOPT_NAIVE:
				BFS_EXEC(blank_fun, union_stopt, find_naive, blank_fun2);

			case STOPT_COMPRESS:
				BFS_EXEC(blank_fun, union_stopt, find_compress, blank_fun2);

			case STOPT_HALVE:
				BFS_EXEC(blank_fun, union_stopt,find_halve, blank_fun2);

			case STOPT_SPLIT:
				BFS_EXEC(blank_fun, union_stopt, find_split, blank_fun2);

			case STOPT_A_HALVE:
				BFS_EXEC(blank_fun, union_stopt,find_a_halve, blank_fun2);

			case STOPT_A_SPLIT:
				BFS_EXEC(blank_fun, union_stopt,find_a_split, blank_fun2);


			case EARLY_NAIVE:
				BFS_EXEC(blank_fun, union_early, find_naive, blank_fun2);
			case EARLY_COMPRESS:
				BFS_EXEC(blank_fun, union_early, find_compress, blank_fun2);
			case EARLY_A_HALVE:
				BFS_EXEC(blank_fun, union_early, find_a_halve, blank_fun2);
			case EARLY_A_SPLIT:
				BFS_EXEC(blank_fun, union_early, find_a_split, blank_fun2);

			case ECL_NAIVE:
				BFS_EXEC(ecl_fun, union_ecl, find_naive, blank_fun2);

			case ECL_COMPRESS:
				BFS_EXEC(ecl_fun, union_ecl,find_compress, blank_fun2);

			case ECL_HALVE:
				BFS_EXEC(ecl_fun, union_ecl,find_halve, blank_fun2);

			case ECL_SPLIT:
				BFS_EXEC(ecl_fun, union_ecl, find_split, blank_fun2);

			case ECL_A_HALVE:
				BFS_EXEC(ecl_fun, union_ecl,find_a_halve, blank_fun2);

			case ECL_A_SPLIT:
				BFS_EXEC(ecl_fun, union_ecl,find_a_split, blank_fun2);

			case AFFOREST_NAIVE:
				BFS_EXEC(blank_fun, union_afforest, find_naive, blank_fun2);

			case AFFOREST_COMPRESS:
				BFS_EXEC(blank_fun, union_afforest,find_compress, blank_fun2);

			case AFFOREST_HALVE:
				BFS_EXEC(blank_fun, union_afforest,find_halve, blank_fun2);

			case AFFOREST_SPLIT:
				BFS_EXEC(blank_fun, union_afforest, find_split, blank_fun2);

			case AFFOREST_A_HALVE:
				BFS_EXEC(blank_fun, union_afforest,find_a_halve, blank_fun2);

			case AFFOREST_A_SPLIT:
				BFS_EXEC(blank_fun, union_afforest,find_a_split, blank_fun2);


			case RAND_NAIVE:
				RAND_BFS_EXEC(blank_fun, union_rand, find_r_naive, blank_fun2);

			case RAND_SPLIT_2:
				RAND_BFS_EXEC(blank_fun, union_rand, find_r_split_2, blank_fun2);


			case REMCAS_NAIVE_SPLICE_CAS:
				BFS_EXEC(blank_fun, union_rem_fence, find_naive, splice_CAS);

			case REMCAS_COMPRESS_SPLICE_CAS:
				BFS_EXEC(blank_fun, union_rem_fence, find_compress, splice_CAS);

			case REMCAS_HALVE_SPLICE_CAS:
				BFS_EXEC(blank_fun, union_rem_fence, find_halve, splice_CAS);

			case REMCAS_SPLIT_SPLICE_CAS:
				BFS_EXEC(blank_fun, union_rem_fence, find_split, splice_CAS);

			case REMCAS_A_HALVE_SPLICE_CAS:
				BFS_EXEC(blank_fun, union_rem_fence, find_a_halve, splice_CAS);

			case REMCAS_A_SPLIT_SPLICE_CAS:
				BFS_EXEC(blank_fun, union_rem_fence, find_a_split, splice_CAS);


			case REMCAS_NAIVE_SPLIT_ONE:
				BFS_EXEC(blank_fun, union_rem, find_naive, split_one);

			case REMCAS_COMPRESS_SPLIT_ONE:
				BFS_EXEC(blank_fun, union_rem, find_compress, split_one);

			case REMCAS_HALVE_SPLIT_ONE:
				BFS_EXEC(blank_fun, union_rem, find_halve, split_one);

			case REMCAS_SPLIT_SPLIT_ONE:
				BFS_EXEC(blank_fun, union_rem, find_split, split_one);

			case REMCAS_A_HALVE_SPLIT_ONE:
				BFS_EXEC(blank_fun, union_rem, find_a_halve, split_one);

			case REMCAS_A_SPLIT_SPLIT_ONE:
				BFS_EXEC(blank_fun, union_rem, find_a_split, split_one);


			case REMCAS_NAIVE_HALVE_ONE:
				BFS_EXEC(blank_fun, union_rem, find_naive, halve_one);

			case REMCAS_COMPRESS_HALVE_ONE:
				BFS_EXEC(blank_fun, union_rem, find_compress, halve_one);

			case REMCAS_HALVE_HALVE_ONE:
				BFS_EXEC(blank_fun, union_rem, find_halve, halve_one);

			case REMCAS_SPLIT_HALVE_ONE:
				BFS_EXEC(blank_fun, union_rem, find_split, halve_one);

			case REMCAS_A_HALVE_HALVE_ONE:
				BFS_EXEC(blank_fun, union_rem, find_a_halve, halve_one);

			case REMCAS_A_SPLIT_HALVE_ONE:
				BFS_EXEC(blank_fun, union_rem, find_a_split, halve_one);


			case REMLOCK_NAIVE_SPLICE_CAS:
				BFS_EXEC(blank_fun, union_rem_lock_fence, find_naive, splice_CAS);

			case REMLOCK_COMPRESS_SPLICE_CAS:
				BFS_EXEC(blank_fun, union_rem_lock_fence, find_compress, splice_CAS);

			case REMLOCK_HALVE_SPLICE_CAS:
				BFS_EXEC(blank_fun, union_rem_lock_fence, find_halve, splice_CAS);

			case REMLOCK_SPLIT_SPLICE_CAS:
				BFS_EXEC(blank_fun, union_rem_lock_fence, find_split, splice_CAS);

			case REMLOCK_A_HALVE_SPLICE_CAS:
				BFS_EXEC(blank_fun, union_rem_lock_fence, find_a_halve, splice_CAS);

			case REMLOCK_A_SPLIT_SPLICE_CAS:
				BFS_EXEC(blank_fun, union_rem_lock_fence, find_a_split, splice_CAS);


			case REMLOCK_NAIVE_SPLIT_ONE:
				BFS_EXEC(blank_fun, union_rem_lock, find_naive, split_one);

			case REMLOCK_COMPRESS_SPLIT_ONE:
				BFS_EXEC(blank_fun, union_rem_lock, find_compress, split_one);

			case REMLOCK_HALVE_SPLIT_ONE:
				BFS_EXEC(blank_fun, union_rem_lock, find_halve, split_one);

			case REMLOCK_SPLIT_SPLIT_ONE:
				BFS_EXEC(blank_fun, union_rem_lock, find_split, split_one);

			case REMLOCK_A_HALVE_SPLIT_ONE:
				BFS_EXEC(blank_fun, union_rem_lock, find_a_halve, split_one);

			case REMLOCK_A_SPLIT_SPLIT_ONE:
				BFS_EXEC(blank_fun, union_rem_lock, find_a_split, split_one);


			case REMLOCK_NAIVE_HALVE_ONE:
				BFS_EXEC(blank_fun, union_rem_lock, find_naive, halve_one);

			case REMLOCK_COMPRESS_HALVE_ONE:
				BFS_EXEC(blank_fun, union_rem_lock, find_compress, halve_one);

			case REMLOCK_HALVE_HALVE_ONE:
				BFS_EXEC(blank_fun, union_rem_lock, find_halve, halve_one);

			case REMLOCK_SPLIT_HALVE_ONE:
				BFS_EXEC(blank_fun, union_rem_lock, find_split, halve_one);

			case REMLOCK_A_HALVE_HALVE_ONE:
				BFS_EXEC(blank_fun, union_rem_lock, find_a_halve, halve_one);

			case REMLOCK_A_SPLIT_HALVE_ONE:
				BFS_EXEC(blank_fun, union_rem_lock, find_a_split, halve_one);



			case SIM_P_U_S:
				SIM_BFS_EXEC(sim_p_u, 0, 0, 0, 0, sampling_phase3_all);
			case SIM_P_R_S:
				SIM_BFS_EXEC(sim_p_r, 0, 0, 0, 1, sampling_phase3_all);
			case SIM_E_U_S:
				SIM_BFS_EXEC(sim_e_u, 0, 0, 0, 0, sampling_phase3_all);

			case SIM_P_U_SS:
				SIM_BFS_EXEC(sim_p_u, 1, 0, 0, 0, sampling_phase3_all);
			case SIM_P_R_SS:
				SIM_BFS_EXEC(sim_p_r, 1, 1, 0, 1, sampling_phase3_all);
			case SIM_E_U_SS:
				SIM_BFS_EXEC(sim_e_u, 1, 0, 0, 0, sampling_phase3_all);

			case STERGIOS:
				STER_BFS_EXEC(ster, 1, 0, 0, 0, sampling_phase3_all);
			case SVA:
				SV_BFS_EXEC(sv, 1, 0, 0, 0, sampling_phase3_all);
			case LPA:
				LP_BFS_EXEC(lp, 1, 0, 0, 0, sampling_phase3_all);

			case BFS_CC:
        			bfs_init(d_input);
				cudaMalloc((void **)&(d_input->_s_vertex), sizeof(uintT));
         			T_START
				uintT s_vertex = 0;
				bfs_run<0>(s_vertex, *d_input);
				while(1) {
					cudaMemset(d_input->_s_vertex, -1, sizeof(uintT));
					get_next_vertex<<<mp, SBSIZE>>>(s_vertex+1, d_input->V, d_input->parent, d_input->_s_vertex);
					cudaMemcpy(&s_vertex, d_input->_s_vertex, sizeof(uintT), cudaMemcpyDeviceToHost);
					if(s_vertex >= (d_input->V)) break;					
					bfs_run<0>(s_vertex, *d_input);
				}			
        			T_END
				cudaMemcpy(d_input->label, d_input->parent, sizeof(uintT)*d_input->V, cudaMemcpyDeviceToDevice);
				cudaFree(d_input->_s_vertex);
				break;
        

		}

		cudaFree(d_input->_Vflag); cudaFree(d_input->_Eflag); cudaFree(d_input->_Vfront); cudaFree(d_input->_Rfront);

	} 
	
	cudaFree(d_input->_Efront); 

	return((double)tot_ms);
}

double run_union_find(struct graph_data *h_input, struct graph_data *d_input)
{
	copy_cpu_to_gpu(h_input, d_input);
	double tot_ms = process(d_input);
	copy_gpu_to_cpu(h_input, d_input);
	return (tot_ms);
}

#endif
