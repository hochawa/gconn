#ifndef __TARJAN_DEFINED
#define __TARJAN_DEFINED

#include "common.h"
#include "kernel.h"
#include "sample.h"

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool sim_e_u (uintT vstat, uintT idx, uintT src0, uintT dst0, struct graph_data d_input)
{
	bool active = (idx < d_input.E);
	bool flag = false;
	if(active) {
		uintT src = PARENT(src0);
		uintT dst = PARENT(dst0);
		if(src != dst) {	
			if(src > dst) { uintT tmp = src; src = dst; dst = tmp; }
			if(src < PARENT(dst)) {
				PARENTW(dst) = src;
				#if defined(SP_TREE)
				//d_input.hook[dst] = idx;
				atomicCAS(&d_input.hook[dst], -1, idx);;
				#endif
				flag = true;
			}
			if(src < PARENT(dst0)) {
				PARENTW(dst0) = src;
				#if defined(SP_TREE)
				//d_input.hook[dst0] = idx;
				atomicCAS(&d_input.hook[dst0], -1, idx);;
				#endif
				flag = true;
			}
		}
	}
	return flag;
}

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool sim_e_u_a (uintT vstat, uintT idx, uintT src0, uintT dst0, struct graph_data d_input)
{
	uintT idx_q = idx/32;
	uintT idx_r = idx - idx_q*32;
	bool active = ((idx < d_input.E) && ((d_input._Efront[idx_q] & (1<<(idx_r))) != 0));
	bool flag = false;
	if(active) {
		uintT src = PARENT(src0);
		uintT dst = PARENT(dst0);
		if(src != dst) {	
			if(src > dst) { uintT tmp = src; src = dst; dst = tmp; }
			if(src < PARENT(dst)) {
				PARENTW(dst) = src;
				#if defined(SP_TREE)
				//d_input.hook[dst] = idx;
				atomicCAS(&d_input.hook[dst], -1, idx);;
				#endif
				flag = true;
			}
			if(src < PARENT(dst0)) {
				PARENTW(dst0) = src;
				#if defined(SP_TREE)
				//d_input.hook[dst0] = idx;
				atomicCAS(&d_input.hook[dst0], -1, idx);;
				#endif
				flag = true;
			}
		}
	}
	return flag;
}

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool sim_p_u (uintT vstat, uintT idx, uintT src0, uintT dst0, struct graph_data d_input)
{
	bool active = (idx < d_input.E);
	bool flag = false;
	if(active) {
		uintT src = PARENT(src0);
		uintT dst = PARENT(dst0);
		if(src != dst) {	
			if(src > dst) { uintT tmp = src; src = dst; dst = tmp; }
			if(src < PARENT(dst)) {
				PARENTW(dst) = src;
				#if defined(SP_TREE)
				//d_input.hook[dst] = idx;
				atomicCAS(&d_input.hook[dst], -1, idx);;
				#endif
				flag = true;
			}
		}
	}
	return flag;
}

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool sim_p_u_a (uintT vstat, uintT idx, uintT src0, uintT dst0, struct graph_data d_input)
{
	uintT idx_q = idx/32;
	uintT idx_r = idx - idx_q*32;
	bool active = ((idx < d_input.E) && ((d_input._Efront[idx_q] & (1<<(idx_r))) != 0));
	bool flag = false;
	if(active) {
		uintT src = PARENT(src0);
		uintT dst = PARENT(dst0);
		if(src != dst) {	
			if(src > dst) { uintT tmp = src; src = dst; dst = tmp; }
			if(src < PARENT(dst)) {
				PARENTW(dst) = src;
				#if defined(SP_TREE)
				//d_input.hook[dst] = idx;
				atomicCAS(&d_input.hook[dst], -1, idx);;
				#endif
				flag = true;
			}
		}
	}
	return flag;
}

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool sim_c_u_a (uintT vstat, uintT idx, uintT src0, uintT dst0, struct graph_data d_input)
{
	uintT idx_q = idx/32;
	uintT idx_r = idx - idx_q*32;
	bool active = ((idx < d_input.E) && ((d_input._Efront[idx_q] & (1<<(idx_r))) != 0));
	bool flag = false;
	if(active) {
		uintT src = PARENT(src0);
		uintT dst = PARENT(dst0);
		if(src0 != dst0) {	
			if(src0 > dst0) { uintT tmp = src0; src0 = dst0; dst0 = tmp; }
			if(src0 < PARENT(dst0)) {
				PARENTW(dst0) = src0;
				#if defined(SP_TREE)
				//d_input.hook[dst0] = idx;
				atomicCAS(&d_input.hook[dst0], -1, idx);;
				#endif
				flag = true;
			}
		}
	}
	return flag;
}

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool sim_p_r (uintT vstat, uintT idx, uintT src0, uintT dst0, struct graph_data d_input)
{
	bool active = (idx < d_input.E);
	bool flag = false;
	if(active) {
		uintT src = PARENT(src0);
		uintT dst = PARENT(dst0);

		uintT src_q = src/32;
		uintT src_r = src - src_q*32;
		uintT dst_q = dst/32;
		uintT dst_r = dst - dst_q*32;

		bool src_root = (d_input._Rfront[src_q] & (1<<src_r));
		bool dst_root = (d_input._Rfront[dst_q] & (1<<dst_r));
		if(src_root || dst_root) {
			if(src_root && dst < src) {
				PARENTW(src) = dst;
				#if defined(SP_TREE)
				//d_input.hook[src] = idx;
				atomicCAS(&d_input.hook[src], -1, idx);;
				#endif
				flag = true;
			}
			if(dst_root && src < dst) {
				PARENTW(dst) = src;
				#if defined(SP_TREE) 
				//d_input.hook[dst] = idx;
				atomicCAS(&d_input.hook[dst], -1, idx);;
				#endif
				flag = true;
			}
		}
	}
	return flag;
}

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool sim_p_r_a (uintT vstat, uintT idx, uintT src0, uintT dst0, struct graph_data d_input)
{
	uintT idx_q = idx/32;
	uintT idx_r = idx - idx_q*32;

	uintT idx2 = idx - d_input.offset;
	uintT idx_q2 = idx2/32;
	uintT idx_r2 = idx2 - idx_q2*32;

	bool active = ((idx < d_input.E) && ((d_input._Efront[idx_q2] & (1<<(idx_r2))) != 0));
	bool flag = false;
	if(active) {
		uintT src = PARENT(src0);
		uintT dst = PARENT(dst0);

		uintT src_q = src/32;
		uintT src_r = src - src_q*32;
		uintT dst_q = dst/32;
		uintT dst_r = dst - dst_q*32;

		bool src_root = (d_input._Rfront[src_q] & (1<<src_r));
		bool dst_root = (d_input._Rfront[dst_q] & (1<<dst_r));
		if(src_root || dst_root) {
			if(src_root && dst < src) {
				PARENTW(src) = dst;
				#if defined(SP_TREE)
				//d_input.hook[src] = idx;
				atomicCAS(&d_input.hook[src], -1, idx);;
				#endif
				flag = true;
			}
			if(dst_root && src < dst) {
				PARENTW(dst) = src;
				#if defined(SP_TREE)
				//d_input.hook[dst] = idx;
				atomicCAS(&d_input.hook[dst], -1, idx);;
				#endif
				flag = true;
			}
		}
	}
	return flag;
}

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool sim_c_r_a (uintT vstat, uintT idx, uintT src0, uintT dst0, struct graph_data d_input)
{
	uintT idx_q = idx/32;
	uintT idx_r = idx - idx_q*32;

	uintT idx2 = idx - d_input.offset;
	uintT idx_q2 = idx2/32;
	uintT idx_r2 = idx2 - idx_q2*32;

	bool active = ((idx < d_input.E) && ((d_input._Efront[idx_q2] & (1<<(idx_r2))) != 0));
	bool flag = false;
	if(active) {
		uintT src = PARENT(src0);
		uintT dst = PARENT(dst0);

		if(src != dst) {
			//uintT src_q = src/32;
			//uintT src_r = src - src_q*32;
			//uintT dst_q = dst/32;
			//uintT dst_r = dst - dst_q*32;

			uintT src_q = src0/32;
			uintT src_r = src0 - src_q*32;
			uintT dst_q = dst0/32;
			uintT dst_r = dst0 - dst_q*32;

			bool src_root = (d_input._Rfront[src_q] & (1<<src_r));
			bool dst_root = (d_input._Rfront[dst_q] & (1<<dst_r));
			if(src_root || dst_root) {
				if(src_root && dst0 < src0) {
					PARENTW(src0) = dst0;
					#if defined(SP_TREE)
					//d_input.hook[src0] = idx;
					atomicCAS(&d_input.hook[src0], -1, idx);
					#endif
					flag = true;
				}
				if(dst_root && src0 < dst0) {
					PARENTW(dst0) = src0;
					#if defined(SP_TREE)
					//d_input.hook[dst0] = idx;
					atomicCAS(&d_input.hook[dst0], -1, idx);
					#endif
					flag = true;
				}
			}
		}
	}
	return flag;
}

__global__ void alter_edges_chunk00 (uintT tot_size, struct graph_data d_input)
{
#ifdef CHUNK_STREAMING
	uintT idx = blockIdx.x * blockDim.x + threadIdx.x + d_input.offset;

	__shared__ uintT sm_Eflag;
	if(threadIdx.x == 0) sm_Eflag = 0;
	__syncthreads();
#else
	uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
#endif  

	 if(idx < CEIL(tot_size,32)*32) {
		uintT idx_q = idx/32;
		uintT idx_r = idx - idx_q*32;

		uintT mask = __activemask();
		uintT inv_mask = 0xFFFFFFFF - mask;
		bool active = ((idx < tot_size) && ((d_input._Efront[idx_q] & (1<<(idx_r))) != 0));
		if (active) {
			uintT src0 = d_input.src_idx[idx];
			uintT dst0 = d_input.dst_idx[idx];
			uintT src = PARENT(src0);
			uintT dst = PARENT(dst0);
			if(src != dst) {
				d_input.src_idx[idx] = src; 
				d_input.dst_idx[idx] = dst;
				//if(src0 != src || dst0 != dst) sm_Eflag = 1;	
			} else active = false;
		}

	}

#ifdef CHUNK_STREAMING
	__syncthreads();
	if(threadIdx.x == 0 && sm_Eflag == 1) {
		*d_input._Eflag = 1;
	}
#endif
}

__global__ void alter_edges_chunk0 (uintT tot_size, struct graph_data d_input)
{
#ifdef CHUNK_STREAMING
	uintT idx = blockIdx.x * blockDim.x + threadIdx.x + d_input.offset;
#else
	uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
#endif  
	 if(idx < CEIL(tot_size,32)*32) {
		uintT idx_q = idx/32;
		uintT idx_r = idx - idx_q*32;

		bool active = (idx < tot_size);
		if (active) {
			uintT src0 = d_input.src_idx[idx];
			uintT dst0 = d_input.dst_idx[idx];
			uintT src = PARENT(src0);
			uintT dst = PARENT(dst0);
			if(src != dst) {
				d_input.src_idx[idx] = src; 
				d_input.dst_idx[idx] = dst;
			} else active = false;
		}
		d_input._Efront[idx_q] = __ballot_sync(-1, active);
	}
}

__global__ void alter_edges_chunk (uintT tot_size, struct graph_data d_input)
{
	__shared__ uintT sm_Eflag;
	if(threadIdx.x == 0) sm_Eflag = 0;
	__syncthreads();

	uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
	 if(idx < CEIL(tot_size,32)*32) {
		uintT idx_q = idx/32;
		uintT idx_r = idx - idx_q*32;

		bool active = ((idx < tot_size) && ((d_input._Efront[idx_q] & (1<<(idx_r))) != 0));
		if (active) {
			uintT src0 = d_input.src_idx[idx+d_input.offset];
			uintT dst0 = d_input.dst_idx[idx+d_input.offset];
			uintT src = PARENT(src0);
			uintT dst = PARENT(dst0);
			if(src != dst) {
				d_input.src_idx[idx+d_input.offset] = src; 
				d_input.dst_idx[idx+d_input.offset] = dst;
				if(src0 != src || dst0 != dst) sm_Eflag = 1;	
			} else active = false;
		}
		d_input._Efront[idx_q] = __ballot_sync(-1, active); 
	}

	__syncthreads();
	if(threadIdx.x == 0 && sm_Eflag == 1) {
		*d_input._Eflag = 1;
	}

}

__global__ void alter_edges_chunk_CS (uintT tot_size, struct graph_data d_input)
{
	__shared__ uintT sm_Eflag;
	if(threadIdx.x == 0) sm_Eflag = 0;
	__syncthreads();

	uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
	 if(idx < CEIL(tot_size,32)*32) {
		uintT idx_q = idx/32;
		uintT idx_r = idx - idx_q*32;

		bool active = ((idx < tot_size) && ((d_input._Efront[idx_q] & (1<<(idx_r))) != 0));
		if (active) {
			uintT src0 = d_input.src_idx[idx+d_input.offset];
			uintT dst0 = d_input.dst_idx[idx+d_input.offset];
			uintT src = PARENT(src0);
			uintT dst = PARENT(dst0);
			if(src != dst) {
				d_input.src_idx[idx+d_input.offset] = src; 
				d_input.dst_idx[idx+d_input.offset] = dst;
				if(src0 != src || dst0 != dst) sm_Eflag = 1;	
			} else active = false;
		}
		d_input._Efront[idx_q] = __ballot_sync(-1, active); 
	}

	__syncthreads();
	if(threadIdx.x == 0 && sm_Eflag == 1) {
		*d_input._Eflag = 1;
	}

}




__global__ void alter_edges (uintT tot_size, struct graph_data d_input)
{
	uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
	 if(idx < CEIL(tot_size,32)*32) {
		uintT idx_q = idx/32;
		uintT idx_r = idx - idx_q*32;

		bool active = ((idx < tot_size) && ((d_input._Efront[idx_q] & (1<<(idx_r))) != 0));
		if (active) {
			uintT src0 = d_input.src_idx[idx];
			uintT dst0 = d_input.dst_idx[idx];
			uintT src = PARENT(src0);
			uintT dst = PARENT(dst0);
			if(src != dst) {
				d_input.src_idx[idx] = src; 
				d_input.dst_idx[idx] = dst;
			} else active = false;
		}
		d_input._Efront[idx_q] = __ballot_sync(-1, active); 
	}
}

__global__ void shortcut_with_front(uintT tot_size, struct graph_data d_input)
{
	__shared__ uintT sm_Vflag;
	uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadIdx.x == 0) sm_Vflag = 0;
	__syncthreads();
        if(idx < CEIL(tot_size,32)*32) {
		uintT idx_q = idx/32;
		uintT idx_r = idx - idx_q*32;

		
		bool active = ((idx < tot_size) && ((d_input._Vfront[idx_q] & (1<<(idx_r))) != 0)  && PARENT(idx) != PARENT(PARENT(idx)));
 		if(active) {
			PARENTW(idx) = PARENT(PARENT(idx)); 
			sm_Vflag = 1;
		}
		d_input._Vfront[idx_q] = __ballot_sync(-1, active);
	}
	__syncthreads();
	if(threadIdx.x == 0 && sm_Vflag == 1) {
		*d_input._Vflag = 1;
	}
}

__global__ void shortcut_without_front(uintT tot_size, struct graph_data d_input)
{
	__shared__ uintT sm_Vflag;
	uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadIdx.x == 0) sm_Vflag = 0;
	__syncthreads();
        if(idx < tot_size) {
     		if(PARENT(idx) != PARENT(PARENT(idx))) {
			PARENTW(idx) = PARENT(PARENT(idx));
			sm_Vflag = 1;
		}
	}
	__syncthreads();
	if(threadIdx.x == 0 && sm_Vflag == 1) {
		*d_input._Vflag = 1;
	}
}

__global__ void find_roots(uintT tot_size, struct graph_data d_input)
{
	uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < CEIL(tot_size,32)*32) {
		uintT idx_q = idx/32;
		uintT idx_r = idx - idx_q*32;

		bool active = ((idx < tot_size) && ((d_input._Rfront[idx_q] & (1<<(idx_r))) != 0) && (idx == PARENT(idx)));
		d_input._Rfront[idx_q] = __ballot_sync(-1, active);
	}
}	

template <STR_fun UDF, STR_fun UDF2, int Vfront_active, int Efront_active, int alter_active, int Rfront_active, int sample_active>
void union_find_tarjan(uintT f_size, struct graph_data d_input)
{
	uintT V_bitsize = CEIL(d_input.V, 32);
	uintT E_bitsize = CEIL(d_input.E, 32);
	uintT Vflag, Eflag;
	uintT V_frontsize = d_input.V;
	uintT E_frontsize = f_size;
	if(E_frontsize == V_frontsize) E_frontsize *= 4;

	if(Efront_active) {
		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
		cudaMemset(d_input._Efront, -1, sizeof(uintT)*E_bitsize);
	} 
	if(Rfront_active) {
		cudaMemset(d_input._Rfront, -1, sizeof(uintT)*V_bitsize);
		
	}
	while(1) {
		UDF<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(f_size, d_input);
		if(sample_active && d_input.is_sym == 0) {
			UDF2<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(f_size, d_input);
		}
		cudaMemcpy(&Eflag, d_input._Eflag, sizeof(uintT), cudaMemcpyDeviceToHost);
		if(Vfront_active) {
			cudaMemset(d_input._Vflag, 0, sizeof(uintT));
			cudaMemset(d_input._Vfront, -1, sizeof(uintT)*V_bitsize);
			while(1) {
				shortcut_with_front<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	
				cudaMemcpy(&Vflag, d_input._Vflag, sizeof(uintT), cudaMemcpyDeviceToHost);
				if (Vflag == 0) break;
				cudaMemset(d_input._Vflag, 0, sizeof(uintT));
			}
		} else {
			cudaMemset(d_input._Vflag, 0, sizeof(uintT));
			shortcut_without_front<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	
			cudaMemcpy(&Vflag, d_input._Vflag, sizeof(uintT), cudaMemcpyDeviceToHost);
		}
		if(Eflag == 0 && Vflag == 0) break;
		if(Rfront_active) {
			find_roots<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	
		}

		
		if(alter_active) {
			alter_edges<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(E_frontsize, d_input);
		}

		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
	}
}

template <STR_fun UDF, STR_fun UDF2, int Vfront_active, int Efront_active, int alter_active, int Rfront_active, int sample_active>
void union_find_tarjan_CHUNK(uintT f_size, struct graph_data d_input)
{
	uintT V_bitsize = CEIL(d_input.V, 32);
	uintT E_bitsize = CEIL(d_input.E, 32);
	uintT Vflag, Eflag;
	uintT V_frontsize = d_input.V;
	uintT E_frontsize = f_size;

	if(Efront_active) {
		cudaMemset(&d_input._Eflag, 0, sizeof(uintT));
		cudaMemset(d_input._Efront, -1, sizeof(uintT)*CEIL(d_input.size,32));
	} 
	if(Rfront_active) {
		if(d_input.offset == 0) cudaMemset(d_input._Rfront, -1, sizeof(uintT)*V_bitsize);
		
	}

	while(1) {
		UDF<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(d_input.offset+f_size, d_input);
		if(sample_active && d_input.is_sym == 0) {
			UDF2<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(d_input.offset+f_size, d_input);
		}
		if(Vfront_active) {
			cudaMemset(d_input._Vflag, 0, sizeof(uintT));
			cudaMemset(d_input._Vfront, -1, sizeof(uintT)*V_bitsize);
			while(1) {
				shortcut_with_front<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	
				cudaMemcpy(&Vflag, d_input._Vflag, sizeof(uintT), cudaMemcpyDeviceToHost);
				if (Vflag == 0) break;
				cudaMemset(d_input._Vflag, 0, sizeof(uintT));
			}
		} else {
			cudaMemset(d_input._Vflag, 0, sizeof(uintT));
			shortcut_without_front<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	
			cudaMemcpy(&Vflag, d_input._Vflag, sizeof(uintT), cudaMemcpyDeviceToHost);
		}


		if(alter_active) {
			alter_edges_chunk<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(d_input.size, d_input);
		}
		cudaMemcpy(&Eflag, d_input._Eflag, sizeof(uintT), cudaMemcpyDeviceToHost);
		if(Eflag == 0 && Vflag == 0) break;

		if(Rfront_active) {
			find_roots<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);
		}

		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
	}
}


template <STR_fun UDF, STR_fun UDF2, int Vfront_active, int Efront_active, int alter_active, int Rfront_active, int sample_active>
void union_find_tarjan_COO_SAMPLE(uintT f_size, struct graph_data d_input)
{
	uintT V_bitsize = CEIL(d_input.V, 32);
	uintT E_bitsize = CEIL(d_input.E, 32);
	uintT Vflag, Eflag;
	uintT V_frontsize = d_input.V;
	uintT E_frontsize = f_size;
	if(E_frontsize == V_frontsize) E_frontsize *= 4;



	bool init=false;
	while(1) {

	if(Efront_active) {
		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
		cudaMemset(d_input._Efront, -1, sizeof(uintT)*E_bitsize);
	} 
	if(Rfront_active) {
		cudaMemset(d_input._Rfront, -1, sizeof(uintT)*V_bitsize);
		
	}



		for(uintT gran = 0; gran < d_input.E; gran += d_input.coo_sample_size) {
			d_input.offset = gran;
			d_input.size = MIN(d_input.E, gran + d_input.coo_sample_size) - gran;
			if(init) edge_relabeling<0><<<CEIL(d_input.size, SBSIZE), SBSIZE>>>(d_input.E, d_input);
			init = true;
			UDF<<<CEIL(d_input.size, SBSIZE), SBSIZE>>>(f_size, d_input);
			if(sample_active && d_input.is_sym == 0) {
				UDF2<<<CEIL(d_input.size, SBSIZE), SBSIZE>>>(f_size, d_input);
			}
		}
		if(Vfront_active) {
			cudaMemset(d_input._Vflag, 0, sizeof(uintT));
			cudaMemset(d_input._Vfront, -1, sizeof(uintT)*V_bitsize);
			while(1) {
				shortcut_with_front<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	
				cudaMemcpy(&Vflag, d_input._Vflag, sizeof(uintT), cudaMemcpyDeviceToHost);
				if (Vflag == 0) break;
				cudaMemset(d_input._Vflag, 0, sizeof(uintT));
			}
		} else {
			cudaMemset(d_input._Vflag, 0, sizeof(uintT));
			shortcut_without_front<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	
			cudaMemcpy(&Vflag, d_input._Vflag, sizeof(uintT), cudaMemcpyDeviceToHost);
		}

		d_input.offset = 0;
		if(alter_active) {
			alter_edges_chunk<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(E_frontsize, d_input);
		}
		cudaMemcpy(&Eflag, d_input._Eflag, sizeof(uintT), cudaMemcpyDeviceToHost);
		if(Eflag == 0 && Vflag == 0) break;


		if(Rfront_active) {
			find_roots<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	
		}

		cudaMemset(d_input._Eflag, 0, sizeof(uintT));

	}
}


#endif
