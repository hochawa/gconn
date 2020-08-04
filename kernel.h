#ifndef __KERNEL_DEFINED
#define __KERNEL_DEFINED

//#ifdef PRE_INCLUDE
#include "common.h"
#include "union_primitives.h"
//#endif
#include <curand.h>
#include <curand_kernel.h>
using namespace cooperative_groups;

template <int is_long>
__global__ void edge_relabeling(uintT E, struct graph_data d_input)
{
        uintT idx = blockIdx.x * blockDim.x + threadIdx.x + d_input.offset;
        if(idx < E) {
		if(is_long == 0) {
			d_input.src_idx[idx] = d_input.parent[d_input.src_idx[idx]];
			d_input.dst_idx[idx] = d_input.parent[d_input.dst_idx[idx]];
		} else if(is_long == 1) {
			if(!(d_input.lparent[d_input.src_idx[idx]] & ULONG_T_MAX)) d_input.src_idx[idx] = (d_input.lparent[d_input.src_idx[idx]] & (UINT_T_MAX-1));
			if(!(d_input.lparent[d_input.dst_idx[idx]] & ULONG_T_MAX)) d_input.dst_idx[idx] = (d_input.lparent[d_input.dst_idx[idx]] & (UINT_T_MAX-1));
		} else {
			d_input.src_idx[idx] = MAX(d_input.parent[d_input.src_idx[idx]+d_input.V], d_input.parent[d_input.src_idx[idx]]);
			d_input.dst_idx[idx] = MAX(d_input.parent[d_input.dst_idx[idx]+d_input.V], d_input.parent[d_input.dst_idx[idx]]);
		}
        }
}


__global__ void rand_gen(struct graph_data d_input, uintT *Efront)
{
        uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
        uintT tot_size = CEIL(d_input.E,32);
        if(idx < tot_size) {
                curandState_t state;
                curand_init(idx, 0, 0, &state);
                Efront[idx] = curand(&state);
        }
}

__global__ void comp_tree(uintT V, struct graph_data d_input)
{
        uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < V) {
		while(1) {
			uintT prev = PARENT(idx);
			PARENTW(idx) = PARENT(prev);
			if(PARENT(idx) == prev) break;
		}
        }

}


__global__ void compL_tree(uintT V, struct graph_data d_input)
{
	uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
	uintT i = idx;
	if(idx < V) {
		if(d_input.lparent[i] & ULONG_T_MAX) return;
		while(1) {
			if(d_input.lparent[i] & ULONG_T_MAX) break;
			uintT pid = (d_input.lparent[i] & (UINT_T_MAX-1));
			i = pid;
		}
		d_input.lparent[idx] = i;
		
	}

}


__global__ void compL_tree00(uintT V, struct graph_data d_input)
{
	uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
	uintT i = idx;
	uintT pid_v;
	if(idx < V) {
		uintT init_pid = (d_input.lparent[i] & (UINT_T_MAX-1));
		while(1) {
			ulongT pid = d_input.lparent[i];
			pid_v = (pid & (UINT_T_MAX-1));
			if(pid & ULONG_T_MAX) break;
			i = pid_v;
		}
	}
}

__device__ inline void update_bin(uintT idx, struct graph_data d_input, int iter)
{
	uintT src = d_input.src_idx[idx];
	uintT dst = d_input.src_idx[idx];

        int bin_src_num = (int)log2f(1+src/BASE_C);
        int bin_dst_num = (int)log2f(1+dst/BASE_C);

	if(d_input.bin_p1[bin_src_num] == 0) d_input.bin_p1[bin_src_num] = iter;
	if(d_input.bin_p1[bin_dst_num] == 0) d_input.bin_p1[bin_dst_num] = iter;
}

template <csr_fun UDF0, union_fun UDF, uintT is_asym, uintT is_finish>
__global__ void union_find_gpu_COO(uintT E, struct graph_data d_input)
{
        uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ uintT sm_Eflag;
	if(is_finish) {
        	if(threadIdx.x == 0) sm_Eflag = 0;
	        __syncthreads();
	}

	#if defined(STREAMING_SYNC)
	grid_group grid = this_grid();
	int iter=1;
    for(; idx < CEIL(E, blockDim.x*gridDim.x)*blockDim.x*gridDim.x; idx += blockDim.x*gridDim.x, iter++) {	

	if(idx < E) update_bin(idx, d_input, iter);
	grid.sync();
	__threadfence();	
	
	for(uintT k=0; k<MAX_BIN; k++) {
		if(d_input.bin_p1[k] == iter) {
			if(threadIdx.x == 0 && blockIdx.x == 0) printf("%d %d\n", k, iter);
			uintT bin_size = powf(2, k)*BASE_C;
		        uintT bin_base = (powf(2,k)-1)*BASE_C;
			for(uintT id=blockIdx.x * blockDim.x + threadIdx.x; id < bin_size; id += blockDim.x * gridDim.x) {
				PARENTW(bin_base + id) = PARENT(bin_base + id);	
			} 
		}
	}
	
	grid.sync();
	__threadfence();	
	#endif

        if(idx < E) {
		uintT sv = UDF0(d_input.src_idx[idx], d_input);
		if(is_asym || d_input.src_idx[idx] < d_input.dst_idx[idx]) {
	                bool r = UDF(sv, idx, d_input.src_idx[idx], d_input.dst_idx[idx], d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
			}
		}
        }
	#if defined(STREAMING_SYNC)
    }
	#endif

	if(is_finish) {
	        __syncthreads();
	        if(threadIdx.x == 0 && sm_Eflag == 1) {
	                *d_input._Eflag = 1;
	        }
	}
}

template <csr_fun UDF0, union_fun UDF, uintT is_asym, uintT is_finish>
__global__ void union_find_gpu_COO_SAMPLE(uintT E, struct graph_data d_input)
{
	__shared__ uintT sm_Eflag;
	if(is_finish) {
        	if(threadIdx.x == 0) sm_Eflag = 0;
	        __syncthreads();
	}
	uintT idx = blockIdx.x * blockDim.x + threadIdx.x + d_input.offset;
	if(idx < E) {
		uintT sv = UDF0(d_input.src_idx[idx], d_input);
		bool r = UDF(sv, idx, d_input.src_idx[idx], d_input.dst_idx[idx], d_input);
		if(is_finish) {
			if(r) sm_Eflag = 1;
		}
	}

	if(is_finish) {
	        __syncthreads();
	        if(threadIdx.x == 0 && sm_Eflag == 1) {
	                *d_input._Eflag = 1;
	        }
	}
}


template <csr_fun UDF0, union_fun UDF, uintT is_asym, uintT is_finish>
__global__ void union_find_gpu_CSR(uintT tot_size, struct graph_data d_input)
{
        __shared__ int buffer1[SBSIZE/4], buffer2[SBSIZE/4], buffer3[SBSIZE/4], buffer4[SBSIZE/4];
        __shared__ uintT buffer_p[3];
	__shared__ uintT sm_Eflag;

        int base_idx = ((blockIdx.x*blockDim.x + threadIdx.x)>>2);
        int warp_id = ((blockIdx.x*blockDim.x + threadIdx.x>>5)&3);
        int index_size, bias; uintT index;

        if(threadIdx.x < 3) {
                buffer_p[threadIdx.x] = 0;
        }
	if(is_finish) {
        	if(threadIdx.x == 0) sm_Eflag = 0;
	}
        __syncthreads();

        if(base_idx < tot_size) {
                bias = 0;
                index = base_idx;
                index_size = d_input.csr_ptr[index+1] - d_input.csr_ptr[index];
                if(index_size >= 32) {
                        bias = index_size - (index_size&31);
                        if((index_size & (32+64+128+256)) && (threadIdx.x&3) == 0) {
                                uintT p = atomicAggInc(&buffer_p[0]);
                                buffer1[p] = index;
                                buffer2[p] = bias;
                        }
                        if(index_size >= SBSIZE) {
                                if((threadIdx.x&3) == 0) {
					uintT p2 = atomicAggInc(&buffer_p[1]);
					buffer3[p2] = index;
					buffer4[p2] = index_size - (index_size&(SBSIZE-1));
				}
                        }
                }
        }
        __syncthreads();

        if(base_idx < tot_size) {
		uintT sv = UDF0(index, d_input);
                for(uintT i=bias+(threadIdx.x&3); i<index_size; i+=4) {
                        uintT dst_idx = d_input.dst_idx[d_input.csr_ptr[index]+i];
			if(is_asym || index < dst_idx) {
	                        bool r = UDF(sv, d_input.csr_ptr[index]+i, index, dst_idx, d_input);
				if(is_finish) {
					if(r) sm_Eflag = 1;
				}
			}
                }
        }

        uintT upper = buffer_p[0];
        uintT i;
        while(1) {
                if((threadIdx.x&31) == 0) i = atomicAdd(&buffer_p[2], 1);
                i = __shfl_sync(-1, i, 0);
                if(i >= upper) break;
                index = buffer1[i];
                int bf2 = buffer2[i];
                int bf22 = bf2 - (bf2&(SBSIZE-1));
		uintT sv = UDF0(index, d_input);
                for(int j=bf22+(threadIdx.x&31);j<bf2;j+=32) {
                        uintT dst_idx = d_input.dst_idx[d_input.csr_ptr[index]+j];
			if(is_asym || index < dst_idx) {
                        	bool r = UDF(sv, d_input.csr_ptr[index]+j, index, dst_idx, d_input);
				if(is_finish) {
					if(r) sm_Eflag = 1;
				}
			}
                }
        }
        for(uintT i=0;i<buffer_p[1];i++) {
                index = buffer3[i];
		uintT sv = UDF0(index, d_input);
                for(int j=threadIdx.x;j<buffer4[i];j+=blockDim.x) {
                        uintT dst_idx = d_input.dst_idx[d_input.csr_ptr[index]+j];
			if(is_asym || index < dst_idx) {
                        	bool r = UDF(sv, d_input.csr_ptr[index]+j, index, dst_idx, d_input);
				if(is_finish) {
					if(r) sm_Eflag = 1;
				}
			}
                }
        }
	if(is_finish) {
	        __syncthreads();
	        if(threadIdx.x == 0 && sm_Eflag == 1) {
	                *d_input._Eflag = 1;
	        }
	}
}

__global__ void relabeling(uintT tot_size, struct graph_data d_input, uintT mc)
{
	uintT idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < tot_size) {
		uintT k = d_input.parent[idx];
		if(k == mc) d_input.parent[idx] = 0;
		else if(k == 0) d_input.parent[idx] = mc;
	}
}

#endif
