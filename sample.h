#ifndef __SAMPLE_DEFINED
#define __SAMPLE_DEFINED

#include "common.h"
#include "kernel.h"
#include "sample_def.h"
#include "ihook_def.h"
#include <curand.h>
#include <curand_kernel.h>

__global__ void gen_ff(uintT um, struct graph_data d_input, uintT *q, uintT *qp)
{
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < um) {
                if(PARENT(index) != d_input.max_c) {
                        int pnt = atomicAggInc(qp);
                        q[pnt] = index;
                } 
        }
}

template <csr_fun UDF0, union_fun UDF>
__global__ void sampling_phase1(uintT tot_size, struct graph_data d_input, uintT k)
{       
        __shared__ int buffer1[SBSIZE/4], buffer2[SBSIZE/4], buffer3[SBSIZE/4], buffer4[SBSIZE/4];
        __shared__ uintT buffer_p[3];
        
        int base_idx = ((blockIdx.x*blockDim.x + threadIdx.x)>>0);
        int warp_id = ((blockIdx.x*blockDim.x + threadIdx.x>>5)&3);
        int index_size, bias; uintT index;
        
        if(base_idx < tot_size) {
                index = base_idx;
                uintT loc = d_input.csr_ptr[index] + k;
                if(loc < d_input.csr_ptr[index+1]) {
			uintT sv = UDF0(index, d_input);
                        uintT dst_idx = d_input.dst_idx[loc];
                        UDF(sv, loc, index, dst_idx, d_input);
                
                }
        }
}

template <csr_fun UDF0, union_fun UDF>
__global__ void sampling_phase11_fusion(uintT tot_size, struct graph_data d_input)
{       
        __shared__ int buffer1[SBSIZE/4], buffer2[SBSIZE/4], buffer3[SBSIZE/4], buffer4[SBSIZE/4];
        __shared__ uintT buffer_p[3];
        
        int base_idx = ((blockIdx.x*blockDim.x + threadIdx.x)/(SSFACT));
        int warp_id = ((blockIdx.x*blockDim.x + threadIdx.x>>5)&3);
        int index_size, bias; uintT index;
        
        if(base_idx < tot_size) {
                index = d_input.queue[base_idx];
                uintT loc = d_input.csr_ptr[index] + (threadIdx.x&(SSFACT-1));
                if(loc < d_input.csr_ptr[index+1]) {
			uintT sv = UDF0(index, d_input);
                        uintT dst_idx = d_input.dst_idx[loc];
                        UDF(sv, loc, index, dst_idx, d_input);
                
                }
        }
}

template <csr_fun UDF0, union_fun UDF>
__global__ void sampling_phase1_fusion(uintT tot_size, struct graph_data d_input)
{       
        
        int base_idx = ((blockIdx.x*blockDim.x + threadIdx.x)>>1);
        int warp_id = ((blockIdx.x*blockDim.x + threadIdx.x>>5)&3);
        int index_size, bias; uintT index;
        
        if(base_idx < tot_size) {
                index = base_idx;
                uintT loc = d_input.csr_ptr[index] + (threadIdx.x&1);
                if(loc < d_input.csr_ptr[index+1]) {
			uintT sv = UDF0(index, d_input);
                        uintT dst_idx = d_input.dst_idx[loc];
                        UDF(sv, loc, index, dst_idx, d_input);
                
                }
        }
}

template <csr_fun UDF0, union_fun UDF>
__global__ void sampling_phase1_fusionX(uintT tot_size, struct graph_data d_input)
{       
        
        int base_idx = blockIdx.x*blockDim.x + (threadIdx.x/2)*2;
        int warp_id = ((blockIdx.x*blockDim.x + threadIdx.x>>5)&3);
        int index_size, bias; uintT index;
        
	if(blockIdx.x < gridDim.x-1) {
		uintT index = base_idx;
                uintT loc = d_input.csr_ptr[index] + (threadIdx.x&1);
		uintT loc2 =  d_input.csr_ptr[index+1] + (threadIdx.x&1);

                uintT dst_idx = d_input.dst_idx[loc];
                uintT dst_idx2 = d_input.dst_idx[loc2];
		if(threadIdx.x & 1) {
			uintT tmp = dst_idx; dst_idx = dst_idx2; dst_idx2 = tmp;
 			index++;
			loc = loc2-1;
		}
		dst_idx2 = __shfl_xor_sync(-1, dst_idx2, 1);
		if(threadIdx.x & 1) {
			uintT tmp = dst_idx; dst_idx = dst_idx2; dst_idx2 = tmp;
		}

		if(loc+1 < d_input.csr_ptr[index+1]) {
			uintT sv = UDF0(index, d_input);
			UDF(sv, loc, index, dst_idx, d_input);
			uintT sv2 = UDF0(index, d_input);
			UDF(sv2, loc+1, index, dst_idx2, d_input);
		}
		else if(loc < d_input.csr_ptr[index+1]) {
			uintT sv = UDF0(index, d_input);
			UDF(sv, loc, index, dst_idx, d_input);
		}

	} else if(base_idx < tot_size) {
                index = base_idx;
                uintT loc = d_input.csr_ptr[index] + (threadIdx.x&1);
                if(loc < d_input.csr_ptr[index+1]) {
			uintT sv = UDF0(index, d_input);
                        uintT dst_idx = d_input.dst_idx[loc];
                        UDF(sv, loc, index, dst_idx, d_input);
                
                }
        }
}



template <csr_fun UDF0, union_fun UDF>
__global__ void sampling_phase1_fusion_rand(uintT tot_size, struct graph_data d_input)
{       
        __shared__ int buffer1[SBSIZE/4], buffer2[SBSIZE/4], buffer3[SBSIZE/4], buffer4[SBSIZE/4];
        __shared__ uintT buffer_p[3];
        
        int base_idx = tot_size - 1 - ((blockIdx.x*blockDim.x + threadIdx.x)>>1);
        int warp_id = ((blockIdx.x*blockDim.x + threadIdx.x>>5)&3);
        int index_size, bias; uintT index;
        
	if(base_idx >= 0) {
		index = base_idx;
		uintT index_size = d_input.csr_ptr[index+1] - d_input.csr_ptr[index];
		if(index_size >= 2) {
			uintT loc = d_input.csr_ptr[index] + (threadIdx.x * 239  % index_size);
			uintT sv = UDF0(index, d_input);
			uintT dst_idx = d_input.dst_idx[loc];
			UDF(sv, loc, index, dst_idx, d_input);
		} else if(index_size == 1) {
			uintT loc = d_input.csr_ptr[index];
			uintT sv = UDF0(index, d_input);
			uintT dst_idx = d_input.dst_idx[loc];
			UDF(sv, loc, index, dst_idx, d_input);
		}
        }
}

// previous version (before kernel fission)
template <csr_fun UDF0, union_fun UDF, uintT long_flag, uintT is_finish>
__global__ void sampling_phase3_prev_version(uintT tot_size, struct graph_data d_input)
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

	bias = 0;
	index = base_idx;

	bool flag;
	if(long_flag == 0) { flag = (base_idx < tot_size && PARENT(index) != d_input.max_c); }
	if(long_flag == 1) { flag = (base_idx < tot_size && (LPARENT(index) & (UINT_T_MAX-1)) != d_input.max_c); }

        if(flag) {
                index_size = d_input.csr_ptr[index+1] - d_input.csr_ptr[index] - d_input.sample_k;
		if(index_size < 0) index_size = 0;
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

        if(flag) {
		uintT sv = UDF0(index, d_input);
                for(uintT i=bias+(threadIdx.x&3); i<index_size; i+=4) {
                        uintT dst_idx = d_input.dst_idx[d_input.csr_ptr[index]+d_input.sample_k+i];
                        bool r = UDF(sv, d_input.csr_ptr[index]+d_input.sample_k+i, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
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
                        uintT dst_idx = d_input.dst_idx[d_input.csr_ptr[index]+d_input.sample_k+j];
                        bool r = UDF(sv, d_input.csr_ptr[index]+d_input.sample_k+j, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
			}
                }
        }
        for(uintT i=0;i<buffer_p[1];i++) {
                index = buffer3[i];
		uintT sv = UDF0(index, d_input);
                for(int j=threadIdx.x;j<buffer4[i];j+=blockDim.x) {
                        uintT dst_idx = d_input.dst_idx[d_input.csr_ptr[index]+d_input.sample_k+j];
                        bool r = UDF(sv, d_input.csr_ptr[index]+d_input.sample_k+j, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
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

// after kernel fission
template <csr_fun UDF0, union_fun UDF, uintT long_flag, uintT is_finish>
__global__ void sampling_phase3(uintT tot_size, struct graph_data d_input)
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

	bias = 0;
	index = base_idx;

	bool flag;
	if(long_flag == 0) { flag = (base_idx < tot_size && PARENT(index) != d_input.max_c); }
	if(long_flag == 1) { flag = (base_idx < tot_size && (LPARENT(index) & (UINT_T_MAX-1)) != d_input.max_c); }
	if(long_flag == 2) { flag = (base_idx < tot_size && (PARENT(index) != d_input.max_c || PARENT_R(index+d_input.V) != d_input.max_c)); }

        if(flag) {
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

        if(flag) {
		uintT sv = UDF0(index, d_input);
                for(uintT i=bias+(threadIdx.x&3); i<index_size; i+=4) {
                        uintT dst_idx = d_input.dst_idx[d_input.csr_ptr[index]+i];
                        bool r = UDF(sv, d_input.csr_ptr[index]+i, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
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
                        bool r = UDF(sv, d_input.csr_ptr[index]+j, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
			}
                }
        }
        for(uintT i=0;i<buffer_p[1];i++) {
                index = buffer3[i];
		uintT sv = UDF0(index, d_input);
                for(int j=threadIdx.x;j<buffer4[i];j+=blockDim.x) {
                        uintT dst_idx = d_input.dst_idx[d_input.csr_ptr[index]+j];
                        bool r = UDF(sv, d_input.csr_ptr[index]+j, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
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

template <csr_fun UDF0, union_fun UDF, uintT long_flag, uintT is_finish>
__global__ void sampling_phase33(uintT tot_size, uintT *q, uintT qp, struct graph_data d_input)
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

	bias = 0;

	bool flag;

	if(long_flag == 0) { flag = (base_idx < qp); }
	if(long_flag == 1) { flag = (base_idx < qp && (LPARENT(index) & (UINT_T_MAX-1)) != d_input.max_c); }
	if(long_flag == 2) { flag = (base_idx < qp && (PARENT(index) != d_input.max_c || PARENT_R(index+d_input.V) != d_input.max_c)); }

        if(flag) {
		index = q[base_idx];
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

        if(flag) {
		uintT sv = UDF0(index, d_input);
                for(uintT i=bias+(threadIdx.x&3); i<index_size; i+=4) {
                        uintT dst_idx = d_input.dst_idx[d_input.csr_ptr[index]+i];
                        bool r = UDF(sv, d_input.csr_ptr[index]+i, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
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
                        bool r = UDF(sv, d_input.csr_ptr[index]+j, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
			}
                }
        }
        for(uintT i=0;i<buffer_p[1];i++) {
                index = buffer3[i];
		uintT sv = UDF0(index, d_input);
                for(int j=threadIdx.x;j<buffer4[i];j+=blockDim.x) {
                        uintT dst_idx = d_input.dst_idx[d_input.csr_ptr[index]+j];
                        bool r = UDF(sv, d_input.csr_ptr[index]+j, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
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



template <csr_fun UDF0, union_fun UDF, uintT long_flag, uintT is_finish>
__global__ void sampling_phase3_all(uintT tot_size, struct graph_data d_input)
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

	bias = 0;
	index = base_idx;

	bool flag;
	if(long_flag == 0) { flag = (base_idx < tot_size && PARENT(index) != d_input.max_c); }
	if(long_flag == 1) { flag = (base_idx < tot_size && (LPARENT(index) & (UINT_T_MAX-1)) != d_input.max_c); }
	if(long_flag == 2) { flag = (base_idx < tot_size && (PARENT(index) != d_input.max_c || PARENT_R(index+d_input.V) != d_input.max_c)); }

        if(flag) {
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

        if(flag) {
		uintT sv = UDF0(index, d_input);
                for(uintT i=bias+(threadIdx.x&3); i<index_size; i+=4) {
                        uintT dst_idx = d_input.dst_idx[d_input.csr_ptr[index]+i];
                        bool r = UDF(sv, d_input.csr_ptr[index]+i, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
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
                        bool r = UDF(sv, d_input.csr_ptr[index]+j, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
			}
                }
        }
        for(uintT i=0;i<buffer_p[1];i++) {
                index = buffer3[i];
		uintT sv = UDF0(index, d_input);
                for(int j=threadIdx.x;j<buffer4[i];j+=blockDim.x) {
                        uintT dst_idx = d_input.dst_idx[d_input.csr_ptr[index]+j];
                        bool r = UDF(sv, d_input.csr_ptr[index]+j, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
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


template <csr_fun UDF0, union_fun UDF, uintT long_flag, uintT is_finish>
__global__ void sampling_inv_phase3(uintT tot_size, struct graph_data d_input)
{
        __shared__ int buffer1[SBSIZE/4], buffer2[SBSIZE/4], buffer3[SBSIZE/4], buffer4[SBSIZE/4];
        __shared__ uintT buffer_p[3];
        __shared__ uintT sm_Eflag;

        if(is_finish) {
                if(threadIdx.x == 0) sm_Eflag = 0;
        }

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

	bias = 0;
	index = base_idx;

	bool flag;
	if(long_flag == 0) { flag = (base_idx < tot_size && PARENT(index) != d_input.max_c); }
	if(long_flag == 1) { flag = (base_idx < tot_size && (LPARENT(index) & (UINT_T_MAX-1)) != d_input.max_c); }
	if(long_flag == 2) { flag = (base_idx < tot_size && (PARENT(index) != d_input.max_c || PARENT_R(index+d_input.V) != d_input.max_c)); }

        if(flag) {
                index_size = d_input.csr_inv_ptr[index+1] - d_input.csr_inv_ptr[index];
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

        if(flag) {
		uintT sv = UDF0(index, d_input);
                for(uintT i=bias+(threadIdx.x&3); i<index_size; i+=4) {
                        uintT dst_idx = d_input.dst_inv_idx[d_input.csr_inv_ptr[index]+i];
                        bool r = UDF(sv, d_input.csr_inv_ptr[index]+i, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
			}
                }
        }

        int upper = buffer_p[0];
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
                        uintT dst_idx = d_input.dst_inv_idx[d_input.csr_inv_ptr[index]+j];
                        bool r = UDF(sv, d_input.csr_inv_ptr[index]+j, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
			}
                }
        }
        for(uintT i=0;i<buffer_p[1];i++) {
                index = buffer3[i];
		uintT sv = UDF0(index, d_input);
                for(int j=threadIdx.x;j<buffer4[i];j+=blockDim.x) {
                        uintT dst_idx = d_input.dst_inv_idx[d_input.csr_inv_ptr[index]+j];
                        bool r = UDF(sv, d_input.csr_inv_ptr[index]+j, index, dst_idx, d_input);
			if(is_finish) {
				if(r) sm_Eflag = 1;
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

__global__ void _find_largest_component(int size, int nv, uintT *parent, ulongT *lparent, uintT *samples)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size) {
		curandState_t state;
		curand_init(idx, 0, 0, &state);
		int k = curand(&state) % nv;
		samples[idx] = parent[k];

	}
}

__global__ void _find_largest_Lcomponent(int size, int nv, uintT *parent, ulongT *lparent, uintT *samples)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size) {
		curandState_t state;
		curand_init(idx, 0, 0, &state);
		int k = curand(&state) % nv;
		samples[idx] = (lparent[k]&(UINT_T_MAX-1));

	}
}


template <int long_flag>
inline uintT find_largest_component(int nv, int sample_size, uintT *parent, ulongT *lparent, uintT *samples, uintT *_samples)
{
#ifdef ADD_TIMING
float tot_ms;
cudaEvent_t event1, event2;
cudaEventCreate(&event1);
cudaEventCreate(&event2);
T_START
#endif
	if(long_flag == 0) {
		_find_largest_component<<<CEIL(sample_size,SBSIZE), SBSIZE>>>(sample_size, nv, parent, lparent, _samples);
	} else {
		_find_largest_Lcomponent<<<CEIL(sample_size,SBSIZE), SBSIZE>>>(sample_size, nv, parent, lparent, _samples);
	}

	cudaMemcpy(samples, _samples, sizeof(uintT)*sample_size, cudaMemcpyDeviceToHost);
	std::sort(samples, samples+sample_size);
	int max_v=-1, max_loc=0, local_v = 0;
	for(int i=0;i<sample_size;i++) {
		if(i<sample_size-1 && samples[i] == samples[i+1]) local_v++;
		else {
			if(local_v > max_v) { max_v = local_v; max_loc = samples[i]; }
			local_v = 0;
		}
	}

	#ifdef ADD_TIMING
T_END
printf("xx: %f\n", tot_ms);
#endif

	return(max_loc);	
}

#endif 
