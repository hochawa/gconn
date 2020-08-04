#ifndef __SV_DEFINED
#define __SV_DEFINED

#include "common.h"
#include "kernel.h"
#include "sample.h"

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool sv (uintT vstat, uintT idx, uintT src0, uintT dst0, struct graph_data d_input)
{
        bool active = (idx < d_input.E);
        bool flag = false;
        if(active) {
                uintT src = PARENT(src0);
                uintT dst = PARENT(dst0);
                if(src != dst) {
                        if(src > dst) { uintT tmp = src; src = dst; dst = tmp; }
                        if(dst == PARENT_R(dst+d_input.V)) {
                                flag = true;
                                uintT k = atomicMin(&PARENTW(dst), src);
                                #if defined(SP_TREE)
                                if(k == dst) d_input.hook[dst] = idx;
                                #endif

                        }
                }
        }
        return flag;
}

// previous version of Shiloach-Vishkin. In the worst case, this version iterates n times.
template <find_fun UDF, apply_fun UDF2>
__device__ inline bool sv_prev (uintT vstat, uintT idx, uintT src0, uintT dst0, struct graph_data d_input)
{
	bool active = (idx < d_input.E);
	bool flag = false;
	if(active) {
		uintT src = PARENT(src0);
		uintT dst = PARENT(dst0);
		if(src != dst) {	
			if(src > dst) { uintT tmp = src; src = dst; dst = tmp; }
			if(dst == PARENT(dst)) {
				flag = true;
				uintT k = atomicMin(&PARENTW(dst), src);
				#if defined(SP_TREE)
				if(k == dst) d_input.hook[dst] = idx;
				#endif

			}
		}
	}
	return flag;
}

__global__ void shortcut_sv(uintT tot_size, struct graph_data d_input)
{
	__shared__ uintT sm_Vflag;
	uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < tot_size) {
		while (PARENT(idx) != PARENT(PARENT(idx))) {
			PARENTW(idx) = PARENT(PARENT(idx));
		}
	}
}

template <STR_fun UDF, STR_fun UDF2, int Vfront_active, int Efront_active, int alter_active, int Rfront_active, int sample_active>
void union_find_sv(uintT f_size, struct graph_data d_input)
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
		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
                cudaMemcpy(&d_input.parent[d_input.V], d_input.parent, sizeof(uintT)*d_input.V, cudaMemcpyDeviceToDevice);
		UDF<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(f_size, d_input);
		if(sample_active && d_input.is_sym == 0) {
			UDF2<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(f_size, d_input);
		}
		cudaMemcpy(&Eflag, d_input._Eflag, sizeof(uintT), cudaMemcpyDeviceToHost);

		shortcut_sv<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	

		if(Eflag == 0) break;


	}
}

template <STR_fun UDF, STR_fun UDF2, int Vfront_active, int Efront_active, int alter_active, int Rfront_active, int sample_active>
void union_find_sv_CHUNK(uintT f_size, struct graph_data d_input)
{
	uintT V_bitsize = CEIL(d_input.V, 32);
	uintT E_bitsize = CEIL(d_input.E, 32);
	uintT Vflag, Eflag;
	uintT V_frontsize = d_input.V;
	uintT E_frontsize = f_size;

	if(Efront_active) {
		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
		if(d_input.offset == 0) cudaMemset(d_input._Efront, -1, sizeof(uintT)*E_bitsize);
	} 
	if(Rfront_active) {
		cudaMemset(d_input._Rfront, -1, sizeof(uintT)*V_bitsize);
		
	}
	while(1) {
		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
                cudaMemcpy(&d_input.parent[d_input.V], d_input.parent, sizeof(uintT)*d_input.V, cudaMemcpyDeviceToDevice);
		UDF<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(d_input.offset+f_size, d_input);
		if(sample_active && d_input.is_sym == 0) {
			UDF2<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(d_input.offset+f_size, d_input);
		}
		cudaMemcpy(&Eflag, d_input._Eflag, sizeof(uintT), cudaMemcpyDeviceToHost);

		shortcut_sv<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	

		if(Eflag == 0) break;


	}
}


template <STR_fun UDF, STR_fun UDF2, int Vfront_active, int Efront_active, int alter_active, int Rfront_active, int sample_active>
void union_find_sv_COO_SAMPLE(uintT f_size, struct graph_data d_input)
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
	bool init=false;
	while(1) {
		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
                cudaMemcpy(&d_input.parent[d_input.V], d_input.parent, sizeof(uintT)*d_input.V, cudaMemcpyDeviceToDevice);
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
		cudaMemcpy(&Eflag, d_input._Eflag, sizeof(uintT), cudaMemcpyDeviceToHost);

		shortcut_sv<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	

		if(Eflag == 0) break;


	}
}

#endif
