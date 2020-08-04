#ifndef __LP_DEFINED
#define __LP_DEFINED

#include "common.h"
#include "kernel.h"
#include "sample.h"

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool lp (uintT vstat, uintT idx, uintT src0, uintT dst0, struct graph_data d_input)
{
	bool active = (idx < d_input.E);
	bool flag = false;
	if(active) {
		uintT src = PARENT(src0);
		uintT dst = PARENT(dst0);
		if(PARENT(src0)  < PARENT(dst0)) {
			if(PARENT(dst0) == atomicMin(&PARENTW(dst0), PARENT(src0))) {
				flag = true;
			}
		}
		else if(PARENT(src0) > PARENT(dst0)) {
			if(PARENT(src0) == atomicMin(&PARENTW(src0), PARENT(dst0))) {
				flag = true;
			}
		}

	}
	return flag;
}

template <STR_fun UDF, STR_fun UDF2, int Vfront_active, int Efront_active, int alter_active, int Rfront_active, int sample_active>
void union_find_lp(uintT f_size, struct graph_data d_input)
{
	uintT V_bitsize = CEIL(d_input.V, 32);
	uintT E_bitsize = CEIL(d_input.E, 32);
	uintT Vflag, Eflag;
	intT V_frontsize = d_input.V;
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
		UDF<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(f_size, d_input);
		if(sample_active && d_input.is_sym == 0) {
			UDF2<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(f_size, d_input);
		}
		cudaMemcpy(&Eflag, d_input._Eflag, sizeof(uintT), cudaMemcpyDeviceToHost);

		if(Eflag == 0) break;
	}
}

template <STR_fun UDF, STR_fun UDF2, int Vfront_active, int Efront_active, int alter_active, int Rfront_active, int sample_active>
void union_find_lp_CHUNK(uintT f_size, struct graph_data d_input)
{
	uintT V_bitsize = CEIL(d_input.V, 32);
	uintT E_bitsize = CEIL(f_size, 32);
	uintT Vflag, Eflag;
	uintT V_frontsize = d_input.V;
	uintT E_frontsize = f_size;

	if(Efront_active) {
		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
		cudaMemset(d_input._Efront, -1, sizeof(uintT)*E_bitsize);
	} 
	if(Rfront_active) {
		cudaMemset(d_input._Rfront, -1, sizeof(uintT)*V_bitsize);
		
	}
	while(1) {
		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
		UDF<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(f_size, d_input);
		if(sample_active && d_input.is_sym == 0) {
			UDF2<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(f_size, d_input);
		}
		cudaMemcpy(&Eflag, d_input._Eflag, sizeof(uintT), cudaMemcpyDeviceToHost);

		if(Eflag == 0) break;
	}
}


template <STR_fun UDF, STR_fun UDF2, int Vfront_active, int Efront_active, int alter_active, int Rfront_active, int sample_active>
void union_find_lp_COO_SAMPLE(uintT f_size, struct graph_data d_input)
{
	uintT V_bitsize = CEIL(d_input.V, 32);
	uintT E_bitsize = CEIL(d_input.E, 32);
	uintT Vflag, Eflag;
	uintT V_frontsize = d_input.V;
	uintT E_frontsize = f_size;
	if(E_frontsize == V_frontsize) E_frontsize *= 4;

	bool init = false;
	if(Efront_active) {
		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
		cudaMemset(d_input._Efront, -1, sizeof(uintT)*E_bitsize);
	} 
	if(Rfront_active) {
		cudaMemset(d_input._Rfront, -1, sizeof(uintT)*V_bitsize);
		
	}

	while(1) {
		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
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

		if(Eflag == 0) break;
	}
}


#endif
