#ifndef __STERGIOU_DEFINED
#define __STERGIOU_DEFINED

#include "common.h"
#include "kernel.h"
#include "sample.h"

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool ster (uintT vstat, uintT idx, uintT src0, uintT dst0, struct graph_data d_input)
{
	uintT idx_q = idx/32;
	uintT idx_r = idx - idx_q*32;
	bool active = (idx < d_input.E);
	bool flag = false;
	if(active) {
		
		if(PARENT(dst0) > PARENT_R(src0+d_input.V)) {
			PARENTW(dst0) = PARENT_R(src0+d_input.V);//parent[dst0] = prev_parent[src0];
			#if defined SP_TREE
			d_input.hook[dst0] = PARENT_R(src0+d_input.V);
			#endif
			flag = true;
		}
		if(d_input.is_sym == 0) {
			if(PARENT(src0) > PARENT_R(dst0 + d_input.V)) {
				PARENTW(src0) = PARENT_R(dst0 + d_input.V); // parent[src0] = prev_parent[dst0];

				#if defined SP_TREE
				d_input.hook[src0] = PARENT_R(dst0+d_input.V);
				#endif
				flag = true;
			}
		}
	}
	return flag;
}

__global__ void shortcut_ster(uintT tot_size, struct graph_data d_input)
{
	__shared__ uintT sm_Vflag;
	uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadIdx.x == 0) sm_Vflag = 0;
	__syncthreads();
        if(idx < tot_size) {
		if(PARENT(idx) < PARENT_R(idx+d_input.V)) {
			PARENTW(idx+d_input.V) = PARENT(idx);
			sm_Vflag = 1;
		}		
		if(PARENT(idx) > PARENT_R(PARENT(idx)+d_input.V)) {
			PARENTW(idx) = PARENT_R(PARENT(idx)+d_input.V);
			sm_Vflag = 1;
		}
	}
	__syncthreads();
	if(threadIdx.x == 0 && sm_Vflag == 1) {
		*d_input._Vflag = 1;
	}
}

template <STR_fun UDF, STR_fun UDF2, int Vfront_active, int Efront_active, int alter_active, int Rfront_active, int sample_active>
void union_find_ster(uintT f_size, struct graph_data d_input)
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
		cudaMemcpy(&d_input.parent[V_frontsize], &d_input.parent[0], sizeof(uintT) * V_frontsize, cudaMemcpyDeviceToDevice); 
		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
		UDF<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(f_size, d_input);
		if(sample_active && d_input.is_sym == 0) {
			UDF2<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(f_size, d_input);
		}
		cudaMemcpy(&Eflag, d_input._Eflag, sizeof(uintT), cudaMemcpyDeviceToHost);

		cudaMemset(d_input._Vflag, 0, sizeof(uintT));
		shortcut_ster<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	
		cudaMemcpy(&Vflag, d_input._Vflag, sizeof(uintT), cudaMemcpyDeviceToHost);

		if(Eflag == 0 && Vflag == 0) break;


	}
}

template <STR_fun UDF, STR_fun UDF2, int Vfront_active, int Efront_active, int alter_active, int Rfront_active, int sample_active>
void union_find_ster_CHUNK(uintT f_size, struct graph_data d_input)
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
		cudaMemcpy(&d_input.parent[V_frontsize], &d_input.parent[0], sizeof(uintT) * V_frontsize, cudaMemcpyDeviceToDevice); 
		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
		UDF<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(d_input.offset+f_size, d_input);
		if(sample_active && d_input.is_sym == 0) {
			UDF2<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(d_input.offset+f_size, d_input);
		}
		cudaMemcpy(&Eflag, d_input._Eflag, sizeof(uintT), cudaMemcpyDeviceToHost);

		cudaMemset(d_input._Vflag, 0, sizeof(uintT));
		shortcut_ster<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	
		cudaMemcpy(&Vflag, d_input._Vflag, sizeof(uintT), cudaMemcpyDeviceToHost);

		if(Eflag == 0 && Vflag == 0) break;

	}
}


template <STR_fun UDF, STR_fun UDF2, int Vfront_active, int Efront_active, int alter_active, int Rfront_active, int sample_active>
void union_find_ster_COO_SAMPLE(uintT f_size, struct graph_data d_input)
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
		cudaMemcpy(&d_input.parent[V_frontsize], &d_input.parent[0], sizeof(uintT) * V_frontsize, cudaMemcpyDeviceToDevice); 
		cudaMemset(d_input._Eflag, 0, sizeof(uintT));
		for(uintT gran = 0; gran < d_input.E; gran += d_input.coo_sample_size) {
			d_input.offset = gran;
			d_input.size = MIN(d_input.E, gran + d_input.coo_sample_size) - gran;
			if(init) edge_relabeling<2><<<CEIL(d_input.size, SBSIZE), SBSIZE>>>(d_input.E, d_input);
			init = true;
			UDF<<<CEIL(d_input.size, SBSIZE), SBSIZE>>>(f_size, d_input);
		}
		if(sample_active && d_input.is_sym == 0) {
			UDF2<<<CEIL(E_frontsize, SBSIZE), SBSIZE>>>(f_size, d_input);
		}
		cudaMemcpy(&Eflag, d_input._Eflag, sizeof(uintT), cudaMemcpyDeviceToHost);

		cudaMemset(d_input._Vflag, 0, sizeof(uintT));
		shortcut_ster<<<CEIL(V_frontsize, SBSIZE), SBSIZE>>>(V_frontsize, d_input);	
		cudaMemcpy(&Vflag, d_input._Vflag, sizeof(uintT), cudaMemcpyDeviceToHost);

		if(Eflag == 0 && Vflag == 0) break;
		d_input.offset = 0;
		d_input.size = d_input.E;
		edge_relabeling<2><<<CEIL(d_input.size, SBSIZE), SBSIZE>>>(d_input.E, d_input);


	}
}


#endif
