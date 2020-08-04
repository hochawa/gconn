#ifndef __COMMON_DEFINED
#define __COMMON_DEFINED

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <functional>
#include <array>
#include <iostream>
#include <cooperative_groups.h>

#define ERR fprintf(stderr, "err\n");
#define uintT unsigned int
#define intT int

#define ulongT unsigned long long int
#define ULONG_T_MAX ((ulongT)1<<63)

#define UINT_T_MAX ((uintT)1<<31)
//#define UINT_T_MAX (1073742418)
#define CEIL(a,b) (((a)+(b)-1)/(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define SBSIZE (512)
#define BIG_THRESHOLD (512)
//#define PRINT(x) ()

#define T_START	cudaDeviceSynchronize();\
		cudaEventRecord(event1,0);\
		init_gpu(d_input);


#define T_START2	cudaDeviceSynchronize();\
		cudaEventRecord(event1,0);\
		init_gpu2(d_input);

#define T_END	cudaEventRecord(event2,0);\
		cudaEventSynchronize(event1);\
		cudaEventSynchronize(event2);\
		cudaEventElapsedTime(&tot_ms, event1, event2);
		//cudaDeviceSynchronize();

#if defined(SCC) || defined(STREAMING) || defined(STREAMING_SYNC) || defined(STREAMING_SIMPLE)
#define CC_GPU(xx) cc_gpu<xx><<<grid_size_final, tb_size>>>(d_input->V, *d_input);
#endif
#if defined(SP_TREE)
#define CC_GPU(xx)
#endif

//#define CC_GPU_CHUNK(xx) cc_gpu_CHUNK<xx><<<CEIL(d_input->size,tb_size), tb_size>>>(d_input->V, *d_input);


#define BASE_C (1024)
#define MAX_BIN (20)

// data formats
#define COO (0)
#define CSR (1)
#define SAMPLE (2)
#define BFS (3)
#define COO_SAMPLE (4)
#define IHOOK (5)

// execution strategies
#define ASYNC_NAIVE (0)
#define ASYNC_COMPRESS (1)
#define ASYNC_HALVE (2)
#define ASYNC_SPLIT (3)
#define ASYNC_A_HALVE (4)
#define ASYNC_A_SPLIT (5)

#define EARLY_NAIVE (2000)
#define EARLY_COMPRESS (2001)
#define EARLY_A_HALVE (2002)
#define EARLY_A_SPLIT (2003)

#define STOPT_NAIVE (21)
#define STOPT_COMPRESS (22)
#define STOPT_HALVE (23)
#define STOPT_SPLIT (24)
#define STOPT_A_HALVE (25)
#define STOPT_A_SPLIT (26)

#define ECL_NAIVE (6)
#define ECL_COMPRESS (7)
#define ECL_HALVE (8)
#define ECL_SPLIT (9)
#define ECL_A_HALVE (10)
#define ECL_A_SPLIT (11)

#define AFFOREST_NAIVE (12)
#define AFFOREST_COMPRESS (13)
#define AFFOREST_HALVE (14)
#define AFFOREST_SPLIT (15)
#define AFFOREST_A_HALVE (16)
#define AFFOREST_A_SPLIT (17)

#define RAND_NAIVE (18)
#define RAND_SPLIT_2 (19)

#define SIM_C_U_S_A (100)
#define SIM_C_R_S_A (101)
#define SIM_P_U_S_A (102)
#define SIM_P_R_S_A (103)
#define SIM_P_U_S (104)
#define SIM_P_R_S (105)
#define SIM_E_U_S_A (106)
#define SIM_E_U_S (107)

#define SIM_C_U_SS_A (108)
#define SIM_C_R_SS_A (109)
#define SIM_P_U_SS_A (110)
#define SIM_P_R_SS_A (111)
#define SIM_P_U_SS (112)
#define SIM_P_R_SS (113)
#define SIM_E_U_SS_A (114)
#define SIM_E_U_SS (115)

#define STERGIOS (116)
#define SVA (117)
#define LPA (118)

#define REMCAS_NAIVE_SPLICE (200)
#define REMCAS_COMPRESS_SPLICE (201)
#define REMCAS_HALVE_SPLICE (202)
#define REMCAS_SPLIT_SPLICE (203)
#define REMCAS_A_HALVE_SPLICE (204)
#define REMCAS_A_SPLIT_SPLICE (205)

#define REMCAS_NAIVE_SPLICE_CAS (206)
#define REMCAS_COMPRESS_SPLICE_CAS (207)
#define REMCAS_HALVE_SPLICE_CAS (208)
#define REMCAS_SPLIT_SPLICE_CAS (209)
#define REMCAS_A_HALVE_SPLICE_CAS (210)
#define REMCAS_A_SPLIT_SPLICE_CAS (211)

#define REMCAS_NAIVE_SPLIT_ONE (212)
#define REMCAS_COMPRESS_SPLIT_ONE (213)
#define REMCAS_HALVE_SPLIT_ONE (214)
#define REMCAS_SPLIT_SPLIT_ONE (215)
#define REMCAS_A_HALVE_SPLIT_ONE (216)
#define REMCAS_A_SPLIT_SPLIT_ONE (217)

#define REMCAS_NAIVE_HALVE_ONE (218)
#define REMCAS_COMPRESS_HALVE_ONE (219)
#define REMCAS_HALVE_HALVE_ONE (220)
#define REMCAS_SPLIT_HALVE_ONE (221)
#define REMCAS_A_HALVE_HALVE_ONE (222)
#define REMCAS_A_SPLIT_HALVE_ONE (223)

#define REMLOCK_NAIVE_SPLICE (300)
#define REMLOCK_COMPRESS_SPLICE (301)
#define REMLOCK_HALVE_SPLICE (302)
#define REMLOCK_SPLIT_SPLICE (303)
#define REMLOCK_A_HALVE_SPLICE (304)
#define REMLOCK_A_SPLIT_SPLICE (305)

#define REMLOCK_NAIVE_SPLICE_CAS (306)
#define REMLOCK_COMPRESS_SPLICE_CAS (307)
#define REMLOCK_HALVE_SPLICE_CAS (308)
#define REMLOCK_SPLIT_SPLICE_CAS (309)
#define REMLOCK_A_HALVE_SPLICE_CAS (310)
#define REMLOCK_A_SPLIT_SPLICE_CAS (311)

#define REMLOCK_NAIVE_SPLIT_ONE (312)
#define REMLOCK_COMPRESS_SPLIT_ONE (313)
#define REMLOCK_HALVE_SPLIT_ONE (314)
#define REMLOCK_SPLIT_SPLIT_ONE (315)
#define REMLOCK_A_HALVE_SPLIT_ONE (316)
#define REMLOCK_A_SPLIT_SPLIT_ONE (317)

#define REMLOCK_NAIVE_HALVE_ONE (318)
#define REMLOCK_COMPRESS_HALVE_ONE (319)
#define REMLOCK_HALVE_HALVE_ONE (320)
#define REMLOCK_SPLIT_HALVE_ONE (321)
#define REMLOCK_A_HALVE_HALVE_ONE (322)
#define REMLOCK_A_SPLIT_HALVE_ONE (323)

#define BFS_CC (400)

#define PARENTW(i) d_input.parent[i]
#define UNINIT (0xFFFFFFFF)

#if defined(SCC) || defined(SP_TREE) || defined(STREAMING_SYNC)
#define PARENT(i) (d_input.parent[i])
#endif
#if defined(STREAMING_SIMPLE)
#define PARENT(i) acti_parent(d_input, i)
#endif

// Is it ok for weak-consistent model
//#define PARENT0(i) (d_input.parent[i])
#define PARENT0(i) PARENT(i)

#if defined(SCC) || defined(SP_TREE)
#define PARENT_ATOM(i) (atomicCAS(&d_input.parent[i], -100, -200))
#define PARENT_R(i) PARENT(i)
#endif
#if defined(STREAMING_SIMPLE)
#define PARENT_ATOM(i) acti_parent_atom(d_input, i)
#define PARENT_R(i) acti_parent_r(d_input, i)
#endif

#if defined(SCC) || defined(SP_TREE) || defined(STREAMING_SYNC)
#define LPARENT(i) (d_input.lparent[i])
#endif
#if defined(STREAMING_SIMPLE)
#define LPARENT(i) acti_lparent(d_input, i)
#endif
#define LPARENT0(i) (LPARENT(i))


// 0: before, 1: progress, 2: done
#if defined(STREAMING)
#define PARENT(i) get_parent_idx(d_input, i) 
#endif

#define GPRINT2(w,x,y) int *w=(int *)malloc(sizeof(int)*(y));\
        fprintf(stderr, "\n");\
        cudaMemcpy(w, x, sizeof(int)*(y), cudaMemcpyDeviceToHost);\
        for(int i=0;i<(y);i++) fprintf(stderr,"%d ", w[i]); fprintf(stderr,"\n");\
        free(w);

#define PATH_SIZE (1048576/4)

//#define PRINT

struct graph_data {
	uintT V; // the number of vertices
	uintT E; // the number of edges
	uintT *csr_ptr;
	uintT *src_idx; // COO structure
	uintT *dst_idx; // COO & CSR structure
	uintT *label; // labels of vertices
	uintT *parent; // parents of vertices
	ulongT *lparent;
	uintT *hook;

	uintT *csr_inv_ptr; // for transposed graph with CSR
	uintT *dst_inv_idx; // for transposed graph with CSR

	uintT format;
	uintT algo;
	uintT is_sym;
	uintT sample_k;
	uintT sample_size;

	uintT *_Vflag;
	uintT *_Eflag;
	uintT *_Vfront;
	uintT *_Efront;
	uintT *_Rfront;

	//uintT front_size;
	uintT mode; 
	uintT max_c;
	uintT max_c2;
	uintT cc_cnt;
	//uintT *_lg_comp_id;
	//uintT *_lg_comp_size;

	//for BFS
	uintT *curr_f;//, *_next_f;
	uintT *bflag, *bflag2;	
	uintT *_fsize;
	uintT *sum_deg;

	//for BFS-CC
	uintT *_s_vertex;

	// for streaming
	uintT *bin_p1;
	uintT *bin_p2;

	// for COO_SAMPLE
	uintT coo_sample_size, offset, size;

	// for streaming_sync
	uintT iter;
	uintT sym_off;
	uintT batch_size;

	#ifdef PATH_LENGTH
	uintT *path_length;
	uintT *length;
	uintT *path_max;
	#endif
	#ifdef CHUNK_STREAMING // streaming with different chunk size
	ulongT chunk_size;
	#endif
	
	uintT *queuep;
	uintT *queue;

	uintT two_parent;
};

// temporary structure for creating a graph
struct v_struct {
	uintT src, dst; 
};

struct vv_struct {
	uintT src, dst;
	uintT idx, valid;
};


typedef __device__ bool union_fun(uintT i, uintT j, uintT src, uintT dst, struct graph_data d_input);
typedef __global__ void STR_fun(uintT src, struct graph_data d_input);
typedef __device__ uintT csr_fun(uintT src, struct graph_data d_input);
typedef __device__ uintT find_fun(uintT i, struct graph_data d_input);
typedef __device__ uintT apply_fun(uintT i, uintT j, struct graph_data d_input);
typedef __global__ void sim_fun(uintT tot_size, uintT *Eflag, uintT *Efront, uintT *Rfront, uintT *src_idx, uintT *dst_idx, uintT *parent);

struct v_struct *generate_graph(int, char **, struct graph_data *);
void mtx_parse(int, char **, struct graph_data *);
double run_union_find(struct graph_data *, struct graph_data *);
bool CC_correctness_check(struct graph_data *);

template <find_fun UDF>
__global__ void cc_gpu(uintT V, struct graph_data d_input)
{       
        uintT idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < V) {
                d_input.label[idx] = UDF(idx, d_input);
        }
}

template <find_fun UDF>
__global__ void cc_gpu_CHUNK(uintT V, struct graph_data d_input)
{       
        uintT idx = blockIdx.x * blockDim.x + threadIdx.x + d_input.offset;
        if(idx < V) {
                d_input.label[idx] = UDF(idx, d_input);
        }
}


#define BIN_TB (MIN(256,BASE_C))

__device__ inline uintT acti_parent(struct graph_data d_input, uintT i)
{
	uintT bin_num = (uintT)log2f(1+i/BASE_C);
	if(d_input.bin_p1[bin_num] == UNINIT) {
		uintT bin_base = (powf(2,bin_num)-1)*BASE_C;
		d_input.bin_p1[bin_num] = bin_base;
	}
	uintT bin_offset = i - d_input.bin_p1[bin_num];
	uintT idx = d_input.bin_p1[bin_num] + bin_offset;
	if(d_input.parent[idx] == UNINIT) {
		while(atomicCAS(&d_input.parent[idx], UNINIT, idx) == UNINIT);
	}
	return d_input.parent[idx];
}



__device__ inline uintT acti_parent_r(struct graph_data d_input, uintT i)
{
	uintT bin_num = (uintT)log2f(1+i/BASE_C);
	if(d_input.bin_p1[bin_num] == UNINIT) {
		uintT bin_base = (powf(2,bin_num)-1)*BASE_C;
		d_input.bin_p1[bin_num] = bin_base;
	}
	uintT bin_offset = i - d_input.bin_p1[bin_num];
	uintT idx = d_input.bin_p1[bin_num] + bin_offset;
	if(d_input.parent[idx] == UNINIT) {
		atomicCAS(&d_input.parent[idx], UNINIT, idx%d_input.V);
	}
	return d_input.parent[idx];
}


__device__ inline uintT acti_parent_atom(struct graph_data d_input, uintT i)
{
	uintT bin_num = (uintT)log2f(1+i/BASE_C);
	if(d_input.bin_p1[bin_num] == UNINIT) {
		uintT bin_base = (powf(2,bin_num)-1)*BASE_C;
		d_input.bin_p1[bin_num] = bin_base;
	}
	uintT bin_offset = i - d_input.bin_p1[bin_num];
	uintT idx = d_input.bin_p1[bin_num] + bin_offset;
	if(d_input.parent[idx] == UNINIT) {
		while(atomicCAS(&d_input.parent[idx], UNINIT, idx) == UNINIT);
	}
	return atomicCAS(&d_input.parent[idx], -100, -200);
}


__device__ inline ulongT acti_lparent(struct graph_data d_input, uintT i)
{
	uintT bin_num = (uintT)log2f(1+i/BASE_C);
	if(d_input.bin_p1[bin_num] == UNINIT) {
		uintT bin_base = (powf(2,bin_num)-1)*BASE_C;
		d_input.bin_p1[bin_num] = bin_base;
	}
	uintT bin_offset = i - d_input.bin_p1[bin_num];
	uintT idx = d_input.bin_p1[bin_num] + bin_offset;
	if(d_input.lparent[idx] == ((ulongT)UNINIT<<32)+UNINIT) {
		while(atomicCAS(&d_input.lparent[idx], ((ulongT)UNINIT<<32)+UNINIT, ((ulongT)idx | ULONG_T_MAX)) == ((ulongT)UNINIT<<32)+UNINIT);
	}
	return d_input.lparent[idx];
}


__device__ inline uintT get_parent_idx(struct graph_data d_input, uintT i) {
	int bin_num = (int)log2f(1+i/BASE_C);

	uintT bin_base = (powf(2,bin_num)-1)*BASE_C;

	while(d_input.bin_p2[bin_num] != 2) {
		while(atomicCAS(&d_input.bin_p2[bin_num], 0, 1) == 0) {
			d_input.bin_p2[bin_num] = 2;
		}
	}

	uintT bin_offset = i - bin_base;

	// compiler may optimize this part
	return(d_input.parent[bin_base+bin_offset]);


}

__device__ inline uintT get_parent_idx0(struct graph_data d_input, uintT i) {
	int bin_num = (int)log2f(1+i/BASE_C);


	uintT bin_base = (powf(2,bin_num)-1)*BASE_C;

	while(atomicCAS(&d_input.bin_p2[bin_num], -100, -200) == 0) {
		if(atomicCAS(&d_input.bin_p1[bin_num], 0, 1)) {
			atomicCAS(&d_input.bin_p2[bin_num], 0, 1);
		}
	}
	uintT bin_offset = i - bin_base;

	// compiler may optimize this part
	return(d_input.parent[bin_base+bin_offset]);
}

__device__ inline int lane_id(void) { return (threadIdx.x&31); }

__device__ inline uintT warp_bcast(uintT mask, int v, int leader) { return __shfl_sync(mask, v, leader); }

__device__ inline uintT atomicAggInc(uintT *ctr) {
        uintT mask = __activemask();
        int leader = __ffs(mask) - 1;
        uintT res;
        if(lane_id() == leader)
                res = atomicAdd(ctr, __popc(mask));
        res = warp_bcast(mask, res, leader);

        return (res + __popc(mask & ((1 << lane_id()) - 1)));
}

__device__ inline void cuda_lock(struct graph_data d_input, uintT k) {
	while(atomicCAS(&d_input.parent[d_input.V+k], 0, 1) != 0);
}

__device__ inline void cuda_unlock(struct graph_data d_input, uintT k) {
	atomicExch(&d_input.parent[d_input.V+k], 0);
}

template <find_fun UDF, apply_fun UDF2>
__device__ inline uintT blank_fun(uintT src, struct graph_data d_input)
{
	return 0;
}

__device__ inline uintT blank_fun2(uintT src, uintT dst, struct graph_data d_input)
{
	return 0;
}



template <find_fun UDF, apply_fun UDF2>
__device__ inline uintT ecl_fun(uintT src, struct graph_data d_input)
{
	uintT vstat = UDF(src, d_input);
	return vstat;
}


__device__ inline uintT find_naive(uintT i, struct graph_data d_input)
{
	#ifdef PATH_LENGTH
	uintT pl = 1;
	#endif
	while(i != PARENT0(i)) {
		i = PARENT0(i);
		#ifdef PATH_LENGTH
		pl++;
		#endif
	}
	#ifdef PATH_LENGTH
		atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
		atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
	#endif
	return i;
}

__device__ inline uintT find_compress(uintT i, struct graph_data d_input)
{
	#ifdef PATH_LENGTH
	uintT pl = 0;
	#endif
	uintT j = i;
	if (PARENT0(j) == j) {
		#ifdef PATH_LENGTH
		pl = 1;
		atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
		atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
		#endif
		return j;
	}
	do {
		#ifdef PATH_LENGTH
		pl++;
		#endif
		j = PARENT0(j);
	} while (PARENT0(j) != j);

	uintT tmp;
	while((tmp=PARENT0(i))>j) {
		PARENTW(i) = j;
		i = tmp;
		#ifdef PATH_LENGTH
		pl++;
		#endif
	}
	#ifdef PATH_LENGTH
		atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
		atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
	#endif
	return j;
}

__device__ inline uintT find_halve(uintT i, struct graph_data d_input)
{
	#ifdef PATH_LENGTH
	uintT pl = 1;
	#endif
	while(1) {
		uintT v = PARENT0(i);
		uintT w = PARENT0(v);
		if(v == w) {
			#ifdef PATH_LENGTH
			pl = 1;
			atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
			atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
			#endif
			return v;
		}
		else {
			PARENTW(i) = w;
			i = PARENT0(i);
		}
		#ifdef PATH_LENGTH
		pl++;
		#endif
	}
	#ifdef PATH_LENGTH
		atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
		atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
	#endif
}

__device__ inline uintT find_split(uintT i, struct graph_data d_input)
{
	#ifdef PATH_LENGTH
	uintT pl = 1;
	#endif
	while(1) {
		uintT v = PARENT0(i);
		uintT w = PARENT0(v);
		if(v == w) {
			#ifdef PATH_LENGTH
			atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
			atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
			#endif
			return v;
		}
		else {
			//atomicCAS(&parent[i], v, w);
			PARENTW(i) = w;
			i = v;	
		}
		#ifdef PATH_LENGTH
		pl++;
		#endif
	}
	#ifdef PATH_LENGTH
		atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
		atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
	#endif
}


__device__ inline uintT find_a_halve(uintT i, struct graph_data d_input)
{
	#ifdef PATH_LENGTH
	uintT pl = 1;
	#endif
	while(1) {
		uintT v = PARENT0(i);
		uintT w = PARENT0(v);
		if(v == w) {
			#ifdef PATH_LENGTH
			atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
			atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
			#endif
			return v;
		}
		else {
			atomicCAS(&PARENTW(i), v, w);
			//parent[i] = w;
			i = PARENT0(i);
		}
		#ifdef PATH_LENGTH
		pl++;
		#endif
	}
	#ifdef PATH_LENGTH
		atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
		atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
	#endif
}

__device__ inline uintT find_a_split(uintT i, struct graph_data d_input)
{
	#ifdef PATH_LENGTH
	uintT pl = 1;
	#endif
	while(1) {
		uintT v = PARENT0(i);
		uintT w = PARENT0(v);
		if(v == w) {
			#ifdef PATH_LENGTH
			atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
			atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
			#endif
			return v;
		}
		else {
			atomicCAS(&PARENTW(i), v, w);
			//parent[i] = w;
			i = v;	
		}
		#ifdef PATH_LENGTH
		pl++;
		#endif
	}
	#ifdef PATH_LENGTH
		atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
		atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
	#endif

}


__device__ inline uintT split_one(uintT i, uintT dummy, struct graph_data d_input)
{
        uintT v = PARENT0(i);
        uintT w = PARENT0(v);
        if (v != w) {
                atomicCAS(&PARENTW(i), v, w);
        }
	#ifdef PATH_LENGTH
	atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 1);
	atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 1);
	#endif
        return v;
}

__device__ inline uintT halve_one(uintT i, uintT dummy, struct graph_data d_input)
{
        uintT v = PARENT0(i);
        uintT w = PARENT0(v); 
        if(v != w) {
                atomicCAS(&PARENTW(i), v, w);
        }
	#ifdef PATH_LENGTH
	atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 1);
	atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 1);
	#endif
        return w;
}

__device__ inline uintT splice(uintT x, uintT y, struct graph_data d_input)
{
        uintT z = PARENT0(x);
	PARENTW(x) = y;
        //if(z > y) atomicCAS(&PARENT(x), x, y);
	#ifdef PATH_LENGTH
	atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 1);
	atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 1);
	#endif
        return z;
}

__device__ inline uintT splice_CAS(uintT x, uintT y, struct graph_data d_input)
{
        uintT z = PARENT0(x); 
        //if(z > y) atomicCAS(&PARENT(x), x, y);
        atomicMin(&PARENTW(x), PARENT0(y));
	#ifdef PATH_LENGTH
	atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 1);
	atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 1);
	#endif
        return z;
}

__device__ inline uintT find_r_naive(uintT i0, struct graph_data d_input)
{
	uintT i = i0;
	#ifdef PATH_LENGTH
	uintT pl = 1;
	#endif
        while(1) {
		if(LPARENT0(i) & ULONG_T_MAX) break;
                uintT pid = (LPARENT0(i) & (UINT_T_MAX-1));
                i = pid;
		#ifdef PATH_LENGTH
		pl++;
		#endif
        }
	#ifdef PATH_LENGTH
		atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
		atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
	#endif
        return i;

}

__device__ inline uintT find_r_split_2(uintT i, struct graph_data d_input)
{
        uintT u = i;
	#ifdef PATH_LENGTH
	uintT pl = 1;
	#endif
        while(1) {
                ulongT u_p = LPARENT0(u);
                if(u_p & ULONG_T_MAX) break;
                uintT v = (u_p & (UINT_T_MAX-1));
                ulongT v_p = LPARENT0(v);
                if(v_p & ULONG_T_MAX) {
			#ifdef PATH_LENGTH
			atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
			atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
			#endif
			return v;
		}
		//cas 1
                uintT w = (v_p & (UINT_T_MAX-1));
                ulongT old_v = (u_p&(ULONG_T_MAX-1));
                ulongT new_v = old_v - v + w;
                atomicCAS(&d_input.lparent[u], old_v, new_v);

		#ifdef PATH_LENGTH
		pl++;
		#endif

                u_p = LPARENT0(u);
                v = (u_p & (UINT_T_MAX-1));
                v_p = LPARENT0(v);
                if(v_p & ULONG_T_MAX) {
			#ifdef PATH_LENGTH
			atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
			atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
			#endif
			return v;
		}
                w = (v_p & (UINT_T_MAX-1));
                old_v = (u_p&(ULONG_T_MAX-1));
                new_v = old_v - v + w;
                atomicCAS(&d_input.lparent[u], old_v, new_v);

                u = v;
		#ifdef PATH_LENGTH
		pl++;
		#endif
        }
	#ifdef PATH_LENGTH
		atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
		atomicAdd(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
	#endif
        return u;
}


#endif
