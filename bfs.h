#ifndef __BFS_DEFINED
#define __BFS_DEFINED

#include "bfs_def.h"

#define VBIAS (1048576*256)
#define VVBIAS (VBIAS-1)

#define SUM_NB (60)
#define SUM_TB (1024)

#define BUF_FACT (4)

__global__ void bfs_push(uintT tot_size, uintT nn1, uintT nn2, uintT nn3, uintT nn4, struct graph_data d_input, uintT *curr_f) 
{
	__shared__ uintT sm_sp1[SBSIZE/8], sm_ep1[SBSIZE/8];
	__shared__ uintT sm_sp2[SBSIZE/8], sm_ep2[SBSIZE/8];
	__shared__ uintT buffer_p[3];

	uintT base_idx = ((blockIdx.x*blockDim.x + threadIdx.x)>>3);
	uintT base = d_input.V / 4;
	uintT warp_id = ((blockIdx.x*blockDim.x + threadIdx.x>>5)&3);
	uintT index_size, bias, index;

	if(threadIdx.x < 3) {
		buffer_p[threadIdx.x] = 0;
	}
	__syncthreads();

	if(base_idx < tot_size) {
		#ifdef PATH_LENGTH
		if(threadIdx.x % 8 == 0) atomicAdd(&d_input.path_length[base_idx%PATH_SIZE], d_input.csr_ptr[index+1] - d_input.csr_ptr[index]);
		#endif

		bias = 0;
                if(base_idx < nn1) index = curr_f[base_idx];
                else if(base_idx < nn1+nn2) index = curr_f[base+base_idx-nn1];
                else if(base_idx < nn1+nn2+nn3) index = curr_f[base*2+base_idx-nn1-nn2];
                else index = curr_f[base*3+base_idx-nn1-nn2-nn3];

		index_size = d_input.csr_ptr[index+1] - d_input.csr_ptr[index];
		if(index_size >= 32) {
			bias = index_size - (index_size&31);
			if((index_size & (32+64+128+256)) && (threadIdx.x&7) == 0) {
				uintT p = atomicAggInc(&buffer_p[0]);
				sm_sp1[p] = d_input.csr_ptr[index] + (index_size/SBSIZE)*SBSIZE;
				sm_ep1[p] = d_input.csr_ptr[index+1] - (index_size%32);
			}
			if(index_size >= SBSIZE) {
				if((threadIdx.x&7) == 0) {
					uintT p = atomicAggInc(&buffer_p[1]);
					sm_sp2[p] = d_input.csr_ptr[index];
					sm_ep2[p] = d_input.csr_ptr[index] + (index_size/SBSIZE)*SBSIZE;
				}
			}
		}

	}
	__syncthreads();

	if(base_idx < tot_size) {
		for(uintT i=bias+(threadIdx.x&7); i<index_size; i+=8) {
			uintT dst_idx = d_input.dst_idx[d_input.csr_ptr[index]+i];
			uintT r = dst_idx/32;
			uintT q = (1<<(dst_idx%32));
			if((d_input.bflag[r] & q) == 0) {
				uintT av = atomicOr(&d_input.bflag[r], q);
				#if defined(SP_TREE)
				if((av & q) == 0) d_input.hook[dst_idx] = d_input.csr_ptr[index]+i;
				#endif
			}
		}
	}

        uintT upper = buffer_p[0];
        uintT i;
        while(1) {
                if((threadIdx.x&31) == 0) i = atomicAdd(&buffer_p[2], 1);
                i = __shfl_sync(-1, i, 0);
                if(i >= upper) break;
		uintT s_p = sm_sp1[i], e_p = sm_ep1[i];
		for(uintT j=s_p+(threadIdx.x&31);j<e_p;j+=32) {
			uintT dst_idx = d_input.dst_idx[j];
			uintT r = dst_idx/32;
			uintT q = (1<<(dst_idx%32));
			if((d_input.bflag[r] & q) == 0) {
				uintT av = atomicOr(&d_input.bflag[r], q);
				#if defined(SP_TREE)
				if((av & q) == 0) d_input.hook[dst_idx] = j;
				#endif
			}
		}
	}
	for(uintT i=0;i<buffer_p[1];i++) {
		uintT s_p = sm_sp2[i], e_p = sm_ep2[i];
		for(uintT j=s_p+threadIdx.x;j<e_p;j+=blockDim.x) {
			uintT dst_idx = d_input.dst_idx[j];
			uintT r = dst_idx/32;
			uintT q = (1<<(dst_idx%32));
			if((d_input.bflag[r] & q) == 0) {
				uintT av = atomicOr(&d_input.bflag[r], q);
				#if defined(SP_TREE)
				if((av & q) == 0) d_input.hook[dst_idx] = j;
				#endif
			}

		}
	}
}


__global__ void bfs_pull(uintT tot_size, uintT nn1, uintT nn2, uintT nn3, uintT nn4, struct graph_data d_input, uintT *curr_f) 
{
	uintT base_idx = ((blockIdx.x*blockDim.x + threadIdx.x)>>0);
	uintT warp_id = ((blockIdx.x*blockDim.x + threadIdx.x>>5)&3);
	uintT base = d_input.V / 4;
	uintT index_size, bias, index;

	if(base_idx < tot_size) {
		bias = 0;
                if(base_idx < nn1) index = curr_f[base_idx];
                else if(base_idx < nn1+nn2) index = curr_f[base+base_idx-nn1];
                else if(base_idx < nn1+nn2+nn3) index = curr_f[base*2+base_idx-nn1-nn2];
                else index = curr_f[base*3+base_idx-nn1-nn2-nn3];

		index_size = d_input.csr_ptr[index+1] - d_input.csr_ptr[index];
		uintT s_p = d_input.csr_ptr[index], e_p = d_input.csr_ptr[index+1];

		for(uintT i=s_p; i<e_p; i++) {
			#ifdef PATH_LENGTH
			atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 1);
			#endif
			uintT dst_idx = d_input.dst_idx[i];
			uintT r = dst_idx/32;
			uintT q = (1<<(dst_idx - r*32));
			if((d_input.bflag[r] & q)) {
				uintT r2 = index/32;
				uintT q2 = (1<<(index - r2*32));
				uintT av = atomicOr(&d_input.bflag2[r2], q2);
				#if defined(SP_TREE)
				if(!(av & q2)) d_input.hook[index] = i;
				#endif
				break;
			}
		}
	}
}

__global__ void gen_rev_frontier(struct graph_data d_input, uintT *curr_f)
{
	__shared__ uintT sm_buf[SBSIZE/32][32];
	__shared__ uintT sm_idx[SBSIZE/32][32];
	__shared__ uintT sm_buf2[SBSIZE/32][32*(BUF_FACT)];

	uintT idx = blockIdx.x*blockDim.x + threadIdx.x;
	uintT warp_id = ((blockIdx.x*blockDim.x + threadIdx.x>>5)&3);
	uintT wid = (threadIdx.x/32);
	uintT lane = (threadIdx.x%32);
	uintT mask;
	uintT fpnt = 0;
	uintT fpnt2 = 0;
	uintT base = d_input.V/4;

	if(idx < CEIL(d_input.V,32) && d_input.bflag[idx] != 0xFFFFFFFF)  {
		mask = __activemask();
		uintT p = fpnt + __popc(mask & ((1 << lane_id()) - 1));
		sm_buf[wid][p] = d_input.bflag[idx];
		sm_idx[wid][p] = idx*32;
	} else {
		mask = 0xFFFFFFFF - __activemask();
	}
	fpnt += __popc(mask);
	for(uintT i=0;i<fpnt;i++) {
		uintT idx = sm_buf[wid][i];
		if((idx & (1<<lane)) == 0) {
			uintT loc = sm_idx[wid][i]+lane;
			mask = __activemask();
			uintT p = fpnt2 + __popc(mask & ((1 << lane_id()) - 1));
			sm_buf2[wid][p] = loc;
		} else {
			mask = 0xFFFFFFFF - __activemask();
		}
		fpnt2 += __popc(mask);
		if(fpnt2 >= (BUF_FACT-1)*32) {
			uintT r = (fpnt2/32)*32;
			uintT k0;
	   		if((threadIdx.x&31) == 0) {
       				 k0 = atomicAdd(&d_input._fsize[warp_id], r);
       			}
       			k0 = __shfl_sync(-1, k0, 0) - (fpnt2-r);
			for(uintT j=fpnt2-r+(threadIdx.x&31); j<fpnt2; j+=32) {
				curr_f[base*warp_id + k0+j] = sm_buf2[wid][j];
			}
			fpnt2 -= r;
		}
	}
	if(fpnt2 > 0) {
		uintT k0;
   		if((threadIdx.x&31) == 0) {
			k0 = atomicAdd(&d_input._fsize[warp_id], fpnt2);
      		}
      		k0 = __shfl_sync(-1, k0, 0);
		for(uintT j=(threadIdx.x&31); j<fpnt2; j+=32) {
			curr_f[base*warp_id + k0+j] = sm_buf2[wid][j];
		}
	}
}

template <uintT is_long>
__global__ void write_dist(uintT seed, struct graph_data d_input, uintT *curr_f)
{
	__shared__ uintT sm_buf[SBSIZE/32][32];
	__shared__ uintT sm_idx[SBSIZE/32][32];
	__shared__ uintT sm_buf2[SBSIZE/32][32*BUF_FACT];

	uintT base = d_input.V/4;
	uintT idx = (blockDim.x*blockIdx.x+threadIdx.x);
	uintT warp_id = ((blockIdx.x*blockDim.x + threadIdx.x>>5)&3);
	uintT wid = (threadIdx.x/32);
	uintT lane = (threadIdx.x%32);
	uintT mask;
	uintT fpnt = 0;
	uintT fpnt2 = 0;

	if(idx < CEIL(d_input.V, 32) && d_input.bflag[idx] != d_input.bflag2[idx])  {
		mask = __activemask();
		uintT p = fpnt + __popc(mask & ((1 << lane_id()) - 1));
		sm_buf[wid][p] = (d_input.bflag[idx] ^ d_input.bflag2[idx]);
		sm_idx[wid][p] = idx*32;
	} else {
		mask = 0xFFFFFFFF - __activemask();
	}
	fpnt += __popc(mask);
	for(uintT i=0;i<fpnt;i++) {
		uintT idx = sm_buf[wid][i];
		if(idx & (1<<lane)) {
			uintT loc = sm_idx[wid][i]+lane;
if(is_long == 0) {
			d_input.parent[loc] = seed;
} else {
			d_input.lparent[loc] = seed;
}
			mask = __activemask();
			uintT p = fpnt2 + __popc(mask & ((1 << lane_id()) - 1));
			sm_buf2[wid][p] = loc;
		} else {
			mask = 0xFFFFFFFF - __activemask();
		}
		fpnt2 += __popc(mask);
		if(fpnt2 >= (BUF_FACT-1)*32) {
			uintT r = (fpnt2/32)*32;
			uintT k0;
	   		if((threadIdx.x&31) == 0) {
       				 k0 = atomicAdd(&d_input._fsize[warp_id], r);
       			}
       			k0 = __shfl_sync(-1, k0, 0) - (fpnt2-r);
			for(uintT j=fpnt2-r+(threadIdx.x&31); j<fpnt2; j+=32) {
				curr_f[base*warp_id + k0+j] = sm_buf2[wid][j];
			}
			fpnt2 -= r;
		}
	}
	if(fpnt2 > 0) {
		uintT k0;
   		if((threadIdx.x&31) == 0) {
			k0 = atomicAdd(&d_input._fsize[warp_id], fpnt2);
      		}
      		k0 = __shfl_sync(-1, k0, 0);
		for(uintT j=(threadIdx.x&31); j<fpnt2; j+=32) {
			curr_f[base*warp_id + k0+j] = sm_buf2[wid][j];
		}
	}
}

template <uintT is_long>
__global__ void write_dist2(uintT seed, struct graph_data d_input, uintT *curr_f)
{
	__shared__ uintT sm_buf[SBSIZE/32][32];
	__shared__ uintT sm_idx[SBSIZE/32][32];
	__shared__ uintT sm_buf2[SBSIZE/32][32*BUF_FACT];

	uintT idx = (blockDim.x*blockIdx.x+threadIdx.x);
	uintT warp_id = ((blockIdx.x*blockDim.x + threadIdx.x>>5)&3);
	uintT wid = (threadIdx.x/32);
	uintT lane = (threadIdx.x%32);
	uintT mask;
	uintT fpnt = 0;
	uintT fpnt2 = 0;
	uintT gpnt = 0;
	uintT base = d_input.V/4;

	if(idx < CEIL(d_input.V, 32) && (d_input.bflag[idx] != d_input.bflag2[idx])) {
		mask = __activemask();
		uintT p = gpnt + __popc(mask & ((1 << lane_id()) - 1));
		sm_buf[wid][p] = (d_input.bflag[idx] ^ d_input.bflag2[idx]);
		sm_idx[wid][p] = idx*32;
	} else {
		mask = 0xFFFFFFFF - __activemask();
	}
	gpnt += __popc(mask);
	for(uintT i=0;i<gpnt;i++) {
		uintT idx = sm_buf[wid][i];
		if(idx & (1<<lane)) {
			uintT loc = sm_idx[wid][i]+lane;
if(is_long == 0) {
			d_input.parent[loc] = seed;//
} else {
			d_input.lparent[loc] = seed;
}
		}
	}

	if(idx < CEIL(d_input.V, 32) && (d_input.bflag2[idx] != 0xFFFFFFFF))  {
		mask = __activemask();
		uintT p = fpnt + __popc(mask & ((1 << lane_id()) - 1));
		sm_buf[wid][p] = 0xFFFFFFFF - d_input.bflag2[idx];
		sm_idx[wid][p] = idx*32;
	} else {
		mask = 0xFFFFFFFF - __activemask();
	}
	fpnt += __popc(mask);
	for(uintT i=0;i<fpnt;i++) {
		uintT idx = sm_buf[wid][i];
		if(idx & (1<<lane)) {
			uintT loc = sm_idx[wid][i]+lane;
			mask = __activemask();
			uintT p = fpnt2 + __popc(mask & ((1 << lane_id()) - 1));
			sm_buf2[wid][p] = loc;
		} else {
			mask = 0xFFFFFFFF - __activemask();
		}
		fpnt2 += __popc(mask);
		if(fpnt2 >= (BUF_FACT-1)*32) {
			uintT r = (fpnt2/32)*32;
			uintT k0;
	   		if((threadIdx.x&31) == 0) {
       				 k0 = atomicAdd(&d_input._fsize[warp_id], r);
       			}
       			k0 = __shfl_sync(-1, k0, 0) - (fpnt2-r);
			for(uintT j=fpnt2-r+(threadIdx.x&31); j<fpnt2; j+=32) {
				curr_f[base*warp_id + k0+j] = sm_buf2[wid][j];
			}
			fpnt2 -= r;
		}
	}
	if(fpnt2 > 0) {
		uintT k0;
   		if((threadIdx.x&31) == 0) {
			k0 = atomicAdd(&d_input._fsize[warp_id], fpnt2);
      		}
      		k0 = __shfl_sync(-1, k0, 0);
		for(uintT j=(threadIdx.x&31); j<fpnt2; j+=32) {
			curr_f[base*warp_id + k0+j] = sm_buf2[wid][j];
		}
	}
}

__global__ void sum_degree_kernel(uintT tot_elt, uintT nn1, uintT nn2, uintT nn3, uintT nn4, struct graph_data d_input, uintT *curr_f)
{
	uintT base_idx = blockIdx.x*blockDim.x + threadIdx.x;
	uintT num_blk = CEIL(tot_elt, SUM_TB);
	uintT accum=0;
	uintT base = d_input.V/4;

	__shared__ uintT sm_accum[32];
	for(uintT i=blockIdx.x; i<num_blk; i+=gridDim.x) {
		uintT index;
		if(base_idx < tot_elt) {
	                if(base_idx < nn1) index = curr_f[base_idx];
	                else if(base_idx < nn1+nn2) index = curr_f[base+base_idx-nn1];
   	        	else if(base_idx < nn1+nn2+nn3) index = curr_f[base*2+base_idx-nn1-nn2];
       		        else index = curr_f[base*3+base_idx-nn1-nn2-nn3];
			accum += d_input.csr_ptr[index+1] - d_input.csr_ptr[index];
		}
	}
	accum += __shfl_down_sync(-1, accum, 16);	
	accum += __shfl_down_sync(-1, accum, 8);	
	accum += __shfl_down_sync(-1, accum, 4);	
	accum += __shfl_down_sync(-1, accum, 2);	
	accum += __shfl_down_sync(-1, accum, 1);
	if(threadIdx.x % 32 == 0) {
		sm_accum[threadIdx.x/32] = accum;	
	}
	__syncthreads();
	if(threadIdx.x < SUM_TB/32) {
		accum = sm_accum[threadIdx.x];
	} else {
		accum = 0;
	}
	__syncwarp();
	if(threadIdx.x < 32) {
		accum += __shfl_down_sync(-1, accum, 16);	
		accum += __shfl_down_sync(-1, accum, 8);	
		accum += __shfl_down_sync(-1, accum, 4);	
		accum += __shfl_down_sync(-1, accum, 2);	
		accum += __shfl_down_sync(-1, accum, 1);
	}
	if(threadIdx.x == 0) {
		d_input.sum_deg[blockIdx.x] = accum;
	}

}

void bfs_init(struct graph_data *d_input)
{
	uintT V_bitsize = CEIL(d_input->V, 32); 
	cudaMalloc((void **) &(d_input->bflag), V_bitsize*sizeof(uintT));
	cudaMalloc((void **) &(d_input->bflag2), V_bitsize*sizeof(uintT));
	cudaMalloc((void **) &(d_input->curr_f), (d_input->V)*sizeof(uintT)*2);
	cudaMalloc((void **) &(d_input->_fsize), sizeof(uintT)*4);
	cudaMalloc((void **) &(d_input->sum_deg), sizeof(uintT)*SUM_TB);
	cudaDeviceSynchronize();
} 


template <uintT is_long>
void bfs_run(uintT seed, struct graph_data d_input)
{
	uintT s_q = seed / 32;
	uintT s_r = (1<<(seed - s_q * 32));
	ulongT seed_parent_val;

	uintT V_bitsize = CEIL(d_input.V, 32); 
	cudaMemset(d_input.bflag, 0, V_bitsize*sizeof(uintT));	
	cudaMemcpy(&d_input.bflag[s_q], &s_r, sizeof(uintT), cudaMemcpyHostToDevice); // assume s = 0;

	cudaMemcpy(d_input.bflag2, d_input.bflag, V_bitsize*sizeof(uintT), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_input.curr_f, &seed, sizeof(uintT), cudaMemcpyHostToDevice);

	uintT tot_size = 1;
	uintT ro = 0;
	uintT fsize[4]={1,0,0,0};
	uintT sum_deg_arr[SUM_NB];
	uintT THRESHOLD = d_input.E / 20;
	while(1) {
		uintT num_blk = MIN(SUM_NB, CEIL(tot_size, SUM_TB));
		sum_degree_kernel<<<num_blk, SUM_TB>>>(tot_size, fsize[0], fsize[1], fsize[2], fsize[3], d_input, &d_input.curr_f[ro]);
		cudaMemcpy(sum_deg_arr, d_input.sum_deg, sizeof(uintT)*num_blk, cudaMemcpyDeviceToHost);
		uintT sum_deg = 0;
		for(uintT i=0;i<num_blk;i++) {
			sum_deg += sum_deg_arr[i];
		}
		if(sum_deg <= THRESHOLD) {
			cudaMemset(d_input._fsize, 0, sizeof(uintT)*4);
			bfs_push<<<CEIL(tot_size, SBSIZE/8), SBSIZE>>>(tot_size, fsize[0], fsize[1], fsize[2], fsize[3], d_input, &d_input.curr_f[ro]);
			write_dist<is_long><<<CEIL(V_bitsize,SBSIZE), SBSIZE>>>(seed, d_input, &d_input.curr_f[d_input.V - ro]);
			cudaMemcpy(d_input.bflag2, d_input.bflag, V_bitsize*sizeof(uintT), cudaMemcpyDeviceToDevice);
			cudaMemcpy(fsize, d_input._fsize, sizeof(uintT)*4, cudaMemcpyDeviceToHost);
			tot_size = fsize[0]+fsize[1]+fsize[2]+fsize[3];
			if(tot_size == 0) break;
			ro = d_input.V - ro;
		} else {
			cudaMemset(d_input._fsize, 0, sizeof(uintT)*4);
			gen_rev_frontier<<<CEIL(V_bitsize,SBSIZE), SBSIZE>>>(d_input, &d_input.curr_f[ro]);
			cudaMemcpy(fsize, d_input._fsize, sizeof(uintT)*4, cudaMemcpyDeviceToHost);
			tot_size = fsize[0]+fsize[1]+fsize[2]+fsize[3];
			if(tot_size == 0) break;
			uintT prev_tot = tot_size;
			while(1) {
				cudaMemset(d_input._fsize, 0, sizeof(uintT)*4);
				bfs_pull<<<CEIL(tot_size,SBSIZE),SBSIZE>>>(tot_size, fsize[0], fsize[1], fsize[2], fsize[3], d_input, &d_input.curr_f[ro]);
				write_dist2<is_long><<<CEIL(V_bitsize,SBSIZE), SBSIZE>>>(seed, d_input, &d_input.curr_f[d_input.V - ro]);
				cudaMemcpy(d_input.bflag, d_input.bflag2, V_bitsize*sizeof(uintT), cudaMemcpyDeviceToDevice);
				cudaMemcpy(fsize, d_input._fsize, sizeof(uintT)*4, cudaMemcpyDeviceToHost);
				prev_tot = tot_size;
				tot_size = fsize[0]+fsize[1]+fsize[2]+fsize[3];
				if(tot_size == prev_tot) break;
				ro = d_input.V - ro;

			}
			break;
		}
	}
	if(is_long == 1) {
		ulongT tmp_val = seed + ((ulongT)1<<32) + ((ulongT)1<<63);
		cudaMemcpy(&d_input.lparent[seed], &tmp_val, sizeof(ulongT), cudaMemcpyHostToDevice);
	}

}

#endif
