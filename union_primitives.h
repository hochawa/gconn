#ifndef __UNION_PRIMITIVE_DEFINED
#define __UNION_PRIMITIVE_DEFINED

#include "common.h"

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_rem_fence(uintT i, uintT idx, uintT src, uintT dst, struct graph_data d_input)
{
        uintT rx = src;
        uintT ry = dst;
        uintT z;
        #if defined(STREAMING_SIMPLE)
        uintT x = PARENT(rx);
        uintT y = PARENT(ry);
        #endif
        while(PARENT_ATOM(rx) != PARENT_ATOM(ry)) {
                if(PARENT_ATOM(rx) < PARENT_ATOM(ry)) {
                        uintT temp = rx; rx = ry; ry = temp;
                }

                if(rx == PARENT_ATOM(rx)) {
                        uintT py = PARENT_ATOM(ry);
                        if(py < rx && rx == atomicCAS(&PARENTW(rx), rx, py)) {
                                #if defined(SP_TREE)
                                d_input.hook[rx] = idx + d_input.sym_off;
                                #endif
                                UDF(src, d_input); 
                                UDF(dst, d_input);

				#if defined(PATH_LENGTH)
				atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
				atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
				#endif

                                return true;
                        }
                } else {
                        rx = UDF2(rx, ry, d_input);
                }
        }

	#if defined(PATH_LENGTH)
	atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
	atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
	#endif

        return true;
}

#if defined (STREAMING_SIMPLE)
template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_rem(uintT i, uintT idx, uintT src, uintT dst, struct graph_data d_input)
#endif
#if defined(SCC) || defined(SP_TREE)
template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_rem_fence_notused(uintT i, uintT idx, uintT src, uintT dst, struct graph_data d_input)
#endif
{
        uintT rx = src;
        uintT ry = dst;
        uintT z;
        #if defined(STREAMING_SIMPLE)
        uintT x = PARENT(rx);
        uintT y = PARENT(ry);
        #endif
        while(PARENT_ATOM(rx) != PARENT_ATOM(ry)) {
                if(PARENT_ATOM(rx) < PARENT_ATOM(ry)) {
                        uintT temp = rx; rx = ry; ry = temp;
                }

                if(rx == PARENT_ATOM(rx)) {
                        uintT py = PARENT_ATOM(ry);
                        if(py < rx && rx == atomicCAS(&PARENTW(rx), rx, py)) {
                                #if defined(SP_TREE)
                                d_input.hook[rx] = idx + d_input.sym_off;
                                #endif
                                UDF(src, d_input); 
                                UDF(dst, d_input);

				#if defined(PATH_LENGTH)
				atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
				atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
				#endif

                                return true;
                        }
                } else {
                        rx = UDF2(rx, ry, d_input);
                }
        }
      
	#if defined(PATH_LENGTH)
	atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
	atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
	#endif

	return true;
}

#if defined(SCC) || defined(SP_TREE)
template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_rem(uintT i, uintT idx, uintT src, uintT dst, struct graph_data d_input)
#endif
#if defined(STREAMING_SIMPLE)
template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_rem_notused(uintT i, uintT idx, uintT src, uintT dst, struct graph_data d_input)
#endif
{
	uintT rx = src;
	uintT ry = dst;
	uintT z;
	#if defined(STREAMING_SIMPLE)
	uintT x = PARENT(rx);
	uintT y = PARENT(ry);
	#endif
	while(PARENT0(rx) != PARENT0(ry)) {
		if(PARENT0(rx) < PARENT0(ry)) {
			uintT temp = rx; rx = ry; ry = temp;
		}
		
		if(rx == PARENT0(rx)) {
			if(rx == atomicCAS(&PARENTW(rx), rx, PARENT0(ry))) {
				#if defined(SP_TREE)
				d_input.hook[rx] = idx + d_input.sym_off;
				#endif
				UDF(src, d_input); 
				UDF(dst, d_input);

				#if defined(PATH_LENGTH)
				atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
				atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
				#endif

				return true;
			}
			

		} else {
			rx = UDF2(rx, ry, d_input);
		}
	}

	#if defined(PATH_LENGTH)
	atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
	atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
	#endif

	return true;
}

#ifdef VOLTA
//For Volta
template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_rem_lock_fence(uintT i, uintT idx, uintT src, uintT dst, struct graph_data d_input)
{
	uintT rx = src;
	uintT ry = dst;
	uintT z;
	#if defined(STREAMING_SIMPLE)
	uintT x = PARENT(rx);
	uintT y = PARENT(ry);
	#endif
	while(PARENT0(rx) != PARENT0(ry)) {
		if(PARENT0(rx) < PARENT0(ry)) {
			uintT temp = rx; rx = ry; ry = temp;
		}
		
		if(rx == PARENT0(rx)) {
			cuda_lock(d_input, rx);
			bool success = false;
			if(rx == atomicCAS(&PARENTW(rx), -100, -200)) {
				uintT py = PARENT0(ry);
				PARENTW(rx) = py;
				__threadfence();
				success = true;
				#if defined(SP_TREE)
				d_input.hook[rx] = idx + d_input.sym_off;
				#endif
				
			}
			cuda_unlock(d_input, rx);
			if(success) {
				UDF(src, d_input); 
				UDF(dst, d_input);
				break;
			}
			
		} else {
			rx = UDF2(rx, ry, d_input);
		}
	}

	#if defined(PATH_LENGTH)
	atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
	atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
	#endif

	return true;
}

// for Volta
template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_rem_lock(uintT i, uintT idx, uintT src, uintT dst, struct graph_data d_input)
{
	uintT rx = src;
	uintT ry = dst;
	uintT z;
	while(PARENT(rx) != PARENT(ry)) {
		if(PARENT0(rx) < PARENT0(ry)) {
			uintT temp = rx; rx = ry; ry = temp;
		}
		
		if(rx == PARENT0(rx)) {
			cuda_lock(d_input, rx);
			bool success = false;
			if(rx == atomicCAS(&PARENTW(rx), -100, -200)) {
				uintT py = PARENT0(ry);
				PARENTW(rx) = py;
				__threadfence();
				success = true;
				#if defined(SP_TREE)
				d_input.hook[rx] = idx + d_input.sym_off;
				#endif
				
			}
			cuda_unlock(d_input, rx);
			if(success) {
				UDF(src, d_input); 
				UDF(dst, d_input);
				break;
			}
			
		} else {
			rx = UDF2(rx, ry, d_input);
		}
	}

	#if defined(PATH_LENGTH)
	atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
	atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
	#endif

	return true;
}
#endif
#ifdef PASCAL
// for Pascal
template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_rem_lock_fence(uintT i, uintT idx, uintT src, uintT dst, struct graph_data d_input)
{
	uintT rx = src;
	uintT ry = dst;
	uintT z;
	#if defined(STREAMING_SIMPLE)
	uintT x = PARENT(rx);
	uintT y = PARENT(ry);
	#endif
	while(PARENT0(rx) != PARENT0(ry)) {
		if(PARENT0(rx) < PARENT0(ry)) {
			uintT temp = rx; rx = ry; ry = temp;
		}
		
		if(rx == PARENT0(rx)) {
			bool success = false;
			for(int i=0;i<32;i++) {
				if(threadIdx.x % 32 == i) {
					cuda_lock(d_input, rx);
					if(rx == atomicCAS(&PARENTW(rx), -100, -200)) {
						uintT py = PARENT0(ry);
						PARENTW(rx) = py;
						__threadfence();
						success = true;
						#if defined(SP_TREE)
						d_input.hook[rx] = idx + d_input.sym_off;
						#endif

					}
					cuda_unlock(d_input, rx);
				}
			}
			if(success) {
				UDF(src, d_input); 
				UDF(dst, d_input);
				break;
			}
			
		} else {
			rx = UDF2(rx, ry, d_input);
		}
	}

	#if defined(PATH_LENGTH)
	atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
	atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
	#endif

	return true;
}

// for Pascal
template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_rem_lock(uintT i, uintT idx, uintT src, uintT dst, struct graph_data d_input)
{
	uintT rx = src;
	uintT ry = dst;
	uintT z;
	while(PARENT(rx) != PARENT(ry)) {
		if(PARENT0(rx) < PARENT0(ry)) {
			uintT temp = rx; rx = ry; ry = temp;
		}
		
		if(rx == PARENT0(rx)) {
			bool success = false;
			for(int i=0;i<32;i++) {
				if(threadIdx.x % 32 == i) {
					cuda_lock(d_input, rx);
					if(rx == atomicCAS(&PARENTW(rx), -100, -200)) {
						uintT py = PARENT0(ry);
						PARENTW(rx) = py;
						__threadfence();
						success = true;
						#if defined(SP_TREE)
						d_input.hook[rx] = idx + d_input.sym_off;
						#endif
					}
					cuda_unlock(d_input, rx);
				}
			}
			if(success) {
				UDF(src, d_input); 
				UDF(dst, d_input);
				break;
			}
			
		} else {
			rx = UDF2(rx, ry, d_input);
		}
	}

	#if defined(PATH_LENGTH)
	atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
	atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
	#endif

	return true;
}
#endif

// instead of using SOA, AOS can improve performance
template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_async(uintT i, uintT idx, uintT src, uintT dst, struct graph_data d_input)
{

        while(1) {
		#if defined(STREAMING_SIMPLE)
			uintT x = PARENT(src);
			uintT y = PARENT(dst);
		#endif
                uintT u = UDF(src, d_input);
                uintT v = UDF(dst, d_input);
                if(u == v) break;
                if(v > u) { uintT temp; temp = u; u = v; v = temp; }
                if(u == atomicCAS(&PARENTW(u),u,v)) {
                //if(PARENT(u) == u && u == atomicCAS(&PARENTW(u),u,v)) {
			#if defined(SP_TREE)
			d_input.hook[u] = idx + d_input.sym_off;
			#endif

			#if defined(PATH_LENGTH)
			atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
			atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
			#endif

			return true;
		} else {
		}
        }
}

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_stopt(uintT i, uintT idx, uintT src, uintT dst, struct graph_data d_input)
{
	bool flag = true;
        while(flag) {
		#if defined(STREAMING_SIMPLE)
			uintT x = PARENT(src);
			uintT y = PARENT(dst);
		#endif
                uintT u = UDF(src, d_input);
                uintT v = UDF(dst, d_input);
                if(u == v) break;
                if(v > u) { uintT temp; temp = u; u = v; v = temp; }
                if(PARENT(u+d_input.V) == 0 && 0 == atomicCAS(&PARENTW(u+d_input.V),0,1)) {
			PARENTW(u) = v;
			#if defined(SP_TREE)
			d_input.hook[u] = idx + d_input.sym_off;
			#endif
			flag = false;
		}
		__threadfence();
        }

	#if defined(PATH_LENGTH)
	atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
	atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
	#endif


	return true;
}

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_early(uintT i, uintT idx, uintT src, uintT dst, struct graph_data d_input)
{
	uintT u = src;
	uintT v = dst;

	#ifdef PATH_LENGTH
	uintT pl = 1;
	#endif
	while(1) {
		if(u == v)  break;
		if(v > u) { uintT temp; temp = u; u = v; v = temp; }
		if(u == atomicCAS(&PARENTW(u),u,v)) {
			#if defined(SP_TREE)
			d_input.hook[u] = idx + d_input.sym_off;
			#endif
			break;
		}
		uintT z = PARENT(u);
		uintT w = PARENT(z);
		atomicCAS(&PARENTW(u),z,w);
		u = z;
		#ifdef PATH_LENGTH
		pl++;
		#endif
		
	}	

	UDF(src, d_input); 
	UDF(dst, d_input);
	#if defined(PATH_LENGTH)
	atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
	atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
	atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
	#endif
	return true;
}


template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_ecl(uintT vstat, uintT idx, uintT src, uintT dst, struct graph_data d_input)
{
		#if defined(STREAMING_SIMPLE)
		uintT x = PARENT(dst);
		#endif
                uintT ostat = UDF(dst, d_input);
                bool repeat;
                do {
                        repeat = false;
                        if(vstat != ostat) {
                                int ret;
                                if(vstat < ostat) {
                                        if ((ret = atomicCAS(&PARENTW(ostat), ostat, vstat)) != ostat) {
                                                ostat = ret;
                                                repeat = true;
                                        } else {
						#if defined(SP_TREE)
						d_input.hook[ostat] = idx + d_input.sym_off;
						#endif
					}
                                } else {
                                        if ((ret = atomicCAS(&PARENTW(vstat), vstat, ostat)) != vstat) {
                                                vstat = ret;
                                                repeat = true;
                                        } else {
						#if defined(SP_TREE)
						d_input.hook[vstat] = idx + d_input.sym_off;
						#endif
					}
                                }
                        }
                } while(repeat);

	#if defined(PATH_LENGTH)
	atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
	atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
	#endif

	return true;
}

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_afforest(uintT vstat, uintT idx, uintT src, uintT dst, struct graph_data d_input)
{
        uintT p1 = PARENT(src);
        uintT p2 = PARENT(dst);

	#ifdef PATH_LENGTH
	uintT pl = 1;
	#endif


        while (p1 != p2) {
                uintT high = p1 > p2 ? p1 : p2;
                uintT low = p1 + (p2 - high);

                uintT prev = atomicCAS(&PARENTW(high), high, low);

		#if defined(SP_TREE)
		if(prev == high) {
			d_input.hook[high] = idx + d_input.sym_off;
		}
		#endif

                if(prev == high || prev == low) break;

                p1 = PARENT0(PARENT0(high));
                p2 = PARENT0(low);
        }

	#if defined(PATH_LENGTH)
	atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
	atomicAdd(&d_input.path_length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], pl);
	atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
	#endif


	return true;
}

template <find_fun UDF, apply_fun UDF2>
__device__ inline bool union_rand(uintT vstat, uintT idx, uintT src, uintT dst, struct graph_data d_input)
{
	#if defined(STREAMING_SIMPLE)
	ulongT x = LPARENT(src);
	ulongT y = LPARENT(dst);
	#endif
        uintT u = UDF(src, d_input);
        uintT v = UDF(dst, d_input);
	uintT idx2 = idx;
	uintT q = (idx2>>5);
        uintT r = idx2 - (q<<5); 
	uintT flag = ((d_input._Efront[q] >> r) & 1);

        while(u != v) {
                //link
                ulongT u_p = LPARENT0(u);
                ulongT v_p = LPARENT0(v);
                unsigned int u_r = (u_p>>32);
                unsigned int v_r = (v_p>>32);
                if(u_r < v_r) {
			ulongT k = atomicCAS(&d_input.lparent[u], u_p | ULONG_T_MAX, (((ulongT)u_r<<32) + v) & (ULONG_T_MAX-1));
			#if defined(SP_TREE)
			if(k == (u_p | ULONG_T_MAX)) {
				d_input.hook[u] = idx + d_input.sym_off;
			}
			#endif
		}
                else if(v_r < u_r) {
			ulongT k = atomicCAS(&d_input.lparent[v], v_p | ULONG_T_MAX,  (((ulongT)v_r<<32) + u) & (ULONG_T_MAX-1));
			#if defined(SP_TREE)
			if(k == (v_p | ULONG_T_MAX)) {
				d_input.hook[v] = idx + d_input.sym_off;
			}
			#endif
	        } else {
                        if(u > v) {
				ulongT k = atomicCAS(&d_input.lparent[u], u_p | ULONG_T_MAX, ((((ulongT)u_r+1)<<32) + v) ^ (((ulongT)flag)<<63)  );
				#if defined(SP_TREE)
				if(k == (u_p | ULONG_T_MAX) &&  !((((((ulongT)u_r+1)<<32) + v) ^ (((ulongT)flag)<<63)) & ULONG_T_MAX)) {
					d_input.hook[u] = idx + d_input.sym_off;
				}
				#endif

			}
                        else {
				ulongT k = atomicCAS(&d_input.lparent[v], v_p | ULONG_T_MAX,  ((((ulongT)v_r+1)<<32) + u) ^ (((ulongT)flag)<<63)  );
				#if defined(SP_TREE)
				if(k == (v_p | ULONG_T_MAX) &&  !((((((ulongT)v_r+1)<<32) + u) ^ (((ulongT)flag)<<63)) & ULONG_T_MAX)) {
					d_input.hook[v] = idx + d_input.sym_off;
				}
				#endif
			}
                }

                u = UDF(u, d_input);
                v = UDF(v, d_input);
        }

	#if defined(PATH_LENGTH)
	atomicMax(&d_input.path_max[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE]);
	atomicExch(&d_input.length[(blockIdx.x*blockDim.x+threadIdx.x)%PATH_SIZE], 0);
	#endif

	return true;
}
__device__ inline void uniteEarly(uintT i, uintT *src, uintT *dst, uintT *parent, uintT *hook)
{
	uintT u = src[i];
	uintT v = dst[i];

	while(1) {
		if(u == v) return;
		if(v < u) { uintT temp; temp = u; u = v; v = temp; }
		if(u == atomicCAS(&parent[u],u,v)) { /*hook[u] = i;*/ return; }
		uintT z = parent[u];
		uintT w = parent[z];
		atomicCAS(&parent[u],z,w);
		u = z;
	}	
}
#endif
