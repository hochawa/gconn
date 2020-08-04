#include "common.h"

//#define REPEAT (128*8)
//#define REPEAT (1)
//#define REPEAT (16)
//#define REPEAT (5)
#define REPEAT (1)


struct graph_data h_input; // host input
struct graph_data d_input; // device input



struct v_struct *g_temp_v;

int compare00(const void *a, const void *b)
{
        if (((struct v_struct *)a)->src > ((struct v_struct *)b)->src) return 1;
        if (((struct v_struct *)a)->src < ((struct v_struct *)b)->src) return -1;
        return (intT)(((struct v_struct *)a)->dst) - (intT)(((struct v_struct *)b)->dst);
}

int compare1(const void *a, const void *b)
{
        return ((double *)a) - ((double *)b);
}


void dfs(int root, int idx, int *visited, uintT *csr_ptr, uintT *csr_idx)
{
//fprintf(stderr, "%d %d\n", root, idx);
	for(int i=csr_ptr[idx]; i<csr_ptr[idx+1]; i++) {
		if(visited[csr_idx[i]] == 0) {
			visited[csr_idx[i]] = 1;
			dfs(root, csr_idx[i], visited, csr_ptr, csr_idx);
		}
	}
}

int main(int argc, char **argv)
{
	double tot_ms[REPEAT];
	double tot_ms_t=0;
	g_temp_v = generate_graph(argc, argv, &h_input);
	struct v_struct *g_temp_inv;


//originally chunk_size = 1000
#ifdef CHUNK_STREAMING
	for(d_input.chunk_size = 10000000; (double)d_input.chunk_size / h_input.E < 9.99; d_input.chunk_size *= 10) {
	if(d_input.chunk_size > h_input.E) d_input.chunk_size = h_input.E;
#endif
	for(int loop=0; loop<REPEAT; loop++) {
	tot_ms[loop] = run_union_find(&h_input, &d_input);


	//#ifdef CHUNK_STREAMING
	//if(loop == REPEAT-1) fprintf(stdout, "%d,%f,", d_input.chunk_size, tot_ms[loop]);
	//#endif

	#ifdef PATH_LENGTH
		uintT *avg_length = (uintT *)malloc(sizeof(uintT)*PATH_SIZE);
		uintT *max_length = (uintT *)malloc(sizeof(uintT)*PATH_SIZE);
		cudaMemcpy(avg_length, d_input.path_length, sizeof(uintT)*PATH_SIZE, cudaMemcpyDeviceToHost);
		cudaMemcpy(max_length, d_input.path_max, sizeof(uintT)*PATH_SIZE, cudaMemcpyDeviceToHost);
		double avg=0; uintT max=0;
		for(int i=0;i<PATH_SIZE;i++) {
			avg += avg_length[i];
			max = MAX(max, max_length[i]);
		}
		fprintf(stdout, "%f,%d,", avg/d_input.E, max);

		//exit(0);
	#endif

#ifndef QUERY_TEST
//#define VALIDATE
#endif

#ifdef VALIDATE //start of validation
  // for static connected components
  #if defined(SCC) || defined(STREAMING) || defined(STREAMING_SYNC) || defined(STREAMING_SIMPLE)
 
	int nv = h_input.V;
  int *visit = (int *)malloc(sizeof(int)*nv);
  memset(visit, -1, sizeof(int)*nv);
  int cnt=0;
  for(int v = 0; v < nv; v++) {
if(h_input.label[v] < 0 || h_input.label[v] >= nv) fprintf(stderr, "ERR %d: %d %d\n",v,  h_input.label[v], nv);
    if(visit[h_input.label[v]] < 0) { visit[h_input.label[v]] = cnt++; }
    visit[v] = visit[h_input.label[v]];
  }

  char fpo_name[300];
  strcpy(fpo_name, argv[1]);
  strcat(fpo_name, ".base");
  FILE *fpo = fopen(fpo_name, "r");

  for(int v =0; v<nv;v++) {
        int t;
        fscanf(fpo, "%d", &t);
        if(t != visit[v]) { printf("FAIL\n"); exit(0);}
	//if(t != visit[v]) { printf("%d %d %d\n", v, t, visit[v]); }
  }
	//printf("SUCC,");
#ifdef PRINT
        printf("PASS\n");
#endif

	free(visit);  

  fclose(fpo);
	
  #endif

  // for spanning trees
  #if defined(SP_TREE)

	// check the # components
	char fpo_name[300];
	strcpy(fpo_name, argv[1]);
	strcat(fpo_name, ".num");
	FILE *fpo = fopen(fpo_name, "r");
	fscanf(fpo, "%d", &h_input.cc_cnt);
	fclose(fpo);    
	uintT tmp_cc_cnt = h_input.cc_cnt;
	uintT sp_e = 0;


	struct v_struct *sp_t = (struct v_struct *)malloc(sizeof(struct v_struct)*h_input.V*2);

	for(uintT i=0;i<h_input.V; i++) {
		uintT v = h_input.hook[i];
		if(h_input.algo == RAND_NAIVE || h_input.algo == RAND_SPLIT_2) {
			if((h_input.lparent[i] & ULONG_T_MAX)) v = -1;
		}
		if(v == -1) {  h_input.cc_cnt--; }
		else {
			sp_t[sp_e].src = g_temp_v[v].src;
			sp_t[sp_e].dst = g_temp_v[v].dst;
			sp_t[sp_e+1].src = g_temp_v[v].dst;
			sp_t[sp_e+1].dst = g_temp_v[v].src;
			sp_e += 2;
		}
	}	
	if(h_input.cc_cnt != 0) { printf("FAIL1 %d\n", h_input.cc_cnt); exit(0);}

        qsort(sp_t, sp_e, sizeof(struct v_struct), compare00);		

	uintT *sp_csr_ptr = (uintT *)malloc(sizeof(uintT)*(h_input.V+1));
	uintT *sp_csr_idx = (uintT *)malloc(sizeof(uintT)*h_input.V*2);
	memset(sp_csr_ptr, 0, sizeof(uintT)*(h_input.V+1));

	for(int i=0;i<sp_e;i++) { 
		sp_csr_idx[i] = sp_t[i].dst;
		sp_csr_ptr[1+sp_t[i].src] = i+1;
	}

	for(int i=1;i<(h_input.V);i++) {
		if(sp_csr_ptr[i] == 0) sp_csr_ptr[i] = sp_csr_ptr[i-1];
	}
	sp_csr_ptr[h_input.V] = sp_e;

        int *c_q, *c_vv;
        int qhead, qtail;
        c_q = (int *)malloc(sizeof(int)*h_input.V);
        c_vv = (int *)malloc(sizeof(int)*h_input.V);
        memset(c_vv, 0, sizeof(int)*h_input.V);

	for(uintT i=0; i<h_input.V; i++) {
		if(c_vv[i] == 0) {
			c_vv[i] = 1;
			tmp_cc_cnt--;
			qhead = 0; qtail = 1;
			c_q[qhead] = i;
			while(1) {
				if(qhead == qtail) break;
				int ii = c_q[qhead];
				qhead++;
				for(int j=sp_csr_ptr[ii]; j<sp_csr_ptr[ii+1]; j++) {
					int k = sp_csr_idx[j];
					if(c_vv[k] == 0) {
						c_vv[k] = 1;
						c_q[qtail] = k;
						qtail++;
					}
				}
			}
		}
	}

	if(tmp_cc_cnt != 0) {printf("FAIL2 %d\n",tmp_cc_cnt); exit(0);}

	free(sp_t);
	free(sp_csr_ptr);
	free(sp_csr_idx);

	free(c_q); free(c_vv);


  #endif
#endif // end of validation

	}


	//#if defined(PATH_LENGTH) || defined(CHUNK_STREAMING)
	#if defined(PATH_LENGTH)
	//exit(0);
	#endif

#ifdef PRINT
	printf("PASS\n");
#endif
qsort(tot_ms, REPEAT, sizeof(double), compare1);
	if(REPEAT % 2 == 1) { 
		#if defined(CHUNK_STREAMING) || defined(STREAMING) || defined(STREAMING_SIMPLE)
		printf("%f,",(double)d_input.E/tot_ms[REPEAT/2]/1000000);
		#else
		printf("%f,", tot_ms[REPEAT/2]);
		#endif
	}
	else {
		#if defined(CHUNK_STREAMING) || defined(STREAMING) || defined(STREAMING_SIMPLE)
		printf("%f,", (double)d_input.E/((tot_ms[(REPEAT-1)/2]+tot_ms[(REPEAT-1)/2+1])/2)/1000000);
		#else
		printf("%f,", (tot_ms[(REPEAT-1)/2]+tot_ms[(REPEAT-1)/2+1])/2);
		#endif
	}

#if defined(CHUNK_STREAMING)
     }
#endif


#if defined(SP_TREE)
	free(g_temp_v);
#endif

  exit(0);


}
