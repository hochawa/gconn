#ifndef __GEN_GRAPH_DEFINED
#define __GEN_GRAPH_DEFINED

#include "common.h"

struct v_struct *temp_v;

int compare0(const void *a, const void *b)
{
        if (((struct v_struct *)a)->src > ((struct v_struct *)b)->src) return 1;
        if (((struct v_struct *)a)->src < ((struct v_struct *)b)->src) return -1;
        return (intT)((long)(((struct v_struct *)a)->dst) - (long)(intT)(((struct v_struct *)b)->dst));
}

int compare0v(const void *a, const void *b)
{
        if (((struct vv_struct *)a)->src > ((struct vv_struct *)b)->src) return 1;
        if (((struct vv_struct *)a)->src < ((struct vv_struct *)b)->src) return -1;
        return (intT)((long)(((struct vv_struct *)a)->dst) - (long)(intT)(((struct vv_struct *)b)->dst));
}

int compare0vv(const void *a, const void *b)
{
        if (((struct vv_struct *)a)->valid > ((struct vv_struct *)b)->valid) return 1;
        if (((struct vv_struct *)a)->valid < ((struct vv_struct *)b)->valid) return -1;
        return (intT)((long)(((struct vv_struct *)a)->idx) - (long)(((struct vv_struct *)b)->idx));
}



void mtx_parse_write(int argc, char **argv, struct graph_data *input)
{
	FILE *fp;
        unsigned int *loc;
        char buf[300];
        int nflag, sflag;
        int pre_count=0;
	uintT i; int dummy;

	fp = fopen(argv[1], "r");
	fgets(buf, 300, fp);
        if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; // symmetric
        else sflag = 0;
        if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
        else if(strstr(buf, "complex") != NULL) nflag = -1;
        else nflag = 1;

	input->is_sym = sflag;

//printf("(%s)\n", argv[4]);

	if(strcmp(argv[2], "SYM") == 0) { input->is_sym = 1; sflag = 1;}
	else if(strcmp(argv[2], "ASYM") != 0) { printf("SYM/ASYM should be listed\n"); exit(0); }

	// data format
	if(strcmp(argv[3], "COO") == 0) input->format = COO;
	else if(strcmp(argv[3], "CSR") == 0) input->format = CSR;
	else if(strcmp(argv[3], "SAMPLE") == 0) input->format = SAMPLE;
	else if(strcmp(argv[3], "BFS") == 0) input->format = BFS;
	else if(strcmp(argv[3], "COO_SAMPLE") == 0) input->format = COO_SAMPLE;
	else if(strcmp(argv[3], "IHOOK") == 0) input->format = IHOOK;
	else { printf("Data Format (COO or CSR) should be specified\n"); exit(0); }

	// execution strategy	
	if(strcmp(argv[4], "ASYNC_NAIVE") == 0) input->algo = ASYNC_NAIVE;
	else if(strcmp(argv[4], "ASYNC_COMPRESS") == 0) input->algo = ASYNC_COMPRESS;
	else if(strcmp(argv[4], "ASYNC_HALVE") == 0) input->algo = ASYNC_HALVE;
	else if(strcmp(argv[4], "ASYNC_SPLIT") == 0) input->algo = ASYNC_SPLIT;
	else if(strcmp(argv[4], "ASYNC_A_HALVE") == 0) input->algo = ASYNC_A_HALVE;
	else if(strcmp(argv[4], "ASYNC_A_SPLIT") == 0) input->algo = ASYNC_A_SPLIT;

	else if(strcmp(argv[4], "EARLY_NAIVE") == 0) input->algo = EARLY_NAIVE;
	else if(strcmp(argv[4], "EARLY_COMPRESS") == 0) input->algo = EARLY_COMPRESS;
	else if(strcmp(argv[4], "EARLY_A_HALVE") == 0) input->algo = EARLY_A_HALVE;
	else if(strcmp(argv[4], "EARLY_A_SPLIT") == 0) input->algo = EARLY_A_SPLIT;

	else if(strcmp(argv[4], "STOPT_NAIVE") == 0) input->algo = STOPT_NAIVE;
	else if(strcmp(argv[4], "STOPT_COMPRESS") == 0) input->algo = STOPT_COMPRESS;
	else if(strcmp(argv[4], "STOPT_HALVE") == 0) input->algo = STOPT_HALVE;
	else if(strcmp(argv[4], "STOPT_SPLIT") == 0) input->algo = STOPT_SPLIT;
	else if(strcmp(argv[4], "STOPT_A_HALVE") == 0) input->algo = STOPT_A_HALVE;
	else if(strcmp(argv[4], "STOPT_A_SPLIT") == 0) input->algo = STOPT_A_SPLIT;


	else if(strcmp(argv[4], "ECL_NAIVE") == 0) input->algo = ECL_NAIVE;
	else if(strcmp(argv[4], "ECL_COMPRESS") == 0) input->algo = ECL_COMPRESS;
	else if(strcmp(argv[4], "ECL_HALVE") == 0) input->algo = ECL_HALVE;
	else if(strcmp(argv[4], "ECL_SPLIT") == 0) input->algo = ECL_SPLIT;
	else if(strcmp(argv[4], "ECL_A_HALVE") == 0) input->algo = ECL_A_HALVE;
	else if(strcmp(argv[4], "ECL_A_SPLIT") == 0) input->algo = ECL_A_SPLIT;

	else if(strcmp(argv[4], "AFFOREST_NAIVE") == 0) input->algo = AFFOREST_NAIVE;
	else if(strcmp(argv[4], "AFFOREST_COMPRESS") == 0) input->algo = AFFOREST_COMPRESS;
	else if(strcmp(argv[4], "AFFOREST_HALVE") == 0) input->algo = AFFOREST_HALVE;
	else if(strcmp(argv[4], "AFFOREST_SPLIT") == 0) input->algo = AFFOREST_SPLIT;
	else if(strcmp(argv[4], "AFFOREST_A_HALVE") == 0) input->algo = AFFOREST_A_HALVE;
	else if(strcmp(argv[4], "AFFOREST_A_SPLIT") == 0) input->algo = AFFOREST_A_SPLIT;

	else if(strcmp(argv[4], "RAND_NAIVE") == 0) input->algo = RAND_NAIVE;
	else if(strcmp(argv[4], "RAND_SPLIT_2") == 0) input->algo = RAND_SPLIT_2;

	else if(strcmp(argv[4], "SIM_C_U_S_A") == 0) input->algo = SIM_C_U_S_A;
	else if(strcmp(argv[4], "SIM_C_R_S_A") == 0) input->algo = SIM_C_R_S_A;
	else if(strcmp(argv[4], "SIM_P_U_S_A") == 0) input->algo = SIM_P_U_S_A;
	else if(strcmp(argv[4], "SIM_P_R_S_A") == 0) input->algo = SIM_P_R_S_A;
	else if(strcmp(argv[4], "SIM_P_U_S") == 0) input->algo = SIM_P_U_S;
	else if(strcmp(argv[4], "SIM_P_R_S") == 0) input->algo = SIM_P_R_S;
	else if(strcmp(argv[4], "SIM_E_U_S_A") == 0) input->algo = SIM_E_U_S_A;
	else if(strcmp(argv[4], "SIM_E_U_S") == 0) input->algo = SIM_E_U_S;

	else if(strcmp(argv[4], "SIM_C_U_SS_A") == 0) input->algo = SIM_C_U_SS_A;
	else if(strcmp(argv[4], "SIM_C_R_SS_A") == 0) input->algo = SIM_C_R_SS_A;
	else if(strcmp(argv[4], "SIM_P_U_SS_A") == 0) input->algo = SIM_P_U_SS_A;
	else if(strcmp(argv[4], "SIM_P_R_SS_A") == 0) input->algo = SIM_P_R_SS_A;
	else if(strcmp(argv[4], "SIM_P_U_SS") == 0) input->algo = SIM_P_U_SS;
	else if(strcmp(argv[4], "SIM_P_R_SS") == 0) input->algo = SIM_P_R_SS;
	else if(strcmp(argv[4], "SIM_E_U_SS_A") == 0) input->algo = SIM_E_U_SS_A;
	else if(strcmp(argv[4], "SIM_E_U_SS") == 0) input->algo = SIM_E_U_SS;

	else if(strcmp(argv[4], "STERGIOS") == 0) input->algo = STERGIOS;
	else if(strcmp(argv[4], "SVA") == 0) input->algo = SVA;
	else if(strcmp(argv[4], "LPA") == 0) input->algo = LPA;

	else if(strcmp(argv[4], "REMCAS_NAIVE_SPLICE") == 0) input->algo = REMCAS_NAIVE_SPLICE;
	else if(strcmp(argv[4], "REMCAS_COMPRESS_SPLICE") == 0) input->algo = REMCAS_COMPRESS_SPLICE;
	else if(strcmp(argv[4], "REMCAS_HALVE_SPLICE") == 0) input->algo = REMCAS_HALVE_SPLICE;
	else if(strcmp(argv[4], "REMCAS_SPLIT_SPLICE") == 0) input->algo = REMCAS_SPLIT_SPLICE;
	else if(strcmp(argv[4], "REMCAS_A_HALVE_SPLICE") == 0) input->algo = REMCAS_A_HALVE_SPLICE;
	else if(strcmp(argv[4], "REMCAS_A_SPLIT_SPLICE") == 0) input->algo = REMCAS_A_SPLIT_SPLICE;

	else if(strcmp(argv[4], "REMCAS_NAIVE_SPLICE_CAS") == 0) input->algo = REMCAS_NAIVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_COMPRESS_SPLICE_CAS") == 0) input->algo = REMCAS_COMPRESS_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_HALVE_SPLICE_CAS") == 0) input->algo = REMCAS_HALVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_SPLIT_SPLICE_CAS") == 0) input->algo = REMCAS_SPLIT_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_A_HALVE_SPLICE_CAS") == 0) input->algo = REMCAS_A_HALVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_A_SPLIT_SPLICE_CAS") == 0) input->algo = REMCAS_A_SPLIT_SPLICE_CAS;

	else if(strcmp(argv[4], "REMCAS_NAIVE_SPLIT_ONE") == 0) input->algo = REMCAS_NAIVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_COMPRESS_SPLIT_ONE") == 0) input->algo = REMCAS_COMPRESS_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_HALVE_SPLIT_ONE") == 0) input->algo = REMCAS_HALVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_SPLIT_SPLIT_ONE") == 0) input->algo = REMCAS_SPLIT_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_A_HALVE_SPLIT_ONE") == 0) input->algo = REMCAS_A_HALVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_A_SPLIT_SPLIT_ONE") == 0) input->algo = REMCAS_A_SPLIT_SPLIT_ONE;

	else if(strcmp(argv[4], "REMCAS_NAIVE_HALVE_ONE") == 0) input->algo = REMCAS_NAIVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_COMPRESS_HALVE_ONE") == 0) input->algo = REMCAS_COMPRESS_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_HALVE_HALVE_ONE") == 0) input->algo = REMCAS_HALVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_SPLIT_HALVE_ONE") == 0) input->algo = REMCAS_SPLIT_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_A_HALVE_HALVE_ONE") == 0) input->algo = REMCAS_A_HALVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_A_SPLIT_HALVE_ONE") == 0) input->algo = REMCAS_A_SPLIT_HALVE_ONE;

	else if(strcmp(argv[4], "REMLOCK_NAIVE_SPLICE") == 0) input->algo = REMLOCK_NAIVE_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_COMPRESS_SPLICE") == 0) input->algo = REMLOCK_COMPRESS_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_HALVE_SPLICE") == 0) input->algo = REMLOCK_HALVE_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_SPLIT_SPLICE") == 0) input->algo = REMLOCK_SPLIT_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_A_HALVE_SPLICE") == 0) input->algo = REMLOCK_A_HALVE_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_A_SPLIT_SPLICE") == 0) input->algo = REMLOCK_A_SPLIT_SPLICE;

	else if(strcmp(argv[4], "REMLOCK_NAIVE_SPLICE_CAS") == 0) input->algo = REMLOCK_NAIVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_COMPRESS_SPLICE_CAS") == 0) input->algo = REMLOCK_COMPRESS_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_HALVE_SPLICE_CAS") == 0) input->algo = REMLOCK_HALVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_SPLIT_SPLICE_CAS") == 0) input->algo = REMLOCK_SPLIT_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_A_HALVE_SPLICE_CAS") == 0) input->algo = REMLOCK_A_HALVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_A_SPLIT_SPLICE_CAS") == 0) input->algo = REMLOCK_A_SPLIT_SPLICE_CAS;

	else if(strcmp(argv[4], "REMLOCK_NAIVE_SPLIT_ONE") == 0) input->algo = REMLOCK_NAIVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_COMPRESS_SPLIT_ONE") == 0) input->algo = REMLOCK_COMPRESS_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_HALVE_SPLIT_ONE") == 0) input->algo = REMLOCK_HALVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_SPLIT_SPLIT_ONE") == 0) input->algo = REMLOCK_SPLIT_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_A_HALVE_SPLIT_ONE") == 0) input->algo = REMLOCK_A_HALVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_A_SPLIT_SPLIT_ONE") == 0) input->algo = REMLOCK_A_SPLIT_SPLIT_ONE;

	else if(strcmp(argv[4], "REMLOCK_NAIVE_HALVE_ONE") == 0) input->algo = REMLOCK_NAIVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_COMPRESS_HALVE_ONE") == 0) input->algo = REMLOCK_COMPRESS_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_HALVE_HALVE_ONE") == 0) input->algo = REMLOCK_HALVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_SPLIT_HALVE_ONE") == 0) input->algo = REMLOCK_SPLIT_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_A_HALVE_HALVE_ONE") == 0) input->algo = REMLOCK_A_HALVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_A_SPLIT_HALVE_ONE") == 0) input->algo = REMLOCK_A_SPLIT_HALVE_ONE;

	else if(strcmp(argv[4], "BFS_CC") == 0) input->algo = BFS_CC;

	else { printf("Strategy (e.g., ASYNC_NAIVE) should be specified\n"); exit(0); }

        while(1) {
                pre_count++;
                fgets(buf, 300, fp);
                if(strstr(buf, "%") == NULL) break;
        }
        fclose(fp);

		

        fp = fopen(argv[1], "r");
        for(i=0;i<pre_count;i++)
                fgets(buf, 300, fp);

        fscanf(fp, "%d %d %d", &(input->V), &dummy, &(input->E));
	if(dummy > (input->V)) (input->V) = dummy;
	uintT prev_ne = (input->E); 
	(input->E) *= (sflag+1);

 if((input->format == SAMPLE || input->format == BFS || input->format == IHOOK) && input->is_sym == 0) {
        temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(2*(input->E)+1+input->V));
 } else {
        temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*((input->E)+1+input->V));
 }

        for(uintT i=0;i<prev_ne;i++) {
                fscanf(fp, "%d %d", &temp_v[i].src, &temp_v[i].dst);
                temp_v[i].src--; temp_v[i].dst--;

                if(temp_v[i].src < 0 || temp_v[i].src >= (input->V) || temp_v[i].dst < 0 || temp_v[i].dst >= (input->V)) { // unsigned -> simplify condition
                        fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].src+1, temp_v[i].dst+1);
                        exit(0);
                }
                if (nflag == 1) {
                        float ftemp;
                        fscanf(fp, " %f ", &ftemp);
                } else if (nflag == 2) { // complex
                        float ftemp1, ftemp2;
                        fscanf(fp, " %f %f ", &ftemp1, &ftemp2);
                }

                if(sflag == 1) {
                        //i++;
                        temp_v[i+prev_ne].src = temp_v[i].dst;
                        temp_v[i+prev_ne].dst = temp_v[i].src;
                }
        }
	if(input->format == COO) {
		struct vv_struct *vv = (struct vv_struct *)malloc(sizeof(struct vv_struct)*(input->E)+(input->V));
		for(uintT i=0; i<input->E; i++) {
			vv[i].src = temp_v[i].src;
			vv[i].dst = temp_v[i].dst;
			vv[i].idx = i;
			vv[i].valid = 0;
		}
		qsort(vv, input->E, sizeof(struct vv_struct), compare0v);

		if(vv[0].src ==  vv[0].dst) vv[0].valid = 1;
		for(i=1;i<(input->E);i++) {
			if(vv[i].src == vv[i-1].src && vv[i].dst == vv[i-1].dst)
				vv[i].valid = 1;
			else if(vv[i].src == vv[i].dst)
				vv[i].valid = 1;
			else vv[i].valid = 0;
		}
		qsort(vv, input->E, sizeof(struct vv_struct), compare0vv);
		for(i=1;i<input->E;i++) {
			if(vv[i].valid == 1) break;
		}
		input->E = i;
		for(i=0;i<input->E;i++) {
			temp_v[i].src = vv[i].src; 
			temp_v[i].dst = vv[i].dst;
		}

	 	char fpo_name[300];
	  	strcpy(fpo_name, argv[1]);
	  	strcat(fpo_name, ".unsorted");
		FILE *fpo = fopen(fpo_name, "wb");
		fwrite(temp_v, sizeof(struct v_struct), input->E, fpo);
		fclose(fpo);

	 	char fpo_name2[300];
	  	strcpy(fpo_name2, argv[1]);
	  	strcat(fpo_name2, ".unsorted_e");
		FILE *fpo2 = fopen(fpo_name2, "w");
		fprintf(fpo2, "%ld", input->E);
		fclose(fpo2);


	}
	

	if(input->format == CSR || input->format == SAMPLE || input->format == BFS || input->format == IHOOK) {
		qsort(temp_v, input->E, sizeof(struct v_struct), compare0);
		loc = (unsigned int *)malloc(sizeof(unsigned int)*((input->E)+1));

		memset(loc, 0, sizeof(unsigned int)*((input->E)+1));
		if(temp_v[0].src !=  temp_v[0].dst) loc[0]=1;
		for(i=1;i<(input->E);i++) {
			if(temp_v[i].src == temp_v[i-1].src && temp_v[i].dst == temp_v[i-1].dst)
				loc[i] = 0;
			else if(temp_v[i].src == temp_v[i].dst)
				loc[i] = 0;
			else loc[i] = 1;
		}
		for(i=1;i<=(input->E);i++)
			loc[i] += loc[i-1];
		for(i=(input->E); i>=1; i--)
			loc[i] = loc[i-1];
		loc[0] = 0;

		for(i=0;i<(input->E);i++) {
			temp_v[loc[i]].src = temp_v[i].src;
			temp_v[loc[i]].dst = temp_v[i].dst;
		}

		(input->E) = loc[input->E];

		for(i=0;i<(input->E);i++) {
			if(temp_v[i].src == temp_v[i].dst) fprintf(stderr, "inp err: %d %d\n", i, temp_v[i].src);
		}
		free(loc);

	 	char fpo_name[300];
	  	strcpy(fpo_name, argv[1]);
	  	strcat(fpo_name, ".sorted");
		FILE *fpo = fopen(fpo_name, "wb");
		fwrite(temp_v, sizeof(struct v_struct), input->E, fpo);
		fclose(fpo);

	 	char fpo_name2[300];
	  	strcpy(fpo_name2, argv[1]);
	  	strcat(fpo_name2, ".sorted_e");
		FILE *fpo2 = fopen(fpo_name2, "w");
		fprintf(fpo2, "%ld", input->E);
		fclose(fpo2);

	}
	exit(0);
}


void mtx_parse_fast(int argc, char **argv, struct graph_data *input)
{
	FILE *fp;
        unsigned int *loc;
        char buf[300];
        int nflag, sflag;
        int pre_count=0;
	uintT i; int dummy;

	fp = fopen(argv[1], "r");
	fgets(buf, 300, fp);
        if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; // symmetric
        else sflag = 0;
        if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
        else if(strstr(buf, "complex") != NULL) nflag = -1;
        else nflag = 1;

	input->is_sym = sflag;

//#define QUERY_TEST
#ifdef QUERY_TEST
        double dump_f = atof(argv[5]);
        double dump = 1 + (double)1/dump_f;
#endif

	if(strcmp(argv[2], "SYM") == 0) { input->is_sym = 1; sflag = 1;}
	else if(strcmp(argv[2], "ASYM") != 0) { printf("SYM/ASYM should be listed\n"); exit(0); }

	// data format
	if(strcmp(argv[3], "COO") == 0) input->format = COO;
	else if(strcmp(argv[3], "CSR") == 0) input->format = CSR;
	else if(strcmp(argv[3], "SAMPLE") == 0) input->format = SAMPLE;
	else if(strcmp(argv[3], "BFS") == 0) input->format = BFS;
	else if(strcmp(argv[3], "COO_SAMPLE") == 0) input->format = COO_SAMPLE;
	else if(strcmp(argv[3], "IHOOK") == 0) input->format = IHOOK;
	else { printf("Data Format (COO or CSR) should be specified\n"); exit(0); }

	// execution strategy	
	if(strcmp(argv[4], "ASYNC_NAIVE") == 0) input->algo = ASYNC_NAIVE;
	else if(strcmp(argv[4], "ASYNC_COMPRESS") == 0) input->algo = ASYNC_COMPRESS;
	else if(strcmp(argv[4], "ASYNC_HALVE") == 0) input->algo = ASYNC_HALVE;
	else if(strcmp(argv[4], "ASYNC_SPLIT") == 0) input->algo = ASYNC_SPLIT;
	else if(strcmp(argv[4], "ASYNC_A_HALVE") == 0) input->algo = ASYNC_A_HALVE;
	else if(strcmp(argv[4], "ASYNC_A_SPLIT") == 0) input->algo = ASYNC_A_SPLIT;

	else if(strcmp(argv[4], "EARLY_NAIVE") == 0) input->algo = EARLY_NAIVE;
	else if(strcmp(argv[4], "EARLY_COMPRESS") == 0) input->algo = EARLY_COMPRESS;
	else if(strcmp(argv[4], "EARLY_A_HALVE") == 0) input->algo = EARLY_A_HALVE;
	else if(strcmp(argv[4], "EARLY_A_SPLIT") == 0) input->algo = EARLY_A_SPLIT;

	else if(strcmp(argv[4], "STOPT_NAIVE") == 0) input->algo = STOPT_NAIVE;
	else if(strcmp(argv[4], "STOPT_COMPRESS") == 0) input->algo = STOPT_COMPRESS;
	else if(strcmp(argv[4], "STOPT_HALVE") == 0) input->algo = STOPT_HALVE;
	else if(strcmp(argv[4], "STOPT_SPLIT") == 0) input->algo = STOPT_SPLIT;
	else if(strcmp(argv[4], "STOPT_A_HALVE") == 0) input->algo = STOPT_A_HALVE;
	else if(strcmp(argv[4], "STOPT_A_SPLIT") == 0) input->algo = STOPT_A_SPLIT;


	else if(strcmp(argv[4], "ECL_NAIVE") == 0) input->algo = ECL_NAIVE;
	else if(strcmp(argv[4], "ECL_COMPRESS") == 0) input->algo = ECL_COMPRESS;
	else if(strcmp(argv[4], "ECL_HALVE") == 0) input->algo = ECL_HALVE;
	else if(strcmp(argv[4], "ECL_SPLIT") == 0) input->algo = ECL_SPLIT;
	else if(strcmp(argv[4], "ECL_A_HALVE") == 0) input->algo = ECL_A_HALVE;
	else if(strcmp(argv[4], "ECL_A_SPLIT") == 0) input->algo = ECL_A_SPLIT;

	else if(strcmp(argv[4], "AFFOREST_NAIVE") == 0) input->algo = AFFOREST_NAIVE;
	else if(strcmp(argv[4], "AFFOREST_COMPRESS") == 0) input->algo = AFFOREST_COMPRESS;
	else if(strcmp(argv[4], "AFFOREST_HALVE") == 0) input->algo = AFFOREST_HALVE;
	else if(strcmp(argv[4], "AFFOREST_SPLIT") == 0) input->algo = AFFOREST_SPLIT;
	else if(strcmp(argv[4], "AFFOREST_A_HALVE") == 0) input->algo = AFFOREST_A_HALVE;
	else if(strcmp(argv[4], "AFFOREST_A_SPLIT") == 0) input->algo = AFFOREST_A_SPLIT;

	else if(strcmp(argv[4], "RAND_NAIVE") == 0) input->algo = RAND_NAIVE;
	else if(strcmp(argv[4], "RAND_SPLIT_2") == 0) input->algo = RAND_SPLIT_2;

	else if(strcmp(argv[4], "SIM_C_U_S_A") == 0) input->algo = SIM_C_U_S_A;
	else if(strcmp(argv[4], "SIM_C_R_S_A") == 0) input->algo = SIM_C_R_S_A;
	else if(strcmp(argv[4], "SIM_P_U_S_A") == 0) input->algo = SIM_P_U_S_A;
	else if(strcmp(argv[4], "SIM_P_R_S_A") == 0) input->algo = SIM_P_R_S_A;
	else if(strcmp(argv[4], "SIM_P_U_S") == 0) input->algo = SIM_P_U_S;
	else if(strcmp(argv[4], "SIM_P_R_S") == 0) input->algo = SIM_P_R_S;
	else if(strcmp(argv[4], "SIM_E_U_S_A") == 0) input->algo = SIM_E_U_S_A;
	else if(strcmp(argv[4], "SIM_E_U_S") == 0) input->algo = SIM_E_U_S;

	else if(strcmp(argv[4], "SIM_C_U_SS_A") == 0) input->algo = SIM_C_U_SS_A;
	else if(strcmp(argv[4], "SIM_C_R_SS_A") == 0) input->algo = SIM_C_R_SS_A;
	else if(strcmp(argv[4], "SIM_P_U_SS_A") == 0) input->algo = SIM_P_U_SS_A;
	else if(strcmp(argv[4], "SIM_P_R_SS_A") == 0) input->algo = SIM_P_R_SS_A;
	else if(strcmp(argv[4], "SIM_P_U_SS") == 0) input->algo = SIM_P_U_SS;
	else if(strcmp(argv[4], "SIM_P_R_SS") == 0) input->algo = SIM_P_R_SS;
	else if(strcmp(argv[4], "SIM_E_U_SS_A") == 0) input->algo = SIM_E_U_SS_A;
	else if(strcmp(argv[4], "SIM_E_U_SS") == 0) input->algo = SIM_E_U_SS;

	else if(strcmp(argv[4], "STERGIOS") == 0) input->algo = STERGIOS;
	else if(strcmp(argv[4], "SVA") == 0) input->algo = SVA;
	else if(strcmp(argv[4], "LPA") == 0) input->algo = LPA;

	else if(strcmp(argv[4], "REMCAS_NAIVE_SPLICE") == 0) input->algo = REMCAS_NAIVE_SPLICE;
	else if(strcmp(argv[4], "REMCAS_COMPRESS_SPLICE") == 0) input->algo = REMCAS_COMPRESS_SPLICE;
	else if(strcmp(argv[4], "REMCAS_HALVE_SPLICE") == 0) input->algo = REMCAS_HALVE_SPLICE;
	else if(strcmp(argv[4], "REMCAS_SPLIT_SPLICE") == 0) input->algo = REMCAS_SPLIT_SPLICE;
	else if(strcmp(argv[4], "REMCAS_A_HALVE_SPLICE") == 0) input->algo = REMCAS_A_HALVE_SPLICE;
	else if(strcmp(argv[4], "REMCAS_A_SPLIT_SPLICE") == 0) input->algo = REMCAS_A_SPLIT_SPLICE;

	else if(strcmp(argv[4], "REMCAS_NAIVE_SPLICE_CAS") == 0) input->algo = REMCAS_NAIVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_COMPRESS_SPLICE_CAS") == 0) input->algo = REMCAS_COMPRESS_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_HALVE_SPLICE_CAS") == 0) input->algo = REMCAS_HALVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_SPLIT_SPLICE_CAS") == 0) input->algo = REMCAS_SPLIT_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_A_HALVE_SPLICE_CAS") == 0) input->algo = REMCAS_A_HALVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_A_SPLIT_SPLICE_CAS") == 0) input->algo = REMCAS_A_SPLIT_SPLICE_CAS;

	else if(strcmp(argv[4], "REMCAS_NAIVE_SPLIT_ONE") == 0) input->algo = REMCAS_NAIVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_COMPRESS_SPLIT_ONE") == 0) input->algo = REMCAS_COMPRESS_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_HALVE_SPLIT_ONE") == 0) input->algo = REMCAS_HALVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_SPLIT_SPLIT_ONE") == 0) input->algo = REMCAS_SPLIT_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_A_HALVE_SPLIT_ONE") == 0) input->algo = REMCAS_A_HALVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_A_SPLIT_SPLIT_ONE") == 0) input->algo = REMCAS_A_SPLIT_SPLIT_ONE;

	else if(strcmp(argv[4], "REMCAS_NAIVE_HALVE_ONE") == 0) input->algo = REMCAS_NAIVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_COMPRESS_HALVE_ONE") == 0) input->algo = REMCAS_COMPRESS_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_HALVE_HALVE_ONE") == 0) input->algo = REMCAS_HALVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_SPLIT_HALVE_ONE") == 0) input->algo = REMCAS_SPLIT_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_A_HALVE_HALVE_ONE") == 0) input->algo = REMCAS_A_HALVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_A_SPLIT_HALVE_ONE") == 0) input->algo = REMCAS_A_SPLIT_HALVE_ONE;

	else if(strcmp(argv[4], "REMLOCK_NAIVE_SPLICE") == 0) input->algo = REMLOCK_NAIVE_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_COMPRESS_SPLICE") == 0) input->algo = REMLOCK_COMPRESS_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_HALVE_SPLICE") == 0) input->algo = REMLOCK_HALVE_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_SPLIT_SPLICE") == 0) input->algo = REMLOCK_SPLIT_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_A_HALVE_SPLICE") == 0) input->algo = REMLOCK_A_HALVE_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_A_SPLIT_SPLICE") == 0) input->algo = REMLOCK_A_SPLIT_SPLICE;

	else if(strcmp(argv[4], "REMLOCK_NAIVE_SPLICE_CAS") == 0) input->algo = REMLOCK_NAIVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_COMPRESS_SPLICE_CAS") == 0) input->algo = REMLOCK_COMPRESS_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_HALVE_SPLICE_CAS") == 0) input->algo = REMLOCK_HALVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_SPLIT_SPLICE_CAS") == 0) input->algo = REMLOCK_SPLIT_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_A_HALVE_SPLICE_CAS") == 0) input->algo = REMLOCK_A_HALVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_A_SPLIT_SPLICE_CAS") == 0) input->algo = REMLOCK_A_SPLIT_SPLICE_CAS;

	else if(strcmp(argv[4], "REMLOCK_NAIVE_SPLIT_ONE") == 0) input->algo = REMLOCK_NAIVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_COMPRESS_SPLIT_ONE") == 0) input->algo = REMLOCK_COMPRESS_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_HALVE_SPLIT_ONE") == 0) input->algo = REMLOCK_HALVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_SPLIT_SPLIT_ONE") == 0) input->algo = REMLOCK_SPLIT_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_A_HALVE_SPLIT_ONE") == 0) input->algo = REMLOCK_A_HALVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_A_SPLIT_SPLIT_ONE") == 0) input->algo = REMLOCK_A_SPLIT_SPLIT_ONE;

	else if(strcmp(argv[4], "REMLOCK_NAIVE_HALVE_ONE") == 0) input->algo = REMLOCK_NAIVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_COMPRESS_HALVE_ONE") == 0) input->algo = REMLOCK_COMPRESS_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_HALVE_HALVE_ONE") == 0) input->algo = REMLOCK_HALVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_SPLIT_HALVE_ONE") == 0) input->algo = REMLOCK_SPLIT_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_A_HALVE_HALVE_ONE") == 0) input->algo = REMLOCK_A_HALVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_A_SPLIT_HALVE_ONE") == 0) input->algo = REMLOCK_A_SPLIT_HALVE_ONE;

	else if(strcmp(argv[4], "BFS_CC") == 0) input->algo = BFS_CC;

	else { printf("Strategy (e.g., ASYNC_NAIVE) should be specified\n"); exit(0); }



        while(1) {
                pre_count++;
                fgets(buf, 300, fp);
                if(strstr(buf, "%") == NULL) break;
        }
        fclose(fp);

        fp = fopen(argv[1], "r");
        for(i=0;i<pre_count;i++)
                fgets(buf, 300, fp);

        fscanf(fp, "%d %d %d", &(input->V), &dummy, &(input->E));
	if(dummy > (input->V)) (input->V) = dummy;
	uintT prev_ne = (input->E); 
	(input->E) *= (sflag+1);

 if((input->format == SAMPLE || input->format == BFS || input->format == IHOOK) && input->is_sym == 0) {
#ifdef QUERY_TEST
        temp_v = (struct v_struct *)malloc(1024+dump*sizeof(struct v_struct)*(2*(input->E)+1+input->V));
#else
        temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(2*(input->E)+1+input->V));
#endif
 } else {
#ifdef QUERY_TEST
        temp_v = (struct v_struct *)malloc(1024+dump*sizeof(struct v_struct)*((input->E)+1+input->V));
#else
        temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*((input->E)+1+input->V));
#endif
 }

	if(input->format == CSR || input->format == SAMPLE || input->format == BFS || input->format == IHOOK) {
		char fpo_name2[300];
		strcpy(fpo_name2, argv[1]);
		strcat(fpo_name2, ".sorted_e");
		FILE *fpo2 = fopen(fpo_name2, "r");
		fscanf(fpo2, "%u", &(input->E));
		fclose(fpo2);

		char fpo_name[300];
		strcpy(fpo_name, argv[1]);
		strcat(fpo_name, ".sorted");
		FILE *fpo = fopen(fpo_name, "rb");
		fread(temp_v, sizeof(struct v_struct), input->E, fpo);
		fclose(fpo);
	} else { // COO or COO_SAMPLE
		char fpo_name2[300];
		strcpy(fpo_name2, argv[1]);
		strcat(fpo_name2, ".unsorted_e");
		FILE *fpo2 = fopen(fpo_name2, "r");
		fscanf(fpo2, "%u", &(input->E));
		fclose(fpo2);

		char fpo_name[300];
		strcpy(fpo_name, argv[1]);
		strcat(fpo_name, ".unsorted");
		FILE *fpo = fopen(fpo_name, "rb");
		fread(temp_v, sizeof(struct v_struct), input->E, fpo);
		fclose(fpo);
#ifdef QUERY_TEST
                uintT pE = input->E;
                input->E *= dump;
                //fprintf(stdout, "(%d),", input->E);
                for(uintT i=pE;i<input->E;i++) {
                        temp_v[i].src = rand()%(input->V);
                        temp_v[i].dst = rand()%(input->V);
                }
		for(uintT i=pE-1;i>=0;i--) {
			uintT k = i * dump;
			uintT src_tmp = temp_v[i].src;
			uintT dst_tmp = temp_v[i].dst;
			temp_v[i].src = temp_v[k].src;
			temp_v[i].dst = temp_v[k].dst;
			temp_v[k].src = src_tmp;
			temp_v[i].dst = dst_tmp;
		}
#endif
	}

}



void mtx_parse(int argc, char **argv, struct graph_data *input)
{
	FILE *fp;
        unsigned int *loc;
        char buf[300];
        int nflag, sflag;
        int pre_count=0;
	uintT i; int dummy;

	fp = fopen(argv[1], "r");
	fgets(buf, 300, fp);
        if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; // symmetric
        else sflag = 0;
        if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
        else if(strstr(buf, "complex") != NULL) nflag = -1;
        else nflag = 1;

	input->is_sym = sflag;


	if(strcmp(argv[2], "SYM") == 0) { input->is_sym = 1; sflag = 1;}
	else if(strcmp(argv[2], "ASYM") != 0) { printf("SYM/ASYM should be listed\n"); exit(0); }

	// data format
	if(strcmp(argv[3], "COO") == 0) input->format = COO;
	else if(strcmp(argv[3], "CSR") == 0) input->format = CSR;
	else if(strcmp(argv[3], "SAMPLE") == 0) input->format = SAMPLE;
	else if(strcmp(argv[3], "BFS") == 0) input->format = BFS;
	else if(strcmp(argv[3], "COO_SAMPLE") == 0) input->format = COO_SAMPLE;
	else if(strcmp(argv[3], "IHOOK") == 0) input->format = IHOOK;
	else { printf("Data Format (COO or CSR) should be specified\n"); exit(0); }

	// execution strategy	
	if(strcmp(argv[4], "ASYNC_NAIVE") == 0) input->algo = ASYNC_NAIVE;
	else if(strcmp(argv[4], "ASYNC_COMPRESS") == 0) input->algo = ASYNC_COMPRESS;
	else if(strcmp(argv[4], "ASYNC_HALVE") == 0) input->algo = ASYNC_HALVE;
	else if(strcmp(argv[4], "ASYNC_SPLIT") == 0) input->algo = ASYNC_SPLIT;
	else if(strcmp(argv[4], "ASYNC_A_HALVE") == 0) input->algo = ASYNC_A_HALVE;
	else if(strcmp(argv[4], "ASYNC_A_SPLIT") == 0) input->algo = ASYNC_A_SPLIT;

	else if(strcmp(argv[4], "EARLY_NAIVE") == 0) input->algo = EARLY_NAIVE;

	else if(strcmp(argv[4], "STOPT_NAIVE") == 0) input->algo = STOPT_NAIVE;
	else if(strcmp(argv[4], "STOPT_COMPRESS") == 0) input->algo = STOPT_COMPRESS;
	else if(strcmp(argv[4], "STOPT_HALVE") == 0) input->algo = STOPT_HALVE;
	else if(strcmp(argv[4], "STOPT_SPLIT") == 0) input->algo = STOPT_SPLIT;
	else if(strcmp(argv[4], "STOPT_A_HALVE") == 0) input->algo = STOPT_A_HALVE;
	else if(strcmp(argv[4], "STOPT_A_SPLIT") == 0) input->algo = STOPT_A_SPLIT;


	else if(strcmp(argv[4], "ECL_NAIVE") == 0) input->algo = ECL_NAIVE;
	else if(strcmp(argv[4], "ECL_COMPRESS") == 0) input->algo = ECL_COMPRESS;
	else if(strcmp(argv[4], "ECL_HALVE") == 0) input->algo = ECL_HALVE;
	else if(strcmp(argv[4], "ECL_SPLIT") == 0) input->algo = ECL_SPLIT;
	else if(strcmp(argv[4], "ECL_A_HALVE") == 0) input->algo = ECL_A_HALVE;
	else if(strcmp(argv[4], "ECL_A_SPLIT") == 0) input->algo = ECL_A_SPLIT;

	else if(strcmp(argv[4], "AFFOREST_NAIVE") == 0) input->algo = AFFOREST_NAIVE;
	else if(strcmp(argv[4], "AFFOREST_COMPRESS") == 0) input->algo = AFFOREST_COMPRESS;
	else if(strcmp(argv[4], "AFFOREST_HALVE") == 0) input->algo = AFFOREST_HALVE;
	else if(strcmp(argv[4], "AFFOREST_SPLIT") == 0) input->algo = AFFOREST_SPLIT;
	else if(strcmp(argv[4], "AFFOREST_A_HALVE") == 0) input->algo = AFFOREST_A_HALVE;
	else if(strcmp(argv[4], "AFFOREST_A_SPLIT") == 0) input->algo = AFFOREST_A_SPLIT;

	else if(strcmp(argv[4], "RAND_NAIVE") == 0) input->algo = RAND_NAIVE;
	else if(strcmp(argv[4], "RAND_SPLIT_2") == 0) input->algo = RAND_SPLIT_2;

	else if(strcmp(argv[4], "SIM_C_U_S_A") == 0) input->algo = SIM_C_U_S_A;
	else if(strcmp(argv[4], "SIM_C_R_S_A") == 0) input->algo = SIM_C_R_S_A;
	else if(strcmp(argv[4], "SIM_P_U_S_A") == 0) input->algo = SIM_P_U_S_A;
	else if(strcmp(argv[4], "SIM_P_R_S_A") == 0) input->algo = SIM_P_R_S_A;
	else if(strcmp(argv[4], "SIM_P_U_S") == 0) input->algo = SIM_P_U_S;
	else if(strcmp(argv[4], "SIM_P_R_S") == 0) input->algo = SIM_P_R_S;
	else if(strcmp(argv[4], "SIM_E_U_S_A") == 0) input->algo = SIM_E_U_S_A;
	else if(strcmp(argv[4], "SIM_E_U_S") == 0) input->algo = SIM_E_U_S;

	else if(strcmp(argv[4], "SIM_C_U_SS_A") == 0) input->algo = SIM_C_U_SS_A;
	else if(strcmp(argv[4], "SIM_C_R_SS_A") == 0) input->algo = SIM_C_R_SS_A;
	else if(strcmp(argv[4], "SIM_P_U_SS_A") == 0) input->algo = SIM_P_U_SS_A;
	else if(strcmp(argv[4], "SIM_P_R_SS_A") == 0) input->algo = SIM_P_R_SS_A;
	else if(strcmp(argv[4], "SIM_P_U_SS") == 0) input->algo = SIM_P_U_SS;
	else if(strcmp(argv[4], "SIM_P_R_SS") == 0) input->algo = SIM_P_R_SS;
	else if(strcmp(argv[4], "SIM_E_U_SS_A") == 0) input->algo = SIM_E_U_SS_A;
	else if(strcmp(argv[4], "SIM_E_U_SS") == 0) input->algo = SIM_E_U_SS;

	else if(strcmp(argv[4], "STERGIOS") == 0) input->algo = STERGIOS;
	else if(strcmp(argv[4], "SVA") == 0) input->algo = SVA;
	else if(strcmp(argv[4], "LPA") == 0) input->algo = LPA;

	else if(strcmp(argv[4], "REMCAS_NAIVE_SPLICE") == 0) input->algo = REMCAS_NAIVE_SPLICE;
	else if(strcmp(argv[4], "REMCAS_COMPRESS_SPLICE") == 0) input->algo = REMCAS_COMPRESS_SPLICE;
	else if(strcmp(argv[4], "REMCAS_HALVE_SPLICE") == 0) input->algo = REMCAS_HALVE_SPLICE;
	else if(strcmp(argv[4], "REMCAS_SPLIT_SPLICE") == 0) input->algo = REMCAS_SPLIT_SPLICE;
	else if(strcmp(argv[4], "REMCAS_A_HALVE_SPLICE") == 0) input->algo = REMCAS_A_HALVE_SPLICE;
	else if(strcmp(argv[4], "REMCAS_A_SPLIT_SPLICE") == 0) input->algo = REMCAS_A_SPLIT_SPLICE;

	else if(strcmp(argv[4], "REMCAS_NAIVE_SPLICE_CAS") == 0) input->algo = REMCAS_NAIVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_COMPRESS_SPLICE_CAS") == 0) input->algo = REMCAS_COMPRESS_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_HALVE_SPLICE_CAS") == 0) input->algo = REMCAS_HALVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_SPLIT_SPLICE_CAS") == 0) input->algo = REMCAS_SPLIT_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_A_HALVE_SPLICE_CAS") == 0) input->algo = REMCAS_A_HALVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMCAS_A_SPLIT_SPLICE_CAS") == 0) input->algo = REMCAS_A_SPLIT_SPLICE_CAS;

	else if(strcmp(argv[4], "REMCAS_NAIVE_SPLIT_ONE") == 0) input->algo = REMCAS_NAIVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_COMPRESS_SPLIT_ONE") == 0) input->algo = REMCAS_COMPRESS_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_HALVE_SPLIT_ONE") == 0) input->algo = REMCAS_HALVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_SPLIT_SPLIT_ONE") == 0) input->algo = REMCAS_SPLIT_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_A_HALVE_SPLIT_ONE") == 0) input->algo = REMCAS_A_HALVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMCAS_A_SPLIT_SPLIT_ONE") == 0) input->algo = REMCAS_A_SPLIT_SPLIT_ONE;

	else if(strcmp(argv[4], "REMCAS_NAIVE_HALVE_ONE") == 0) input->algo = REMCAS_NAIVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_COMPRESS_HALVE_ONE") == 0) input->algo = REMCAS_COMPRESS_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_HALVE_HALVE_ONE") == 0) input->algo = REMCAS_HALVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_SPLIT_HALVE_ONE") == 0) input->algo = REMCAS_SPLIT_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_A_HALVE_HALVE_ONE") == 0) input->algo = REMCAS_A_HALVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMCAS_A_SPLIT_HALVE_ONE") == 0) input->algo = REMCAS_A_SPLIT_HALVE_ONE;

	else if(strcmp(argv[4], "REMLOCK_NAIVE_SPLICE") == 0) input->algo = REMLOCK_NAIVE_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_COMPRESS_SPLICE") == 0) input->algo = REMLOCK_COMPRESS_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_HALVE_SPLICE") == 0) input->algo = REMLOCK_HALVE_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_SPLIT_SPLICE") == 0) input->algo = REMLOCK_SPLIT_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_A_HALVE_SPLICE") == 0) input->algo = REMLOCK_A_HALVE_SPLICE;
	else if(strcmp(argv[4], "REMLOCK_A_SPLIT_SPLICE") == 0) input->algo = REMLOCK_A_SPLIT_SPLICE;

	else if(strcmp(argv[4], "REMLOCK_NAIVE_SPLICE_CAS") == 0) input->algo = REMLOCK_NAIVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_COMPRESS_SPLICE_CAS") == 0) input->algo = REMLOCK_COMPRESS_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_HALVE_SPLICE_CAS") == 0) input->algo = REMLOCK_HALVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_SPLIT_SPLICE_CAS") == 0) input->algo = REMLOCK_SPLIT_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_A_HALVE_SPLICE_CAS") == 0) input->algo = REMLOCK_A_HALVE_SPLICE_CAS;
	else if(strcmp(argv[4], "REMLOCK_A_SPLIT_SPLICE_CAS") == 0) input->algo = REMLOCK_A_SPLIT_SPLICE_CAS;

	else if(strcmp(argv[4], "REMLOCK_NAIVE_SPLIT_ONE") == 0) input->algo = REMLOCK_NAIVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_COMPRESS_SPLIT_ONE") == 0) input->algo = REMLOCK_COMPRESS_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_HALVE_SPLIT_ONE") == 0) input->algo = REMLOCK_HALVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_SPLIT_SPLIT_ONE") == 0) input->algo = REMLOCK_SPLIT_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_A_HALVE_SPLIT_ONE") == 0) input->algo = REMLOCK_A_HALVE_SPLIT_ONE;
	else if(strcmp(argv[4], "REMLOCK_A_SPLIT_SPLIT_ONE") == 0) input->algo = REMLOCK_A_SPLIT_SPLIT_ONE;

	else if(strcmp(argv[4], "REMLOCK_NAIVE_HALVE_ONE") == 0) input->algo = REMLOCK_NAIVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_COMPRESS_HALVE_ONE") == 0) input->algo = REMLOCK_COMPRESS_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_HALVE_HALVE_ONE") == 0) input->algo = REMLOCK_HALVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_SPLIT_HALVE_ONE") == 0) input->algo = REMLOCK_SPLIT_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_A_HALVE_HALVE_ONE") == 0) input->algo = REMLOCK_A_HALVE_HALVE_ONE;
	else if(strcmp(argv[4], "REMLOCK_A_SPLIT_HALVE_ONE") == 0) input->algo = REMLOCK_A_SPLIT_HALVE_ONE;

	else if(strcmp(argv[4], "BFS_CC") == 0) input->algo = BFS_CC;

	else { printf("Strategy (e.g., ASYNC_NAIVE) should be specified\n"); exit(0); }

        while(1) {
                pre_count++;
                fgets(buf, 300, fp);
                if(strstr(buf, "%") == NULL) break;
        }
        fclose(fp);

        fp = fopen(argv[1], "r");
        for(i=0;i<pre_count;i++)
                fgets(buf, 300, fp);

        fscanf(fp, "%d %d %d", &(input->V), &dummy, &(input->E));
	if(dummy > (input->V)) (input->V) = dummy;
	uintT prev_ne = (input->E); 
	(input->E) *= (sflag+1);

 if((input->format == SAMPLE || input->format == BFS || input->format == IHOOK) && input->is_sym == 0) {
        temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(2*(input->E)+1+input->V));
 } else {
        temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*((input->E)+1+input->V));
 }


        for(uintT i=0;i<prev_ne;i++) {
                fscanf(fp, "%d %d", &temp_v[i].src, &temp_v[i].dst);
                temp_v[i].src--; temp_v[i].dst--;

                if(temp_v[i].src < 0 || temp_v[i].src >= (input->V) || temp_v[i].dst < 0 || temp_v[i].dst >= (input->V)) { // unsigned -> simplify condition
                        fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].src+1, temp_v[i].dst+1);
                        exit(0);
                }
                if (nflag == 1) {
                        float ftemp;
                        fscanf(fp, " %f ", &ftemp);
                } else if (nflag == 2) { // complex
                        float ftemp1, ftemp2;
                        fscanf(fp, " %f %f ", &ftemp1, &ftemp2);
                }

                if(sflag == 1) {
                        //i++;
                        temp_v[i+prev_ne].src = temp_v[i].dst;
                        temp_v[i+prev_ne].dst = temp_v[i].src;
                }
        }

	if(input->format == CSR || input->format == SAMPLE || input->format == BFS || input->format == IHOOK) {
		qsort(temp_v, input->E, sizeof(struct v_struct), compare0);
		loc = (unsigned int *)malloc(sizeof(unsigned int)*((input->E)+1));

		memset(loc, 0, sizeof(unsigned int)*((input->E)+1));
		loc[0]=1;
		for(i=1;i<(input->E);i++) {
			if(temp_v[i].src == temp_v[i-1].src && temp_v[i].dst == temp_v[i-1].dst)
				loc[i] = 0;
			else if(temp_v[i].src == temp_v[i].dst)
				loc[i] = 0;
			else loc[i] = 1;
		}
		for(i=1;i<=(input->E);i++)
			loc[i] += loc[i-1];
		for(i=(input->E); i>=1; i--)
			loc[i] = loc[i-1];
		loc[0] = 0;

		for(i=0;i<(input->E);i++) {
			temp_v[loc[i]].src = temp_v[i].src;
			temp_v[loc[i]].dst = temp_v[i].dst;
		}

		(input->E) = loc[input->E];
		free(loc);
	}
}

void init_graph(struct graph_data *input)
{
	uintT i;
        (input->dst_idx) = (uintT *)malloc(sizeof(uintT)*(input->E));
	if(input->format == CSR || input->format == SAMPLE || input->format == BFS || input->format == IHOOK) {
 	       (input->csr_ptr) = (uintT *)malloc(sizeof(uintT)*((input->V)+1));
		memset(input->csr_ptr, 0, sizeof(uintT)*(input->V));

		for(i=0;i<(input->E);i++) {
			input->dst_idx[i] = temp_v[i].dst;
			input->csr_ptr[1+temp_v[i].src] = i+1;
		}

		for(i=1;i<(input->V);i++) {
			if(input->csr_ptr[i] == 0) input->csr_ptr[i] = input->csr_ptr[i-1];
		}

		input->csr_ptr[input->V] = (input->E);

		if((input->format == SAMPLE || input->format == BFS || input->format == IHOOK) && input->is_sym == 0) {
			memcpy(&temp_v[input->E], temp_v, sizeof(struct v_struct) * (input->E));

			for(i=input->E;i<2*(input->E);i++) {
				uintT tmp = temp_v[i].src; temp_v[i].src = temp_v[i].dst; temp_v[i].dst = tmp;
			}

			qsort(&temp_v[input->E], input->E, sizeof(struct v_struct), compare0);

	 	       (input->csr_inv_ptr) = (uintT *)malloc(sizeof(uintT)*((input->V)+1));
	 	       (input->dst_inv_idx) = (uintT *)malloc(sizeof(uintT)*((input->E)+0));
			memset(input->csr_inv_ptr, 0, sizeof(uintT)*(input->V));

			for(i=0;i<(input->E);i++) {
				input->dst_inv_idx[i] = temp_v[i+input->E].dst;
				input->csr_inv_ptr[1+temp_v[i+input->E].src] = i+1;
			}

			for(i=1;i<(input->V);i++) {
				if(input->csr_inv_ptr[i] == 0) input->csr_inv_ptr[i] = input->csr_inv_ptr[i-1];
			}

			input->csr_inv_ptr[input->V] = (input->E);
		}

	}
	if(input->format == COO || input->format == COO_SAMPLE) {
		(input->src_idx) = (uintT *)malloc(sizeof(uintT)*(input->E));

		for(i=0; i<(input->E); i++) {
			input->src_idx[i] = temp_v[i].src;
			input->dst_idx[i] = temp_v[i].dst;
		}


	}

	(input->label) = (uintT *)malloc(sizeof(uintT)*(input->V));
	(input->hook) = (uintT *)malloc(sizeof(uintT)*(input->V));
	if(input->algo == RAND_NAIVE || input->algo == RAND_SPLIT_2) {
		(input->lparent) = (ulongT *)malloc(sizeof(ulongT)*(input->V)*1);
	} else {
		(input->parent) = (uintT *)malloc(sizeof(uintT)*(input->V)*1);
	}
  
	for(i=0; i<(input->V); i++) {
		input->label[i] = i;

		if(input->algo == RAND_NAIVE || input->algo == RAND_SPLIT_2) {
			input->lparent[i] = i;
               		input->lparent[i] |= ULONG_T_MAX;
		} else {
			input->parent[i] = i;
		}

		input->hook[i] = i;//UINT_T_MAX;
		#if defined (SP_TREE)
		input->hook[i] = -1;
		#endif
	}	
#if defined(SCC) || defined(STREAMING) || defined(STREAMING_SYNC) || defined(STREAMING_SIMPLE)
	free(temp_v);
#endif
}

struct v_struct *generate_graph(int argc, char **argv, struct graph_data *input)
{
	//mtx_parse_write(argc, argv, input);
	mtx_parse_fast(argc, argv, input);
	init_graph(input);	
	return (temp_v);
}

#endif
