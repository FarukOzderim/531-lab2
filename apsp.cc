#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <assert.h>
#include <cstdint>
#include <algorithm>

#define INF 200

int main(int argc, char** argv) {
    int size, myRank, rc;
    rc = MPI_Init(&argc, &argv);
    assert(!rc);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    
    int n, m, w, k, base, myStart, myEnd;
    uint8_t *d;
    MPI_Status statuses[n];
    MPI_Request requests[n];
    int starts[n];
    int ends[n];

    if(0 == myRank){
        int a,b;
        FILE *infile = fopen(argv[1], "r");
        fscanf(infile, "%d %d", &n, &m);
        d = (uint8_t *) malloc(sizeof(uint8_t *) * n * n);
        for (int i = 0; i < n * n; ++i) d[i] = INF;
        for (int i = 0; i < m; ++i) {
            fscanf(infile, "%d %d %d", &a, &b, &w);
            d[a * n + b] = d[b * n + a] = w;
        }
        fclose(infile);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    
    if(0 != myRank){
        d = (uint8_t *) malloc(sizeof(uint8_t *) * n * n);
    }
    MPI_Bcast(d, n*n, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    
    base = (n+size-1)/size;
    myStart = myRank*base;
    myEnd = std::min(myStart + base, n);

    for (int k = 0; k < n; ++k){
        MPI_Bcast(d + (k)*n, n/2, MPI_UINT8_T, k/base, MPI_COMM_WORLD);
        for (int i = myStart; i < myEnd; ++i) 
            for (int j = 0; j < n/2; ++j) 
                if ((w = d[i * n + k] + d[k * n + j]) < d[i * n + j]) 
                    d[i * n + j] = w;
        MPI_Bcast(d + (k)*n + n/2, n-n/2, MPI_UINT8_T, k/base, MPI_COMM_WORLD);
        for (int i = myStart; i < myEnd; ++i) 
            for (int j = n/2; j < n; ++j) 
                if ((w = d[i * n + k] + d[k * n + j]) < d[i * n + j]) 
                    d[i * n + j] = w;
    }
    
    if(0 == myRank){
        for(int rank = 1; rank<size; rank++){
            starts[rank] = rank*base;
            ends[rank] = std::min(starts[rank] + base, n);
            MPI_Irecv(d + (starts[rank])*n, (ends[rank]-starts[rank])*n, MPI_UINT8_T, rank, 2*n, MPI_COMM_WORLD, &requests[rank]);
        }   
    }
    //This way we won't do copy after Irecv as it will directly write it to d
    MPI_Barrier(MPI_COMM_WORLD);

    if(0 != myRank){
        MPI_Isend(d + (myStart)*n, (myEnd-myStart)*n, MPI_UINT8_T, 0, 2*n, MPI_COMM_WORLD, &requests[myRank]);
        MPI_Wait(&requests[myRank], &statuses[myRank]);
    }  
    
    if(0 == myRank){
        FILE *outfile = fopen(argv[2], "w");
        for (int i = myStart; i < myEnd; ++i) 
                for (int j = 0; j < n; ++j) 
                    fprintf(outfile, "%d%s",
                        (i == j ? 0 : d[i * n + j]),
                        (j == n - 1 ? " \n" : " ")
                    );

        for (int rank = 1; rank<size; rank++){
            MPI_Wait(&requests[rank], &statuses[rank]);
            for (int i = starts[rank]; i < ends[rank]; ++i) 
                for (int j = 0; j < n; ++j) 
                    fprintf(outfile, "%d%s",
                        (i == j ? 0 : d[i * n + j]),
                        (j == n - 1 ? " \n" : " ")
                    );


        }
    }

    free(d);
    MPI_Finalize();
}