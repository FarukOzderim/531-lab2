#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <assert.h>
#include <cstdint>

#define INF 200

int main(int argc, char** argv) {
    int size, myRank, rc;
    rc = MPI_Init(&argc, &argv);
    assert(!rc);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    
    int n, m;
    uint8_t *d;
    int a,b,w;
    if(0 == myRank){
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
    
    int base = (n+size-1)/size;
    int myStart = myRank*base;
    int myEnd = std::min(myStart + base, n);

    for (int k = 0; k < n; ++k){
        MPI_Bcast(d + (k)*n, n, MPI_UINT8_T, k/base, MPI_COMM_WORLD);
        for (int i = myStart; i < myEnd; ++i) 
            for (int j = 0; j < n; ++j) 
                if ((w = d[i * n + k] + d[k * n + j]) < d[i * n + j]) 
                    d[i * n + j] = w;
    }
    
    MPI_Status status;
    if(0 == myRank) 
        for(int rank = 1; rank<size; rank++){
            int targetStart = rank*base;
            int targetEnd = std::min(targetStart + base, n);
            MPI_Recv(d + (targetStart)*n, (targetEnd-targetStart)*n, MPI_UINT8_T, rank, rank, MPI_COMM_WORLD, &status);
        }
    else  
        MPI_Send(d + (myStart)*n, (myEnd-myStart)*n, MPI_UINT8_T, 0, myRank, MPI_COMM_WORLD);


    if(0 == myRank){
        FILE *outfile = fopen(argv[2], "w");
        for (int i = 0; i < n; ++i) 
            for (int j = 0; j < n; ++j) 
                fprintf(outfile, "%d%s",
                    (i == j ? 0 : d[i * n + j]),
                    (j == n - 1 ? " \n" : " ")
                );
    }
    free(d);
    MPI_Finalize();
}