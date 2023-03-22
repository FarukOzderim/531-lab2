make -j
mpirun -np 4 ./apsp ./dataset/50-1225.in cur.out
diff ./dataset/50-1225.out cur.out