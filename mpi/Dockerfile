FROM intel/oneapi-basekit

WORKDIR /example

COPY example-mpi.cpp example-mpi.cpp

RUN icpx -fsycl -lmpi example-mpi.cpp -O3 -o example

ENTRYPOINT ["mpiexec", "./example"]