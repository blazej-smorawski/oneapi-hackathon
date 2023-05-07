FROM intel/oneapi-basekit

WORKDIR /example

COPY example.cpp example.cpp

RUN icpx -fsycl -lmpi example-mpi.cpp -O3 -o example

ENTRYPOINT ./example