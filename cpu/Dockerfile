FROM intel/oneapi-basekit

WORKDIR /example

COPY example-cpu.cpp example.cpp

RUN icpx example.cpp -O3 -o example

ENTRYPOINT ["./example"]