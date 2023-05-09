FROM intel/oneapi-basekit

WORKDIR /example

COPY example.cpp example.cpp

RUN icpx example.cpp -O0 -o example

ENTRYPOINT ["./example"]