FROM intel/oneapi-basekit

WORKDIR /example

COPY example.cpp example.cpp

RUN icpx -fsycl example.cpp -O3 -o example

ENTRYPOINT ["./example"]