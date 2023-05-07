FROM intel/oneapi-basekit

WORKDIR /example

COPY example.cpp example.cpp

RUN icpx -fsycl example.cpp -o example

CMD sycl-ls; ./example