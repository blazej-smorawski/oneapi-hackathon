FROM intel/oneapi-basekit as builder
WORKDIR /example
COPY example-mpi.cpp example-mpi.cpp
RUN icpx -fsycl -lmpi example-mpi.cpp -O3 -o example


FROM bash AS downloader
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -O key.PUB


FROM mpioperator/intel:latest
COPY --from=downloader key.PUB /tmp/key.PUB
RUN apt update \
    && apt install -y --no-install-recommends gnupg2 ca-certificates \
    && apt-key add /tmp/key.PUB \
    && rm /tmp/key.PUB \
    && echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt remove -y gnupg2 ca-certificates \
    && apt autoremove -y \
    && apt update \
    && apt install -y --no-install-recommends \
        dnsutils \
        intel-oneapi-runtime-libs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /example/example /home/mpiuser/example