ARG BASE_LABEL

FROM mpioperator/intel-builder:${BASE_LABEL} as builder

COPY example-mpi.cpp /src/example-mpi.cpp
RUN bash -c "source /opt/intel/oneapi/setvars.sh && icpx -fsycl -lmpi /src/example-mpi.cpp -O3 -o /example"

FROM mpioperator/intel:${BASE_LABEL}

RUN apt-get update && apt-get upgrade -y && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg gpg-agent software-properties-common && \
  rm -rf /var/lib/apt/lists/*
# repository to install Intel(R) oneAPI Libraries
RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | apt-key add -
RUN echo "deb [trusted=yes] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update && apt-get upgrade -y && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg gpg-agent software-properties-common && \
  rm -rf /var/lib/apt/lists/*
# repository to install Intel(R) GPU drivers
RUN curl -fsSL https://repositories.intel.com/graphics/intel-graphics.key | apt-key add -
RUN echo "deb [trusted=yes arch=amd64] https://repositories.intel.com/graphics/ubuntu focal-devel main" > /etc/apt/sources.list.d/intel-graphics.list

RUN apt-get update && apt-get upgrade -y && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    sudo intel-oneapi-runtime-dpcpp-cpp intel-level-zero-gpu level-zero && \
  rm -rf /var/lib/apt/lists/*

COPY --from=builder /example /home/mpiuser/example
