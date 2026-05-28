#!/usr/bin/env bash
# Build the C++ gRPC tokenizer client.
#
# Prefers system gRPC (`libgrpc++-dev` + `protobuf-compiler-grpc`) so the build
# finishes in seconds. Falls back gracefully if `gRPC` ships a CMake CONFIG.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

need_apt=()
if ! command -v cmake >/dev/null 2>&1; then need_apt+=(cmake); fi
if ! pkg-config --exists grpc++ 2>/dev/null \
     && [ ! -d /usr/lib/x86_64-linux-gnu/cmake/grpc ]; then
  need_apt+=(libgrpc++-dev protobuf-compiler-grpc)
fi

if [ "${#need_apt[@]}" -gt 0 ]; then
  echo "Installing: ${need_apt[*]}"
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y "${need_apt[@]}"
  else
    apt-get update -qq
    apt-get install -y "${need_apt[@]}"
  fi
fi

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j"$(nproc)"

echo "Built: $HERE/build/cpp_client"
