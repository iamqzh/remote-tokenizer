cmake_minimum_required(VERSION 3.16)
project(tokenizer_cpp_client CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Resolve protobuf + gRPC. Prefer find_package(gRPC CONFIG) (set when a CONFIG
# package is installed), otherwise fall back to pkg-config + Protobuf module —
# which is what the Ubuntu `libgrpc++-dev` / `protobuf-compiler-grpc` packages
# ship.
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG QUIET)

if(NOT gRPC_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GRPCPP REQUIRED grpc++)
endif()

find_program(PROTOC_EXECUTABLE protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../proto/tokenizer.proto")
set(PROTO_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../../proto")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${GENERATED_DIR})

set(GEN_PB_CC   "${GENERATED_DIR}/tokenizer.pb.cc")
set(GEN_PB_H    "${GENERATED_DIR}/tokenizer.pb.h")
set(GEN_GRPC_CC "${GENERATED_DIR}/tokenizer.grpc.pb.cc")
set(GEN_GRPC_H  "${GENERATED_DIR}/tokenizer.grpc.pb.h")

add_custom_command(
  OUTPUT  ${GEN_PB_CC} ${GEN_PB_H} ${GEN_GRPC_CC} ${GEN_GRPC_H}
  COMMAND ${PROTOC_EXECUTABLE}
    --cpp_out=${GENERATED_DIR}
    --grpc_out=${GENERATED_DIR}
    --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
    -I${PROTO_DIR}
    ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
  COMMENT "Generating C++ protobuf + gRPC stubs"
  VERBATIM
)

add_executable(cpp_client
  main.cc
  ${GEN_PB_CC}
  ${GEN_GRPC_CC}
)

target_include_directories(cpp_client PRIVATE ${GENERATED_DIR})

if(gRPC_FOUND)
  target_link_libraries(cpp_client PRIVATE gRPC::grpc++ protobuf::libprotobuf)
else()
  target_include_directories(cpp_client PRIVATE
    ${GRPCPP_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
  )
  target_link_directories(cpp_client PRIVATE ${GRPCPP_LIBRARY_DIRS})
  target_link_libraries(cpp_client PRIVATE
    ${GRPCPP_LIBRARIES}
    ${Protobuf_LIBRARIES}
  )
endif()
