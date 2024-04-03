# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own rvtensor dialect.
add_mlir_aggregate(RVTensorCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  # TODO: Remove this in favor of showing fine grained dialect registration
  # (once available).
  MLIRCAPIRegisterEverything
  RVTensorCAPI
)

add_llvm_executable(rvtensor-capi-test
  rvtensor-capi-test.c
)
llvm_update_compile_flags(rvtensor-capi-test)
target_link_libraries(rvtensor-capi-test
  PRIVATE RVTensorCAPITestLib)
