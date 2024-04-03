add_mlir_public_c_api_library(RVTensorCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRRVTensor
)
