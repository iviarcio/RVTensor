add_mlir_dialect_library(MLIRRVTensor
        RVTensorTypes.cpp
        RVTensorDialect.cpp
        RVTensorOps.cpp
        RVTensorPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/RVTensor

        DEPENDS
        MLIRRVTensorOpsIncGen
        MLIRRVTensorPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
