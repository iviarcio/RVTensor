add_mlir_dialect_library(MLIRRVMath
        RVMathTypes.cpp
        RVMathDialect.cpp
        RVMathOps.cpp
        RVMathPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/RVMath

        DEPENDS
        MLIRRVMathOpsIncGen
        MLIRRVMathPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
