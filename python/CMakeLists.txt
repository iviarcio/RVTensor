include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_rvmath`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_rvmath.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(RVMathPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT RVMathPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_rvmath"
  TD_FILE dialects/RVMathOps.td
  SOURCES
    dialects/rvmath.py
  DIALECT_NAME rvmath)

declare_mlir_python_extension(RVMathPythonSources.Extension
  MODULE_NAME _rvmathDialects
  ADD_TO_PARENT RVMathPythonSources
  SOURCES
    RVMathExtension.cpp
  EMBED_CAPI_LINK_LIBS
    RVMathCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(RVMathPythonCAPI
  INSTALL_COMPONENT RVMathPythonModules
  INSTALL_DESTINATION python_packages/rvmath/mlir_rvmath/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/rvmath/mlir_rvmath/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    RVMathPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(RVMathPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/rvmath/mlir_rvmath"
  INSTALL_PREFIX "python_packages/rvmath/mlir_rvmath"
  DECLARED_SOURCES
    RVMathPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    RVMathPythonCAPI
  )
