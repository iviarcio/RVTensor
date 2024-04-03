include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_rvtensor`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_rvtensor.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(RVTensorPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT RVTensorPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_rvtensor"
  TD_FILE dialects/RVTensorOps.td
  SOURCES
    dialects/rvtensor.py
  DIALECT_NAME rvtensor)

declare_mlir_python_extension(RVTensorPythonSources.Extension
  MODULE_NAME _rvtensorDialects
  ADD_TO_PARENT RVTensorPythonSources
  SOURCES
    RVTensorExtension.cpp
  EMBED_CAPI_LINK_LIBS
    RVTensorCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(RVTensorPythonCAPI
  INSTALL_COMPONENT RVTensorPythonModules
  INSTALL_DESTINATION python_packages/rvtensor/mlir_rvtensor/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/rvtensor/mlir_rvtensor/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    RVTensorPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(RVTensorPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/rvtensor/mlir_rvtensor"
  INSTALL_PREFIX "python_packages/rvtensor/mlir_rvtensor"
  DECLARED_SOURCES
    RVTensorPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    RVTensorPythonCAPI
  )
