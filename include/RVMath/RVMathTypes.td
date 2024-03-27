//===- RVMathTypes.td - RVMath dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RVMATH_TYPES
#define RVMATH_TYPES

include "mlir/IR/AttrTypeBase.td"
include "RVMath/RVMathDialect.td"

//===----------------------------------------------------------------------===//
// RVMath type definitions
//===----------------------------------------------------------------------===//

class RVMath_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<RVMath_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def RVMath_CustomType : RVMath_Type<"Custom", "custom"> {
    let summary = "RVMath custom type";
    let description = "Custom type in rvmath dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // RVMATH_TYPES
