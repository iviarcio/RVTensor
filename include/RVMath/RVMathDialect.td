//===- RVMathDialect.td - RVMath dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RVMATH_DIALECT
#define RVMATH_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// RVMath dialect definition.
//===----------------------------------------------------------------------===//

def RVMath_Dialect : Dialect {
    let name = "rvmath";
    let summary = "A rvmath out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an out-of-tree MLIR RVMath dialect designed to operate
        the Convolutions & Matrix Multiplications on The Risc-V architecture.
    }];
    let cppNamespace = "::mlir::rvmath";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base rvmath operation definition.
//===----------------------------------------------------------------------===//

class RVMath_Op<string mnemonic, list<Trait> traits = []> :
        Op<RVMath_Dialect, mnemonic, traits>;

#endif // RVMATH_DIALECT
