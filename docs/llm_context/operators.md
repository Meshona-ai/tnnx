# Operators

## Summary

Semantic schema count: 69. ONNX ingest map count: 70 spellings into 67 semantic ops. JAX and MLX dispatch coverage match `SEMANTIC_SCHEMAS` at review base.

`RELU6` and `SILU` have schema/codegen/shape support but no direct ONNX map entry. Keep them as `VERIFY` until a lowering or source path proves how they enter IR.

## Operator Matrix

| Semantic op | ONNX source | Shape prop | JAX | MLX | Disposition |
| --- | --- | --- | --- | --- | --- |
| ADD | Add | yes | yes | yes | KEEP |
| AND | And | yes | yes | yes | KEEP |
| ARANGE | Range | yes | yes | yes | KEEP |
| AVGPOOL | AveragePool, GlobalAveragePool | yes | yes | yes | KEEP |
| BATCHNORM | BatchNormalization | yes | yes | yes | KEEP |
| CAST | Cast | yes | yes | yes | KEEP |
| CLIP | Clip | yes | yes | yes | KEEP |
| CONCAT | Concat | yes | yes | yes | KEEP |
| CONSTANT_OF_SHAPE | ConstantOfShape | yes | yes | yes | KEEP |
| CONV2D | Conv | yes | yes | yes | KEEP |
| COS | Cos | yes | yes | yes | KEEP |
| CUMSUM | CumSum | yes | yes | yes | KEEP |
| DEQUANTIZE | DequantizeLinear | yes | yes | yes | KEEP |
| DIV | Div | yes | yes | yes | KEEP |
| EQUAL | Equal | yes | yes | yes | KEEP |
| ERF | Erf | yes | yes | yes | KEEP |
| EXP | Exp | yes | yes | yes | KEEP |
| EXPAND | Expand | yes | yes | yes | KEEP |
| FLATTEN | Flatten | yes | yes | yes | KEEP |
| GATHER | Gather | yes | yes | yes | KEEP |
| GATHERELEMENTS | GatherElements | yes | yes | yes | KEEP |
| GELU | Gelu | yes | yes | yes | KEEP |
| GEMM | Gemm | yes | yes | yes | KEEP |
| GREATER | Greater | yes | yes | yes | KEEP |
| GREATEROREQUAL | GreaterOrEqual | yes | yes | yes | KEEP |
| GROUPQUERYATTENTION | GroupQueryAttention | yes | yes | yes | KEEP |
| IDENTITY | Identity | yes | yes | yes | KEEP |
| INSTANCENORM | InstanceNormalization | yes | yes | yes | KEEP |
| ISNAN | IsNaN | yes | yes | yes | KEEP |
| LAYERNORM | LayerNormalization | yes | yes | yes | KEEP |
| LESS | Less | yes | yes | yes | KEEP |
| LESSOREQUAL | LessOrEqual | yes | yes | yes | KEEP |
| LOG | Log | yes | yes | yes | KEEP |
| MATMUL | MatMul | yes | yes | yes | KEEP |
| MAXPOOL | GlobalMaxPool, MaxPool | yes | yes | yes | KEEP |
| MISH | Mish | yes | yes | yes | KEEP |
| MOD | Mod | yes | yes | yes | KEEP |
| MUL | Mul | yes | yes | yes | KEEP |
| NEG | Neg | yes | yes | yes | KEEP |
| PAD | Pad | yes | yes | yes | KEEP |
| POW | Pow | yes | yes | yes | KEEP |
| QUANTIZE | QuantizeLinear | yes | yes | yes | KEEP |
| RECIPROCAL | Reciprocal | yes | yes | yes | KEEP |
| REDUCEMEAN | ReduceMean | yes | yes | yes | KEEP |
| REDUCESUM | ReduceSum | yes | yes | yes | KEEP |
| RELU | Relu | yes | yes | yes | KEEP |
| RELU6 | none | yes | yes | yes | VERIFY |
| RESHAPE | Reshape | yes | yes | yes | KEEP |
| RMSNORM | RMSNormalization, SimplifiedLayerNormalization | yes | yes | yes | KEEP |
| ROTARYEMBEDDING | RotaryEmbedding | yes | yes | yes | KEEP |
| SCATTERND | ScatterND | yes | yes | yes | KEEP |
| SHAPE | Shape | yes | yes | yes | KEEP |
| SIGMOID | Sigmoid | yes | yes | yes | KEEP |
| SILU | none | yes | yes | yes | VERIFY |
| SIN | Sin | yes | yes | yes | KEEP |
| SKIPRMSNORM | SkipSimplifiedLayerNormalization | yes | yes | yes | KEEP |
| SLICE | Slice | yes | yes | yes | KEEP |
| SOFTMAX | Softmax | yes | yes | yes | KEEP |
| SOFTPLUS | Softplus | yes | yes | yes | KEEP |
| SPLIT | Split | yes | yes | yes | KEEP |
| SQRT | Sqrt | yes | yes | yes | KEEP |
| SQUEEZE | Squeeze | yes | yes | yes | KEEP |
| SUB | Sub | yes | yes | yes | KEEP |
| TANH | Tanh | yes | yes | yes | KEEP |
| TRANSPOSE | Transpose | yes | yes | yes | KEEP |
| TRILU | Trilu | yes | yes | yes | KEEP |
| UNSQUEEZE | Unsqueeze | yes | yes | yes | KEEP |
| UPSAMPLE | Resize | yes | yes | yes | KEEP |
| WHERE | Where | yes | yes | yes | KEEP |

## Unsupported-Op Behavior

Unknown ONNX ops raise `UnsupportedOpError` during ingest. This happens before pruning, so a dead unsupported ONNX node can still block ingest until task T07/T08 clarifies pruning and IR invariants.

## Add Or Update An Operator

1. Add or update the ONNX mapping in `src/tnnx/ingest/op_map.py`.
2. Add schema arity and attr types in `src/tnnx/ir/schema.py`.
3. Add shape propagation in `src/tnnx/passes/shape_prop.py` or explicitly document same-shape/broadcast behavior.
4. Add JAX emission and tests.
5. Add MLX emission and tests.
6. Add runtime parity and snapshots where generated source changes.
7. Update this page and machine indexes.
