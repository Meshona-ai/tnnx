# Supported Operators

Generated from `src/tnnx/ir/schema.py`, `src/tnnx/ingest/op_map.py`, and backend dispatch.

- Semantic schema ops: 69
- ONNX spellings mapped: 70
- JAX dispatch coverage: 69/69
- MLX dispatch coverage: 69/69

| Semantic op | ONNX source | JAX | MLX | Disposition |
| --- | --- | --- | --- | --- |
| ADD | Add | yes | yes | ready |
| AND | And | yes | yes | ready |
| ARANGE | Range | yes | yes | ready |
| AVGPOOL | AveragePool, GlobalAveragePool | yes | yes | ready |
| BATCHNORM | BatchNormalization | yes | yes | ready |
| CAST | Cast | yes | yes | ready |
| CLIP | Clip | yes | yes | ready |
| CONCAT | Concat | yes | yes | ready |
| CONSTANT_OF_SHAPE | ConstantOfShape | yes | yes | ready |
| CONV2D | Conv | yes | yes | ready |
| COS | Cos | yes | yes | ready |
| CUMSUM | CumSum | yes | yes | ready |
| DEQUANTIZE | DequantizeLinear | yes | yes | ready |
| DIV | Div | yes | yes | ready |
| EQUAL | Equal | yes | yes | ready |
| ERF | Erf | yes | yes | ready |
| EXP | Exp | yes | yes | ready |
| EXPAND | Expand | yes | yes | ready |
| FLATTEN | Flatten | yes | yes | ready |
| GATHER | Gather | yes | yes | ready |
| GATHERELEMENTS | GatherElements | yes | yes | ready |
| GELU | Gelu | yes | yes | ready |
| GEMM | Gemm | yes | yes | ready |
| GREATER | Greater | yes | yes | ready |
| GREATEROREQUAL | GreaterOrEqual | yes | yes | ready |
| GROUPQUERYATTENTION | GroupQueryAttention | yes | yes | ready |
| IDENTITY | Identity | yes | yes | ready |
| INSTANCENORM | InstanceNormalization | yes | yes | ready |
| ISNAN | IsNaN | yes | yes | ready |
| LAYERNORM | LayerNormalization | yes | yes | ready |
| LESS | Less | yes | yes | ready |
| LESSOREQUAL | LessOrEqual | yes | yes | ready |
| LOG | Log | yes | yes | ready |
| MATMUL | MatMul | yes | yes | ready |
| MAXPOOL | GlobalMaxPool, MaxPool | yes | yes | ready |
| MISH | Mish | yes | yes | ready |
| MOD | Mod | yes | yes | ready |
| MUL | Mul | yes | yes | ready |
| NEG | Neg | yes | yes | ready |
| PAD | Pad | yes | yes | ready |
| POW | Pow | yes | yes | ready |
| QUANTIZE | QuantizeLinear | yes | yes | ready |
| RECIPROCAL | Reciprocal | yes | yes | ready |
| REDUCEMEAN | ReduceMean | yes | yes | ready |
| REDUCESUM | ReduceSum | yes | yes | ready |
| RELU | Relu | yes | yes | ready |
| RELU6 | internal/generated IR only | yes | yes | internal-only |
| RESHAPE | Reshape | yes | yes | ready |
| RMSNORM | RMSNormalization, SimplifiedLayerNormalization | yes | yes | ready |
| ROTARYEMBEDDING | RotaryEmbedding | yes | yes | ready |
| SCATTERND | ScatterND | yes | yes | ready |
| SHAPE | Shape | yes | yes | ready |
| SIGMOID | Sigmoid | yes | yes | ready |
| SILU | internal/generated IR only | yes | yes | internal-only |
| SIN | Sin | yes | yes | ready |
| SKIPRMSNORM | SkipSimplifiedLayerNormalization | yes | yes | ready |
| SLICE | Slice | yes | yes | ready |
| SOFTMAX | Softmax | yes | yes | ready |
| SOFTPLUS | Softplus | yes | yes | ready |
| SPLIT | Split | yes | yes | ready |
| SQRT | Sqrt | yes | yes | ready |
| SQUEEZE | Squeeze | yes | yes | ready |
| SUB | Sub | yes | yes | ready |
| TANH | Tanh | yes | yes | ready |
| TRANSPOSE | Transpose | yes | yes | ready |
| TRILU | Trilu | yes | yes | ready |
| UNSQUEEZE | Unsqueeze | yes | yes | ready |
| UPSAMPLE | Resize | yes | yes | ready |
| WHERE | Where | yes | yes | ready |
