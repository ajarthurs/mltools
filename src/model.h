// Refer to `schemas/tflite/tflite_v3.fbs` for more details.
#ifndef MLTOOLS_MODEL_H
#define MLTOOLS_MODEL_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum builtin_operator
{
  BO_ADD = 0,
  BO_AVERAGE_POOL_2D = 1,
  BO_CONCATENATION = 2,
  BO_CONV_2D = 3,
  BO_DEPTHWISE_CONV_2D = 4,
  BO_DEPTH_TO_SPACE = 5,
  BO_DEQUANTIZE = 6,
  BO_EMBEDDING_LOOKUP = 7,
  BO_FLOOR = 8,
  BO_FULLY_CONNECTED = 9,
  BO_HASHTABLE_LOOKUP = 10,
  BO_L2_NORMALIZATION = 11,
  BO_L2_POOL_2D = 12,
  BO_LOCAL_RESPONSE_NORMALIZATION = 13,
  BO_LOGISTIC = 14,
  BO_LSH_PROJECTION = 15,
  BO_LSTM = 16,
  BO_MAX_POOL_2D = 17,
  BO_MUL = 18,
  BO_RELU = 19,
  BO_RELU_N1_TO_1 = 20,
  BO_RELU6 = 21,
  BO_RESHAPE = 22,
  BO_RESIZE_BILINEAR = 23,
  BO_RNN = 24,
  BO_SOFTMAX = 25,
  BO_SPACE_TO_DEPTH = 26,
  BO_SVDF = 27,
  BO_TANH = 28,
  BO_CONCAT_EMBEDDINGS = 29,
  BO_SKIP_GRAM = 30,
  BO_CALL = 31,
  BO_CUSTOM = 32,
  BO_EMBEDDING_LOOKUP_SPARSE = 33,
  BO_PAD = 34,
  BO_UNIDIRECTIONAL_SEQUENCE_RNN = 35,
  BO_GATHER = 36,
  BO_BATCH_TO_SPACE_ND = 37,
  BO_SPACE_TO_BATCH_ND = 38,
  BO_TRANSPOSE = 39,
  BO_MEAN = 40,
  BO_SUB = 41,
  BO_DIV = 42,
  BO_SQUEEZE = 43,
  BO_UNIDIRECTIONAL_SEQUENCE_LSTM = 44,
  BO_STRIDED_SLICE = 45,
  BO_BIDIRECTIONAL_SEQUENCE_RNN = 46,
  BO_EXP = 47,
  BO_TOPK_V2 = 48,
  BO_SPLIT = 49,
  BO_LOG_SOFTMAX = 50,
  BO_DELEGATE = 51,
  BO_BIDIRECTIONAL_SEQUENCE_LSTM = 52,
  BO_CAST = 53,
  BO_PRELU = 54,
  BO_MAXIMUM = 55,
  BO_ARG_MAX = 56,
  BO_MINIMUM = 57,
  BO_LESS = 58,
  BO_NEG = 59,
  BO_PADV2 = 60,
  BO_GREATER = 61,
  BO_GREATER_EQUAL = 62,
  BO_LESS_EQUAL = 63,
  BO_SELECT = 64,
  BO_SLICE = 65,
  BO_SIN = 66,
  BO_TRANSPOSE_CONV = 67,
  BO_SPARSE_TO_DENSE = 68,
  BO_TILE = 69,
  BO_EXPAND_DIMS = 70,
  BO_EQUAL = 71,
  BO_NOT_EQUAL = 72,
  BO_LOG = 73,
  BO_SUM = 74,
  BO_SQRT = 75,
  BO_RSQRT = 76,
  BO_SHAPE = 77,
  BO_POW = 78,
  BO_ARG_MIN = 79,
  BO_FAKE_QUANT = 80,
  BO_REDUCE_PROD = 81,
  BO_REDUCE_MAX = 82,
  BO_PACK = 83,
  BO_LOGICAL_OR = 84,
  BO_ONE_HOT = 85,
  BO_LOGICAL_AND = 86,
  BO_LOGICAL_NOT = 87,
  BO_UNPACK = 88,
  BO_REDUCE_MIN = 89,
  BO_FLOOR_DIV = 90,
  BO_REDUCE_ANY = 91,
  BO_SQUARE = 92,
  BO_ZEROS_LIKE = 93,
  BO_FILL = 94,
  BO_FLOOR_MOD = 95,
  BO_RANGE = 96,
  BO_RESIZE_NEAREST_NEIGHBOR = 97,
  BO_LEAKY_RELU = 98,
  BO_SQUARED_DIFFERENCE = 99,
  BO_MIRROR_PAD = 100,
  BO_ABS = 101,
  BO_SPLIT_V = 102,
  BO_UNIQUE = 103,
  BO_CEIL = 104,
  BO_REVERSE_V2 = 105,
  BO_ADD_N = 106,
  BO_GATHER_ND = 107,
  BO_COS = 108,
  BO_WHERE = 109,
  BO_RANK = 110,
  BO_ELU = 111,
  BO_REVERSE_SEQUENCE = 112,
  BO_MATRIX_DIAG = 113,
  BO_QUANTIZE = 114,
  BO_MATRIX_SET_DIAG = 115,
  BO_ROUND = 116,
  BO_HARD_SWISH = 117,
  BO_IF = 118,
  BO_WHILE = 119
};

enum padding
{
  P_SAME,
  P_VALID
};

enum activation_function_type
{
  AFT_NONE = 0,
  AFT_RELU = 1,
  AFT_RELU_N1_TO_1 = 2,
  AFT_RELU6 = 3,
  AFT_TANH = 4,
  AFT_SIGN_BIT = 5
};

struct conv2d_options
{
  enum padding padding;
  int8_t stride_w;
  int8_t stride_h;
  enum activation_function_type fused_activation_function;
  int8_t dilation_w_factor; // default: 1
  int8_t dilation_h_factor; // default: 1
};

struct depthwise_conv2d_options
{
  enum padding padding;
  int8_t stride_w;
  int8_t stride_h;
  int8_t depth_multiplier;
  enum activation_function_type fused_activation_function;
  int8_t dilation_w_factor; // default: 1
  int8_t dilation_h_factor; // default: 1
};

union builtin_options
{
  struct conv2d_options conv2d_options;
  struct depthwise_conv2d_options depthwise_conv2d_options;
};

struct metadata
{
  const char *name;
  uint16_t buffer;
};

struct operator
{
  uint8_t opcode_index;
  int8_t *inputs;
  int8_t *outputs;
  union builtin_options builtin_options;
};

struct operator_code
{
  enum builtin_operator builtin_code;
  //const char *custom_code;
};

struct tensor
{
};

struct subgraph
{
  struct tensor *tensors;
  uint8_t *inputs;
  uint8_t *outputs;
  struct operator *operators;
  const char *name;
};

struct model
{
  uint8_t version;
  const char *description;
  struct metadata *metadata;
  struct operator_code *operator_codes;
  struct subgraph *subgraphs;
  uint8_t *buffers[];
};

uint8_t *serialize_to_flatbuffer(const struct model *m);
struct model *deserialize_from_flatbuffer(const uint8_t *buf);

#endif //ifndef MLTOOLS_MODEL_H
