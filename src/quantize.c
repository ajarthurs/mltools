// Quantize
// Populates an input model with quantization parameters.

//#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <flatcc/flatcc.h>

#include "exceptions.h"
#include "schemas/tflite/tflite_v3_builder.h"
#include "schemas/tflite/tflite_v3_reader.h"

static struct
{
  tflite_Model_table_t in_model;
  char *in_model_buf;
  FILE *out_model_file;
  flatcc_builder_t tflite_model_builder;
  bool tflite_model_builder_initialized;
} app;

static void apply_constraints()
{
}

static void print_usage()
{
  printf("quantize IN_FILE OUT_FILE\n");
  printf("  Performs post-training quantization on the model stored at IN_FILE, writes\n  resulting model into OUT_FILE.\n");
}

// Release all resources held by this application.
static void release_app()
{
  if(app.in_model_buf)
  {
    free(app.in_model_buf);
    app.in_model_buf = NULL;
  }
  if(app.out_model_file)
  {
    fclose(app.out_model_file);
    app.out_model_file = NULL;
  }
  if(app.tflite_model_builder_initialized)
  {
    flatcc_builder_clear(&(app.tflite_model_builder));
    app.tflite_model_builder_initialized = false;
  }
  printf("Released application's resources.\n");
}

// Initialize application with argv-style arguments.
static void init_app(
    int argc,
    char *argv[]
    )
{
  FILE *in_model_file;
  struct stat in_model_stat;
  if(argc != 3)
  {
    print_usage();
    errno = EINVAL;
    ERROR("Requires exactly 2 arguments");
  }
  app.in_model_buf = NULL;
  app.out_model_file = NULL;
  app.tflite_model_builder_initialized = false;
  if(stat(argv[1], &in_model_stat) != 0)
    ERRORF("%s", argv[1]);
  if((in_model_file = fopen(argv[1], "rb")) == NULL)
    ERRORF("%s", argv[1]);
  atexit(release_app);
  if((app.in_model_buf = malloc(in_model_stat.st_size)) == NULL)
    ERROR();
  fread(app.in_model_buf, sizeof(char), in_model_stat.st_size, in_model_file);
  if(fclose(in_model_file) != 0)
    ERROR();
  app.in_model = tflite_Model_as_root(app.in_model_buf);
  if((app.out_model_file = fopen(argv[2], "wb")) == NULL)
    ERRORF("%s", argv[2]);
  if(flatcc_builder_init(&(app.tflite_model_builder)) != 0)
  {
    if(errno == 0)
      errno = ENOSYS; // `flatcc_builder_init` not implemented
    ERROR();
  }
  app.tflite_model_builder_initialized = true;
}

static void quantize_biases()
{
}

static bool has_quant_params(
    tflite_Tensor_table_t tensor
    )
{
}

static void quantize_op_input(
    tflite_Tensor_table_t op_input
    )
{
  flatbuffers_int32_vec_t shape = tflite_Tensor_shape(op_input);
  tflite_QuantizationParameters_table_t qparams = tflite_Tensor_quantization(op_input);
  flatbuffers_float_vec_t qmins = tflite_QuantizationParameters_min(qparams),
                          qmaxes = tflite_QuantizationParameters_max(qparams),
                          qscales = tflite_QuantizationParameters_scale(qparams);
  flatbuffers_int64_vec_t qzps = tflite_QuantizationParameters_zero_point(qparams);
  //printf("\n");
}

// Quantize weights and each operator's I/O.
static void quantize_weights_io(
    tflite_Model_table_t in_model
    )
{
  tflite_SubGraph_vec_t subgraphs = tflite_Model_subgraphs(in_model);
  printf("model_version=%d\n",
      tflite_Model_version(in_model)
      );
  for(
      size_t subgraph_idx = 0;
      subgraph_idx < tflite_SubGraph_vec_len(subgraphs);
      subgraph_idx++
     )
  {
    tflite_SubGraph_table_t subgraph = tflite_SubGraph_vec_at(subgraphs, subgraph_idx);
    tflite_Operator_vec_t operators = tflite_SubGraph_operators(subgraph);
    tflite_Tensor_vec_t tensors = tflite_SubGraph_tensors(subgraph);
    printf(
        " num_operators=%d\n"
        " num_tensors=%d\n",
        tflite_Operator_vec_len(operators),
        tflite_Tensor_vec_len(tensors)
        );
    for(
        size_t operator_idx = 0;
        operator_idx < tflite_Operator_vec_len(operators);
        operator_idx++
        )
    {
      tflite_Operator_table_t operator = tflite_Operator_vec_at(operators, operator_idx);
      flatbuffers_int32_vec_t op_inputs = tflite_Operator_inputs(operator),
                              op_outputs = tflite_Operator_outputs(operator);
      for(
        size_t vec_idx = 0;
        vec_idx < flatbuffers_int32_vec_len(op_inputs);
        vec_idx++
        )
      {
        int32_t tensor_idx = flatbuffers_int32_vec_at(op_inputs, vec_idx);
        tflite_Tensor_table_t op_input = tflite_Tensor_vec_at(tensors, tensor_idx);
        quantize_op_input(op_input);
      }
    }
  }
}

static void set_io_types()
{
}

static void set_opcode_version()
{
}

static void finish_model_buffer()
{
}

int main(
    int argc,
    char *argv[]
    )
{
  init_app(argc, argv);
  quantize_weights_io(app.in_model);
  apply_constraints();
  quantize_biases();
  set_opcode_version();
  set_io_types();
  finish_model_buffer();
  return EXIT_SUCCESS;
}
