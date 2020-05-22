// Replicate model.
// Transform a single-batched model into a multi-batched model. It is assumed that the first axis is the batch dimension throughout the model.

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
  tflite_Model_table_t in_model;      // input model TF Lite flatbuffers structure
  char *in_model_buf;                 // input model buffer (referenced by `in_model`)
  uint16_t batch_size;                // target batch size
  FILE *out_model_file;               // output model file
  flatcc_builder_t *tflite_builder;   // TF Lite flatbuffers serializer
  bool *are_tensors_on_datapath;      // boolean array indicating which tensors are on the datapath
  bool *are_tensors_shape_param;      // boolean array indicating which tensors store shape parameters
  bool *are_buffers_on_datapath;      // boolean array indicating which buffers are part of the datapath
  bool *are_buffers_shape_param;      // boolean array indicating which buffers store shape parameters
} app;

static void print_usage()
{
  printf("replicate IN_FILE N OUT_FILE\n");
  printf("  Replicates the model stored at IN_FILE and writes an N-batch model into OUT_FILE.\n");
}

// Release all resources held by this application. Registered on exit by `init_app()`.
static void release_app()
{
  if(app.in_model_buf != NULL)
  {
    free(app.in_model_buf);
    app.in_model_buf = NULL;
  }
  if(app.out_model_file != NULL)
  {
    fclose(app.out_model_file);
    app.out_model_file = NULL;
  }
  if(app.tflite_builder != NULL)
  {
    flatcc_builder_clear(app.tflite_builder);
    free(app.tflite_builder);
    app.tflite_builder = NULL;
  }
  if(app.are_tensors_on_datapath != NULL)
  {
    free(app.are_tensors_on_datapath);
    app.are_tensors_on_datapath = NULL;
  }
  if(app.are_tensors_shape_param != NULL)
  {
    free(app.are_tensors_shape_param);
    app.are_tensors_shape_param = NULL;
  }
  if(app.are_buffers_on_datapath != NULL)
  {
    free(app.are_buffers_on_datapath);
    app.are_buffers_on_datapath = NULL;
  }
  if(app.are_buffers_shape_param != NULL)
  {
    free(app.are_buffers_shape_param);
    app.are_buffers_shape_param = NULL;
  }
#ifdef DEBUG_REPLICATE_C
  printf("Released application's resources.\n");
#endif //ifdef DEBUG_REPLICATE_C
}

// Initialize application with argv-style arguments.
static void init_app(
    int argc,
    char *argv[]
    )
{
  FILE *in_model_file;
  struct stat in_model_stat;
  if(argc != 4)
  {
    print_usage();
    errno = EINVAL;
    ERROR("Requires exactly 3 arguments");
  }
  app.in_model_buf = NULL;
  app.out_model_file = NULL;
  app.tflite_builder = NULL;
  app.are_tensors_on_datapath = NULL;
  app.are_tensors_shape_param = NULL;
  app.are_buffers_on_datapath = NULL;
  app.are_buffers_shape_param = NULL;
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
  app.batch_size = strtol(argv[2], NULL, 10);
  if(errno != 0)
    ERROR();
  if((app.out_model_file = fopen(argv[3], "wb")) == NULL)
    ERRORF("%s", argv[3]);
  if((app.tflite_builder = malloc(sizeof(flatcc_builder_t))) == NULL)
    ERROR();
  if(flatcc_builder_init(app.tflite_builder) != 0)
  {
    if(errno == 0)
      errno = ENOSYS; // `flatcc_builder_init` not implemented
    ERROR();
  }
}

static void replicate_shape(
    flatcc_builder_t *tflite_builder,
    flatbuffers_int32_vec_t in_shape
    )
{
  uint8_t in_shape_size = flatbuffers_int32_vec_len(in_shape);
  int32_t *out_shape;
  out_shape = flatbuffers_int32_vec_extend(tflite_builder, in_shape_size);
  if(out_shape == NULL)
    ERROR();
  out_shape[0] = app.batch_size;
  for(
      uint8_t shape_idx = 1;
      shape_idx < in_shape_size;
      shape_idx++
     )
  {
    out_shape[shape_idx] = in_shape[shape_idx];
  }
}

static tflite_Operator_vec_ref_t replicate_operators_io(
    flatcc_builder_t *tflite_builder,
    tflite_Operator_vec_t in_operators,
    tflite_OperatorCode_vec_t opcodes,
    tflite_Tensor_vec_t in_tensors
    )
{
  if(tflite_SubGraph_operators_start(tflite_builder))
    ERROR();
  for(
      uint16_t operator_idx = 0;
      operator_idx < tflite_Operator_vec_len(in_operators);
      operator_idx++
     )
  {
    tflite_Operator_table_t in_operator = tflite_Operator_vec_at(in_operators, operator_idx);
    uint8_t opcode_idx = tflite_Operator_opcode_index(in_operator);
    tflite_OperatorCode_table_t opcode = tflite_OperatorCode_vec_at(opcodes, opcode_idx);
    tflite_BuiltinOperator_enum_t op = tflite_OperatorCode_builtin_code(opcode);
    tflite_Tensor_vec_t inputs = tflite_Operator_inputs(in_operator);
    tflite_Tensor_vec_t outputs = tflite_Operator_outputs(in_operator);
    tflite_SubGraph_operators_push_start(tflite_builder);
    if(
        tflite_Operator_opcode_index_pick(tflite_builder, in_operator) ||
        tflite_Operator_inputs_pick(tflite_builder, in_operator) ||
        tflite_Operator_outputs_pick(tflite_builder, in_operator) ||
        //tflite_Operator_builtin_options_pick(tflite_builder, in_operator) ||
        tflite_Operator_custom_options_pick(tflite_builder, in_operator) ||
        tflite_Operator_mutating_variable_inputs_pick(tflite_builder, in_operator) ||
        tflite_Operator_intermediates_pick(tflite_builder, in_operator) ||
        tflite_Operator_custom_options_format_pick(tflite_builder, in_operator)
      )
      ERROR();
    if(
        op == tflite_BuiltinOperator_CONV_2D ||
        op == tflite_BuiltinOperator_DEPTHWISE_CONV_2D
      )
    {
      app.are_tensors_on_datapath[inputs[0]] = true;  // data
      app.are_tensors_on_datapath[inputs[1]] = false; // filter/weight/kernel
      app.are_tensors_on_datapath[inputs[2]] = false; // bias
      app.are_tensors_on_datapath[outputs[0]] = true; // data
      if(tflite_Operator_builtin_options_pick(tflite_builder, in_operator))
        ERROR();
    }
    else if(op == tflite_BuiltinOperator_RESHAPE)
    {
      tflite_ReshapeOptions_table_t in_options = tflite_Operator_builtin_options(in_operator);
      flatbuffers_int32_vec_t in_new_shape;
      app.are_tensors_on_datapath[inputs[0]] = true;  // data
      app.are_tensors_on_datapath[inputs[1]] = false; // shape
      app.are_tensors_shape_param[inputs[1]] = true;  // shape
      app.are_tensors_on_datapath[outputs[0]] = true; // data
      if(
        tflite_ReshapeOptions_start(tflite_builder) ||
        tflite_ReshapeOptions_new_shape_start(tflite_builder)
        )
        ERROR();
      in_new_shape = tflite_ReshapeOptions_new_shape(in_options);
      replicate_shape(tflite_builder, in_new_shape);
      tflite_ReshapeOptions_new_shape_end(tflite_builder);
      tflite_Operator_builtin_options_add(
          tflite_builder,
          tflite_BuiltinOptions_as_ReshapeOptions(
            tflite_ReshapeOptions_end(tflite_builder)
            )
          );
    }
    else if(op == tflite_BuiltinOperator_CONCATENATION)
    {
      for(
        uint8_t input_idx = 0;
        input_idx < tflite_Tensor_vec_len(inputs);
        input_idx++
        )
      {
        app.are_tensors_on_datapath[inputs[input_idx]] = true;  // data
      }
      app.are_tensors_on_datapath[outputs[0]] = true; // data
      if(tflite_Operator_builtin_options_pick(tflite_builder, in_operator))
        ERROR();
    }
    else if(op == tflite_BuiltinOperator_LOGISTIC)
    {
      app.are_tensors_on_datapath[inputs[0]] = true;  // data
      app.are_tensors_on_datapath[outputs[0]] = true; // data
      if(tflite_Operator_builtin_options_pick(tflite_builder, in_operator))
        ERROR();
    }
    else if(tflite_Operator_builtin_options_pick(tflite_builder, in_operator))
      ERROR();
#ifdef DEBUG_REPLICATE_C
    if(op == tflite_BuiltinOperator_CONV_2D)
      printf("Looking at a convolutional operator.\n");
    else if(op == tflite_BuiltinOperator_DEPTHWISE_CONV_2D)
      printf("Looking at a depthwise convolutional operator.\n");
    else if(op == tflite_BuiltinOperator_RESHAPE)
      printf("Looking at a reshape operator.\n");
    else if(op == tflite_BuiltinOperator_CONCATENATION)
      printf("Looking at a concatenaton operator.\n");
    else if(op == tflite_BuiltinOperator_LOGISTIC)
      printf("Looking at a logistic operator.\n");
    else
      printf("UNSUPPORTED OP: '%s'\n", tflite_BuiltinOperator_name(op));
#endif //ifdef DEBUG_REPLICATE_C
    tflite_SubGraph_operators_push_end(tflite_builder);
  }
  tflite_SubGraph_operators_end(tflite_builder);
}

static void replicate_tensors(
    flatcc_builder_t *tflite_builder,
    tflite_Tensor_vec_t in_tensors
    )
{
  if(tflite_SubGraph_tensors_start(tflite_builder))
    ERROR();
  for(
      uint16_t tensor_idx = 0;
      tensor_idx < tflite_Tensor_vec_len(in_tensors);
      tensor_idx++
     )
  {
    tflite_Tensor_table_t in_tensor = tflite_Tensor_vec_at(in_tensors, tensor_idx);
    uint32_t buffer_idx = tflite_Tensor_buffer(in_tensor);
    flatbuffers_int32_vec_t in_shape = tflite_Tensor_shape(in_tensor);
    tflite_Tensor_vec_push_start(tflite_builder);
    if(
        //tflite_Tensor_shape_pick(tflite_builder, in_tensor) ||
        tflite_Tensor_buffer_pick(tflite_builder, in_tensor) ||
        tflite_Tensor_name_pick(tflite_builder, in_tensor) ||
        tflite_Tensor_quantization_pick(tflite_builder, in_tensor) ||
        tflite_Tensor_type_pick(tflite_builder, in_tensor) ||
        tflite_Tensor_is_variable_pick(tflite_builder, in_tensor)
      )
      ERROR();
    app.are_buffers_on_datapath[buffer_idx] = app.are_tensors_on_datapath[tensor_idx];
    app.are_buffers_shape_param[buffer_idx] = app.are_tensors_shape_param[tensor_idx];
    if(app.are_tensors_on_datapath[tensor_idx])
    {
      tflite_Tensor_shape_start(tflite_builder);
      replicate_shape(tflite_builder, in_shape);
      tflite_Tensor_shape_end(tflite_builder);
    }
    else if(tflite_Tensor_shape_pick(tflite_builder, in_tensor))
      ERROR();
    tflite_Tensor_vec_push_end(tflite_builder);
  }
  tflite_SubGraph_tensors_end(tflite_builder);
}

static void replicate_subgraphs(
    flatcc_builder_t *tflite_builder,
    tflite_SubGraph_vec_t in_subgraphs
    )
{
  if(tflite_Model_subgraphs_start(tflite_builder))
    ERROR();
  for(
      uint8_t subgraph_idx = 0;
      subgraph_idx < tflite_SubGraph_vec_len(in_subgraphs);
      subgraph_idx++
     )
  {
    tflite_SubGraph_table_t in_subgraph = tflite_SubGraph_vec_at(in_subgraphs, subgraph_idx);
    tflite_Tensor_vec_t in_tensors;
    tflite_Operator_vec_t in_operators;
    tflite_OperatorCode_vec_t opcodes;
    tflite_Model_subgraphs_push_start(tflite_builder);
    if(
        //tflite_SubGraph_tensors_pick(tflite_builder, in_subgraph) ||
        tflite_SubGraph_inputs_pick(tflite_builder, in_subgraph) ||
        tflite_SubGraph_outputs_pick(tflite_builder, in_subgraph) ||
        //tflite_SubGraph_operators_pick(tflite_builder, in_subgraph) ||
        tflite_SubGraph_name_pick(tflite_builder, in_subgraph)
      )
      ERROR();
    in_tensors = tflite_SubGraph_tensors(in_subgraph);
    app.are_tensors_on_datapath = calloc(tflite_Tensor_vec_len(in_tensors), sizeof(bool));
    app.are_tensors_shape_param = calloc(tflite_Tensor_vec_len(in_tensors), sizeof(bool));
    in_operators = tflite_SubGraph_operators(in_subgraph);
    opcodes = tflite_Model_operator_codes(app.in_model);
    replicate_operators_io(tflite_builder, in_operators, opcodes, in_tensors);
    replicate_tensors(tflite_builder, in_tensors);
    tflite_Model_subgraphs_push_end(tflite_builder);
    free(app.are_tensors_on_datapath); app.are_tensors_on_datapath = NULL;
    free(app.are_tensors_shape_param); app.are_tensors_shape_param = NULL;
  }
  tflite_Model_subgraphs_end(tflite_builder);
}

static void replicate_buffers(
    flatcc_builder_t *tflite_builder,
    tflite_Buffer_vec_t in_buffers
    )
{
  if(tflite_Model_buffers_start(tflite_builder))
    ERROR();
  for(
      uint32_t buffer_idx = 0;
      buffer_idx < tflite_Buffer_vec_len(in_buffers);
      buffer_idx++
     )
  {
    tflite_Buffer_table_t in_buffer = tflite_Buffer_vec_at(in_buffers, buffer_idx);
    tflite_Buffer_vec_push_start(tflite_builder);
    if(app.are_buffers_on_datapath[buffer_idx])
    {
      flatbuffers_uint8_vec_t data = tflite_Buffer_data(in_buffer);
      uint32_t data_size = flatbuffers_uint8_vec_len(data);
      uint8_t *out_data;
      if(tflite_Buffer_data_start(tflite_builder))
        ERROR();
      out_data = tflite_Buffer_data_extend(tflite_builder, app.batch_size * data_size);
      if(out_data == NULL)
        ERROR();
      for(
        uint8_t batch_idx = 0;
        batch_idx < app.batch_size;
        batch_idx++
        )
      {
        uint32_t batch_offset = batch_idx * data_size;
        for(
          uint32_t data_idx = 0;
          data_idx < data_size;
          data_idx++
          )
        {
          out_data[batch_offset + data_idx] = data[data_idx];
        }
      }
      tflite_Buffer_data_end(tflite_builder);
    }
    else if(app.are_buffers_shape_param[buffer_idx])
    {
      flatbuffers_uint8_vec_t data = tflite_Buffer_data(in_buffer);
      uint32_t data_size = flatbuffers_uint8_vec_len(data);
      uint8_t *out_data;
      if(tflite_Buffer_data_start(tflite_builder))
        ERROR();
      out_data = tflite_Buffer_data_extend(tflite_builder, data_size);
      if(out_data == NULL)
        ERROR();
      out_data[0] = app.batch_size;
      for(
          uint32_t data_idx = 1;
          data_idx < data_size;
          data_idx++
         )
      {
        out_data[data_idx] = data[data_idx];
      }
      tflite_Buffer_data_end(tflite_builder);
    }
    else if(tflite_Buffer_data_pick(tflite_builder, in_buffer))
      ERROR();
    tflite_Buffer_vec_push_end(tflite_builder);
  }
  tflite_Model_buffers_end(tflite_builder);
}

static tflite_Model_ref_t replicate_model(
    flatcc_builder_t *tflite_builder,
    tflite_Model_table_t in_model
    )
{
  tflite_SubGraph_vec_t in_subgraphs;
  tflite_Buffer_vec_t in_buffers;
  if(
      tflite_Model_start_as_root(tflite_builder) ||
      tflite_Model_version_pick(tflite_builder, in_model) ||
      tflite_Model_operator_codes_pick(tflite_builder, in_model) ||
      //tflite_Model_subgraphs_pick(tflite_builder, in_model) ||
      tflite_Model_description_pick(tflite_builder, in_model) ||
      //tflite_Model_buffers_pick(tflite_builder, in_model) ||
      tflite_Model_metadata_buffer_pick(tflite_builder, in_model) ||
      tflite_Model_metadata_pick(tflite_builder, in_model)
    )
    ERROR();
  in_subgraphs = tflite_Model_subgraphs(in_model);
  in_buffers = tflite_Model_buffers(in_model);
  app.are_buffers_on_datapath = calloc(tflite_Buffer_vec_len(in_buffers), sizeof(bool));
  app.are_buffers_shape_param = calloc(tflite_Buffer_vec_len(in_buffers), sizeof(bool));
  replicate_subgraphs(tflite_builder, in_subgraphs);
  replicate_buffers(tflite_builder, in_buffers);
  tflite_Model_end_as_root(tflite_builder);
  free(app.are_buffers_on_datapath); app.are_buffers_on_datapath = NULL;
  free(app.are_buffers_shape_param); app.are_buffers_shape_param = NULL;
}

int main(
    int argc,
    char *argv[]
    )
{
  tflite_Model_ref_t out_model;
  uint8_t *out_buf;
  size_t out_size;
  init_app(argc, argv);
  out_model = replicate_model(app.tflite_builder, app.in_model);
  if((out_buf = flatcc_builder_finalize_aligned_buffer(app.tflite_builder, &out_size)) == NULL)
    ERROR();
  fwrite(out_buf, sizeof(uint8_t), out_size, app.out_model_file);
  flatcc_builder_aligned_free(out_buf);
  return EXIT_SUCCESS;
}
