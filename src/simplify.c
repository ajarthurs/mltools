// Simplify model.
// Convert int8 tensors to uint8 and per-axis quantization to per-tensor.

//#include <getopt.h>
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
  FILE *out_model_file;               // output model file
  flatcc_builder_t *tflite_builder;   // TF Lite flatbuffers serializer
} app;

static void print_usage()
{
  printf("simplify IN_FILE OUT_FILE\n");
  printf("  Simplifies the model stored at IN_FILE and writes resulting model into OUT_FILE.\n");
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
#ifdef DEBUG_SIMPLIFY_C
  printf("Released application's resources.\n");
#endif
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
  app.tflite_builder = NULL;
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
  if((app.tflite_builder = malloc(sizeof(flatcc_builder_t))) == NULL)
    ERROR();
  if(flatcc_builder_init(app.tflite_builder) != 0)
  {
    if(errno == 0)
      errno = ENOSYS; // `flatcc_builder_init` not implemented
    ERROR();
  }
}

static tflite_QuantizationParameters_ref_t simplify_quantization(
    flatcc_builder_t *tflite_builder,
    tflite_QuantizationParameters_table_t in_quant
    )
{
}

static tflite_Tensor_vec_ref_t simplify_tensors_quantization(
    flatcc_builder_t *tflite_builder,
    tflite_Tensor_vec_t in_tensors
    )
{
  if(tflite_Tensor_vec_start(tflite_builder))
    ERROR();
  for(
      uint16_t tensor_idx = 0;
      tensor_idx < tflite_Tensor_vec_len(in_tensors);
      tensor_idx++
     )
  {
    tflite_Tensor_table_t in_tensor = tflite_Tensor_vec_at(in_tensors, tensor_idx);
    tflite_Quantization_table_t in_quant;
    tflite_Quantization_ref_t quant_ref;
    if(
        tflite_Tensor_start(tflite_builder) ||
        tflite_Tensor_shape_pick(tflite_builder, in_tensor) ||
        tflite_Tensor_buffer_pick(tflite_builder, in_tensor) ||
        tflite_Tensor_name_pick(tflite_builder, in_tensor) ||
        //tflite_Tensor_quantization_pick(tflite_builder, in_tensor) ||
        //tflite_Tensor_type_pick(tflite_builder, in_tensor) ||
        tflite_Tensor_is_variable_pick(tflite_builder, in_tensor)
      )
      ERROR();
    in_quant = tflite_Tensor_quantization(in_tensor);
    quant_ref = simplify_quantization(tflite_builder, in_quant);
    if(
        tflite_Tensor_quantization_add(tflite_builder, quant_ref) ||
        tflite_Tensor_type_add(tflite_builder, tflite_TensorType_UINT8)
      )
      ERROR();
    tflite_Tensor_vec_push(tflite_builder, tflite_Tensor_end(tflite_builder));
  }
  return tflite_Tensor_vec_end(tflite_builder);
}

static tflite_SubGraph_vec_ref_t simplify_subgraphs_quantization(
    flatcc_builder_t *tflite_builder,
    tflite_SubGraph_vec_t in_subgraphs
    )
{
  if(tflite_SubGraph_vec_start(tflite_builder))
    ERROR();
  for(
      uint8_t subgraph_idx = 0;
      subgraph_idx < tflite_SubGraph_vec_len(in_subgraphs);
      subgraph_idx++
     )
  {
    tflite_SubGraph_table_t in_subgraph = tflite_SubGraph_vec_at(in_subgraphs, subgraph_idx);
    tflite_Tensor_vec_t in_tensors;
    tflite_Tensor_vec_ref_t tensor_vec;
    if(
        tflite_SubGraph_start(tflite_builder) ||
        //tflite_SubGraph_tensors_pick(tflite_builder, in_subgraph) ||
        tflite_SubGraph_inputs_pick(tflite_builder, in_subgraph) ||
        tflite_SubGraph_outputs_pick(tflite_builder, in_subgraph) ||
        tflite_SubGraph_operators_pick(tflite_builder, in_subgraph) ||
        tflite_SubGraph_name_pick(tflite_builder, in_subgraph)
      )
      ERROR();
    in_tensors = tflite_SubGraph_tensors(in_subgraph);
    tensor_vec = simplify_tensors_quantization(tflite_builder, in_tensors);
    tflite_SubGraph_tensors_add(tflite_builder, tensor_vec);
    tflite_SubGraph_vec_push(tflite_builder, tflite_SubGraph_end(tflite_builder));
  }
  return tflite_SubGraph_vec_end(tflite_builder);
}

static tflite_Model_ref_t simplify_model_quantization(
    flatcc_builder_t *tflite_builder,
    tflite_Model_table_t in_model
    )
{
  tflite_SubGraph_vec_t subgraphs;
  tflite_SubGraph_vec_ref_t subgraph_vec;
  if(
      tflite_Model_start_as_root(tflite_builder) ||
      tflite_Model_version_pick(tflite_builder, in_model) ||
      tflite_Model_operator_codes_pick(tflite_builder, in_model) ||
      //tflite_Model_subgraphs_pick(tflite_builder, in_model) ||
      tflite_Model_description_pick(tflite_builder, in_model) ||
      tflite_Model_buffers_pick(tflite_builder, in_model) ||
      tflite_Model_metadata_buffer_pick(tflite_builder, in_model) ||
      tflite_Model_metadata_pick(tflite_builder, in_model)
    )
    ERROR();
  subgraphs = tflite_Model_subgraphs(in_model);
  subgraph_vec = simplify_subgraphs_quantization(tflite_builder, subgraphs);
  tflite_Model_subgraphs_add(tflite_builder, subgraph_vec);
  tflite_Model_end_as_root(tflite_builder);
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

// Mutate-and-serialize a TF Lite model.
static tflite_Model_ref_t mutate_tflite_model(
    flatbuffers_builder_t *tflite_builder,
    tflite_Model_table_t in_model
    )
{
  //tflite_Model_ref_t out_model;
  //tflite_SubGraph_vec_t subgraphs;
  ////if((out_model = tflite_Model_clone_as_root(tflite_builder, in_model)) == 0)
  ////  ERROR();
  ////return out_model;
  //subgraphs = tflite_Model_subgraphs(in_model);
  //  ERROR();
  //tflite_Model_version(in_model)
  //for(
  //    size_t subgraph_idx = 0;
  //    subgraph_idx < tflite_SubGraph_vec_len(subgraphs);
  //    subgraph_idx++
  //   )
  //{
  //  tflite_SubGraph_table_t subgraph = tflite_SubGraph_vec_at(subgraphs, subgraph_idx);
  //  tflite_Operator_vec_t operators = tflite_SubGraph_operators(subgraph);
  //  tflite_Tensor_vec_t tensors = tflite_SubGraph_tensors(subgraph);
  //  printf(
  //      " num_operators=%d\n"
  //      " num_tensors=%d\n",
  //      tflite_Operator_vec_len(operators),
  //      tflite_Tensor_vec_len(tensors)
  //      );
  //  for(
  //      size_t operator_idx = 0;
  //      operator_idx < tflite_Operator_vec_len(operators);
  //      operator_idx++
  //      )
  //  {
  //    tflite_Operator_table_t operator = tflite_Operator_vec_at(operators, operator_idx);
  //    flatbuffers_int32_vec_t op_inputs = tflite_Operator_inputs(operator),
  //                            op_outputs = tflite_Operator_outputs(operator);
  //    for(
  //      size_t vec_idx = 0;
  //      vec_idx < flatbuffers_int32_vec_len(op_inputs);
  //      vec_idx++
  //      )
  //    {
  //      int32_t tensor_idx = flatbuffers_int32_vec_at(op_inputs, vec_idx);
  //      tflite_Tensor_table_t op_input = tflite_Tensor_vec_at(tensors, tensor_idx);
  //      quantize_op_input(op_input);
  //    }
  //  }
  //}
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
  out_model = simplify_model_quantization(app.tflite_builder, app.in_model);
  if((out_buf = flatcc_builder_finalize_aligned_buffer(app.tflite_builder, &out_size)) == NULL)
    ERROR();
  fwrite(out_buf, sizeof(uint8_t), out_size, app.out_model_file);
  flatcc_builder_aligned_free(out_buf);
  return EXIT_SUCCESS;
}
