// Clone model.

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
  printf("clone IN_FILE OUT_FILE\n");
  printf("  Clones the model stored at IN_FILE and writes to OUT_FILE.\n");
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
#ifdef DEBUG_REPLICATE_C
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
    ERROR("Requires exactly 3 arguments");
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

static tflite_Model_ref_t clone_model(
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
      tflite_Model_subgraphs_pick(tflite_builder, in_model) ||
      tflite_Model_description_pick(tflite_builder, in_model) ||
      tflite_Model_buffers_pick(tflite_builder, in_model) ||
      tflite_Model_metadata_buffer_pick(tflite_builder, in_model) ||
      tflite_Model_metadata_pick(tflite_builder, in_model)
    )
    ERROR();
  tflite_Model_end_as_root(tflite_builder);
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
  out_model = clone_model(app.tflite_builder, app.in_model);
  if((out_buf = flatcc_builder_finalize_aligned_buffer(app.tflite_builder, &out_size)) == NULL)
    ERROR();
  fwrite(out_buf, sizeof(uint8_t), out_size, app.out_model_file);
  flatcc_builder_aligned_free(out_buf);
  return EXIT_SUCCESS;
}
