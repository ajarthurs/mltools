// Test TFLite JSON parse.

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <flatcc/flatcc.h>

#include "../tflite/tflite_v3_builder.h"
#include "../tflite/tflite_v3_json_parser.h"

#define BUF_SIZE (33554432 * sizeof(char))
#define MODEL "deeplab-noargmax"

int main()
{
  flatcc_builder_t builder;
  flatcc_json_parser_t json_parser;
  FILE *json = fopen(MODEL ".json", "r"),
       *tflite = fopen(MODEL ".tflite", "wb");
  char *b = malloc(BUF_SIZE);
  size_t s;
  if(
      !b ||
      !json ||
      !tflite
    )
    goto cleanup;
  // Load JSON file.
  fread(b, sizeof(char), BUF_SIZE, json);
  fclose(json); json = NULL;
  // Initialize TFLite structure.
  flatcc_builder_init(&builder);
  // Parse JSON and update the TFLite structure (`builder`).
  if(tflite_v3_parse_json(
        &builder,
        &json_parser,
        b,
        BUF_SIZE,
        flatcc_json_parser_f_skip_unknown
        ) != 0) // error while parsing JSON
  {
    fprintf(
        stderr,
        MODEL ".json:%d:%d: %s\n",
        (int)json_parser.line,
        (int)(json_parser.error_loc - json_parser.line_start + 1),
        flatcc_json_parser_error_string(json_parser.error)
        );
    goto cleanup_all;
  }
  free(b); b = NULL;// release JSON input string
  // Serialize TFLite structure and write buffer to file (`tflite`).
  b = flatcc_builder_finalize_aligned_buffer(
      &builder,
      &s
      );
  fwrite(b, sizeof(char), s, tflite);
  // Done.
  fclose(tflite); tflite = NULL;
  flatcc_builder_aligned_free(b); b = NULL;
cleanup_all:
  flatcc_builder_clear(&builder);
cleanup:
  if(b) free(b);
  if(json) fclose(json);
  if(tflite) fclose(tflite);
  return errno;
}
