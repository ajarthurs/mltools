// Test TFLite JSON print.
// Convert TFLite->JSON (Pretty-format).

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <flatcc/flatcc.h>

#include "../tflite/tflite_v3_reader.h"
#include "../tflite/tflite_v3_json_printer.h"

#define BUF_SIZE (4194304 * sizeof(char))

int main()
{
  char *b = malloc(BUF_SIZE);
  flatcc_json_printer_t json_printer;
  FILE *f = fopen("deeplab.tflite", "rb"),
       *json = fopen("deeplab.json", "w");
  if(
      !b ||
      !f ||
      !json
    )
    goto cleanup;
  // Load TFLite file.
  fread(b, sizeof(char), BUF_SIZE, f);
  fclose(f); f = NULL;
  // Initialize JSON printer, specifying output file (`json`) and formatting.
  flatcc_json_printer_init(
      &json_printer,
      json
      );
  flatcc_json_printer_set_flags(
      &json_printer,
      flatcc_json_printer_f_pretty
      );
  // Print TFLite buffer to JSON file.
  tflite_v3_print_json(
      &json_printer,
      b,
      BUF_SIZE
      );
  flatcc_json_printer_flush(&json_printer);
cleanup_all:
  flatcc_json_printer_clear(&json_printer);
cleanup:
  if(b) free(b);
  if(f) fclose(f);
  if(json) fclose(json);
  return errno;
}
