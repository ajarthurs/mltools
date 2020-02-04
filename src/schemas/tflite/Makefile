# TFLite flatbuffer schema Makefile.

# Settings.
VERSION=3
FLATBUFFERS_SCHEMA := tflite_v$(VERSION).fbs
HDRS := flatbuffers_common_builder.h flatbuffers_common_reader.h tflite_v$(VERSION)_json_printer.h tflite_v$(VERSION)_json_parser.h tflite_v$(VERSION)_builder.h tflite_v$(VERSION)_reader.h tflite_v$(VERSION)_verifier.h

all clean: FORCE
FORCE:

all: $(HDRS)

clean:
	rm -f $(HDRS)

# Generate headers.
$(HDRS): $(FLATBUFFERS_SCHEMA)
	flatcc -a --json -o . $(FLATBUFFERS_SCHEMA)
