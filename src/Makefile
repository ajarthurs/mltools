# Top-level sources.

# Settings.
PROGRAMS := quantize
SUBDIRS := schemas
TFLITE_SCHEMA_HDRS := $(wildcard schemas/tflite/*.h)

all clean tests: $(SUBDIRS) FORCE
FORCE:

all: $(PROGRAMS)

tests: FORCE
	$(MAKE) -C $@

clean:
	rm -f $(PROGRAMS)

# Compile programs.
quantize: quantize.c $(TFLITE_SCHEMA_HDRS)
	$(CC) $(CFLAGS) -o $@ $< -lflatccrt_d

# Recurse into each subdirectory, passing along the targets specified at command-line.
$(SUBDIRS): FORCE
	$(MAKE) -C $@ $(MAKECMDGOALS)
