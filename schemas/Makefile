# Schemas Makefile
# A schema is codebase that interfaces a specific file format (e.g., the TFLite flatbuffer format).

SUBDIRS := tflite

# Top-level targets.
all clean: $(SUBDIRS) FORCE
FORCE:

# General tests recipes.
tests: FORCE
	$(MAKE) -C $@

# Recurse into each subdirectory, passing along the targets specified at command-line.
$(SUBDIRS): FORCE
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean: FORCE
	$(MAKE) -C tests $(MAKECMDGOALS)
