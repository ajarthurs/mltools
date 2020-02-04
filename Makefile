# Top-level Makefile

SUBDIRS := src

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
