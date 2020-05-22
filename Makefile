# Top-level Makefile

SUBDIRS := src

# Top-level targets.
all clean tests: $(SUBDIRS) FORCE
FORCE:

# Recurse into each subdirectory, passing along the targets specified at command-line.
$(SUBDIRS): FORCE
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean: FORCE
	$(MAKE) -C $@ $(MAKECMDGOALS)
