# Top-level schema tests.

# Settings.
TESTS := test_tflite_json_print test_tflite_json_parse

all clean: FORCE
FORCE:

all: $(TESTS)

clean:
	rm -f $(TESTS)

# Compile tests.
$(TESTS): % : %.c
	$(CC) $(CFLAGS) -ggdb3 -o $@ $< -lflatccrt_d
