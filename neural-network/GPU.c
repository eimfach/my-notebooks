/*
 * Headers
*/

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>


/*
 * Initialisation
*/

struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);

/*
 * Arrays
*/

struct futhark_f32_1d ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int64_t dim0);
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0);
int futhark_free_f32_1d(struct futhark_context *ctx,
                        struct futhark_f32_1d *arr);
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data);
char *futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                struct futhark_f32_1d *arr);
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr);
struct futhark_f32_2d ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int64_t dim0,
                                          int64_t dim1);
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1);
int futhark_free_f32_2d(struct futhark_context *ctx,
                        struct futhark_f32_2d *arr);
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data);
char *futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                struct futhark_f32_2d *arr);
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr);

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry_dot(struct futhark_context *ctx, struct futhark_f32_2d **out0,
                      const struct futhark_f32_2d *in0, const
                      struct futhark_f32_2d *in1);
int futhark_entry_matsubstract(struct futhark_context *ctx,
                               struct futhark_f32_2d **out0, const
                               struct futhark_f32_2d *in0, const
                               struct futhark_f32_2d *in1);
int futhark_entry_matadd(struct futhark_context *ctx,
                         struct futhark_f32_2d **out0, const
                         struct futhark_f32_2d *in0, const
                         struct futhark_f32_2d *in1);
int futhark_entry_matmultiply(struct futhark_context *ctx,
                              struct futhark_f32_2d **out0, const
                              struct futhark_f32_2d *in0, const
                              struct futhark_f32_2d *in1);
int futhark_entry_lvecmul(struct futhark_context *ctx,
                          struct futhark_f32_1d **out0, const
                          struct futhark_f32_1d *in0, const
                          struct futhark_f32_2d *in1);
int futhark_entry_sigmoid(struct futhark_context *ctx,
                          struct futhark_f32_2d **out0, const
                          struct futhark_f32_2d *in0);
int futhark_entry_lmatsubstract(struct futhark_context *ctx,
                                struct futhark_f32_2d **out0, const float in0,
                                const struct futhark_f32_2d *in1);
int futhark_entry_lmatadd(struct futhark_context *ctx,
                          struct futhark_f32_2d **out0, const float in0, const
                          struct futhark_f32_2d *in1);
int futhark_entry_substract2(struct futhark_context *ctx,
                             struct futhark_f32_1d **out0, const
                             struct futhark_f32_1d *in0, const
                             struct futhark_f32_1d *in1);
int futhark_entry_add2(struct futhark_context *ctx,
                       struct futhark_f32_1d **out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1);
int futhark_entry_lmatmultiply(struct futhark_context *ctx,
                               struct futhark_f32_2d **out0, const float in0,
                               const struct futhark_f32_2d *in1);
int futhark_entry_multiply2(struct futhark_context *ctx,
                            struct futhark_f32_1d **out0, const
                            struct futhark_f32_1d *in0, const
                            struct futhark_f32_1d *in1);
int futhark_entry_substract(struct futhark_context *ctx,
                            struct futhark_f32_1d **out0, const float in0, const
                            struct futhark_f32_1d *in1);
int futhark_entry_add(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                      const float in0, const struct futhark_f32_1d *in1);
int futhark_entry_multiply(struct futhark_context *ctx,
                           struct futhark_f32_1d **out0, const float in0, const
                           struct futhark_f32_1d *in1);
int futhark_entry_divide(struct futhark_context *ctx,
                         struct futhark_f32_1d **out0, const float in0, const
                         struct futhark_f32_1d *in1);
int futhark_entry_negation(struct futhark_context *ctx,
                           struct futhark_f32_1d **out0, const
                           struct futhark_f32_1d *in0);
int futhark_entry_exp(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                      const struct futhark_f32_1d *in0);
int futhark_entry_transp(struct futhark_context *ctx,
                         struct futhark_f32_2d **out0, const
                         struct futhark_f32_2d *in0);

/*
 * Miscellaneous
*/

void futhark_debugging_report(struct futhark_context *ctx);
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
// Start of panic.h.

#include <stdarg.h>

static const char *fut_progname;

static void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
        fprintf(stderr, "%s: ", fut_progname);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* For generating arbitrary-sized error messages.  It is the callers
   responsibility to free the buffer at some point. */
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + (size_t)vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*) malloc(needed);
  va_start(vl, s); /* Must re-init. */
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}

// End of panic.h.

// Start of timing.h.

// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

// End of timing.h.

#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
// Start of values.h.

//// Text I/O

typedef int (*writer)(FILE*, void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces() {
  int c;
  do {
    c = getchar();
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(char *buf, int bufsize) {
 start:
  skipspaces();

  int i = 0;
  while (i < bufsize) {
    int c = getchar();
    buf[i] = (char)c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getchar());
      goto start;
    } else if (!constituent((char)c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, stdin);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(char *buf, int bufsize, const char* expected) {
  next_token(buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    (size_t)(reader->n_elems_space * reader->elem_size));
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(char *buf, int bufsize,
                                struct array_reader *reader, int64_t dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc((size_t)dims, sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc((size_t)dims, sizeof(int64_t));

  while (1) {
    next_token(buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    if (!next_token_is(buf, bufsize, "[")) {
      return 1;
    }

    next_token(buf, bufsize);

    if (sscanf(buf, "%"SCNu64, &shape[i]) != 1) {
      return 1;
    }

    if (!next_token_is(buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(buf, bufsize, ")")) {
    return 1;
  }

  // Check whether the array really is empty.
  for (int i = 0; i < dims; i++) {
    if (shape[i] == 0) {
      return 0;
    }
  }

  // Not an empty array!
  return 1;
}

static int read_str_array(int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, (size_t)(elem_size*reader.n_elems_space));
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNi8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = (int8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNu8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = (uint8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

static void flip_bytes(int elem_size, unsigned char *elem) {
  for (int j=0; j<elem_size/2; j++) {
    unsigned char head = elem[j];
    int tail_index = elem_size-1-j;
    elem[j] = elem[tail_index];
    elem[tail_index] = head;
  }
}

// On Windows we need to explicitly set the file mode to not mangle
// newline characters.  On *nix there is no difference.
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static void set_binary_mode(FILE *f) {
  setmode(fileno(f), O_BINARY);
}
#else
static void set_binary_mode(FILE *f) {
  (void)f;
}
#endif

static int read_byte(void* dest) {
  int num_elems_read = fread(dest, 1, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int64_t size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary() {
  skipspaces();
  int c = getchar();
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(&bin_version);

    if (ret != 0) { panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, stdin);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum() {
  char read_binname[4];

  int num_matched = scanf("%4c", read_binname);
  if (num_matched != 1) { panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum();
  if (bin_type != expected_type) {
    panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum();
  if (expected_type != bin_primtype) {
    panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  int64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    int64_t bin_shape;
    ret = fread(&bin_shape, sizeof(bin_shape), 1, stdin);
    if (ret != 1) {
      panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i);
    }
    if (IS_BIG_ENDIAN) {
      flip_bytes(sizeof(bin_shape), (unsigned char*) &bin_shape);
    }
    elem_count *= bin_shape;
    shape[i] = bin_shape;
  }

  int64_t elem_size = expected_type->size;
  void* tmp = realloc(*data, (size_t)(elem_count * elem_size));
  if (tmp == NULL) {
    panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  int64_t num_elems_read = (int64_t)fread(*data, (size_t)elem_size, (size_t)elem_count, stdin);
  if (num_elems_read != elem_count) {
    panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    flip_bytes(elem_size, (unsigned char*) *data);
  }

  return 0;
}

static int read_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary()) {
    return read_str_array(expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(expected_type, data, shape, dims);
  }
}

static int write_str_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = (int64_t)shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int8_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      printf("empty(");
      for (int64_t i = 0; i < rank; i++) {
        printf("[%"PRIi64"]", shape[i]);
      }
      printf("%s", elem_type->type_name);
      printf(")");
    } else if (rank==1) {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    } else {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    }
  }
  return 0;
}

static int write_bin_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fputs(elem_type->binname, out);
  if (shape != NULL) {
    fwrite(shape, sizeof(int64_t), (size_t)rank, out);
  }

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, (size_t)elem_type->size, (size_t)num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type, void *data, int64_t *shape, int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary()) {
    char buf[100];
    next_token(buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(expected_type);
    int64_t elem_size = expected_type->size;
    int num_elems_read = fread(dest, (size_t)elem_size, 1, stdin);
    if (IS_BIG_ENDIAN) {
      flip_bytes(elem_size, (unsigned char*) dest);
    }
    return num_elems_read == 1 ? 0 : 1;
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

// End of values.h.

static int binary_output = 0;
static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
// Start of tuning.h.

static char* load_tuning_file(const char *fname,
                              void *cfg,
                              int (*set_size)(void*, const char*, size_t)) {
  const int max_line_len = 1024;
  char* line = (char*) malloc(max_line_len);

  FILE *f = fopen(fname, "r");

  if (f == NULL) {
    snprintf(line, max_line_len, "Cannot open file: %s", strerror(errno));
    return line;
  }

  int lineno = 0;
  while (fgets(line, max_line_len, f) != NULL) {
    lineno++;
    char *eql = strstr(line, "=");
    if (eql) {
      *eql = 0;
      int value = atoi(eql+1);
      if (set_size(cfg, line, value) != 0) {
        strncpy(eql+1, line, max_line_len-strlen(line)-1);
        snprintf(line, max_line_len, "Unknown name '%s' on line %d.", eql+1, lineno);
        return line;
      }
    } else {
      snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
               lineno);
      return line;
    }
  }

  free(line);

  return NULL;
}

// End of tuning.h.

int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"debugging", no_argument, NULL, 3},
                                           {"log", no_argument, NULL, 4},
                                           {"entry-point", required_argument,
                                            NULL, 5}, {"binary-output",
                                                       no_argument, NULL, 6},
                                           {0, 0, 0, 0}};
    
    while ((ch = getopt_long(argc, argv, ":t:r:DLe:b", long_options, NULL)) !=
           -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                panic(1, "Cannot open %s: %s\n", optarg, strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                panic(1, "Need a positive number of runs, not %s\n", optarg);
        }
        if (ch == 3 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 4 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 5 || ch == 'e') {
            if (entry_point != NULL)
                entry_point = optarg;
        }
        if (ch == 6 || ch == 'b')
            binary_output = 1;
        if (ch == ':')
            panic(-1, "Missing argument for option %s\n", argv[optind - 1]);
        if (ch == '?') {
            fprintf(stderr, "Usage: %s: %s\n", fut_progname,
                    "[-t/--write-runtime-to FILE] [-r/--runs INT] [-D/--debugging] [-L/--log] [-e/--entry-point NAME] [-b/--binary-output]");
            panic(1, "Unknown option: %s\n", argv[optind - 1]);
        }
    }
    return optind;
}
static void futrts_cli_entry_dot(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_2d *read_value_10129;
    int64_t read_shape_10130[2];
    float *read_arr_10131 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10131, read_shape_10130, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_10132;
    int64_t read_shape_10133[2];
    float *read_arr_10134 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10134, read_shape_10133, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_10135;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_10129 = futhark_new_f32_2d(ctx, read_arr_10131,
                                                      read_shape_10130[0],
                                                      read_shape_10130[1])) !=
            0);
        assert((read_value_10132 = futhark_new_f32_2d(ctx, read_arr_10134,
                                                      read_shape_10133[0],
                                                      read_shape_10133[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_dot(ctx, &result_10135, read_value_10129,
                              read_value_10132);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_10129) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_10132) == 0);
        assert(futhark_free_f32_2d(ctx, result_10135) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_10129 = futhark_new_f32_2d(ctx, read_arr_10131,
                                                      read_shape_10130[0],
                                                      read_shape_10130[1])) !=
            0);
        assert((read_value_10132 = futhark_new_f32_2d(ctx, read_arr_10134,
                                                      read_shape_10133[0],
                                                      read_shape_10133[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_dot(ctx, &result_10135, read_value_10129,
                              read_value_10132);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_10129) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_10132) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_10135) == 0);
        }
    }
    free(read_arr_10131);
    free(read_arr_10134);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_10135)[0] *
                            futhark_shape_f32_2d(ctx, result_10135)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_10135, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_10135), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_10135) == 0);
}
static void futrts_cli_entry_matsubstract(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_2d *read_value_10136;
    int64_t read_shape_10137[2];
    float *read_arr_10138 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10138, read_shape_10137, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_10139;
    int64_t read_shape_10140[2];
    float *read_arr_10141 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10141, read_shape_10140, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_10142;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_10136 = futhark_new_f32_2d(ctx, read_arr_10138,
                                                      read_shape_10137[0],
                                                      read_shape_10137[1])) !=
            0);
        assert((read_value_10139 = futhark_new_f32_2d(ctx, read_arr_10141,
                                                      read_shape_10140[0],
                                                      read_shape_10140[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_matsubstract(ctx, &result_10142, read_value_10136,
                                       read_value_10139);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_10136) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_10139) == 0);
        assert(futhark_free_f32_2d(ctx, result_10142) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_10136 = futhark_new_f32_2d(ctx, read_arr_10138,
                                                      read_shape_10137[0],
                                                      read_shape_10137[1])) !=
            0);
        assert((read_value_10139 = futhark_new_f32_2d(ctx, read_arr_10141,
                                                      read_shape_10140[0],
                                                      read_shape_10140[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_matsubstract(ctx, &result_10142, read_value_10136,
                                       read_value_10139);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_10136) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_10139) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_10142) == 0);
        }
    }
    free(read_arr_10138);
    free(read_arr_10141);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_10142)[0] *
                            futhark_shape_f32_2d(ctx, result_10142)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_10142, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_10142), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_10142) == 0);
}
static void futrts_cli_entry_matadd(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_2d *read_value_10143;
    int64_t read_shape_10144[2];
    float *read_arr_10145 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10145, read_shape_10144, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_10146;
    int64_t read_shape_10147[2];
    float *read_arr_10148 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10148, read_shape_10147, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_10149;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_10143 = futhark_new_f32_2d(ctx, read_arr_10145,
                                                      read_shape_10144[0],
                                                      read_shape_10144[1])) !=
            0);
        assert((read_value_10146 = futhark_new_f32_2d(ctx, read_arr_10148,
                                                      read_shape_10147[0],
                                                      read_shape_10147[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_matadd(ctx, &result_10149, read_value_10143,
                                 read_value_10146);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_10143) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_10146) == 0);
        assert(futhark_free_f32_2d(ctx, result_10149) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_10143 = futhark_new_f32_2d(ctx, read_arr_10145,
                                                      read_shape_10144[0],
                                                      read_shape_10144[1])) !=
            0);
        assert((read_value_10146 = futhark_new_f32_2d(ctx, read_arr_10148,
                                                      read_shape_10147[0],
                                                      read_shape_10147[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_matadd(ctx, &result_10149, read_value_10143,
                                 read_value_10146);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_10143) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_10146) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_10149) == 0);
        }
    }
    free(read_arr_10145);
    free(read_arr_10148);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_10149)[0] *
                            futhark_shape_f32_2d(ctx, result_10149)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_10149, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_10149), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_10149) == 0);
}
static void futrts_cli_entry_matmultiply(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_2d *read_value_10150;
    int64_t read_shape_10151[2];
    float *read_arr_10152 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10152, read_shape_10151, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_10153;
    int64_t read_shape_10154[2];
    float *read_arr_10155 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10155, read_shape_10154, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_10156;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_10150 = futhark_new_f32_2d(ctx, read_arr_10152,
                                                      read_shape_10151[0],
                                                      read_shape_10151[1])) !=
            0);
        assert((read_value_10153 = futhark_new_f32_2d(ctx, read_arr_10155,
                                                      read_shape_10154[0],
                                                      read_shape_10154[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_matmultiply(ctx, &result_10156, read_value_10150,
                                      read_value_10153);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_10150) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_10153) == 0);
        assert(futhark_free_f32_2d(ctx, result_10156) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_10150 = futhark_new_f32_2d(ctx, read_arr_10152,
                                                      read_shape_10151[0],
                                                      read_shape_10151[1])) !=
            0);
        assert((read_value_10153 = futhark_new_f32_2d(ctx, read_arr_10155,
                                                      read_shape_10154[0],
                                                      read_shape_10154[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_matmultiply(ctx, &result_10156, read_value_10150,
                                      read_value_10153);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_10150) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_10153) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_10156) == 0);
        }
    }
    free(read_arr_10152);
    free(read_arr_10155);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_10156)[0] *
                            futhark_shape_f32_2d(ctx, result_10156)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_10156, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_10156), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_10156) == 0);
}
static void futrts_cli_entry_lvecmul(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_1d *read_value_10157;
    int64_t read_shape_10158[1];
    float *read_arr_10159 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10159, read_shape_10158, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_10160;
    int64_t read_shape_10161[2];
    float *read_arr_10162 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10162, read_shape_10161, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_10163;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_10157 = futhark_new_f32_1d(ctx, read_arr_10159,
                                                      read_shape_10158[0])) !=
            0);
        assert((read_value_10160 = futhark_new_f32_2d(ctx, read_arr_10162,
                                                      read_shape_10161[0],
                                                      read_shape_10161[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_lvecmul(ctx, &result_10163, read_value_10157,
                                  read_value_10160);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_10157) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_10160) == 0);
        assert(futhark_free_f32_1d(ctx, result_10163) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_10157 = futhark_new_f32_1d(ctx, read_arr_10159,
                                                      read_shape_10158[0])) !=
            0);
        assert((read_value_10160 = futhark_new_f32_2d(ctx, read_arr_10162,
                                                      read_shape_10161[0],
                                                      read_shape_10161[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_lvecmul(ctx, &result_10163, read_value_10157,
                                  read_value_10160);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_10157) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_10160) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_10163) == 0);
        }
    }
    free(read_arr_10159);
    free(read_arr_10162);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_10163)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_10163, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_10163), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_10163) == 0);
}
static void futrts_cli_entry_sigmoid(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_2d *read_value_10164;
    int64_t read_shape_10165[2];
    float *read_arr_10166 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10166, read_shape_10165, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_10167;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_10164 = futhark_new_f32_2d(ctx, read_arr_10166,
                                                      read_shape_10165[0],
                                                      read_shape_10165[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_sigmoid(ctx, &result_10167, read_value_10164);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_10164) == 0);
        assert(futhark_free_f32_2d(ctx, result_10167) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_10164 = futhark_new_f32_2d(ctx, read_arr_10166,
                                                      read_shape_10165[0],
                                                      read_shape_10165[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_sigmoid(ctx, &result_10167, read_value_10164);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_10164) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_10167) == 0);
        }
    }
    free(read_arr_10166);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_10167)[0] *
                            futhark_shape_f32_2d(ctx, result_10167)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_10167, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_10167), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_10167) == 0);
}
static void futrts_cli_entry_lmatsubstract(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    float read_value_10168;
    
    if (read_scalar(&f32_info, &read_value_10168) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_10169;
    int64_t read_shape_10170[2];
    float *read_arr_10171 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10171, read_shape_10170, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_10172;
    
    if (perform_warmup) {
        int r;
        
        ;
        assert((read_value_10169 = futhark_new_f32_2d(ctx, read_arr_10171,
                                                      read_shape_10170[0],
                                                      read_shape_10170[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_lmatsubstract(ctx, &result_10172, read_value_10168,
                                        read_value_10169);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_10169) == 0);
        assert(futhark_free_f32_2d(ctx, result_10172) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        assert((read_value_10169 = futhark_new_f32_2d(ctx, read_arr_10171,
                                                      read_shape_10170[0],
                                                      read_shape_10170[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_lmatsubstract(ctx, &result_10172, read_value_10168,
                                        read_value_10169);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_10169) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_10172) == 0);
        }
    }
    ;
    free(read_arr_10171);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_10172)[0] *
                            futhark_shape_f32_2d(ctx, result_10172)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_10172, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_10172), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_10172) == 0);
}
static void futrts_cli_entry_lmatadd(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    float read_value_10173;
    
    if (read_scalar(&f32_info, &read_value_10173) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_10174;
    int64_t read_shape_10175[2];
    float *read_arr_10176 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10176, read_shape_10175, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_10177;
    
    if (perform_warmup) {
        int r;
        
        ;
        assert((read_value_10174 = futhark_new_f32_2d(ctx, read_arr_10176,
                                                      read_shape_10175[0],
                                                      read_shape_10175[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_lmatadd(ctx, &result_10177, read_value_10173,
                                  read_value_10174);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_10174) == 0);
        assert(futhark_free_f32_2d(ctx, result_10177) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        assert((read_value_10174 = futhark_new_f32_2d(ctx, read_arr_10176,
                                                      read_shape_10175[0],
                                                      read_shape_10175[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_lmatadd(ctx, &result_10177, read_value_10173,
                                  read_value_10174);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_10174) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_10177) == 0);
        }
    }
    ;
    free(read_arr_10176);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_10177)[0] *
                            futhark_shape_f32_2d(ctx, result_10177)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_10177, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_10177), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_10177) == 0);
}
static void futrts_cli_entry_substract2(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_1d *read_value_10178;
    int64_t read_shape_10179[1];
    float *read_arr_10180 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10180, read_shape_10179, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_10181;
    int64_t read_shape_10182[1];
    float *read_arr_10183 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10183, read_shape_10182, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_10184;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_10178 = futhark_new_f32_1d(ctx, read_arr_10180,
                                                      read_shape_10179[0])) !=
            0);
        assert((read_value_10181 = futhark_new_f32_1d(ctx, read_arr_10183,
                                                      read_shape_10182[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_substract2(ctx, &result_10184, read_value_10178,
                                     read_value_10181);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_10178) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_10181) == 0);
        assert(futhark_free_f32_1d(ctx, result_10184) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_10178 = futhark_new_f32_1d(ctx, read_arr_10180,
                                                      read_shape_10179[0])) !=
            0);
        assert((read_value_10181 = futhark_new_f32_1d(ctx, read_arr_10183,
                                                      read_shape_10182[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_substract2(ctx, &result_10184, read_value_10178,
                                     read_value_10181);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_10178) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_10181) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_10184) == 0);
        }
    }
    free(read_arr_10180);
    free(read_arr_10183);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_10184)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_10184, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_10184), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_10184) == 0);
}
static void futrts_cli_entry_add2(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_1d *read_value_10185;
    int64_t read_shape_10186[1];
    float *read_arr_10187 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10187, read_shape_10186, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_10188;
    int64_t read_shape_10189[1];
    float *read_arr_10190 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10190, read_shape_10189, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_10191;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_10185 = futhark_new_f32_1d(ctx, read_arr_10187,
                                                      read_shape_10186[0])) !=
            0);
        assert((read_value_10188 = futhark_new_f32_1d(ctx, read_arr_10190,
                                                      read_shape_10189[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_add2(ctx, &result_10191, read_value_10185,
                               read_value_10188);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_10185) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_10188) == 0);
        assert(futhark_free_f32_1d(ctx, result_10191) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_10185 = futhark_new_f32_1d(ctx, read_arr_10187,
                                                      read_shape_10186[0])) !=
            0);
        assert((read_value_10188 = futhark_new_f32_1d(ctx, read_arr_10190,
                                                      read_shape_10189[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_add2(ctx, &result_10191, read_value_10185,
                               read_value_10188);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_10185) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_10188) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_10191) == 0);
        }
    }
    free(read_arr_10187);
    free(read_arr_10190);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_10191)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_10191, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_10191), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_10191) == 0);
}
static void futrts_cli_entry_lmatmultiply(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    float read_value_10192;
    
    if (read_scalar(&f32_info, &read_value_10192) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_10193;
    int64_t read_shape_10194[2];
    float *read_arr_10195 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10195, read_shape_10194, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_10196;
    
    if (perform_warmup) {
        int r;
        
        ;
        assert((read_value_10193 = futhark_new_f32_2d(ctx, read_arr_10195,
                                                      read_shape_10194[0],
                                                      read_shape_10194[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_lmatmultiply(ctx, &result_10196, read_value_10192,
                                       read_value_10193);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_10193) == 0);
        assert(futhark_free_f32_2d(ctx, result_10196) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        assert((read_value_10193 = futhark_new_f32_2d(ctx, read_arr_10195,
                                                      read_shape_10194[0],
                                                      read_shape_10194[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_lmatmultiply(ctx, &result_10196, read_value_10192,
                                       read_value_10193);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_10193) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_10196) == 0);
        }
    }
    ;
    free(read_arr_10195);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_10196)[0] *
                            futhark_shape_f32_2d(ctx, result_10196)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_10196, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_10196), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_10196) == 0);
}
static void futrts_cli_entry_multiply2(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_1d *read_value_10197;
    int64_t read_shape_10198[1];
    float *read_arr_10199 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10199, read_shape_10198, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_10200;
    int64_t read_shape_10201[1];
    float *read_arr_10202 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10202, read_shape_10201, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_10203;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_10197 = futhark_new_f32_1d(ctx, read_arr_10199,
                                                      read_shape_10198[0])) !=
            0);
        assert((read_value_10200 = futhark_new_f32_1d(ctx, read_arr_10202,
                                                      read_shape_10201[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_multiply2(ctx, &result_10203, read_value_10197,
                                    read_value_10200);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_10197) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_10200) == 0);
        assert(futhark_free_f32_1d(ctx, result_10203) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_10197 = futhark_new_f32_1d(ctx, read_arr_10199,
                                                      read_shape_10198[0])) !=
            0);
        assert((read_value_10200 = futhark_new_f32_1d(ctx, read_arr_10202,
                                                      read_shape_10201[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_multiply2(ctx, &result_10203, read_value_10197,
                                    read_value_10200);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_10197) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_10200) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_10203) == 0);
        }
    }
    free(read_arr_10199);
    free(read_arr_10202);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_10203)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_10203, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_10203), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_10203) == 0);
}
static void futrts_cli_entry_substract(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    float read_value_10204;
    
    if (read_scalar(&f32_info, &read_value_10204) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_10205;
    int64_t read_shape_10206[1];
    float *read_arr_10207 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10207, read_shape_10206, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_10208;
    
    if (perform_warmup) {
        int r;
        
        ;
        assert((read_value_10205 = futhark_new_f32_1d(ctx, read_arr_10207,
                                                      read_shape_10206[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_substract(ctx, &result_10208, read_value_10204,
                                    read_value_10205);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_10205) == 0);
        assert(futhark_free_f32_1d(ctx, result_10208) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        assert((read_value_10205 = futhark_new_f32_1d(ctx, read_arr_10207,
                                                      read_shape_10206[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_substract(ctx, &result_10208, read_value_10204,
                                    read_value_10205);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_10205) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_10208) == 0);
        }
    }
    ;
    free(read_arr_10207);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_10208)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_10208, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_10208), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_10208) == 0);
}
static void futrts_cli_entry_add(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    float read_value_10209;
    
    if (read_scalar(&f32_info, &read_value_10209) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_10210;
    int64_t read_shape_10211[1];
    float *read_arr_10212 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10212, read_shape_10211, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_10213;
    
    if (perform_warmup) {
        int r;
        
        ;
        assert((read_value_10210 = futhark_new_f32_1d(ctx, read_arr_10212,
                                                      read_shape_10211[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_add(ctx, &result_10213, read_value_10209,
                              read_value_10210);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_10210) == 0);
        assert(futhark_free_f32_1d(ctx, result_10213) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        assert((read_value_10210 = futhark_new_f32_1d(ctx, read_arr_10212,
                                                      read_shape_10211[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_add(ctx, &result_10213, read_value_10209,
                              read_value_10210);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_10210) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_10213) == 0);
        }
    }
    ;
    free(read_arr_10212);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_10213)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_10213, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_10213), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_10213) == 0);
}
static void futrts_cli_entry_multiply(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    float read_value_10214;
    
    if (read_scalar(&f32_info, &read_value_10214) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_10215;
    int64_t read_shape_10216[1];
    float *read_arr_10217 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10217, read_shape_10216, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_10218;
    
    if (perform_warmup) {
        int r;
        
        ;
        assert((read_value_10215 = futhark_new_f32_1d(ctx, read_arr_10217,
                                                      read_shape_10216[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_multiply(ctx, &result_10218, read_value_10214,
                                   read_value_10215);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_10215) == 0);
        assert(futhark_free_f32_1d(ctx, result_10218) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        assert((read_value_10215 = futhark_new_f32_1d(ctx, read_arr_10217,
                                                      read_shape_10216[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_multiply(ctx, &result_10218, read_value_10214,
                                   read_value_10215);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_10215) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_10218) == 0);
        }
    }
    ;
    free(read_arr_10217);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_10218)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_10218, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_10218), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_10218) == 0);
}
static void futrts_cli_entry_divide(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    float read_value_10219;
    
    if (read_scalar(&f32_info, &read_value_10219) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_10220;
    int64_t read_shape_10221[1];
    float *read_arr_10222 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10222, read_shape_10221, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_10223;
    
    if (perform_warmup) {
        int r;
        
        ;
        assert((read_value_10220 = futhark_new_f32_1d(ctx, read_arr_10222,
                                                      read_shape_10221[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_divide(ctx, &result_10223, read_value_10219,
                                 read_value_10220);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_10220) == 0);
        assert(futhark_free_f32_1d(ctx, result_10223) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        assert((read_value_10220 = futhark_new_f32_1d(ctx, read_arr_10222,
                                                      read_shape_10221[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_divide(ctx, &result_10223, read_value_10219,
                                 read_value_10220);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_10220) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_10223) == 0);
        }
    }
    ;
    free(read_arr_10222);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_10223)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_10223, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_10223), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_10223) == 0);
}
static void futrts_cli_entry_negation(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_1d *read_value_10224;
    int64_t read_shape_10225[1];
    float *read_arr_10226 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10226, read_shape_10225, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_10227;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_10224 = futhark_new_f32_1d(ctx, read_arr_10226,
                                                      read_shape_10225[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_negation(ctx, &result_10227, read_value_10224);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_10224) == 0);
        assert(futhark_free_f32_1d(ctx, result_10227) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_10224 = futhark_new_f32_1d(ctx, read_arr_10226,
                                                      read_shape_10225[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_negation(ctx, &result_10227, read_value_10224);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_10224) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_10227) == 0);
        }
    }
    free(read_arr_10226);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_10227)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_10227, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_10227), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_10227) == 0);
}
static void futrts_cli_entry_exp(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_1d *read_value_10228;
    int64_t read_shape_10229[1];
    float *read_arr_10230 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10230, read_shape_10229, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_10231;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_10228 = futhark_new_f32_1d(ctx, read_arr_10230,
                                                      read_shape_10229[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_exp(ctx, &result_10231, read_value_10228);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_10228) == 0);
        assert(futhark_free_f32_1d(ctx, result_10231) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_10228 = futhark_new_f32_1d(ctx, read_arr_10230,
                                                      read_shape_10229[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_exp(ctx, &result_10231, read_value_10228);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_10228) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_10231) == 0);
        }
    }
    free(read_arr_10230);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_10231)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_10231, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_10231), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_10231) == 0);
}
static void futrts_cli_entry_transp(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_2d *read_value_10232;
    int64_t read_shape_10233[2];
    float *read_arr_10234 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10234, read_shape_10233, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_10235;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_10232 = futhark_new_f32_2d(ctx, read_arr_10234,
                                                      read_shape_10233[0],
                                                      read_shape_10233[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_transp(ctx, &result_10235, read_value_10232);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_10232) == 0);
        assert(futhark_free_f32_2d(ctx, result_10235) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_10232 = futhark_new_f32_2d(ctx, read_arr_10234,
                                                      read_shape_10233[0],
                                                      read_shape_10233[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_transp(ctx, &result_10235, read_value_10232);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_10232) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_10235) == 0);
        }
    }
    free(read_arr_10234);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_10235)[0] *
                            futhark_shape_f32_2d(ctx, result_10235)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_10235, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_10235), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_10235) == 0);
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="dot", .fun =
                                                futrts_cli_entry_dot}, {.name =
                                                                        "matsubstract",
                                                                        .fun =
                                                                        futrts_cli_entry_matsubstract},
                                               {.name ="matadd", .fun =
                                                futrts_cli_entry_matadd},
                                               {.name ="matmultiply", .fun =
                                                futrts_cli_entry_matmultiply},
                                               {.name ="lvecmul", .fun =
                                                futrts_cli_entry_lvecmul},
                                               {.name ="sigmoid", .fun =
                                                futrts_cli_entry_sigmoid},
                                               {.name ="lmatsubstract", .fun =
                                                futrts_cli_entry_lmatsubstract},
                                               {.name ="lmatadd", .fun =
                                                futrts_cli_entry_lmatadd},
                                               {.name ="substract2", .fun =
                                                futrts_cli_entry_substract2},
                                               {.name ="add2", .fun =
                                                futrts_cli_entry_add2}, {.name =
                                                                         "lmatmultiply",
                                                                         .fun =
                                                                         futrts_cli_entry_lmatmultiply},
                                               {.name ="multiply2", .fun =
                                                futrts_cli_entry_multiply2},
                                               {.name ="substract", .fun =
                                                futrts_cli_entry_substract},
                                               {.name ="add", .fun =
                                                futrts_cli_entry_add}, {.name =
                                                                        "multiply",
                                                                        .fun =
                                                                        futrts_cli_entry_multiply},
                                               {.name ="divide", .fun =
                                                futrts_cli_entry_divide},
                                               {.name ="negation", .fun =
                                                futrts_cli_entry_negation},
                                               {.name ="exp", .fun =
                                                futrts_cli_entry_exp}, {.name =
                                                                        "transp",
                                                                        .fun =
                                                                        futrts_cli_entry_transp}};
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    if (entry_point != NULL) {
        int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
        entry_point_fun *entry_point_fun = NULL;
        
        for (int i = 0; i < num_entry_points; i++) {
            if (strcmp(entry_points[i].name, entry_point) == 0) {
                entry_point_fun = entry_points[i].fun;
                break;
            }
        }
        if (entry_point_fun == NULL) {
            fprintf(stderr,
                    "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                    entry_point);
            for (int i = 0; i < num_entry_points; i++)
                fprintf(stderr, "%s\n", entry_points[i].name);
            return 1;
        }
        entry_point_fun(ctx);
        if (runtime_file != NULL)
            fclose(runtime_file);
        futhark_debugging_report(ctx);
    }
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
// Start of lock.h.

/* A very simple cross-platform implementation of locks.  Uses
   pthreads on Unix and some Windows thing there.  Futhark's
   host-level code is not multithreaded, but user code may be, so we
   need some mechanism for ensuring atomic access to API functions.
   This is that mechanism.  It is not exposed to user code at all, so
   we do not have to worry about name collisions. */

#ifdef _WIN32

typedef HANDLE lock_t;

static lock_t create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  /* Default security attributes. */
                      FALSE, /* Initially unlocked. */
                      NULL); /* Unnamed. */
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
/* Assuming POSIX */

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  /* Nothing to do for pthreads. */
  (void)lock;
}

#endif

// End of lock.h.

struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
struct futhark_context_config {
    int debugging;
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  (struct futhark_context_config *) malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->debugging = 0;
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg);
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int detail)
{
    cfg->debugging = detail;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int detail)
{
    /* Does nothing for this backend. */
    (void) cfg;
    (void) detail;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    int profiling;
    lock_t lock;
    char *error;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
} ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    ctx->detail_memory = cfg->debugging;
    ctx->debugging = cfg->debugging;
    ctx->error = NULL;
    create_lock(&ctx->lock);
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_lock(&ctx->lock);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    (void) ctx;
    return 0;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
void futhark_context_pause_profiling(struct futhark_context *ctx)
{
    (void) ctx;
}
void futhark_context_unpause_profiling(struct futhark_context *ctx)
{
    (void) ctx;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "default space",
              ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
void futhark_debugging_report(struct futhark_context *ctx)
{
    if (ctx->detail_memory || ctx->profiling) {
        fprintf(stderr, "Peak memory usage for default space: %lld bytes.\n",
                (long long) ctx->peak_mem_usage_default);
    }
    if (ctx->profiling) { }
}
static int futrts_dot(struct futhark_context *ctx,
                      struct memblock *out_mem_p_10082,
                      int32_t *out_out_arrsizze_10083,
                      int32_t *out_out_arrsizze_10084,
                      struct memblock a_mem_9986, struct memblock b_mem_9987,
                      int32_t sizze_9942, int32_t sizze_9943,
                      int32_t sizze_9944, int32_t sizze_9945);
static int futrts_matsubstract(struct futhark_context *ctx,
                               struct memblock *out_mem_p_10085,
                               int32_t *out_out_arrsizze_10086,
                               int32_t *out_out_arrsizze_10087,
                               struct memblock x_mem_9986,
                               struct memblock y_mem_9987, int32_t sizze_9913,
                               int32_t sizze_9914, int32_t sizze_9915,
                               int32_t sizze_9916);
static int futrts_matadd(struct futhark_context *ctx,
                         struct memblock *out_mem_p_10088,
                         int32_t *out_out_arrsizze_10089,
                         int32_t *out_out_arrsizze_10090,
                         struct memblock x_mem_9986, struct memblock y_mem_9987,
                         int32_t sizze_9884, int32_t sizze_9885,
                         int32_t sizze_9886, int32_t sizze_9887);
static int futrts_matmultiply(struct futhark_context *ctx,
                              struct memblock *out_mem_p_10091,
                              int32_t *out_out_arrsizze_10092,
                              int32_t *out_out_arrsizze_10093,
                              struct memblock x_mem_9986,
                              struct memblock y_mem_9987, int32_t sizze_9855,
                              int32_t sizze_9856, int32_t sizze_9857,
                              int32_t sizze_9858);
static int futrts_lvecmul(struct futhark_context *ctx,
                          struct memblock *out_mem_p_10094,
                          int32_t *out_out_arrsizze_10095,
                          struct memblock u_mem_9986,
                          struct memblock b_mem_9987, int32_t sizze_9833,
                          int32_t sizze_9834, int32_t sizze_9835);
static int futrts_sigmoid(struct futhark_context *ctx,
                          struct memblock *out_mem_p_10096,
                          int32_t *out_out_arrsizze_10097,
                          int32_t *out_out_arrsizze_10098,
                          struct memblock x_mem_9986, int32_t sizze_9822,
                          int32_t sizze_9823);
static int futrts_lmatsubstract(struct futhark_context *ctx,
                                struct memblock *out_mem_p_10099,
                                int32_t *out_out_arrsizze_10100,
                                int32_t *out_out_arrsizze_10101,
                                struct memblock x_mem_9986, int32_t sizze_9813,
                                int32_t sizze_9814, float d_9815);
static int futrts_lmatadd(struct futhark_context *ctx,
                          struct memblock *out_mem_p_10102,
                          int32_t *out_out_arrsizze_10103,
                          int32_t *out_out_arrsizze_10104,
                          struct memblock x_mem_9986, int32_t sizze_9804,
                          int32_t sizze_9805, float s_9806);
static int futrts_substract2(struct futhark_context *ctx,
                             struct memblock *out_mem_p_10105,
                             int32_t *out_out_arrsizze_10106,
                             struct memblock x_mem_9986,
                             struct memblock y_mem_9987, int32_t sizze_9789,
                             int32_t sizze_9790);
static int futrts_add2(struct futhark_context *ctx,
                       struct memblock *out_mem_p_10107,
                       int32_t *out_out_arrsizze_10108,
                       struct memblock x_mem_9986, struct memblock y_mem_9987,
                       int32_t sizze_9774, int32_t sizze_9775);
static int futrts_lmatmultiply(struct futhark_context *ctx,
                               struct memblock *out_mem_p_10109,
                               int32_t *out_out_arrsizze_10110,
                               int32_t *out_out_arrsizze_10111,
                               struct memblock x_mem_9986, int32_t sizze_9765,
                               int32_t sizze_9766, float p_9767);
static int futrts_multiply2(struct futhark_context *ctx,
                            struct memblock *out_mem_p_10112,
                            int32_t *out_out_arrsizze_10113,
                            struct memblock x_mem_9986,
                            struct memblock y_mem_9987, int32_t sizze_9750,
                            int32_t sizze_9751);
static int futrts_substract(struct futhark_context *ctx,
                            struct memblock *out_mem_p_10114,
                            int32_t *out_out_arrsizze_10115,
                            struct memblock x_mem_9986, int32_t sizze_9744,
                            float d_9745);
static int futrts_add(struct futhark_context *ctx,
                      struct memblock *out_mem_p_10116,
                      int32_t *out_out_arrsizze_10117,
                      struct memblock x_mem_9986, int32_t sizze_9738,
                      float s_9739);
static int futrts_multiply(struct futhark_context *ctx,
                           struct memblock *out_mem_p_10118,
                           int32_t *out_out_arrsizze_10119,
                           struct memblock x_mem_9986, int32_t sizze_9732,
                           float m_9733);
static int futrts_divide(struct futhark_context *ctx,
                         struct memblock *out_mem_p_10120,
                         int32_t *out_out_arrsizze_10121,
                         struct memblock x_mem_9986, int32_t sizze_9726,
                         float d_9727);
static int futrts_negation(struct futhark_context *ctx,
                           struct memblock *out_mem_p_10122,
                           int32_t *out_out_arrsizze_10123,
                           struct memblock x_mem_9986, int32_t sizze_9721);
static int futrts_exp(struct futhark_context *ctx,
                      struct memblock *out_mem_p_10124,
                      int32_t *out_out_arrsizze_10125,
                      struct memblock x_mem_9986, int32_t sizze_9716);
static int futrts_transp(struct futhark_context *ctx,
                         struct memblock *out_mem_p_10126,
                         int32_t *out_out_arrsizze_10127,
                         int32_t *out_out_arrsizze_10128,
                         struct memblock x_mem_9986, int32_t sizze_9712,
                         int32_t sizze_9713);
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline bool ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline bool ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline bool ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline bool ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline bool ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline bool ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline bool ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline bool ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline bool slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline bool slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline bool slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline bool slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline bool sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline bool sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline bool sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline bool sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((uint8_t) (uint8_t) x)
#define zext_i8_i16(x) ((uint16_t) (uint8_t) x)
#define zext_i8_i32(x) ((uint32_t) (uint8_t) x)
#define zext_i8_i64(x) ((uint64_t) (uint8_t) x)
#define zext_i16_i8(x) ((uint8_t) (uint16_t) x)
#define zext_i16_i16(x) ((uint16_t) (uint16_t) x)
#define zext_i16_i32(x) ((uint32_t) (uint16_t) x)
#define zext_i16_i64(x) ((uint64_t) (uint16_t) x)
#define zext_i32_i8(x) ((uint8_t) (uint32_t) x)
#define zext_i32_i16(x) ((uint16_t) (uint32_t) x)
#define zext_i32_i32(x) ((uint32_t) (uint32_t) x)
#define zext_i32_i64(x) ((uint64_t) (uint32_t) x)
#define zext_i64_i8(x) ((uint8_t) (uint64_t) x)
#define zext_i64_i16(x) ((uint16_t) (uint64_t) x)
#define zext_i64_i32(x) ((uint32_t) (uint64_t) x)
#define zext_i64_i64(x) ((uint64_t) (uint64_t) x)
#if defined(__OPENCL_VERSION__)
int32_t futrts_popc8(int8_t x)
{
    return popcount(x);
}
int32_t futrts_popc16(int16_t x)
{
    return popcount(x);
}
int32_t futrts_popc32(int32_t x)
{
    return popcount(x);
}
int32_t futrts_popc64(int64_t x)
{
    return popcount(x);
}
#elif defined(__CUDA_ARCH__)
int32_t futrts_popc8(int8_t x)
{
    return __popc(zext_i8_i32(x));
}
int32_t futrts_popc16(int16_t x)
{
    return __popc(zext_i16_i32(x));
}
int32_t futrts_popc32(int32_t x)
{
    return __popc(x);
}
int32_t futrts_popc64(int64_t x)
{
    return __popcll(x);
}
#else
int32_t futrts_popc8(int8_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
int32_t futrts_popc16(int16_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
int32_t futrts_popc32(int32_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
int32_t futrts_popc64(int64_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
#endif
#if defined(__OPENCL_VERSION__)
int32_t futrts_clzz8(int8_t x)
{
    return clz(x);
}
int32_t futrts_clzz16(int16_t x)
{
    return clz(x);
}
int32_t futrts_clzz32(int32_t x)
{
    return clz(x);
}
int32_t futrts_clzz64(int64_t x)
{
    return clz(x);
}
#elif defined(__CUDA_ARCH__)
int32_t futrts_clzz8(int8_t x)
{
    return __clz(zext_i8_i32(x)) - 24;
}
int32_t futrts_clzz16(int16_t x)
{
    return __clz(zext_i16_i32(x)) - 16;
}
int32_t futrts_clzz32(int32_t x)
{
    return __clz(x);
}
int32_t futrts_clzz64(int64_t x)
{
    return __clzll(x);
}
#else
int32_t futrts_clzz8(int8_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
int32_t futrts_clzz16(int16_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
int32_t futrts_clzz32(int32_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
int32_t futrts_clzz64(int64_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
#endif
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return fmin(x, y);
}
static inline float fmax32(float x, float y)
{
    return fmax(x, y);
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline bool cmplt32(float x, float y)
{
    return x < y;
}
static inline bool cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return (float) x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return (float) x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return (float) x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return (float) x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return (float) x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return (float) x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return (float) x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return (float) x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return (uint64_t) x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return fmin(x, y);
}
static inline double fmax64(double x, double y)
{
    return fmax(x, y);
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline bool cmplt64(double x, double y)
{
    return x < y;
}
static inline bool cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return (double) x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return (double) x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return (double) x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return (double) x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return (double) x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return (double) x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return (double) x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return (double) x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return (uint64_t) x;
}
static inline float fpconv_f32_f32(float x)
{
    return (float) x;
}
static inline double fpconv_f32_f64(float x)
{
    return (double) x;
}
static inline float fpconv_f64_f32(double x)
{
    return (float) x;
}
static inline double fpconv_f64_f64(double x)
{
    return (double) x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline bool futrts_isnan32(float x)
{
    return isnan(x);
}
static inline bool futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
#ifdef __OPENCL_VERSION__
static inline float fmod32(float x, float y)
{
    return fmod(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline float futrts_floor32(float x)
{
    return floor(x);
}
static inline float futrts_ceil32(float x)
{
    return ceil(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return mix(v0, v1, t);
}
#else
static inline float fmod32(float x, float y)
{
    return fmodf(x, y);
}
static inline float futrts_round32(float x)
{
    return rintf(x);
}
static inline float futrts_floor32(float x)
{
    return floorf(x);
}
static inline float futrts_ceil32(float x)
{
    return ceilf(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return v0 + (v1 - v0) * t;
}
#endif
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_gamma64(double x)
{
    return tgamma(x);
}
static inline double futrts_lgamma64(double x)
{
    return lgamma(x);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline double futrts_ceil64(double x)
{
    return ceil(x);
}
static inline double futrts_floor64(double x)
{
    return floor(x);
}
static inline bool futrts_isnan64(double x)
{
    return isnan(x);
}
static inline bool futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float fmod64(float x, float y)
{
    return fmod(x, y);
}
#ifdef __OPENCL_VERSION__
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return mix(v0, v1, t);
}
#else
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return v0 + (v1 - v0) * t;
}
#endif
static int futrts_dot(struct futhark_context *ctx,
                      struct memblock *out_mem_p_10082,
                      int32_t *out_out_arrsizze_10083,
                      int32_t *out_out_arrsizze_10084,
                      struct memblock a_mem_9986, struct memblock b_mem_9987,
                      int32_t sizze_9942, int32_t sizze_9943,
                      int32_t sizze_9944, int32_t sizze_9945)
{
    struct memblock out_mem_10076;
    
    out_mem_10076.references = NULL;
    
    int32_t out_arrsizze_10077;
    int32_t out_arrsizze_10078;
    bool dim_zzero_9949 = 0 == sizze_9944;
    bool dim_zzero_9950 = 0 == sizze_9943;
    bool both_empty_9951 = dim_zzero_9949 && dim_zzero_9950;
    bool dim_match_9952 = sizze_9943 == sizze_9944;
    bool empty_or_match_9953 = both_empty_9951 || dim_match_9952;
    bool empty_or_match_cert_9954;
    
    if (!empty_or_match_9953) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  GPU.fut:10:19-30\n   #1  GPU.fut:15:21-38\n   #2  GPU.fut:15:3-16:17\n   #3  GPU.fut:25:18-33\n   #4  GPU.fut:25:3-36\n   #5  GPU.fut:24:1-25:36\n");
        if (memblock_unref(ctx, &out_mem_10076, "out_mem_10076") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_9989 = sext_i32_i64(sizze_9942);
    int64_t binop_y_9990 = sext_i32_i64(sizze_9945);
    int64_t binop_x_9991 = binop_x_9989 * binop_y_9990;
    int64_t bytes_9988 = 4 * binop_x_9991;
    struct memblock mem_9992;
    
    mem_9992.references = NULL;
    if (memblock_alloc(ctx, &mem_9992, bytes_9988, "mem_9992"))
        return 1;
    for (int32_t i_9975 = 0; i_9975 < sizze_9942; i_9975++) {
        for (int32_t i_9971 = 0; i_9971 < sizze_9945; i_9971++) {
            float res_9960;
            float redout_9967 = 0.0F;
            
            for (int32_t i_9968 = 0; i_9968 < sizze_9943; i_9968++) {
                float x_9964 = ((float *) a_mem_9986.mem)[i_9975 * sizze_9943 +
                                                          i_9968];
                float x_9965 = ((float *) b_mem_9987.mem)[i_9968 * sizze_9945 +
                                                          i_9971];
                float res_9966 = x_9964 * x_9965;
                float res_9963 = res_9966 + redout_9967;
                float redout_tmp_10081 = res_9963;
                
                redout_9967 = redout_tmp_10081;
            }
            res_9960 = redout_9967;
            ((float *) mem_9992.mem)[i_9975 * sizze_9945 + i_9971] = res_9960;
        }
    }
    out_arrsizze_10077 = sizze_9942;
    out_arrsizze_10078 = sizze_9945;
    if (memblock_set(ctx, &out_mem_10076, &mem_9992, "mem_9992") != 0)
        return 1;
    (*out_mem_p_10082).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10082, &out_mem_10076, "out_mem_10076") !=
        0)
        return 1;
    *out_out_arrsizze_10083 = out_arrsizze_10077;
    *out_out_arrsizze_10084 = out_arrsizze_10078;
    if (memblock_unref(ctx, &mem_9992, "mem_9992") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10076, "out_mem_10076") != 0)
        return 1;
    return 0;
}
static int futrts_matsubstract(struct futhark_context *ctx,
                               struct memblock *out_mem_p_10085,
                               int32_t *out_out_arrsizze_10086,
                               int32_t *out_out_arrsizze_10087,
                               struct memblock x_mem_9986,
                               struct memblock y_mem_9987, int32_t sizze_9913,
                               int32_t sizze_9914, int32_t sizze_9915,
                               int32_t sizze_9916)
{
    struct memblock out_mem_10071;
    
    out_mem_10071.references = NULL;
    
    int32_t out_arrsizze_10072;
    int32_t out_arrsizze_10073;
    bool dim_zzero_9919 = 0 == sizze_9915;
    bool dim_zzero_9920 = 0 == sizze_9916;
    bool old_empty_9921 = dim_zzero_9919 || dim_zzero_9920;
    bool dim_zzero_9922 = 0 == sizze_9913;
    bool new_empty_9923 = dim_zzero_9920 || dim_zzero_9922;
    bool both_empty_9924 = old_empty_9921 && new_empty_9923;
    bool dim_match_9925 = sizze_9913 == sizze_9915;
    bool empty_or_match_9926 = both_empty_9924 || dim_match_9925;
    bool empty_or_match_cert_9927;
    
    if (!empty_or_match_9926) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  GPU.fut:150:3-21\n   #1  GPU.fut:149:1-150:21\n");
        if (memblock_unref(ctx, &out_mem_10071, "out_mem_10071") != 0)
            return 1;
        return 1;
    }
    
    bool dim_zzero_9929 = 0 == sizze_9914;
    bool both_empty_9930 = dim_zzero_9920 && dim_zzero_9929;
    bool dim_match_9931 = sizze_9914 == sizze_9916;
    bool empty_or_match_9932 = both_empty_9930 || dim_match_9931;
    bool empty_or_match_cert_9933;
    
    if (!empty_or_match_9932) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  GPU.fut:141:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPU.fut:150:3-21\n   #4  GPU.fut:149:1-150:21\n");
        if (memblock_unref(ctx, &out_mem_10071, "out_mem_10071") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_9989 = sext_i32_i64(sizze_9913);
    int64_t binop_y_9990 = sext_i32_i64(sizze_9914);
    int64_t binop_x_9991 = binop_x_9989 * binop_y_9990;
    int64_t bytes_9988 = 4 * binop_x_9991;
    struct memblock mem_9992;
    
    mem_9992.references = NULL;
    if (memblock_alloc(ctx, &mem_9992, bytes_9988, "mem_9992"))
        return 1;
    for (int32_t i_9973 = 0; i_9973 < sizze_9913; i_9973++) {
        for (int32_t i_9969 = 0; i_9969 < sizze_9914; i_9969++) {
            float x_9939 = ((float *) x_mem_9986.mem)[i_9973 * sizze_9914 +
                                                      i_9969];
            float x_9940 = ((float *) y_mem_9987.mem)[i_9973 * sizze_9916 +
                                                      i_9969];
            float res_9941 = x_9939 - x_9940;
            
            ((float *) mem_9992.mem)[i_9973 * sizze_9914 + i_9969] = res_9941;
        }
    }
    out_arrsizze_10072 = sizze_9913;
    out_arrsizze_10073 = sizze_9914;
    if (memblock_set(ctx, &out_mem_10071, &mem_9992, "mem_9992") != 0)
        return 1;
    (*out_mem_p_10085).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10085, &out_mem_10071, "out_mem_10071") !=
        0)
        return 1;
    *out_out_arrsizze_10086 = out_arrsizze_10072;
    *out_out_arrsizze_10087 = out_arrsizze_10073;
    if (memblock_unref(ctx, &mem_9992, "mem_9992") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10071, "out_mem_10071") != 0)
        return 1;
    return 0;
}
static int futrts_matadd(struct futhark_context *ctx,
                         struct memblock *out_mem_p_10088,
                         int32_t *out_out_arrsizze_10089,
                         int32_t *out_out_arrsizze_10090,
                         struct memblock x_mem_9986, struct memblock y_mem_9987,
                         int32_t sizze_9884, int32_t sizze_9885,
                         int32_t sizze_9886, int32_t sizze_9887)
{
    struct memblock out_mem_10066;
    
    out_mem_10066.references = NULL;
    
    int32_t out_arrsizze_10067;
    int32_t out_arrsizze_10068;
    bool dim_zzero_9890 = 0 == sizze_9886;
    bool dim_zzero_9891 = 0 == sizze_9887;
    bool old_empty_9892 = dim_zzero_9890 || dim_zzero_9891;
    bool dim_zzero_9893 = 0 == sizze_9884;
    bool new_empty_9894 = dim_zzero_9891 || dim_zzero_9893;
    bool both_empty_9895 = old_empty_9892 && new_empty_9894;
    bool dim_match_9896 = sizze_9884 == sizze_9886;
    bool empty_or_match_9897 = both_empty_9895 || dim_match_9896;
    bool empty_or_match_cert_9898;
    
    if (!empty_or_match_9897) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  GPU.fut:114:3-15\n   #1  GPU.fut:113:1-114:15\n");
        if (memblock_unref(ctx, &out_mem_10066, "out_mem_10066") != 0)
            return 1;
        return 1;
    }
    
    bool dim_zzero_9900 = 0 == sizze_9885;
    bool both_empty_9901 = dim_zzero_9891 && dim_zzero_9900;
    bool dim_match_9902 = sizze_9885 == sizze_9887;
    bool empty_or_match_9903 = both_empty_9901 || dim_match_9902;
    bool empty_or_match_cert_9904;
    
    if (!empty_or_match_9903) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  GPU.fut:105:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPU.fut:114:3-15\n   #4  GPU.fut:113:1-114:15\n");
        if (memblock_unref(ctx, &out_mem_10066, "out_mem_10066") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_9989 = sext_i32_i64(sizze_9884);
    int64_t binop_y_9990 = sext_i32_i64(sizze_9885);
    int64_t binop_x_9991 = binop_x_9989 * binop_y_9990;
    int64_t bytes_9988 = 4 * binop_x_9991;
    struct memblock mem_9992;
    
    mem_9992.references = NULL;
    if (memblock_alloc(ctx, &mem_9992, bytes_9988, "mem_9992"))
        return 1;
    for (int32_t i_9973 = 0; i_9973 < sizze_9884; i_9973++) {
        for (int32_t i_9969 = 0; i_9969 < sizze_9885; i_9969++) {
            float x_9910 = ((float *) x_mem_9986.mem)[i_9973 * sizze_9885 +
                                                      i_9969];
            float x_9911 = ((float *) y_mem_9987.mem)[i_9973 * sizze_9887 +
                                                      i_9969];
            float res_9912 = x_9910 + x_9911;
            
            ((float *) mem_9992.mem)[i_9973 * sizze_9885 + i_9969] = res_9912;
        }
    }
    out_arrsizze_10067 = sizze_9884;
    out_arrsizze_10068 = sizze_9885;
    if (memblock_set(ctx, &out_mem_10066, &mem_9992, "mem_9992") != 0)
        return 1;
    (*out_mem_p_10088).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10088, &out_mem_10066, "out_mem_10066") !=
        0)
        return 1;
    *out_out_arrsizze_10089 = out_arrsizze_10067;
    *out_out_arrsizze_10090 = out_arrsizze_10068;
    if (memblock_unref(ctx, &mem_9992, "mem_9992") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10066, "out_mem_10066") != 0)
        return 1;
    return 0;
}
static int futrts_matmultiply(struct futhark_context *ctx,
                              struct memblock *out_mem_p_10091,
                              int32_t *out_out_arrsizze_10092,
                              int32_t *out_out_arrsizze_10093,
                              struct memblock x_mem_9986,
                              struct memblock y_mem_9987, int32_t sizze_9855,
                              int32_t sizze_9856, int32_t sizze_9857,
                              int32_t sizze_9858)
{
    struct memblock out_mem_10061;
    
    out_mem_10061.references = NULL;
    
    int32_t out_arrsizze_10062;
    int32_t out_arrsizze_10063;
    bool dim_zzero_9861 = 0 == sizze_9857;
    bool dim_zzero_9862 = 0 == sizze_9858;
    bool old_empty_9863 = dim_zzero_9861 || dim_zzero_9862;
    bool dim_zzero_9864 = 0 == sizze_9855;
    bool new_empty_9865 = dim_zzero_9862 || dim_zzero_9864;
    bool both_empty_9866 = old_empty_9863 && new_empty_9865;
    bool dim_match_9867 = sizze_9855 == sizze_9857;
    bool empty_or_match_9868 = both_empty_9866 || dim_match_9867;
    bool empty_or_match_cert_9869;
    
    if (!empty_or_match_9868) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  GPU.fut:80:3-20\n   #1  GPU.fut:79:1-80:20\n");
        if (memblock_unref(ctx, &out_mem_10061, "out_mem_10061") != 0)
            return 1;
        return 1;
    }
    
    bool dim_zzero_9871 = 0 == sizze_9856;
    bool both_empty_9872 = dim_zzero_9862 && dim_zzero_9871;
    bool dim_match_9873 = sizze_9856 == sizze_9858;
    bool empty_or_match_9874 = both_empty_9872 || dim_match_9873;
    bool empty_or_match_cert_9875;
    
    if (!empty_or_match_9874) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  GPU.fut:71:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPU.fut:80:3-20\n   #4  GPU.fut:79:1-80:20\n");
        if (memblock_unref(ctx, &out_mem_10061, "out_mem_10061") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_9989 = sext_i32_i64(sizze_9855);
    int64_t binop_y_9990 = sext_i32_i64(sizze_9856);
    int64_t binop_x_9991 = binop_x_9989 * binop_y_9990;
    int64_t bytes_9988 = 4 * binop_x_9991;
    struct memblock mem_9992;
    
    mem_9992.references = NULL;
    if (memblock_alloc(ctx, &mem_9992, bytes_9988, "mem_9992"))
        return 1;
    for (int32_t i_9973 = 0; i_9973 < sizze_9855; i_9973++) {
        for (int32_t i_9969 = 0; i_9969 < sizze_9856; i_9969++) {
            float x_9881 = ((float *) x_mem_9986.mem)[i_9973 * sizze_9856 +
                                                      i_9969];
            float x_9882 = ((float *) y_mem_9987.mem)[i_9973 * sizze_9858 +
                                                      i_9969];
            float res_9883 = x_9881 * x_9882;
            
            ((float *) mem_9992.mem)[i_9973 * sizze_9856 + i_9969] = res_9883;
        }
    }
    out_arrsizze_10062 = sizze_9855;
    out_arrsizze_10063 = sizze_9856;
    if (memblock_set(ctx, &out_mem_10061, &mem_9992, "mem_9992") != 0)
        return 1;
    (*out_mem_p_10091).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10091, &out_mem_10061, "out_mem_10061") !=
        0)
        return 1;
    *out_out_arrsizze_10092 = out_arrsizze_10062;
    *out_out_arrsizze_10093 = out_arrsizze_10063;
    if (memblock_unref(ctx, &mem_9992, "mem_9992") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10061, "out_mem_10061") != 0)
        return 1;
    return 0;
}
static int futrts_lvecmul(struct futhark_context *ctx,
                          struct memblock *out_mem_p_10094,
                          int32_t *out_out_arrsizze_10095,
                          struct memblock u_mem_9986,
                          struct memblock b_mem_9987, int32_t sizze_9833,
                          int32_t sizze_9834, int32_t sizze_9835)
{
    struct memblock out_mem_10057;
    
    out_mem_10057.references = NULL;
    
    int32_t out_arrsizze_10058;
    bool dim_zzero_9839 = 0 == sizze_9834;
    bool dim_zzero_9840 = 0 == sizze_9833;
    bool both_empty_9841 = dim_zzero_9839 && dim_zzero_9840;
    bool dim_match_9842 = sizze_9833 == sizze_9834;
    bool empty_or_match_9843 = both_empty_9841 || dim_match_9842;
    bool empty_or_match_cert_9844;
    
    if (!empty_or_match_9843) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  GPU.fut:10:19-30\n   #1  GPU.fut:15:21-38\n   #2  GPU.fut:15:3-16:17\n   #3  GPU.fut:14:1-16:17\n");
        if (memblock_unref(ctx, &out_mem_10057, "out_mem_10057") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_9989 = sext_i32_i64(sizze_9835);
    int64_t bytes_9988 = 4 * binop_x_9989;
    struct memblock mem_9990;
    
    mem_9990.references = NULL;
    if (memblock_alloc(ctx, &mem_9990, bytes_9988, "mem_9990"))
        return 1;
    for (int32_t i_9971 = 0; i_9971 < sizze_9835; i_9971++) {
        float res_9848;
        float redout_9967 = 0.0F;
        
        for (int32_t i_9968 = 0; i_9968 < sizze_9833; i_9968++) {
            float x_9852 = ((float *) u_mem_9986.mem)[i_9968];
            float x_9853 = ((float *) b_mem_9987.mem)[i_9968 * sizze_9835 +
                                                      i_9971];
            float res_9854 = x_9852 * x_9853;
            float res_9851 = res_9854 + redout_9967;
            float redout_tmp_10060 = res_9851;
            
            redout_9967 = redout_tmp_10060;
        }
        res_9848 = redout_9967;
        ((float *) mem_9990.mem)[i_9971] = res_9848;
    }
    out_arrsizze_10058 = sizze_9835;
    if (memblock_set(ctx, &out_mem_10057, &mem_9990, "mem_9990") != 0)
        return 1;
    (*out_mem_p_10094).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10094, &out_mem_10057, "out_mem_10057") !=
        0)
        return 1;
    *out_out_arrsizze_10095 = out_arrsizze_10058;
    if (memblock_unref(ctx, &mem_9990, "mem_9990") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10057, "out_mem_10057") != 0)
        return 1;
    return 0;
}
static int futrts_sigmoid(struct futhark_context *ctx,
                          struct memblock *out_mem_p_10096,
                          int32_t *out_out_arrsizze_10097,
                          int32_t *out_out_arrsizze_10098,
                          struct memblock x_mem_9986, int32_t sizze_9822,
                          int32_t sizze_9823)
{
    struct memblock out_mem_10052;
    
    out_mem_10052.references = NULL;
    
    int32_t out_arrsizze_10053;
    int32_t out_arrsizze_10054;
    int64_t binop_x_9988 = sext_i32_i64(sizze_9822);
    int64_t binop_y_9989 = sext_i32_i64(sizze_9823);
    int64_t binop_x_9990 = binop_x_9988 * binop_y_9989;
    int64_t bytes_9987 = 4 * binop_x_9990;
    struct memblock mem_9991;
    
    mem_9991.references = NULL;
    if (memblock_alloc(ctx, &mem_9991, bytes_9987, "mem_9991"))
        return 1;
    for (int32_t i_9973 = 0; i_9973 < sizze_9822; i_9973++) {
        for (int32_t i_9969 = 0; i_9969 < sizze_9823; i_9969++) {
            float x_9828 = ((float *) x_mem_9986.mem)[i_9973 * sizze_9823 +
                                                      i_9969];
            float res_9829 = 0.0F - x_9828;
            float res_9830 = fpow32(2.7182817F, res_9829);
            float res_9831 = 1.0F + res_9830;
            float res_9832 = 1.0F / res_9831;
            
            ((float *) mem_9991.mem)[i_9973 * sizze_9823 + i_9969] = res_9832;
        }
    }
    out_arrsizze_10053 = sizze_9822;
    out_arrsizze_10054 = sizze_9823;
    if (memblock_set(ctx, &out_mem_10052, &mem_9991, "mem_9991") != 0)
        return 1;
    (*out_mem_p_10096).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10096, &out_mem_10052, "out_mem_10052") !=
        0)
        return 1;
    *out_out_arrsizze_10097 = out_arrsizze_10053;
    *out_out_arrsizze_10098 = out_arrsizze_10054;
    if (memblock_unref(ctx, &mem_9991, "mem_9991") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10052, "out_mem_10052") != 0)
        return 1;
    return 0;
}
static int futrts_lmatsubstract(struct futhark_context *ctx,
                                struct memblock *out_mem_p_10099,
                                int32_t *out_out_arrsizze_10100,
                                int32_t *out_out_arrsizze_10101,
                                struct memblock x_mem_9986, int32_t sizze_9813,
                                int32_t sizze_9814, float d_9815)
{
    struct memblock out_mem_10047;
    
    out_mem_10047.references = NULL;
    
    int32_t out_arrsizze_10048;
    int32_t out_arrsizze_10049;
    int64_t binop_x_9988 = sext_i32_i64(sizze_9813);
    int64_t binop_y_9989 = sext_i32_i64(sizze_9814);
    int64_t binop_x_9990 = binop_x_9988 * binop_y_9989;
    int64_t bytes_9987 = 4 * binop_x_9990;
    struct memblock mem_9991;
    
    mem_9991.references = NULL;
    if (memblock_alloc(ctx, &mem_9991, bytes_9987, "mem_9991"))
        return 1;
    for (int32_t i_9973 = 0; i_9973 < sizze_9813; i_9973++) {
        for (int32_t i_9969 = 0; i_9969 < sizze_9814; i_9969++) {
            float x_9820 = ((float *) x_mem_9986.mem)[i_9973 * sizze_9814 +
                                                      i_9969];
            float res_9821 = d_9815 - x_9820;
            
            ((float *) mem_9991.mem)[i_9973 * sizze_9814 + i_9969] = res_9821;
        }
    }
    out_arrsizze_10048 = sizze_9813;
    out_arrsizze_10049 = sizze_9814;
    if (memblock_set(ctx, &out_mem_10047, &mem_9991, "mem_9991") != 0)
        return 1;
    (*out_mem_p_10099).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10099, &out_mem_10047, "out_mem_10047") !=
        0)
        return 1;
    *out_out_arrsizze_10100 = out_arrsizze_10048;
    *out_out_arrsizze_10101 = out_arrsizze_10049;
    if (memblock_unref(ctx, &mem_9991, "mem_9991") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10047, "out_mem_10047") != 0)
        return 1;
    return 0;
}
static int futrts_lmatadd(struct futhark_context *ctx,
                          struct memblock *out_mem_p_10102,
                          int32_t *out_out_arrsizze_10103,
                          int32_t *out_out_arrsizze_10104,
                          struct memblock x_mem_9986, int32_t sizze_9804,
                          int32_t sizze_9805, float s_9806)
{
    struct memblock out_mem_10042;
    
    out_mem_10042.references = NULL;
    
    int32_t out_arrsizze_10043;
    int32_t out_arrsizze_10044;
    int64_t binop_x_9988 = sext_i32_i64(sizze_9804);
    int64_t binop_y_9989 = sext_i32_i64(sizze_9805);
    int64_t binop_x_9990 = binop_x_9988 * binop_y_9989;
    int64_t bytes_9987 = 4 * binop_x_9990;
    struct memblock mem_9991;
    
    mem_9991.references = NULL;
    if (memblock_alloc(ctx, &mem_9991, bytes_9987, "mem_9991"))
        return 1;
    for (int32_t i_9973 = 0; i_9973 < sizze_9804; i_9973++) {
        for (int32_t i_9969 = 0; i_9969 < sizze_9805; i_9969++) {
            float x_9811 = ((float *) x_mem_9986.mem)[i_9973 * sizze_9805 +
                                                      i_9969];
            float res_9812 = s_9806 + x_9811;
            
            ((float *) mem_9991.mem)[i_9973 * sizze_9805 + i_9969] = res_9812;
        }
    }
    out_arrsizze_10043 = sizze_9804;
    out_arrsizze_10044 = sizze_9805;
    if (memblock_set(ctx, &out_mem_10042, &mem_9991, "mem_9991") != 0)
        return 1;
    (*out_mem_p_10102).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10102, &out_mem_10042, "out_mem_10042") !=
        0)
        return 1;
    *out_out_arrsizze_10103 = out_arrsizze_10043;
    *out_out_arrsizze_10104 = out_arrsizze_10044;
    if (memblock_unref(ctx, &mem_9991, "mem_9991") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10042, "out_mem_10042") != 0)
        return 1;
    return 0;
}
static int futrts_substract2(struct futhark_context *ctx,
                             struct memblock *out_mem_p_10105,
                             int32_t *out_out_arrsizze_10106,
                             struct memblock x_mem_9986,
                             struct memblock y_mem_9987, int32_t sizze_9789,
                             int32_t sizze_9790)
{
    struct memblock out_mem_10039;
    
    out_mem_10039.references = NULL;
    
    int32_t out_arrsizze_10040;
    bool dim_zzero_9793 = 0 == sizze_9790;
    bool dim_zzero_9794 = 0 == sizze_9789;
    bool both_empty_9795 = dim_zzero_9793 && dim_zzero_9794;
    bool dim_match_9796 = sizze_9789 == sizze_9790;
    bool empty_or_match_9797 = both_empty_9795 || dim_match_9796;
    bool empty_or_match_cert_9798;
    
    if (!empty_or_match_9797) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  GPU.fut:141:3-14\n   #1  GPU.fut:140:1-141:14\n");
        if (memblock_unref(ctx, &out_mem_10039, "out_mem_10039") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_9989 = sext_i32_i64(sizze_9789);
    int64_t bytes_9988 = 4 * binop_x_9989;
    struct memblock mem_9990;
    
    mem_9990.references = NULL;
    if (memblock_alloc(ctx, &mem_9990, bytes_9988, "mem_9990"))
        return 1;
    for (int32_t i_9969 = 0; i_9969 < sizze_9789; i_9969++) {
        float x_9801 = ((float *) x_mem_9986.mem)[i_9969];
        float x_9802 = ((float *) y_mem_9987.mem)[i_9969];
        float res_9803 = x_9801 - x_9802;
        
        ((float *) mem_9990.mem)[i_9969] = res_9803;
    }
    out_arrsizze_10040 = sizze_9789;
    if (memblock_set(ctx, &out_mem_10039, &mem_9990, "mem_9990") != 0)
        return 1;
    (*out_mem_p_10105).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10105, &out_mem_10039, "out_mem_10039") !=
        0)
        return 1;
    *out_out_arrsizze_10106 = out_arrsizze_10040;
    if (memblock_unref(ctx, &mem_9990, "mem_9990") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10039, "out_mem_10039") != 0)
        return 1;
    return 0;
}
static int futrts_add2(struct futhark_context *ctx,
                       struct memblock *out_mem_p_10107,
                       int32_t *out_out_arrsizze_10108,
                       struct memblock x_mem_9986, struct memblock y_mem_9987,
                       int32_t sizze_9774, int32_t sizze_9775)
{
    struct memblock out_mem_10036;
    
    out_mem_10036.references = NULL;
    
    int32_t out_arrsizze_10037;
    bool dim_zzero_9778 = 0 == sizze_9775;
    bool dim_zzero_9779 = 0 == sizze_9774;
    bool both_empty_9780 = dim_zzero_9778 && dim_zzero_9779;
    bool dim_match_9781 = sizze_9774 == sizze_9775;
    bool empty_or_match_9782 = both_empty_9780 || dim_match_9781;
    bool empty_or_match_cert_9783;
    
    if (!empty_or_match_9782) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  GPU.fut:105:3-14\n   #1  GPU.fut:104:1-105:14\n");
        if (memblock_unref(ctx, &out_mem_10036, "out_mem_10036") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_9989 = sext_i32_i64(sizze_9774);
    int64_t bytes_9988 = 4 * binop_x_9989;
    struct memblock mem_9990;
    
    mem_9990.references = NULL;
    if (memblock_alloc(ctx, &mem_9990, bytes_9988, "mem_9990"))
        return 1;
    for (int32_t i_9969 = 0; i_9969 < sizze_9774; i_9969++) {
        float x_9786 = ((float *) x_mem_9986.mem)[i_9969];
        float x_9787 = ((float *) y_mem_9987.mem)[i_9969];
        float res_9788 = x_9786 + x_9787;
        
        ((float *) mem_9990.mem)[i_9969] = res_9788;
    }
    out_arrsizze_10037 = sizze_9774;
    if (memblock_set(ctx, &out_mem_10036, &mem_9990, "mem_9990") != 0)
        return 1;
    (*out_mem_p_10107).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10107, &out_mem_10036, "out_mem_10036") !=
        0)
        return 1;
    *out_out_arrsizze_10108 = out_arrsizze_10037;
    if (memblock_unref(ctx, &mem_9990, "mem_9990") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10036, "out_mem_10036") != 0)
        return 1;
    return 0;
}
static int futrts_lmatmultiply(struct futhark_context *ctx,
                               struct memblock *out_mem_p_10109,
                               int32_t *out_out_arrsizze_10110,
                               int32_t *out_out_arrsizze_10111,
                               struct memblock x_mem_9986, int32_t sizze_9765,
                               int32_t sizze_9766, float p_9767)
{
    struct memblock out_mem_10031;
    
    out_mem_10031.references = NULL;
    
    int32_t out_arrsizze_10032;
    int32_t out_arrsizze_10033;
    int64_t binop_x_9988 = sext_i32_i64(sizze_9765);
    int64_t binop_y_9989 = sext_i32_i64(sizze_9766);
    int64_t binop_x_9990 = binop_x_9988 * binop_y_9989;
    int64_t bytes_9987 = 4 * binop_x_9990;
    struct memblock mem_9991;
    
    mem_9991.references = NULL;
    if (memblock_alloc(ctx, &mem_9991, bytes_9987, "mem_9991"))
        return 1;
    for (int32_t i_9973 = 0; i_9973 < sizze_9765; i_9973++) {
        for (int32_t i_9969 = 0; i_9969 < sizze_9766; i_9969++) {
            float x_9772 = ((float *) x_mem_9986.mem)[i_9973 * sizze_9766 +
                                                      i_9969];
            float res_9773 = p_9767 * x_9772;
            
            ((float *) mem_9991.mem)[i_9973 * sizze_9766 + i_9969] = res_9773;
        }
    }
    out_arrsizze_10032 = sizze_9765;
    out_arrsizze_10033 = sizze_9766;
    if (memblock_set(ctx, &out_mem_10031, &mem_9991, "mem_9991") != 0)
        return 1;
    (*out_mem_p_10109).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10109, &out_mem_10031, "out_mem_10031") !=
        0)
        return 1;
    *out_out_arrsizze_10110 = out_arrsizze_10032;
    *out_out_arrsizze_10111 = out_arrsizze_10033;
    if (memblock_unref(ctx, &mem_9991, "mem_9991") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10031, "out_mem_10031") != 0)
        return 1;
    return 0;
}
static int futrts_multiply2(struct futhark_context *ctx,
                            struct memblock *out_mem_p_10112,
                            int32_t *out_out_arrsizze_10113,
                            struct memblock x_mem_9986,
                            struct memblock y_mem_9987, int32_t sizze_9750,
                            int32_t sizze_9751)
{
    struct memblock out_mem_10028;
    
    out_mem_10028.references = NULL;
    
    int32_t out_arrsizze_10029;
    bool dim_zzero_9754 = 0 == sizze_9751;
    bool dim_zzero_9755 = 0 == sizze_9750;
    bool both_empty_9756 = dim_zzero_9754 && dim_zzero_9755;
    bool dim_match_9757 = sizze_9750 == sizze_9751;
    bool empty_or_match_9758 = both_empty_9756 || dim_match_9757;
    bool empty_or_match_cert_9759;
    
    if (!empty_or_match_9758) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  GPU.fut:71:3-14\n   #1  GPU.fut:70:1-71:14\n");
        if (memblock_unref(ctx, &out_mem_10028, "out_mem_10028") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_9989 = sext_i32_i64(sizze_9750);
    int64_t bytes_9988 = 4 * binop_x_9989;
    struct memblock mem_9990;
    
    mem_9990.references = NULL;
    if (memblock_alloc(ctx, &mem_9990, bytes_9988, "mem_9990"))
        return 1;
    for (int32_t i_9969 = 0; i_9969 < sizze_9750; i_9969++) {
        float x_9762 = ((float *) x_mem_9986.mem)[i_9969];
        float x_9763 = ((float *) y_mem_9987.mem)[i_9969];
        float res_9764 = x_9762 * x_9763;
        
        ((float *) mem_9990.mem)[i_9969] = res_9764;
    }
    out_arrsizze_10029 = sizze_9750;
    if (memblock_set(ctx, &out_mem_10028, &mem_9990, "mem_9990") != 0)
        return 1;
    (*out_mem_p_10112).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10112, &out_mem_10028, "out_mem_10028") !=
        0)
        return 1;
    *out_out_arrsizze_10113 = out_arrsizze_10029;
    if (memblock_unref(ctx, &mem_9990, "mem_9990") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10028, "out_mem_10028") != 0)
        return 1;
    return 0;
}
static int futrts_substract(struct futhark_context *ctx,
                            struct memblock *out_mem_p_10114,
                            int32_t *out_out_arrsizze_10115,
                            struct memblock x_mem_9986, int32_t sizze_9744,
                            float d_9745)
{
    struct memblock out_mem_10025;
    
    out_mem_10025.references = NULL;
    
    int32_t out_arrsizze_10026;
    int64_t binop_x_9988 = sext_i32_i64(sizze_9744);
    int64_t bytes_9987 = 4 * binop_x_9988;
    struct memblock mem_9989;
    
    mem_9989.references = NULL;
    if (memblock_alloc(ctx, &mem_9989, bytes_9987, "mem_9989"))
        return 1;
    for (int32_t i_9969 = 0; i_9969 < sizze_9744; i_9969++) {
        float x_9748 = ((float *) x_mem_9986.mem)[i_9969];
        float res_9749 = d_9745 - x_9748;
        
        ((float *) mem_9989.mem)[i_9969] = res_9749;
    }
    out_arrsizze_10026 = sizze_9744;
    if (memblock_set(ctx, &out_mem_10025, &mem_9989, "mem_9989") != 0)
        return 1;
    (*out_mem_p_10114).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10114, &out_mem_10025, "out_mem_10025") !=
        0)
        return 1;
    *out_out_arrsizze_10115 = out_arrsizze_10026;
    if (memblock_unref(ctx, &mem_9989, "mem_9989") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10025, "out_mem_10025") != 0)
        return 1;
    return 0;
}
static int futrts_add(struct futhark_context *ctx,
                      struct memblock *out_mem_p_10116,
                      int32_t *out_out_arrsizze_10117,
                      struct memblock x_mem_9986, int32_t sizze_9738,
                      float s_9739)
{
    struct memblock out_mem_10022;
    
    out_mem_10022.references = NULL;
    
    int32_t out_arrsizze_10023;
    int64_t binop_x_9988 = sext_i32_i64(sizze_9738);
    int64_t bytes_9987 = 4 * binop_x_9988;
    struct memblock mem_9989;
    
    mem_9989.references = NULL;
    if (memblock_alloc(ctx, &mem_9989, bytes_9987, "mem_9989"))
        return 1;
    for (int32_t i_9969 = 0; i_9969 < sizze_9738; i_9969++) {
        float x_9742 = ((float *) x_mem_9986.mem)[i_9969];
        float res_9743 = s_9739 + x_9742;
        
        ((float *) mem_9989.mem)[i_9969] = res_9743;
    }
    out_arrsizze_10023 = sizze_9738;
    if (memblock_set(ctx, &out_mem_10022, &mem_9989, "mem_9989") != 0)
        return 1;
    (*out_mem_p_10116).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10116, &out_mem_10022, "out_mem_10022") !=
        0)
        return 1;
    *out_out_arrsizze_10117 = out_arrsizze_10023;
    if (memblock_unref(ctx, &mem_9989, "mem_9989") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10022, "out_mem_10022") != 0)
        return 1;
    return 0;
}
static int futrts_multiply(struct futhark_context *ctx,
                           struct memblock *out_mem_p_10118,
                           int32_t *out_out_arrsizze_10119,
                           struct memblock x_mem_9986, int32_t sizze_9732,
                           float m_9733)
{
    struct memblock out_mem_10019;
    
    out_mem_10019.references = NULL;
    
    int32_t out_arrsizze_10020;
    int64_t binop_x_9988 = sext_i32_i64(sizze_9732);
    int64_t bytes_9987 = 4 * binop_x_9988;
    struct memblock mem_9989;
    
    mem_9989.references = NULL;
    if (memblock_alloc(ctx, &mem_9989, bytes_9987, "mem_9989"))
        return 1;
    for (int32_t i_9969 = 0; i_9969 < sizze_9732; i_9969++) {
        float x_9736 = ((float *) x_mem_9986.mem)[i_9969];
        float res_9737 = m_9733 * x_9736;
        
        ((float *) mem_9989.mem)[i_9969] = res_9737;
    }
    out_arrsizze_10020 = sizze_9732;
    if (memblock_set(ctx, &out_mem_10019, &mem_9989, "mem_9989") != 0)
        return 1;
    (*out_mem_p_10118).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10118, &out_mem_10019, "out_mem_10019") !=
        0)
        return 1;
    *out_out_arrsizze_10119 = out_arrsizze_10020;
    if (memblock_unref(ctx, &mem_9989, "mem_9989") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10019, "out_mem_10019") != 0)
        return 1;
    return 0;
}
static int futrts_divide(struct futhark_context *ctx,
                         struct memblock *out_mem_p_10120,
                         int32_t *out_out_arrsizze_10121,
                         struct memblock x_mem_9986, int32_t sizze_9726,
                         float d_9727)
{
    struct memblock out_mem_10016;
    
    out_mem_10016.references = NULL;
    
    int32_t out_arrsizze_10017;
    int64_t binop_x_9988 = sext_i32_i64(sizze_9726);
    int64_t bytes_9987 = 4 * binop_x_9988;
    struct memblock mem_9989;
    
    mem_9989.references = NULL;
    if (memblock_alloc(ctx, &mem_9989, bytes_9987, "mem_9989"))
        return 1;
    for (int32_t i_9969 = 0; i_9969 < sizze_9726; i_9969++) {
        float x_9730 = ((float *) x_mem_9986.mem)[i_9969];
        float res_9731 = d_9727 / x_9730;
        
        ((float *) mem_9989.mem)[i_9969] = res_9731;
    }
    out_arrsizze_10017 = sizze_9726;
    if (memblock_set(ctx, &out_mem_10016, &mem_9989, "mem_9989") != 0)
        return 1;
    (*out_mem_p_10120).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10120, &out_mem_10016, "out_mem_10016") !=
        0)
        return 1;
    *out_out_arrsizze_10121 = out_arrsizze_10017;
    if (memblock_unref(ctx, &mem_9989, "mem_9989") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10016, "out_mem_10016") != 0)
        return 1;
    return 0;
}
static int futrts_negation(struct futhark_context *ctx,
                           struct memblock *out_mem_p_10122,
                           int32_t *out_out_arrsizze_10123,
                           struct memblock x_mem_9986, int32_t sizze_9721)
{
    struct memblock out_mem_10013;
    
    out_mem_10013.references = NULL;
    
    int32_t out_arrsizze_10014;
    int64_t binop_x_9988 = sext_i32_i64(sizze_9721);
    int64_t bytes_9987 = 4 * binop_x_9988;
    struct memblock mem_9989;
    
    mem_9989.references = NULL;
    if (memblock_alloc(ctx, &mem_9989, bytes_9987, "mem_9989"))
        return 1;
    for (int32_t i_9969 = 0; i_9969 < sizze_9721; i_9969++) {
        float x_9724 = ((float *) x_mem_9986.mem)[i_9969];
        float res_9725 = 0.0F - x_9724;
        
        ((float *) mem_9989.mem)[i_9969] = res_9725;
    }
    out_arrsizze_10014 = sizze_9721;
    if (memblock_set(ctx, &out_mem_10013, &mem_9989, "mem_9989") != 0)
        return 1;
    (*out_mem_p_10122).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10122, &out_mem_10013, "out_mem_10013") !=
        0)
        return 1;
    *out_out_arrsizze_10123 = out_arrsizze_10014;
    if (memblock_unref(ctx, &mem_9989, "mem_9989") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10013, "out_mem_10013") != 0)
        return 1;
    return 0;
}
static int futrts_exp(struct futhark_context *ctx,
                      struct memblock *out_mem_p_10124,
                      int32_t *out_out_arrsizze_10125,
                      struct memblock x_mem_9986, int32_t sizze_9716)
{
    struct memblock out_mem_10010;
    
    out_mem_10010.references = NULL;
    
    int32_t out_arrsizze_10011;
    int64_t binop_x_9988 = sext_i32_i64(sizze_9716);
    int64_t bytes_9987 = 4 * binop_x_9988;
    struct memblock mem_9989;
    
    mem_9989.references = NULL;
    if (memblock_alloc(ctx, &mem_9989, bytes_9987, "mem_9989"))
        return 1;
    for (int32_t i_9969 = 0; i_9969 < sizze_9716; i_9969++) {
        float x_9719 = ((float *) x_mem_9986.mem)[i_9969];
        float res_9720 = fpow32(2.7182817F, x_9719);
        
        ((float *) mem_9989.mem)[i_9969] = res_9720;
    }
    out_arrsizze_10011 = sizze_9716;
    if (memblock_set(ctx, &out_mem_10010, &mem_9989, "mem_9989") != 0)
        return 1;
    (*out_mem_p_10124).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10124, &out_mem_10010, "out_mem_10010") !=
        0)
        return 1;
    *out_out_arrsizze_10125 = out_arrsizze_10011;
    if (memblock_unref(ctx, &mem_9989, "mem_9989") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10010, "out_mem_10010") != 0)
        return 1;
    return 0;
}
static int futrts_transp(struct futhark_context *ctx,
                         struct memblock *out_mem_p_10126,
                         int32_t *out_out_arrsizze_10127,
                         int32_t *out_out_arrsizze_10128,
                         struct memblock x_mem_9986, int32_t sizze_9712,
                         int32_t sizze_9713)
{
    struct memblock out_mem_10005;
    
    out_mem_10005.references = NULL;
    
    int32_t out_arrsizze_10006;
    int32_t out_arrsizze_10007;
    int64_t binop_x_9988 = sext_i32_i64(sizze_9713);
    int64_t binop_y_9989 = sext_i32_i64(sizze_9712);
    int64_t binop_x_9990 = binop_x_9988 * binop_y_9989;
    int64_t bytes_9987 = 4 * binop_x_9990;
    struct memblock mem_9991;
    
    mem_9991.references = NULL;
    if (memblock_alloc(ctx, &mem_9991, bytes_9987, "mem_9991"))
        return 1;
    for (int32_t i_10008 = 0; i_10008 < sizze_9713; i_10008++) {
        for (int32_t i_10009 = 0; i_10009 < sizze_9712; i_10009++) {
            ((float *) mem_9991.mem)[i_10008 * sizze_9712 + i_10009] =
                ((float *) x_mem_9986.mem)[i_10009 * sizze_9713 + i_10008];
        }
    }
    out_arrsizze_10006 = sizze_9713;
    out_arrsizze_10007 = sizze_9712;
    if (memblock_set(ctx, &out_mem_10005, &mem_9991, "mem_9991") != 0)
        return 1;
    (*out_mem_p_10126).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_10126, &out_mem_10005, "out_mem_10005") !=
        0)
        return 1;
    *out_out_arrsizze_10127 = out_arrsizze_10006;
    *out_out_arrsizze_10128 = out_arrsizze_10007;
    if (memblock_unref(ctx, &mem_9991, "mem_9991") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_10005, "out_mem_10005") != 0)
        return 1;
    return 0;
}
struct futhark_f32_1d {
    struct memblock mem;
    int64_t shape[1];
} ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, (size_t) dim0 * sizeof(float),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + 0, (size_t) dim0 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, (size_t) dim0 * sizeof(float),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + offset, (size_t) dim0 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, (size_t) arr->shape[0] * sizeof(float));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                struct futhark_f32_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_f32_2d {
    struct memblock mem;
    int64_t shape[2];
} ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int64_t dim0,
                                          int64_t dim1)
{
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr =
                          (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, (size_t) (dim0 * dim1) * sizeof(float),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    memmove(arr->mem.mem + 0, data + 0, (size_t) (dim0 * dim1) * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1)
{
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr =
                          (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, (size_t) (dim0 * dim1) * sizeof(float),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    memmove(arr->mem.mem + 0, data + offset, (size_t) (dim0 * dim1) *
            sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, (size_t) (arr->shape[0] *
                                                  arr->shape[1]) *
            sizeof(float));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                struct futhark_f32_2d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr)
{
    (void) ctx;
    return arr->shape;
}
int futhark_entry_dot(struct futhark_context *ctx, struct futhark_f32_2d **out0,
                      const struct futhark_f32_2d *in0, const
                      struct futhark_f32_2d *in1)
{
    struct memblock a_mem_9986;
    
    a_mem_9986.references = NULL;
    
    struct memblock b_mem_9987;
    
    b_mem_9987.references = NULL;
    
    int32_t sizze_9942;
    int32_t sizze_9943;
    int32_t sizze_9944;
    int32_t sizze_9945;
    struct memblock out_mem_10076;
    
    out_mem_10076.references = NULL;
    
    int32_t out_arrsizze_10077;
    int32_t out_arrsizze_10078;
    
    lock_lock(&ctx->lock);
    a_mem_9986 = in0->mem;
    sizze_9942 = in0->shape[0];
    sizze_9943 = in0->shape[1];
    b_mem_9987 = in1->mem;
    sizze_9944 = in1->shape[0];
    sizze_9945 = in1->shape[1];
    
    int ret = futrts_dot(ctx, &out_mem_10076, &out_arrsizze_10077,
                         &out_arrsizze_10078, a_mem_9986, b_mem_9987,
                         sizze_9942, sizze_9943, sizze_9944, sizze_9945);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out0)->mem = out_mem_10076;
        (*out0)->shape[0] = out_arrsizze_10077;
        (*out0)->shape[1] = out_arrsizze_10078;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_matsubstract(struct futhark_context *ctx,
                               struct futhark_f32_2d **out0, const
                               struct futhark_f32_2d *in0, const
                               struct futhark_f32_2d *in1)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    struct memblock y_mem_9987;
    
    y_mem_9987.references = NULL;
    
    int32_t sizze_9913;
    int32_t sizze_9914;
    int32_t sizze_9915;
    int32_t sizze_9916;
    struct memblock out_mem_10071;
    
    out_mem_10071.references = NULL;
    
    int32_t out_arrsizze_10072;
    int32_t out_arrsizze_10073;
    
    lock_lock(&ctx->lock);
    x_mem_9986 = in0->mem;
    sizze_9913 = in0->shape[0];
    sizze_9914 = in0->shape[1];
    y_mem_9987 = in1->mem;
    sizze_9915 = in1->shape[0];
    sizze_9916 = in1->shape[1];
    
    int ret = futrts_matsubstract(ctx, &out_mem_10071, &out_arrsizze_10072,
                                  &out_arrsizze_10073, x_mem_9986, y_mem_9987,
                                  sizze_9913, sizze_9914, sizze_9915,
                                  sizze_9916);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out0)->mem = out_mem_10071;
        (*out0)->shape[0] = out_arrsizze_10072;
        (*out0)->shape[1] = out_arrsizze_10073;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_matadd(struct futhark_context *ctx,
                         struct futhark_f32_2d **out0, const
                         struct futhark_f32_2d *in0, const
                         struct futhark_f32_2d *in1)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    struct memblock y_mem_9987;
    
    y_mem_9987.references = NULL;
    
    int32_t sizze_9884;
    int32_t sizze_9885;
    int32_t sizze_9886;
    int32_t sizze_9887;
    struct memblock out_mem_10066;
    
    out_mem_10066.references = NULL;
    
    int32_t out_arrsizze_10067;
    int32_t out_arrsizze_10068;
    
    lock_lock(&ctx->lock);
    x_mem_9986 = in0->mem;
    sizze_9884 = in0->shape[0];
    sizze_9885 = in0->shape[1];
    y_mem_9987 = in1->mem;
    sizze_9886 = in1->shape[0];
    sizze_9887 = in1->shape[1];
    
    int ret = futrts_matadd(ctx, &out_mem_10066, &out_arrsizze_10067,
                            &out_arrsizze_10068, x_mem_9986, y_mem_9987,
                            sizze_9884, sizze_9885, sizze_9886, sizze_9887);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out0)->mem = out_mem_10066;
        (*out0)->shape[0] = out_arrsizze_10067;
        (*out0)->shape[1] = out_arrsizze_10068;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_matmultiply(struct futhark_context *ctx,
                              struct futhark_f32_2d **out0, const
                              struct futhark_f32_2d *in0, const
                              struct futhark_f32_2d *in1)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    struct memblock y_mem_9987;
    
    y_mem_9987.references = NULL;
    
    int32_t sizze_9855;
    int32_t sizze_9856;
    int32_t sizze_9857;
    int32_t sizze_9858;
    struct memblock out_mem_10061;
    
    out_mem_10061.references = NULL;
    
    int32_t out_arrsizze_10062;
    int32_t out_arrsizze_10063;
    
    lock_lock(&ctx->lock);
    x_mem_9986 = in0->mem;
    sizze_9855 = in0->shape[0];
    sizze_9856 = in0->shape[1];
    y_mem_9987 = in1->mem;
    sizze_9857 = in1->shape[0];
    sizze_9858 = in1->shape[1];
    
    int ret = futrts_matmultiply(ctx, &out_mem_10061, &out_arrsizze_10062,
                                 &out_arrsizze_10063, x_mem_9986, y_mem_9987,
                                 sizze_9855, sizze_9856, sizze_9857,
                                 sizze_9858);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out0)->mem = out_mem_10061;
        (*out0)->shape[0] = out_arrsizze_10062;
        (*out0)->shape[1] = out_arrsizze_10063;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_lvecmul(struct futhark_context *ctx,
                          struct futhark_f32_1d **out0, const
                          struct futhark_f32_1d *in0, const
                          struct futhark_f32_2d *in1)
{
    struct memblock u_mem_9986;
    
    u_mem_9986.references = NULL;
    
    struct memblock b_mem_9987;
    
    b_mem_9987.references = NULL;
    
    int32_t sizze_9833;
    int32_t sizze_9834;
    int32_t sizze_9835;
    struct memblock out_mem_10057;
    
    out_mem_10057.references = NULL;
    
    int32_t out_arrsizze_10058;
    
    lock_lock(&ctx->lock);
    u_mem_9986 = in0->mem;
    sizze_9833 = in0->shape[0];
    b_mem_9987 = in1->mem;
    sizze_9834 = in1->shape[0];
    sizze_9835 = in1->shape[1];
    
    int ret = futrts_lvecmul(ctx, &out_mem_10057, &out_arrsizze_10058,
                             u_mem_9986, b_mem_9987, sizze_9833, sizze_9834,
                             sizze_9835);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_10057;
        (*out0)->shape[0] = out_arrsizze_10058;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_sigmoid(struct futhark_context *ctx,
                          struct futhark_f32_2d **out0, const
                          struct futhark_f32_2d *in0)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    int32_t sizze_9822;
    int32_t sizze_9823;
    struct memblock out_mem_10052;
    
    out_mem_10052.references = NULL;
    
    int32_t out_arrsizze_10053;
    int32_t out_arrsizze_10054;
    
    lock_lock(&ctx->lock);
    x_mem_9986 = in0->mem;
    sizze_9822 = in0->shape[0];
    sizze_9823 = in0->shape[1];
    
    int ret = futrts_sigmoid(ctx, &out_mem_10052, &out_arrsizze_10053,
                             &out_arrsizze_10054, x_mem_9986, sizze_9822,
                             sizze_9823);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out0)->mem = out_mem_10052;
        (*out0)->shape[0] = out_arrsizze_10053;
        (*out0)->shape[1] = out_arrsizze_10054;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_lmatsubstract(struct futhark_context *ctx,
                                struct futhark_f32_2d **out0, const float in0,
                                const struct futhark_f32_2d *in1)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    int32_t sizze_9813;
    int32_t sizze_9814;
    float d_9815;
    struct memblock out_mem_10047;
    
    out_mem_10047.references = NULL;
    
    int32_t out_arrsizze_10048;
    int32_t out_arrsizze_10049;
    
    lock_lock(&ctx->lock);
    d_9815 = in0;
    x_mem_9986 = in1->mem;
    sizze_9813 = in1->shape[0];
    sizze_9814 = in1->shape[1];
    
    int ret = futrts_lmatsubstract(ctx, &out_mem_10047, &out_arrsizze_10048,
                                   &out_arrsizze_10049, x_mem_9986, sizze_9813,
                                   sizze_9814, d_9815);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out0)->mem = out_mem_10047;
        (*out0)->shape[0] = out_arrsizze_10048;
        (*out0)->shape[1] = out_arrsizze_10049;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_lmatadd(struct futhark_context *ctx,
                          struct futhark_f32_2d **out0, const float in0, const
                          struct futhark_f32_2d *in1)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    int32_t sizze_9804;
    int32_t sizze_9805;
    float s_9806;
    struct memblock out_mem_10042;
    
    out_mem_10042.references = NULL;
    
    int32_t out_arrsizze_10043;
    int32_t out_arrsizze_10044;
    
    lock_lock(&ctx->lock);
    s_9806 = in0;
    x_mem_9986 = in1->mem;
    sizze_9804 = in1->shape[0];
    sizze_9805 = in1->shape[1];
    
    int ret = futrts_lmatadd(ctx, &out_mem_10042, &out_arrsizze_10043,
                             &out_arrsizze_10044, x_mem_9986, sizze_9804,
                             sizze_9805, s_9806);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out0)->mem = out_mem_10042;
        (*out0)->shape[0] = out_arrsizze_10043;
        (*out0)->shape[1] = out_arrsizze_10044;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_substract2(struct futhark_context *ctx,
                             struct futhark_f32_1d **out0, const
                             struct futhark_f32_1d *in0, const
                             struct futhark_f32_1d *in1)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    struct memblock y_mem_9987;
    
    y_mem_9987.references = NULL;
    
    int32_t sizze_9789;
    int32_t sizze_9790;
    struct memblock out_mem_10039;
    
    out_mem_10039.references = NULL;
    
    int32_t out_arrsizze_10040;
    
    lock_lock(&ctx->lock);
    x_mem_9986 = in0->mem;
    sizze_9789 = in0->shape[0];
    y_mem_9987 = in1->mem;
    sizze_9790 = in1->shape[0];
    
    int ret = futrts_substract2(ctx, &out_mem_10039, &out_arrsizze_10040,
                                x_mem_9986, y_mem_9987, sizze_9789, sizze_9790);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_10039;
        (*out0)->shape[0] = out_arrsizze_10040;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_add2(struct futhark_context *ctx,
                       struct futhark_f32_1d **out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    struct memblock y_mem_9987;
    
    y_mem_9987.references = NULL;
    
    int32_t sizze_9774;
    int32_t sizze_9775;
    struct memblock out_mem_10036;
    
    out_mem_10036.references = NULL;
    
    int32_t out_arrsizze_10037;
    
    lock_lock(&ctx->lock);
    x_mem_9986 = in0->mem;
    sizze_9774 = in0->shape[0];
    y_mem_9987 = in1->mem;
    sizze_9775 = in1->shape[0];
    
    int ret = futrts_add2(ctx, &out_mem_10036, &out_arrsizze_10037, x_mem_9986,
                          y_mem_9987, sizze_9774, sizze_9775);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_10036;
        (*out0)->shape[0] = out_arrsizze_10037;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_lmatmultiply(struct futhark_context *ctx,
                               struct futhark_f32_2d **out0, const float in0,
                               const struct futhark_f32_2d *in1)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    int32_t sizze_9765;
    int32_t sizze_9766;
    float p_9767;
    struct memblock out_mem_10031;
    
    out_mem_10031.references = NULL;
    
    int32_t out_arrsizze_10032;
    int32_t out_arrsizze_10033;
    
    lock_lock(&ctx->lock);
    p_9767 = in0;
    x_mem_9986 = in1->mem;
    sizze_9765 = in1->shape[0];
    sizze_9766 = in1->shape[1];
    
    int ret = futrts_lmatmultiply(ctx, &out_mem_10031, &out_arrsizze_10032,
                                  &out_arrsizze_10033, x_mem_9986, sizze_9765,
                                  sizze_9766, p_9767);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out0)->mem = out_mem_10031;
        (*out0)->shape[0] = out_arrsizze_10032;
        (*out0)->shape[1] = out_arrsizze_10033;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_multiply2(struct futhark_context *ctx,
                            struct futhark_f32_1d **out0, const
                            struct futhark_f32_1d *in0, const
                            struct futhark_f32_1d *in1)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    struct memblock y_mem_9987;
    
    y_mem_9987.references = NULL;
    
    int32_t sizze_9750;
    int32_t sizze_9751;
    struct memblock out_mem_10028;
    
    out_mem_10028.references = NULL;
    
    int32_t out_arrsizze_10029;
    
    lock_lock(&ctx->lock);
    x_mem_9986 = in0->mem;
    sizze_9750 = in0->shape[0];
    y_mem_9987 = in1->mem;
    sizze_9751 = in1->shape[0];
    
    int ret = futrts_multiply2(ctx, &out_mem_10028, &out_arrsizze_10029,
                               x_mem_9986, y_mem_9987, sizze_9750, sizze_9751);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_10028;
        (*out0)->shape[0] = out_arrsizze_10029;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_substract(struct futhark_context *ctx,
                            struct futhark_f32_1d **out0, const float in0, const
                            struct futhark_f32_1d *in1)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    int32_t sizze_9744;
    float d_9745;
    struct memblock out_mem_10025;
    
    out_mem_10025.references = NULL;
    
    int32_t out_arrsizze_10026;
    
    lock_lock(&ctx->lock);
    d_9745 = in0;
    x_mem_9986 = in1->mem;
    sizze_9744 = in1->shape[0];
    
    int ret = futrts_substract(ctx, &out_mem_10025, &out_arrsizze_10026,
                               x_mem_9986, sizze_9744, d_9745);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_10025;
        (*out0)->shape[0] = out_arrsizze_10026;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_add(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                      const float in0, const struct futhark_f32_1d *in1)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    int32_t sizze_9738;
    float s_9739;
    struct memblock out_mem_10022;
    
    out_mem_10022.references = NULL;
    
    int32_t out_arrsizze_10023;
    
    lock_lock(&ctx->lock);
    s_9739 = in0;
    x_mem_9986 = in1->mem;
    sizze_9738 = in1->shape[0];
    
    int ret = futrts_add(ctx, &out_mem_10022, &out_arrsizze_10023, x_mem_9986,
                         sizze_9738, s_9739);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_10022;
        (*out0)->shape[0] = out_arrsizze_10023;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_multiply(struct futhark_context *ctx,
                           struct futhark_f32_1d **out0, const float in0, const
                           struct futhark_f32_1d *in1)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    int32_t sizze_9732;
    float m_9733;
    struct memblock out_mem_10019;
    
    out_mem_10019.references = NULL;
    
    int32_t out_arrsizze_10020;
    
    lock_lock(&ctx->lock);
    m_9733 = in0;
    x_mem_9986 = in1->mem;
    sizze_9732 = in1->shape[0];
    
    int ret = futrts_multiply(ctx, &out_mem_10019, &out_arrsizze_10020,
                              x_mem_9986, sizze_9732, m_9733);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_10019;
        (*out0)->shape[0] = out_arrsizze_10020;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_divide(struct futhark_context *ctx,
                         struct futhark_f32_1d **out0, const float in0, const
                         struct futhark_f32_1d *in1)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    int32_t sizze_9726;
    float d_9727;
    struct memblock out_mem_10016;
    
    out_mem_10016.references = NULL;
    
    int32_t out_arrsizze_10017;
    
    lock_lock(&ctx->lock);
    d_9727 = in0;
    x_mem_9986 = in1->mem;
    sizze_9726 = in1->shape[0];
    
    int ret = futrts_divide(ctx, &out_mem_10016, &out_arrsizze_10017,
                            x_mem_9986, sizze_9726, d_9727);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_10016;
        (*out0)->shape[0] = out_arrsizze_10017;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_negation(struct futhark_context *ctx,
                           struct futhark_f32_1d **out0, const
                           struct futhark_f32_1d *in0)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    int32_t sizze_9721;
    struct memblock out_mem_10013;
    
    out_mem_10013.references = NULL;
    
    int32_t out_arrsizze_10014;
    
    lock_lock(&ctx->lock);
    x_mem_9986 = in0->mem;
    sizze_9721 = in0->shape[0];
    
    int ret = futrts_negation(ctx, &out_mem_10013, &out_arrsizze_10014,
                              x_mem_9986, sizze_9721);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_10013;
        (*out0)->shape[0] = out_arrsizze_10014;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_exp(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                      const struct futhark_f32_1d *in0)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    int32_t sizze_9716;
    struct memblock out_mem_10010;
    
    out_mem_10010.references = NULL;
    
    int32_t out_arrsizze_10011;
    
    lock_lock(&ctx->lock);
    x_mem_9986 = in0->mem;
    sizze_9716 = in0->shape[0];
    
    int ret = futrts_exp(ctx, &out_mem_10010, &out_arrsizze_10011, x_mem_9986,
                         sizze_9716);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_10010;
        (*out0)->shape[0] = out_arrsizze_10011;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_transp(struct futhark_context *ctx,
                         struct futhark_f32_2d **out0, const
                         struct futhark_f32_2d *in0)
{
    struct memblock x_mem_9986;
    
    x_mem_9986.references = NULL;
    
    int32_t sizze_9712;
    int32_t sizze_9713;
    struct memblock out_mem_10005;
    
    out_mem_10005.references = NULL;
    
    int32_t out_arrsizze_10006;
    int32_t out_arrsizze_10007;
    
    lock_lock(&ctx->lock);
    x_mem_9986 = in0->mem;
    sizze_9712 = in0->shape[0];
    sizze_9713 = in0->shape[1];
    
    int ret = futrts_transp(ctx, &out_mem_10005, &out_arrsizze_10006,
                            &out_arrsizze_10007, x_mem_9986, sizze_9712,
                            sizze_9713);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out0)->mem = out_mem_10005;
        (*out0)->shape[0] = out_arrsizze_10006;
        (*out0)->shape[1] = out_arrsizze_10007;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
