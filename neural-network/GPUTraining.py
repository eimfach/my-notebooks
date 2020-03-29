import sys
import numpy as np
import ctypes as ct
# Stub code for OpenCL setup.

import pyopencl as cl
import numpy as np
import sys

if cl.version.VERSION < (2015,2):
    raise Exception('Futhark requires at least PyOpenCL version 2015.2.  Installed version is %s.' %
                    cl.version.VERSION_TEXT)

def parse_preferred_device(s):
    pref_num = 0
    if len(s) > 1 and s[0] == '#':
        i = 1
        while i < len(s):
            if not s[i].isdigit():
                break
            else:
                pref_num = pref_num * 10 + int(s[i])
            i += 1
        while i < len(s) and s[i].isspace():
            i += 1
        return (s[i:], pref_num)
    else:
        return (s, 0)

def get_prefered_context(interactive=False, platform_pref=None, device_pref=None):
    if device_pref != None:
        (device_pref, device_num) = parse_preferred_device(device_pref)
    else:
        device_num = 0

    if interactive:
        return cl.create_some_context(interactive=True)

    def blacklisted(p, d):
        return platform_pref == None and device_pref == None and \
            p.name == "Apple" and d.name.find("Intel(R) Core(TM)") >= 0
    def platform_ok(p):
        return not platform_pref or p.name.find(platform_pref) >= 0
    def device_ok(d):
        return not device_pref or d.name.find(device_pref) >= 0

    device_matches = 0

    for p in cl.get_platforms():
        if not platform_ok(p):
            continue
        for d in p.get_devices():
            if blacklisted(p,d) or not device_ok(d):
                continue
            if device_matches == device_num:
                return cl.Context(devices=[d])
            else:
                device_matches += 1
    raise Exception('No OpenCL platform and device matching constraints found.')

def size_assignment(s):
    name, value = s.split('=')
    return (name, int(value))

def check_types(self, required_types):
    if 'f64' in required_types:
        if self.device.get_info(cl.device_info.PREFERRED_VECTOR_WIDTH_DOUBLE) == 0:
            raise Exception('Program uses double-precision floats, but this is not supported on chosen device: %s' % self.device.name)

def apply_size_heuristics(self, size_heuristics, sizes):
    for (platform_name, device_type, size, value) in size_heuristics:
        if sizes[size] == None \
           and self.platform.name.find(platform_name) >= 0 \
           and self.device.type == device_type:
               if type(value) == str:
                   sizes[size] = self.device.get_info(getattr(cl.device_info,value))
               else:
                   sizes[size] = value
    return sizes

def initialise_opencl_object(self,
                             program_src='',
                             command_queue=None,
                             interactive=False,
                             platform_pref=None,
                             device_pref=None,
                             default_group_size=None,
                             default_num_groups=None,
                             default_tile_size=None,
                             default_threshold=None,
                             size_heuristics=[],
                             required_types=[],
                             all_sizes={},
                             user_sizes={}):
    if command_queue is None:
        self.ctx = get_prefered_context(interactive, platform_pref, device_pref)
        self.queue = cl.CommandQueue(self.ctx)
    else:
        self.ctx = command_queue.context
        self.queue = command_queue
    self.device = self.queue.device
    self.platform = self.device.platform
    self.pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))
    device_type = self.device.type

    check_types(self, required_types)

    max_group_size = int(self.device.max_work_group_size)
    max_tile_size = int(np.sqrt(self.device.max_work_group_size))

    self.max_group_size = max_group_size
    self.max_tile_size = max_tile_size
    self.max_threshold = 0
    self.max_num_groups = 0
    self.max_local_memory = int(self.device.local_mem_size)
    self.free_list = {}

    if 'default_group_size' in sizes:
        default_group_size = sizes['default_group_size']
        del sizes['default_group_size']

    if 'default_num_groups' in sizes:
        default_num_groups = sizes['default_num_groups']
        del sizes['default_num_groups']

    if 'default_tile_size' in sizes:
        default_tile_size = sizes['default_tile_size']
        del sizes['default_tile_size']

    if 'default_threshold' in sizes:
        default_threshold = sizes['default_threshold']
        del sizes['default_threshold']

    default_group_size_set = default_group_size != None
    default_tile_size_set = default_tile_size != None
    default_sizes = apply_size_heuristics(self, size_heuristics,
                                          {'group_size': default_group_size,
                                           'tile_size': default_tile_size,
                                           'num_groups': default_num_groups,
                                           'lockstep_width': None,
                                           'threshold': default_threshold})
    default_group_size = default_sizes['group_size']
    default_num_groups = default_sizes['num_groups']
    default_threshold = default_sizes['threshold']
    default_tile_size = default_sizes['tile_size']
    lockstep_width = default_sizes['lockstep_width']

    if default_group_size > max_group_size:
        if default_group_size_set:
            sys.stderr.write('Note: Device limits group size to {} (down from {})\n'.
                             format(max_tile_size, default_group_size))
        default_group_size = max_group_size

    if default_tile_size > max_tile_size:
        if default_tile_size_set:
            sys.stderr.write('Note: Device limits tile size to {} (down from {})\n'.
                             format(max_tile_size, default_tile_size))
        default_tile_size = max_tile_size

    for (k,v) in user_sizes.items():
        if k in all_sizes:
            all_sizes[k]['value'] = v
        else:
            raise Exception('Unknown size: {}\nKnown sizes: {}'.format(k, ' '.join(all_sizes.keys())))

    self.sizes = {}
    for (k,v) in all_sizes.items():
        if v['class'] == 'group_size':
            max_value = max_group_size
            default_value = default_group_size
        elif v['class'] == 'num_groups':
            max_value = max_group_size # Intentional!
            default_value = default_num_groups
        elif v['class'] == 'tile_size':
            max_value = max_tile_size
            default_value = default_tile_size
        elif v['class'].startswith('threshold'):
            max_value = None
            default_value = default_threshold
        else:
            # Bespoke sizes have no limit or default.
            max_value = None
        if v['value'] == None:
            self.sizes[k] = default_value
        elif max_value != None and v['value'] > max_value:
            sys.stderr.write('Note: Device limits {} to {} (down from {}\n'.
                             format(k, max_value, v['value']))
            self.sizes[k] = max_value
        else:
            self.sizes[k] = v['value']

    # XXX: we perform only a subset of z-encoding here.  Really, the
    # compiler should provide us with the variables to which
    # parameters are mapped.
    if (len(program_src) >= 0):
        return cl.Program(self.ctx, program_src).build(
            ["-DLOCKSTEP_WIDTH={}".format(lockstep_width)]
            + ["-D{}={}".format(s.replace('z', 'zz').replace('.', 'zi'),v) for (s,v) in self.sizes.items()])

def opencl_alloc(self, min_size, tag):
    min_size = 1 if min_size == 0 else min_size
    assert min_size > 0
    return self.pool.allocate(min_size)

def opencl_free_all(self):
    self.pool.free_held()
import pyopencl.array
import time
import argparse
sizes = {}
synchronous = False
preferred_platform = None
preferred_device = None
default_threshold = None
default_group_size = None
default_num_groups = None
default_tile_size = None
fut_opencl_src = """#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
__kernel void dummy_kernel(__global unsigned char *dummy, int n)
{
    const int thread_gid = get_global_id(0);
    
    if (thread_gid >= n)
        return;
}
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef uchar uint8_t;
typedef ushort uint16_t;
typedef uint uint32_t;
typedef ulong uint64_t;
#ifdef cl_nv_pragma_unroll
static inline void mem_fence_global()
{
    asm("membar.gl;");
}
#else
static inline void mem_fence_global()
{
    mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
#endif
static inline void mem_fence_local()
{
    mem_fence(CLK_LOCAL_MEM_FENCE);
}
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
__kernel void map_transpose_f32(__local volatile
                                int64_t *block_11_backing_aligned_0,
                                int32_t destoffset_1, int32_t srcoffset_3,
                                int32_t num_arrays_4, int32_t x_elems_5,
                                int32_t y_elems_6, int32_t in_elems_7,
                                int32_t out_elems_8, int32_t mulx_9,
                                int32_t muly_10, __global
                                unsigned char *destmem_0, __global
                                unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_11_backing_0 = (__local volatile
                                                          char *) block_11_backing_aligned_0;
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_global_id_0_37;
    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;
    
    if (slt32(x_index_31, x_elems_5)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, y_elems_6) && slt32(index_in_35,
                                                                 in_elems_7)) {
                ((__local float *) block_11)[(get_local_id_1_39 + j_43 * 8) *
                                             33 + get_local_id_0_38] =
                    ((__global float *) srcmem_2)[idata_offset_34 +
                                                  index_in_35];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;
    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;
    if (slt32(x_index_31, y_elems_6)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, x_elems_5) && slt32(index_out_36,
                                                                 out_elems_8)) {
                ((__global float *) destmem_0)[odata_offset_33 + index_out_36] =
                    ((__local float *) block_11)[get_local_id_0_38 * 33 +
                                                 get_local_id_1_39 + j_43 * 8];
            }
        }
    }
}
__kernel void map_transpose_f32_low_height(__local volatile
                                           int64_t *block_11_backing_aligned_0,
                                           int32_t destoffset_1,
                                           int32_t srcoffset_3,
                                           int32_t num_arrays_4,
                                           int32_t x_elems_5, int32_t y_elems_6,
                                           int32_t in_elems_7,
                                           int32_t out_elems_8, int32_t mulx_9,
                                           int32_t muly_10, __global
                                           unsigned char *destmem_0, __global
                                           unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_11_backing_0 = (__local volatile
                                                          char *) block_11_backing_aligned_0;
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_0_38 +
            srem32(get_local_id_1_39, mulx_9) * 16;
    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,
                                                          mulx_9);
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&
                                         slt32(index_in_35, in_elems_7))) {
        ((__local float *) block_11)[get_local_id_1_39 * 17 +
                                     get_local_id_0_38] = ((__global
                                                            float *) srcmem_2)[idata_offset_34 +
                                                                               index_in_35];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_9);
    y_index_32 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_1_39 +
        srem32(get_local_id_0_38, mulx_9) * 16;
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&
                                         slt32(index_out_36, out_elems_8))) {
        ((__global float *) destmem_0)[odata_offset_33 + index_out_36] =
            ((__local float *) block_11)[get_local_id_0_38 * 17 +
                                         get_local_id_1_39];
    }
}
__kernel void map_transpose_f32_low_width(__local volatile
                                          int64_t *block_11_backing_aligned_0,
                                          int32_t destoffset_1,
                                          int32_t srcoffset_3,
                                          int32_t num_arrays_4,
                                          int32_t x_elems_5, int32_t y_elems_6,
                                          int32_t in_elems_7,
                                          int32_t out_elems_8, int32_t mulx_9,
                                          int32_t muly_10, __global
                                          unsigned char *destmem_0, __global
                                          unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_11_backing_0 = (__local volatile
                                                          char *) block_11_backing_aligned_0;
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,
                                                          muly_10);
    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_10 + get_local_id_1_39 +
            srem32(get_local_id_0_38, muly_10) * 16;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&
                                         slt32(index_in_35, in_elems_7))) {
        ((__local float *) block_11)[get_local_id_1_39 * 17 +
                                     get_local_id_0_38] = ((__global
                                                            float *) srcmem_2)[idata_offset_34 +
                                                                               index_in_35];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 * muly_10 + get_local_id_0_38 +
        srem32(get_local_id_1_39, muly_10) * 16;
    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_10);
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&
                                         slt32(index_out_36, out_elems_8))) {
        ((__global float *) destmem_0)[odata_offset_33 + index_out_36] =
            ((__local float *) block_11)[get_local_id_0_38 * 17 +
                                         get_local_id_1_39];
    }
}
__kernel void map_transpose_f32_small(__local volatile
                                      int64_t *block_11_backing_aligned_0,
                                      int32_t destoffset_1, int32_t srcoffset_3,
                                      int32_t num_arrays_4, int32_t x_elems_5,
                                      int32_t y_elems_6, int32_t in_elems_7,
                                      int32_t out_elems_8, int32_t mulx_9,
                                      int32_t muly_10, __global
                                      unsigned char *destmem_0, __global
                                      unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_11_backing_0 = (__local volatile
                                                          char *) block_11_backing_aligned_0;
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *
                                          x_elems_5) * (y_elems_6 * x_elems_5);
    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *
                                        x_elems_5), y_elems_6);
    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;
    
    if (slt32(get_global_id_0_37, in_elems_7)) {
        ((__global float *) destmem_0)[odata_offset_33 + index_out_36] =
            ((__global float *) srcmem_2)[idata_offset_34 + index_in_35];
    }
}
__kernel void segred_nonseg_13152(__local volatile
                                  int64_t *sync_arr_mem_13434_backing_aligned_0,
                                  int32_t sizze_12930, int32_t sizze_12932,
                                  int32_t sizze_12933, int32_t sizze_12934,
                                  int32_t sizze_12935, int32_t sizze_12936,
                                  int32_t sizze_12937, int32_t sizze_12938,
                                  float lr_12939, int32_t num_groups_13146,
                                  __global unsigned char *wih_mem_13263,
                                  __global unsigned char *who_mem_13264,
                                  __global unsigned char *mem_13271, __global
                                  unsigned char *mem_13276,
                                  int32_t num_threads_13277, __global
                                  unsigned char *mem_13285, __global
                                  unsigned char *mem_13293, __global
                                  unsigned char *mem_13298, __global
                                  unsigned char *mem_13301, __global
                                  unsigned char *mem_13306, __global
                                  unsigned char *mem_13311, __global
                                  unsigned char *mem_13314, __global
                                  unsigned char *mem_13317, __global
                                  unsigned char *mem_13322, __global
                                  unsigned char *mem_13325, __global
                                  unsigned char *mem_13328, __global
                                  unsigned char *mem_13333, __global
                                  unsigned char *mem_13336, __global
                                  unsigned char *mem_13341, __global
                                  unsigned char *mem_13344, __global
                                  unsigned char *mem_13347, __global
                                  unsigned char *mem_13352, __global
                                  unsigned char *mem_13355, __global
                                  unsigned char *mem_13362, __global
                                  unsigned char *mem_13369, __global
                                  unsigned char *counter_mem_13422, __global
                                  unsigned char *group_res_arr_mem_13424,
                                  __global
                                  unsigned char *group_res_arr_mem_13426)
{
    const int32_t segred_group_sizze_13144 = mainzisegred_group_sizze_13143;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_13434_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_13434_backing_aligned_0;
    int32_t global_tid_13429;
    int32_t local_tid_13430;
    int32_t group_sizze_13433;
    int32_t wave_sizze_13432;
    int32_t group_tid_13431;
    
    global_tid_13429 = get_global_id(0);
    local_tid_13430 = get_local_id(0);
    group_sizze_13433 = get_local_size(0);
    wave_sizze_13432 = LOCKSTEP_WIDTH;
    group_tid_13431 = get_group_id(0);
    
    int32_t phys_tid_13152 = global_tid_13429;
    __local char *sync_arr_mem_13434;
    
    sync_arr_mem_13434 = (__local char *) sync_arr_mem_13434_backing_0;
    
    int32_t dummy_13150 = 0;
    int32_t gtid_13151;
    
    gtid_13151 = 0;
    
    int32_t chunk_sizze_13438;
    int32_t starting_point_13439 = phys_tid_13152 * squot32(sizze_12933 +
                                                            segred_group_sizze_13144 *
                                                            num_groups_13146 -
                                                            1,
                                                            segred_group_sizze_13144 *
                                                            num_groups_13146);
    int32_t remaining_elements_13440 = sizze_12933 - starting_point_13439;
    
    if (sle32(remaining_elements_13440, 0) || sle32(sizze_12933,
                                                    starting_point_13439)) {
        chunk_sizze_13438 = 0;
    } else {
        if (slt32(sizze_12933, (phys_tid_13152 + 1) * squot32(sizze_12933 +
                                                              segred_group_sizze_13144 *
                                                              num_groups_13146 -
                                                              1,
                                                              segred_group_sizze_13144 *
                                                              num_groups_13146))) {
            chunk_sizze_13438 = sizze_12933 - phys_tid_13152 *
                squot32(sizze_12933 + segred_group_sizze_13144 *
                        num_groups_13146 - 1, segred_group_sizze_13144 *
                        num_groups_13146);
        } else {
            chunk_sizze_13438 = squot32(sizze_12933 + segred_group_sizze_13144 *
                                        num_groups_13146 - 1,
                                        segred_group_sizze_13144 *
                                        num_groups_13146);
        }
    }
    // neutral-initialise the accumulators
    {
        for (int32_t i_13441 = 0; i_13441 < sizze_12934; i_13441++) {
            for (int32_t i_13442 = 0; i_13442 < sizze_12935; i_13442++) {
                ((__global float *) group_res_arr_mem_13424)[group_tid_13431 *
                                                             (segred_group_sizze_13144 *
                                                              sizze_12935 *
                                                              sizze_12934) +
                                                             segred_group_sizze_13144 *
                                                             sizze_12935 * 0 +
                                                             segred_group_sizze_13144 *
                                                             0 +
                                                             local_tid_13430 +
                                                             (i_13441 *
                                                              (segred_group_sizze_13144 *
                                                               sizze_12935) +
                                                              i_13442 *
                                                              segred_group_sizze_13144)] =
                    ((__global float *) wih_mem_13263)[i_13441 * sizze_12930 +
                                                       i_13442];
            }
        }
        for (int32_t i_13443 = 0; i_13443 < sizze_12937; i_13443++) {
            for (int32_t i_13444 = 0; i_13444 < sizze_12938; i_13444++) {
                ((__global float *) group_res_arr_mem_13426)[group_tid_13431 *
                                                             (segred_group_sizze_13144 *
                                                              sizze_12938 *
                                                              sizze_12937) +
                                                             segred_group_sizze_13144 *
                                                             sizze_12938 * 0 +
                                                             segred_group_sizze_13144 *
                                                             0 +
                                                             local_tid_13430 +
                                                             (i_13443 *
                                                              (segred_group_sizze_13144 *
                                                               sizze_12938) +
                                                              i_13444 *
                                                              segred_group_sizze_13144)] =
                    ((__global float *) who_mem_13264)[i_13443 * sizze_12932 +
                                                       i_13444];
            }
        }
    }
    for (int32_t i_13592 = 0; i_13592 < squot32(sizze_12933 +
                                                segred_group_sizze_13144 *
                                                num_groups_13146 - 1,
                                                segred_group_sizze_13144 *
                                                num_groups_13146); i_13592++) {
        gtid_13151 = local_tid_13430 + (squot32(phys_tid_13152,
                                                segred_group_sizze_13144) *
                                        squot32(sizze_12933 +
                                                segred_group_sizze_13144 *
                                                num_groups_13146 - 1,
                                                segred_group_sizze_13144 *
                                                num_groups_13146) + i_13592) *
            segred_group_sizze_13144;
        if (slt32(gtid_13151, sizze_12933)) {
            // apply map function
            {
                // save map-out results
                { }
                // load accumulator
                {
                    for (int32_t i_13593 = 0; i_13593 < sizze_12934;
                         i_13593++) {
                        for (int32_t i_13594 = 0; i_13594 < sizze_12935;
                             i_13594++) {
                            ((__global float *) mem_13285)[phys_tid_13152 *
                                                           (sizze_12935 *
                                                            sizze_12934) +
                                                           (i_13593 *
                                                            sizze_12935 +
                                                            i_13594)] =
                                ((__global
                                  float *) group_res_arr_mem_13424)[group_tid_13431 *
                                                                    (segred_group_sizze_13144 *
                                                                     sizze_12935 *
                                                                     sizze_12934) +
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935 *
                                                                    0 +
                                                                    segred_group_sizze_13144 *
                                                                    0 +
                                                                    local_tid_13430 +
                                                                    (i_13593 *
                                                                     (segred_group_sizze_13144 *
                                                                      sizze_12935) +
                                                                     i_13594 *
                                                                     segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13595 = 0; i_13595 < sizze_12937;
                         i_13595++) {
                        for (int32_t i_13596 = 0; i_13596 < sizze_12938;
                             i_13596++) {
                            ((__global float *) mem_13293)[phys_tid_13152 *
                                                           (sizze_12938 *
                                                            sizze_12937) +
                                                           (i_13595 *
                                                            sizze_12938 +
                                                            i_13596)] =
                                ((__global
                                  float *) group_res_arr_mem_13426)[group_tid_13431 *
                                                                    (segred_group_sizze_13144 *
                                                                     sizze_12938 *
                                                                     sizze_12937) +
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12938 *
                                                                    0 +
                                                                    segred_group_sizze_13144 *
                                                                    0 +
                                                                    local_tid_13430 +
                                                                    (i_13595 *
                                                                     (segred_group_sizze_13144 *
                                                                      sizze_12938) +
                                                                     i_13596 *
                                                                     segred_group_sizze_13144)];
                        }
                    }
                }
                // load new values
                {
                    for (int32_t i_13597 = 0; i_13597 < sizze_12934;
                         i_13597++) {
                        for (int32_t i_13598 = 0; i_13598 < sizze_12935;
                             i_13598++) {
                            ((__global float *) mem_13285)[(phys_tid_13152 +
                                                            num_threads_13277) *
                                                           (sizze_12935 *
                                                            sizze_12934) +
                                                           (i_13597 *
                                                            sizze_12935 +
                                                            i_13598)] =
                                ((__global float *) mem_13271)[gtid_13151 +
                                                               (i_13597 *
                                                                (sizze_12933 *
                                                                 sizze_12935) +
                                                                i_13598 *
                                                                sizze_12933)];
                        }
                    }
                    for (int32_t i_13599 = 0; i_13599 < sizze_12937;
                         i_13599++) {
                        for (int32_t i_13600 = 0; i_13600 < sizze_12938;
                             i_13600++) {
                            ((__global float *) mem_13293)[(phys_tid_13152 +
                                                            num_threads_13277) *
                                                           (sizze_12938 *
                                                            sizze_12937) +
                                                           (i_13599 *
                                                            sizze_12938 +
                                                            i_13600)] =
                                ((__global float *) mem_13276)[gtid_13151 +
                                                               (i_13599 *
                                                                (sizze_12936 *
                                                                 sizze_12938) +
                                                                i_13600 *
                                                                sizze_12936)];
                        }
                    }
                }
                // apply reduction operator
                {
                    for (int32_t i_13158 = 0; i_13158 < sizze_12934;
                         i_13158++) {
                        for (int32_t i_13162 = 0; i_13162 < sizze_12935;
                             i_13162++) {
                            float res_13016;
                            float redout_13164 = 0.0F;
                            
                            for (int32_t i_13165 = 0; i_13165 < sizze_12935;
                                 i_13165++) {
                                float x_13020 = ((__global
                                                  float *) mem_13285)[phys_tid_13152 *
                                                                      (sizze_12935 *
                                                                       sizze_12934) +
                                                                      (i_13158 *
                                                                       sizze_12935 +
                                                                       i_13165)];
                                float x_13021 = ((__global
                                                  float *) mem_13285)[(phys_tid_13152 +
                                                                       num_threads_13277) *
                                                                      (sizze_12935 *
                                                                       sizze_12934) +
                                                                      (i_13165 *
                                                                       sizze_12935 +
                                                                       i_13162)];
                                float res_13022 = x_13020 * x_13021;
                                float res_13019 = res_13022 + redout_13164;
                                float redout_tmp_13603 = res_13019;
                                
                                redout_13164 = redout_tmp_13603;
                            }
                            res_13016 = redout_13164;
                            
                            float res_13023 = 0.0F - res_13016;
                            float res_13024 = fpow32(2.7182817F, res_13023);
                            float res_13025 = 1.0F + res_13024;
                            float res_13026 = 1.0F / res_13025;
                            
                            ((__global float *) mem_13301)[phys_tid_13152 +
                                                           i_13162 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13026;
                        }
                        for (int32_t i_13604 = 0; i_13604 < sizze_12935;
                             i_13604++) {
                            ((__global float *) mem_13298)[phys_tid_13152 +
                                                           i_13158 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12935) +
                                                           i_13604 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13301)[phys_tid_13152 +
                                                               i_13604 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13170 = 0; i_13170 < sizze_12937;
                         i_13170++) {
                        for (int32_t i_13175 = 0; i_13175 < sizze_12935;
                             i_13175++) {
                            float res_13035;
                            float redout_13177 = 0.0F;
                            
                            for (int32_t i_13178 = 0; i_13178 < sizze_12938;
                                 i_13178++) {
                                float x_13039 = ((__global
                                                  float *) mem_13293)[phys_tid_13152 *
                                                                      (sizze_12938 *
                                                                       sizze_12937) +
                                                                      (i_13170 *
                                                                       sizze_12938 +
                                                                       i_13178)];
                                float x_13040 = ((__global
                                                  float *) mem_13298)[phys_tid_13152 +
                                                                      (i_13178 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13175 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13041 = x_13039 * x_13040;
                                float res_13038 = res_13041 + redout_13177;
                                float redout_tmp_13608 = res_13038;
                                
                                redout_13177 = redout_tmp_13608;
                            }
                            res_13035 = redout_13177;
                            
                            float res_13042 = 0.0F - res_13035;
                            float res_13043 = fpow32(2.7182817F, res_13042);
                            float res_13044 = 1.0F + res_13043;
                            float res_13045 = 1.0F / res_13044;
                            
                            ((__global float *) mem_13314)[phys_tid_13152 +
                                                           i_13175 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13045;
                        }
                        for (int32_t i_13181 = 0; i_13181 < sizze_12938;
                             i_13181++) {
                            float x_13048 = ((__global
                                              float *) mem_13293)[(phys_tid_13152 +
                                                                   num_threads_13277) *
                                                                  (sizze_12938 *
                                                                   sizze_12937) +
                                                                  (i_13170 *
                                                                   sizze_12938 +
                                                                   i_13181)];
                            float x_13049 = ((__global
                                              float *) mem_13314)[phys_tid_13152 +
                                                                  i_13181 *
                                                                  (num_groups_13146 *
                                                                   segred_group_sizze_13144)];
                            float res_13050 = x_13048 - x_13049;
                            
                            ((__global float *) mem_13317)[phys_tid_13152 +
                                                           i_13181 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13050;
                        }
                        for (int32_t i_13610 = 0; i_13610 < sizze_12938;
                             i_13610++) {
                            ((__global float *) mem_13306)[phys_tid_13152 +
                                                           i_13170 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12938) +
                                                           i_13610 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13317)[phys_tid_13152 +
                                                               i_13610 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                        for (int32_t i_13611 = 0; i_13611 < sizze_12935;
                             i_13611++) {
                            ((__global float *) mem_13311)[phys_tid_13152 +
                                                           i_13170 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12935) +
                                                           i_13611 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13314)[phys_tid_13152 +
                                                               i_13611 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13185 = 0; i_13185 < sizze_12937;
                         i_13185++) {
                        for (int32_t i_13189 = 0; i_13189 < sizze_12938;
                             i_13189++) {
                            float x_13058 = ((__global
                                              float *) mem_13306)[phys_tid_13152 +
                                                                  (i_13185 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12938) +
                                                                   i_13189 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float x_13059 = ((__global
                                              float *) mem_13311)[phys_tid_13152 +
                                                                  (i_13185 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935) +
                                                                   i_13189 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float res_13061 = x_13058 * x_13059;
                            float res_13062 = 1.0F - x_13059;
                            float res_13063 = res_13061 * res_13062;
                            
                            ((__global float *) mem_13325)[phys_tid_13152 +
                                                           i_13189 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13063;
                        }
                        for (int32_t i_13193 = 0; i_13193 < sizze_12938;
                             i_13193++) {
                            float x_13078 = ((__global
                                              float *) mem_13293)[phys_tid_13152 *
                                                                  (sizze_12938 *
                                                                   sizze_12937) +
                                                                  (i_13185 *
                                                                   sizze_12938 +
                                                                   i_13193)];
                            float res_13080;
                            float redout_13195 = 0.0F;
                            
                            for (int32_t i_13196 = 0; i_13196 < sizze_12938;
                                 i_13196++) {
                                float x_13084 = ((__global
                                                  float *) mem_13325)[phys_tid_13152 +
                                                                      i_13196 *
                                                                      (num_groups_13146 *
                                                                       segred_group_sizze_13144)];
                                float x_13085 = ((__global
                                                  float *) mem_13298)[phys_tid_13152 +
                                                                      (i_13193 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13196 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13086 = x_13084 * x_13085;
                                float res_13083 = res_13086 + redout_13195;
                                float redout_tmp_13615 = res_13083;
                                
                                redout_13195 = redout_tmp_13615;
                            }
                            res_13080 = redout_13195;
                            
                            float res_13087 = lr_12939 * res_13080;
                            float res_13088 = x_13078 + res_13087;
                            
                            ((__global float *) mem_13328)[phys_tid_13152 +
                                                           i_13193 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13088;
                        }
                        for (int32_t i_13616 = 0; i_13616 < sizze_12938;
                             i_13616++) {
                            ((__global float *) mem_13322)[phys_tid_13152 +
                                                           i_13185 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12938) +
                                                           i_13616 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13328)[phys_tid_13152 +
                                                               i_13616 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13199 = 0; i_13199 < sizze_12934;
                         i_13199++) {
                        for (int32_t i_13203 = 0; i_13203 < sizze_12935;
                             i_13203++) {
                            float x_13093 = ((__global
                                              float *) mem_13298)[phys_tid_13152 +
                                                                  (i_13199 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935) +
                                                                   i_13203 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float res_13094 = 1.0F - x_13093;
                            
                            ((__global float *) mem_13336)[phys_tid_13152 +
                                                           i_13203 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13094;
                        }
                        for (int32_t i_13619 = 0; i_13619 < sizze_12935;
                             i_13619++) {
                            ((__global float *) mem_13333)[phys_tid_13152 +
                                                           i_13199 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12935) +
                                                           i_13619 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13336)[phys_tid_13152 +
                                                               i_13619 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13207 = 0; i_13207 < sizze_12938;
                         i_13207++) {
                        for (int32_t i_13211 = 0; i_13211 < sizze_12938;
                             i_13211++) {
                            float x_13106 = ((__global
                                              float *) mem_13298)[phys_tid_13152 +
                                                                  (i_13207 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935) +
                                                                   i_13211 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float x_13107 = ((__global
                                              float *) mem_13333)[phys_tid_13152 +
                                                                  (i_13207 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935) +
                                                                   i_13211 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float res_13108;
                            float redout_13213 = 0.0F;
                            
                            for (int32_t i_13214 = 0; i_13214 < sizze_12937;
                                 i_13214++) {
                                float x_13112 = ((__global
                                                  float *) mem_13293)[phys_tid_13152 *
                                                                      (sizze_12938 *
                                                                       sizze_12937) +
                                                                      (i_13214 *
                                                                       sizze_12938 +
                                                                       i_13207)];
                                float x_13113 = ((__global
                                                  float *) mem_13306)[phys_tid_13152 +
                                                                      (i_13214 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12938) +
                                                                       i_13211 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13114 = x_13112 * x_13113;
                                float res_13111 = res_13114 + redout_13213;
                                float redout_tmp_13622 = res_13111;
                                
                                redout_13213 = redout_tmp_13622;
                            }
                            res_13108 = redout_13213;
                            
                            float res_13115 = x_13106 * res_13108;
                            float res_13116 = x_13107 * res_13115;
                            
                            ((__global float *) mem_13344)[phys_tid_13152 +
                                                           i_13211 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13116;
                        }
                        for (int32_t i_13217 = 0; i_13217 < sizze_12934;
                             i_13217++) {
                            float res_13120;
                            float redout_13219 = 0.0F;
                            
                            for (int32_t i_13220 = 0; i_13220 < sizze_12938;
                                 i_13220++) {
                                float x_13124 = ((__global
                                                  float *) mem_13344)[phys_tid_13152 +
                                                                      i_13220 *
                                                                      (num_groups_13146 *
                                                                       segred_group_sizze_13144)];
                                float x_13125 = ((__global
                                                  float *) mem_13285)[(phys_tid_13152 +
                                                                       num_threads_13277) *
                                                                      (sizze_12935 *
                                                                       sizze_12934) +
                                                                      (i_13217 *
                                                                       sizze_12935 +
                                                                       i_13220)];
                                float res_13126 = x_13124 * x_13125;
                                float res_13123 = res_13126 + redout_13219;
                                float redout_tmp_13624 = res_13123;
                                
                                redout_13219 = redout_tmp_13624;
                            }
                            res_13120 = redout_13219;
                            
                            float res_13127 = lr_12939 * res_13120;
                            
                            ((__global float *) mem_13347)[phys_tid_13152 +
                                                           i_13217 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13127;
                        }
                        for (int32_t i_13625 = 0; i_13625 < sizze_12934;
                             i_13625++) {
                            ((__global float *) mem_13341)[phys_tid_13152 +
                                                           i_13207 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12934) +
                                                           i_13625 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13347)[phys_tid_13152 +
                                                               i_13625 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13223 = 0; i_13223 < sizze_12934;
                         i_13223++) {
                        for (int32_t i_13227 = 0; i_13227 < sizze_12935;
                             i_13227++) {
                            float x_13134 = ((__global
                                              float *) mem_13285)[phys_tid_13152 *
                                                                  (sizze_12935 *
                                                                   sizze_12934) +
                                                                  (i_13223 *
                                                                   sizze_12935 +
                                                                   i_13227)];
                            float x_13135 = ((__global
                                              float *) mem_13341)[phys_tid_13152 +
                                                                  (i_13223 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12934) +
                                                                   i_13227 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float res_13136 = x_13134 + x_13135;
                            
                            ((__global float *) mem_13355)[phys_tid_13152 +
                                                           i_13227 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13136;
                        }
                        for (int32_t i_13628 = 0; i_13628 < sizze_12935;
                             i_13628++) {
                            ((__global float *) mem_13352)[phys_tid_13152 +
                                                           i_13223 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12935) +
                                                           i_13628 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13355)[phys_tid_13152 +
                                                               i_13628 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                    }
                    // store in accumulator
                    {
                        for (int32_t i_13629 = 0; i_13629 < sizze_12934;
                             i_13629++) {
                            for (int32_t i_13630 = 0; i_13630 < sizze_12935;
                                 i_13630++) {
                                ((__global
                                  float *) group_res_arr_mem_13424)[group_tid_13431 *
                                                                    (segred_group_sizze_13144 *
                                                                     sizze_12935 *
                                                                     sizze_12934) +
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935 *
                                                                    0 +
                                                                    segred_group_sizze_13144 *
                                                                    0 +
                                                                    local_tid_13430 +
                                                                    (i_13629 *
                                                                     (segred_group_sizze_13144 *
                                                                      sizze_12935) +
                                                                     i_13630 *
                                                                     segred_group_sizze_13144)] =
                                    ((__global
                                      float *) mem_13352)[phys_tid_13152 +
                                                          (i_13629 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12935) +
                                                           i_13630 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144))];
                            }
                        }
                        for (int32_t i_13631 = 0; i_13631 < sizze_12937;
                             i_13631++) {
                            for (int32_t i_13632 = 0; i_13632 < sizze_12938;
                                 i_13632++) {
                                ((__global
                                  float *) group_res_arr_mem_13426)[group_tid_13431 *
                                                                    (segred_group_sizze_13144 *
                                                                     sizze_12938 *
                                                                     sizze_12937) +
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12938 *
                                                                    0 +
                                                                    segred_group_sizze_13144 *
                                                                    0 +
                                                                    local_tid_13430 +
                                                                    (i_13631 *
                                                                     (segred_group_sizze_13144 *
                                                                      sizze_12938) +
                                                                     i_13632 *
                                                                     segred_group_sizze_13144)] =
                                    ((__global
                                      float *) mem_13322)[phys_tid_13152 +
                                                          (i_13631 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12938) +
                                                           i_13632 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144))];
                            }
                        }
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            for (int32_t i_13633 = 0; i_13633 < sizze_12934; i_13633++) {
                for (int32_t i_13634 = 0; i_13634 < sizze_12935; i_13634++) {
                    ((__global float *) mem_13285)[phys_tid_13152 *
                                                   (sizze_12935 * sizze_12934) +
                                                   (i_13633 * sizze_12935 +
                                                    i_13634)] = ((__global
                                                                  float *) group_res_arr_mem_13424)[group_tid_13431 *
                                                                                                    (segred_group_sizze_13144 *
                                                                                                     sizze_12935 *
                                                                                                     sizze_12934) +
                                                                                                    segred_group_sizze_13144 *
                                                                                                    sizze_12935 *
                                                                                                    0 +
                                                                                                    segred_group_sizze_13144 *
                                                                                                    0 +
                                                                                                    local_tid_13430 +
                                                                                                    (i_13633 *
                                                                                                     (segred_group_sizze_13144 *
                                                                                                      sizze_12935) +
                                                                                                     i_13634 *
                                                                                                     segred_group_sizze_13144)];
                }
            }
            for (int32_t i_13635 = 0; i_13635 < sizze_12937; i_13635++) {
                for (int32_t i_13636 = 0; i_13636 < sizze_12938; i_13636++) {
                    ((__global float *) mem_13293)[phys_tid_13152 *
                                                   (sizze_12938 * sizze_12937) +
                                                   (i_13635 * sizze_12938 +
                                                    i_13636)] = ((__global
                                                                  float *) group_res_arr_mem_13426)[group_tid_13431 *
                                                                                                    (segred_group_sizze_13144 *
                                                                                                     sizze_12938 *
                                                                                                     sizze_12937) +
                                                                                                    segred_group_sizze_13144 *
                                                                                                    sizze_12938 *
                                                                                                    0 +
                                                                                                    segred_group_sizze_13144 *
                                                                                                    0 +
                                                                                                    local_tid_13430 +
                                                                                                    (i_13635 *
                                                                                                     (segred_group_sizze_13144 *
                                                                                                      sizze_12938) +
                                                                                                     i_13636 *
                                                                                                     segred_group_sizze_13144)];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_13637;
        int32_t skip_waves_13638;
        
        offset_13637 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_13430, segred_group_sizze_13144)) {
                for (int32_t i_13639 = 0; i_13639 < sizze_12934; i_13639++) {
                    for (int32_t i_13640 = 0; i_13640 < sizze_12935;
                         i_13640++) {
                        ((__global float *) mem_13285)[phys_tid_13152 *
                                                       (sizze_12935 *
                                                        sizze_12934) +
                                                       (i_13639 * sizze_12935 +
                                                        i_13640)] = ((__global
                                                                      float *) mem_13285)[(global_tid_13429 +
                                                                                           offset_13637) *
                                                                                          (sizze_12935 *
                                                                                           sizze_12934) +
                                                                                          (i_13639 *
                                                                                           sizze_12935 +
                                                                                           i_13640)];
                    }
                }
                for (int32_t i_13641 = 0; i_13641 < sizze_12937; i_13641++) {
                    for (int32_t i_13642 = 0; i_13642 < sizze_12938;
                         i_13642++) {
                        ((__global float *) mem_13293)[phys_tid_13152 *
                                                       (sizze_12938 *
                                                        sizze_12937) +
                                                       (i_13641 * sizze_12938 +
                                                        i_13642)] = ((__global
                                                                      float *) mem_13293)[(global_tid_13429 +
                                                                                           offset_13637) *
                                                                                          (sizze_12938 *
                                                                                           sizze_12937) +
                                                                                          (i_13641 *
                                                                                           sizze_12938 +
                                                                                           i_13642)];
                    }
                }
            }
        }
        offset_13637 = 1;
        while (slt32(offset_13637, wave_sizze_13432)) {
            if (slt32(local_tid_13430 + offset_13637,
                      segred_group_sizze_13144) && ((local_tid_13430 -
                                                     squot32(local_tid_13430,
                                                             wave_sizze_13432) *
                                                     wave_sizze_13432) & (2 *
                                                                          offset_13637 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    for (int32_t i_13643 = 0; i_13643 < sizze_12934;
                         i_13643++) {
                        for (int32_t i_13644 = 0; i_13644 < sizze_12935;
                             i_13644++) {
                            ((volatile __global
                              float *) mem_13285)[(phys_tid_13152 +
                                                   num_threads_13277) *
                                                  (sizze_12935 * sizze_12934) +
                                                  (i_13643 * sizze_12935 +
                                                   i_13644)] =
                                ((volatile __global
                                  float *) mem_13285)[(global_tid_13429 +
                                                       offset_13637) *
                                                      (sizze_12935 *
                                                       sizze_12934) + (i_13643 *
                                                                       sizze_12935 +
                                                                       i_13644)];
                        }
                    }
                    for (int32_t i_13645 = 0; i_13645 < sizze_12937;
                         i_13645++) {
                        for (int32_t i_13646 = 0; i_13646 < sizze_12938;
                             i_13646++) {
                            ((volatile __global
                              float *) mem_13293)[(phys_tid_13152 +
                                                   num_threads_13277) *
                                                  (sizze_12938 * sizze_12937) +
                                                  (i_13645 * sizze_12938 +
                                                   i_13646)] =
                                ((volatile __global
                                  float *) mem_13293)[(global_tid_13429 +
                                                       offset_13637) *
                                                      (sizze_12938 *
                                                       sizze_12937) + (i_13645 *
                                                                       sizze_12938 +
                                                                       i_13646)];
                        }
                    }
                }
                // apply reduction operation
                {
                    for (int32_t i_13452 = 0; i_13452 < sizze_12934;
                         i_13452++) {
                        for (int32_t i_13456 = 0; i_13456 < sizze_12935;
                             i_13456++) {
                            float res_13457;
                            float redout_13458 = 0.0F;
                            
                            for (int32_t i_13459 = 0; i_13459 < sizze_12935;
                                 i_13459++) {
                                float x_13460 = ((volatile __global
                                                  float *) mem_13285)[phys_tid_13152 *
                                                                      (sizze_12935 *
                                                                       sizze_12934) +
                                                                      (i_13452 *
                                                                       sizze_12935 +
                                                                       i_13459)];
                                float x_13461 = ((volatile __global
                                                  float *) mem_13285)[(phys_tid_13152 +
                                                                       num_threads_13277) *
                                                                      (sizze_12935 *
                                                                       sizze_12934) +
                                                                      (i_13459 *
                                                                       sizze_12935 +
                                                                       i_13456)];
                                float res_13462 = x_13460 * x_13461;
                                float res_13463 = res_13462 + redout_13458;
                                float redout_tmp_13649 = res_13463;
                                
                                redout_13458 = redout_tmp_13649;
                            }
                            res_13457 = redout_13458;
                            
                            float res_13464 = 0.0F - res_13457;
                            float res_13465 = fpow32(2.7182817F, res_13464);
                            float res_13466 = 1.0F + res_13465;
                            float res_13467 = 1.0F / res_13466;
                            
                            ((volatile __global
                              float *) mem_13301)[phys_tid_13152 + i_13456 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                res_13467;
                        }
                        for (int32_t i_13650 = 0; i_13650 < sizze_12935;
                             i_13650++) {
                            ((volatile __global
                              float *) mem_13298)[phys_tid_13152 + i_13452 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144 *
                                                   sizze_12935) + i_13650 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                ((volatile __global
                                  float *) mem_13301)[phys_tid_13152 + i_13650 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13476 = 0; i_13476 < sizze_12937;
                         i_13476++) {
                        for (int32_t i_13480 = 0; i_13480 < sizze_12935;
                             i_13480++) {
                            float res_13481;
                            float redout_13482 = 0.0F;
                            
                            for (int32_t i_13483 = 0; i_13483 < sizze_12938;
                                 i_13483++) {
                                float x_13484 = ((volatile __global
                                                  float *) mem_13293)[phys_tid_13152 *
                                                                      (sizze_12938 *
                                                                       sizze_12937) +
                                                                      (i_13476 *
                                                                       sizze_12938 +
                                                                       i_13483)];
                                float x_13485 = ((volatile __global
                                                  float *) mem_13298)[phys_tid_13152 +
                                                                      (i_13483 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13480 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13486 = x_13484 * x_13485;
                                float res_13487 = res_13486 + redout_13482;
                                float redout_tmp_13654 = res_13487;
                                
                                redout_13482 = redout_tmp_13654;
                            }
                            res_13481 = redout_13482;
                            
                            float res_13488 = 0.0F - res_13481;
                            float res_13489 = fpow32(2.7182817F, res_13488);
                            float res_13490 = 1.0F + res_13489;
                            float res_13491 = 1.0F / res_13490;
                            
                            ((volatile __global
                              float *) mem_13314)[phys_tid_13152 + i_13480 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                res_13491;
                        }
                        for (int32_t i_13496 = 0; i_13496 < sizze_12938;
                             i_13496++) {
                            float x_13497 = ((volatile __global
                                              float *) mem_13293)[(phys_tid_13152 +
                                                                   num_threads_13277) *
                                                                  (sizze_12938 *
                                                                   sizze_12937) +
                                                                  (i_13476 *
                                                                   sizze_12938 +
                                                                   i_13496)];
                            float x_13498 = ((volatile __global
                                              float *) mem_13314)[phys_tid_13152 +
                                                                  i_13496 *
                                                                  (num_groups_13146 *
                                                                   segred_group_sizze_13144)];
                            float res_13499 = x_13497 - x_13498;
                            
                            ((volatile __global
                              float *) mem_13317)[phys_tid_13152 + i_13496 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                res_13499;
                        }
                        for (int32_t i_13656 = 0; i_13656 < sizze_12938;
                             i_13656++) {
                            ((volatile __global
                              float *) mem_13306)[phys_tid_13152 + i_13476 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144 *
                                                   sizze_12938) + i_13656 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                ((volatile __global
                                  float *) mem_13317)[phys_tid_13152 + i_13656 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)];
                        }
                        for (int32_t i_13657 = 0; i_13657 < sizze_12935;
                             i_13657++) {
                            ((volatile __global
                              float *) mem_13311)[phys_tid_13152 + i_13476 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144 *
                                                   sizze_12935) + i_13657 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                ((volatile __global
                                  float *) mem_13314)[phys_tid_13152 + i_13657 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13506 = 0; i_13506 < sizze_12937;
                         i_13506++) {
                        for (int32_t i_13510 = 0; i_13510 < sizze_12938;
                             i_13510++) {
                            float x_13511 = ((volatile __global
                                              float *) mem_13306)[phys_tid_13152 +
                                                                  (i_13506 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12938) +
                                                                   i_13510 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float x_13512 = ((volatile __global
                                              float *) mem_13311)[phys_tid_13152 +
                                                                  (i_13506 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935) +
                                                                   i_13510 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float res_13513 = x_13511 * x_13512;
                            float res_13514 = 1.0F - x_13512;
                            float res_13515 = res_13513 * res_13514;
                            
                            ((volatile __global
                              float *) mem_13325)[phys_tid_13152 + i_13510 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                res_13515;
                        }
                        for (int32_t i_13520 = 0; i_13520 < sizze_12938;
                             i_13520++) {
                            float x_13521 = ((volatile __global
                                              float *) mem_13293)[phys_tid_13152 *
                                                                  (sizze_12938 *
                                                                   sizze_12937) +
                                                                  (i_13506 *
                                                                   sizze_12938 +
                                                                   i_13520)];
                            float res_13522;
                            float redout_13523 = 0.0F;
                            
                            for (int32_t i_13524 = 0; i_13524 < sizze_12938;
                                 i_13524++) {
                                float x_13525 = ((volatile __global
                                                  float *) mem_13325)[phys_tid_13152 +
                                                                      i_13524 *
                                                                      (num_groups_13146 *
                                                                       segred_group_sizze_13144)];
                                float x_13526 = ((volatile __global
                                                  float *) mem_13298)[phys_tid_13152 +
                                                                      (i_13520 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13524 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13527 = x_13525 * x_13526;
                                float res_13528 = res_13527 + redout_13523;
                                float redout_tmp_13661 = res_13528;
                                
                                redout_13523 = redout_tmp_13661;
                            }
                            res_13522 = redout_13523;
                            
                            float res_13529 = lr_12939 * res_13522;
                            float res_13530 = x_13521 + res_13529;
                            
                            ((volatile __global
                              float *) mem_13328)[phys_tid_13152 + i_13520 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                res_13530;
                        }
                        for (int32_t i_13662 = 0; i_13662 < sizze_12938;
                             i_13662++) {
                            ((volatile __global
                              float *) mem_13322)[phys_tid_13152 + i_13506 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144 *
                                                   sizze_12938) + i_13662 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                ((volatile __global
                                  float *) mem_13328)[phys_tid_13152 + i_13662 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13536 = 0; i_13536 < sizze_12934;
                         i_13536++) {
                        for (int32_t i_13540 = 0; i_13540 < sizze_12935;
                             i_13540++) {
                            float x_13541 = ((volatile __global
                                              float *) mem_13298)[phys_tid_13152 +
                                                                  (i_13536 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935) +
                                                                   i_13540 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float res_13542 = 1.0F - x_13541;
                            
                            ((volatile __global
                              float *) mem_13336)[phys_tid_13152 + i_13540 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                res_13542;
                        }
                        for (int32_t i_13665 = 0; i_13665 < sizze_12935;
                             i_13665++) {
                            ((volatile __global
                              float *) mem_13333)[phys_tid_13152 + i_13536 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144 *
                                                   sizze_12935) + i_13665 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                ((volatile __global
                                  float *) mem_13336)[phys_tid_13152 + i_13665 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13548 = 0; i_13548 < sizze_12938;
                         i_13548++) {
                        for (int32_t i_13552 = 0; i_13552 < sizze_12938;
                             i_13552++) {
                            float x_13553 = ((volatile __global
                                              float *) mem_13298)[phys_tid_13152 +
                                                                  (i_13548 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935) +
                                                                   i_13552 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float x_13554 = ((volatile __global
                                              float *) mem_13333)[phys_tid_13152 +
                                                                  (i_13548 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935) +
                                                                   i_13552 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float res_13555;
                            float redout_13556 = 0.0F;
                            
                            for (int32_t i_13557 = 0; i_13557 < sizze_12937;
                                 i_13557++) {
                                float x_13558 = ((volatile __global
                                                  float *) mem_13293)[phys_tid_13152 *
                                                                      (sizze_12938 *
                                                                       sizze_12937) +
                                                                      (i_13557 *
                                                                       sizze_12938 +
                                                                       i_13548)];
                                float x_13559 = ((volatile __global
                                                  float *) mem_13306)[phys_tid_13152 +
                                                                      (i_13557 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12938) +
                                                                       i_13552 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13560 = x_13558 * x_13559;
                                float res_13561 = res_13560 + redout_13556;
                                float redout_tmp_13668 = res_13561;
                                
                                redout_13556 = redout_tmp_13668;
                            }
                            res_13555 = redout_13556;
                            
                            float res_13562 = x_13553 * res_13555;
                            float res_13563 = x_13554 * res_13562;
                            
                            ((volatile __global
                              float *) mem_13344)[phys_tid_13152 + i_13552 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                res_13563;
                        }
                        for (int32_t i_13568 = 0; i_13568 < sizze_12934;
                             i_13568++) {
                            float res_13569;
                            float redout_13570 = 0.0F;
                            
                            for (int32_t i_13571 = 0; i_13571 < sizze_12938;
                                 i_13571++) {
                                float x_13572 = ((volatile __global
                                                  float *) mem_13344)[phys_tid_13152 +
                                                                      i_13571 *
                                                                      (num_groups_13146 *
                                                                       segred_group_sizze_13144)];
                                float x_13573 = ((volatile __global
                                                  float *) mem_13285)[(phys_tid_13152 +
                                                                       num_threads_13277) *
                                                                      (sizze_12935 *
                                                                       sizze_12934) +
                                                                      (i_13568 *
                                                                       sizze_12935 +
                                                                       i_13571)];
                                float res_13574 = x_13572 * x_13573;
                                float res_13575 = res_13574 + redout_13570;
                                float redout_tmp_13670 = res_13575;
                                
                                redout_13570 = redout_tmp_13670;
                            }
                            res_13569 = redout_13570;
                            
                            float res_13576 = lr_12939 * res_13569;
                            
                            ((volatile __global
                              float *) mem_13347)[phys_tid_13152 + i_13568 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                res_13576;
                        }
                        for (int32_t i_13671 = 0; i_13671 < sizze_12934;
                             i_13671++) {
                            ((volatile __global
                              float *) mem_13341)[phys_tid_13152 + i_13548 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144 *
                                                   sizze_12934) + i_13671 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                ((volatile __global
                                  float *) mem_13347)[phys_tid_13152 + i_13671 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13582 = 0; i_13582 < sizze_12934;
                         i_13582++) {
                        for (int32_t i_13586 = 0; i_13586 < sizze_12935;
                             i_13586++) {
                            float x_13587 = ((volatile __global
                                              float *) mem_13285)[phys_tid_13152 *
                                                                  (sizze_12935 *
                                                                   sizze_12934) +
                                                                  (i_13582 *
                                                                   sizze_12935 +
                                                                   i_13586)];
                            float x_13588 = ((volatile __global
                                              float *) mem_13341)[phys_tid_13152 +
                                                                  (i_13582 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12934) +
                                                                   i_13586 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float res_13589 = x_13587 + x_13588;
                            
                            ((volatile __global
                              float *) mem_13355)[phys_tid_13152 + i_13586 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                res_13589;
                        }
                        for (int32_t i_13674 = 0; i_13674 < sizze_12935;
                             i_13674++) {
                            ((volatile __global
                              float *) mem_13352)[phys_tid_13152 + i_13582 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144 *
                                                   sizze_12935) + i_13674 *
                                                  (num_groups_13146 *
                                                   segred_group_sizze_13144)] =
                                ((volatile __global
                                  float *) mem_13355)[phys_tid_13152 + i_13674 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13675 = 0; i_13675 < sizze_12934;
                         i_13675++) {
                        for (int32_t i_13676 = 0; i_13676 < sizze_12935;
                             i_13676++) {
                            ((volatile __global
                              float *) mem_13285)[phys_tid_13152 *
                                                  (sizze_12935 * sizze_12934) +
                                                  (i_13675 * sizze_12935 +
                                                   i_13676)] =
                                ((volatile __global
                                  float *) mem_13352)[phys_tid_13152 +
                                                      (i_13675 *
                                                       (num_groups_13146 *
                                                        segred_group_sizze_13144 *
                                                        sizze_12935) + i_13676 *
                                                       (num_groups_13146 *
                                                        segred_group_sizze_13144))];
                        }
                    }
                    for (int32_t i_13677 = 0; i_13677 < sizze_12937;
                         i_13677++) {
                        for (int32_t i_13678 = 0; i_13678 < sizze_12938;
                             i_13678++) {
                            ((volatile __global
                              float *) mem_13293)[phys_tid_13152 *
                                                  (sizze_12938 * sizze_12937) +
                                                  (i_13677 * sizze_12938 +
                                                   i_13678)] =
                                ((volatile __global
                                  float *) mem_13322)[phys_tid_13152 +
                                                      (i_13677 *
                                                       (num_groups_13146 *
                                                        segred_group_sizze_13144 *
                                                        sizze_12938) + i_13678 *
                                                       (num_groups_13146 *
                                                        segred_group_sizze_13144))];
                        }
                    }
                }
                // write result of operation
                { }
            }
            offset_13637 *= 2;
        }
        skip_waves_13638 = 1;
        while (slt32(skip_waves_13638, squot32(segred_group_sizze_13144 +
                                               wave_sizze_13432 - 1,
                                               wave_sizze_13432))) {
            barrier(CLK_GLOBAL_MEM_FENCE);
            offset_13637 = skip_waves_13638 * wave_sizze_13432;
            if (slt32(local_tid_13430 + offset_13637,
                      segred_group_sizze_13144) && ((local_tid_13430 -
                                                     squot32(local_tid_13430,
                                                             wave_sizze_13432) *
                                                     wave_sizze_13432) == 0 &&
                                                    (squot32(local_tid_13430,
                                                             wave_sizze_13432) &
                                                     (2 * skip_waves_13638 -
                                                      1)) == 0)) {
                // read array element
                {
                    for (int32_t i_13679 = 0; i_13679 < sizze_12934;
                         i_13679++) {
                        for (int32_t i_13680 = 0; i_13680 < sizze_12935;
                             i_13680++) {
                            ((__global float *) mem_13285)[(phys_tid_13152 +
                                                            num_threads_13277) *
                                                           (sizze_12935 *
                                                            sizze_12934) +
                                                           (i_13679 *
                                                            sizze_12935 +
                                                            i_13680)] =
                                ((__global
                                  float *) mem_13285)[(global_tid_13429 +
                                                       offset_13637) *
                                                      (sizze_12935 *
                                                       sizze_12934) + (i_13679 *
                                                                       sizze_12935 +
                                                                       i_13680)];
                        }
                    }
                    for (int32_t i_13681 = 0; i_13681 < sizze_12937;
                         i_13681++) {
                        for (int32_t i_13682 = 0; i_13682 < sizze_12938;
                             i_13682++) {
                            ((__global float *) mem_13293)[(phys_tid_13152 +
                                                            num_threads_13277) *
                                                           (sizze_12938 *
                                                            sizze_12937) +
                                                           (i_13681 *
                                                            sizze_12938 +
                                                            i_13682)] =
                                ((__global
                                  float *) mem_13293)[(global_tid_13429 +
                                                       offset_13637) *
                                                      (sizze_12938 *
                                                       sizze_12937) + (i_13681 *
                                                                       sizze_12938 +
                                                                       i_13682)];
                        }
                    }
                }
                // apply reduction operation
                {
                    for (int32_t i_13452 = 0; i_13452 < sizze_12934;
                         i_13452++) {
                        for (int32_t i_13456 = 0; i_13456 < sizze_12935;
                             i_13456++) {
                            float res_13457;
                            float redout_13458 = 0.0F;
                            
                            for (int32_t i_13459 = 0; i_13459 < sizze_12935;
                                 i_13459++) {
                                float x_13460 = ((__global
                                                  float *) mem_13285)[phys_tid_13152 *
                                                                      (sizze_12935 *
                                                                       sizze_12934) +
                                                                      (i_13452 *
                                                                       sizze_12935 +
                                                                       i_13459)];
                                float x_13461 = ((__global
                                                  float *) mem_13285)[(phys_tid_13152 +
                                                                       num_threads_13277) *
                                                                      (sizze_12935 *
                                                                       sizze_12934) +
                                                                      (i_13459 *
                                                                       sizze_12935 +
                                                                       i_13456)];
                                float res_13462 = x_13460 * x_13461;
                                float res_13463 = res_13462 + redout_13458;
                                float redout_tmp_13685 = res_13463;
                                
                                redout_13458 = redout_tmp_13685;
                            }
                            res_13457 = redout_13458;
                            
                            float res_13464 = 0.0F - res_13457;
                            float res_13465 = fpow32(2.7182817F, res_13464);
                            float res_13466 = 1.0F + res_13465;
                            float res_13467 = 1.0F / res_13466;
                            
                            ((__global float *) mem_13301)[phys_tid_13152 +
                                                           i_13456 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13467;
                        }
                        for (int32_t i_13686 = 0; i_13686 < sizze_12935;
                             i_13686++) {
                            ((__global float *) mem_13298)[phys_tid_13152 +
                                                           i_13452 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12935) +
                                                           i_13686 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13301)[phys_tid_13152 +
                                                               i_13686 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13476 = 0; i_13476 < sizze_12937;
                         i_13476++) {
                        for (int32_t i_13480 = 0; i_13480 < sizze_12935;
                             i_13480++) {
                            float res_13481;
                            float redout_13482 = 0.0F;
                            
                            for (int32_t i_13483 = 0; i_13483 < sizze_12938;
                                 i_13483++) {
                                float x_13484 = ((__global
                                                  float *) mem_13293)[phys_tid_13152 *
                                                                      (sizze_12938 *
                                                                       sizze_12937) +
                                                                      (i_13476 *
                                                                       sizze_12938 +
                                                                       i_13483)];
                                float x_13485 = ((__global
                                                  float *) mem_13298)[phys_tid_13152 +
                                                                      (i_13483 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13480 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13486 = x_13484 * x_13485;
                                float res_13487 = res_13486 + redout_13482;
                                float redout_tmp_13690 = res_13487;
                                
                                redout_13482 = redout_tmp_13690;
                            }
                            res_13481 = redout_13482;
                            
                            float res_13488 = 0.0F - res_13481;
                            float res_13489 = fpow32(2.7182817F, res_13488);
                            float res_13490 = 1.0F + res_13489;
                            float res_13491 = 1.0F / res_13490;
                            
                            ((__global float *) mem_13314)[phys_tid_13152 +
                                                           i_13480 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13491;
                        }
                        for (int32_t i_13496 = 0; i_13496 < sizze_12938;
                             i_13496++) {
                            float x_13497 = ((__global
                                              float *) mem_13293)[(phys_tid_13152 +
                                                                   num_threads_13277) *
                                                                  (sizze_12938 *
                                                                   sizze_12937) +
                                                                  (i_13476 *
                                                                   sizze_12938 +
                                                                   i_13496)];
                            float x_13498 = ((__global
                                              float *) mem_13314)[phys_tid_13152 +
                                                                  i_13496 *
                                                                  (num_groups_13146 *
                                                                   segred_group_sizze_13144)];
                            float res_13499 = x_13497 - x_13498;
                            
                            ((__global float *) mem_13317)[phys_tid_13152 +
                                                           i_13496 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13499;
                        }
                        for (int32_t i_13692 = 0; i_13692 < sizze_12938;
                             i_13692++) {
                            ((__global float *) mem_13306)[phys_tid_13152 +
                                                           i_13476 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12938) +
                                                           i_13692 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13317)[phys_tid_13152 +
                                                               i_13692 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                        for (int32_t i_13693 = 0; i_13693 < sizze_12935;
                             i_13693++) {
                            ((__global float *) mem_13311)[phys_tid_13152 +
                                                           i_13476 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12935) +
                                                           i_13693 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13314)[phys_tid_13152 +
                                                               i_13693 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13506 = 0; i_13506 < sizze_12937;
                         i_13506++) {
                        for (int32_t i_13510 = 0; i_13510 < sizze_12938;
                             i_13510++) {
                            float x_13511 = ((__global
                                              float *) mem_13306)[phys_tid_13152 +
                                                                  (i_13506 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12938) +
                                                                   i_13510 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float x_13512 = ((__global
                                              float *) mem_13311)[phys_tid_13152 +
                                                                  (i_13506 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935) +
                                                                   i_13510 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float res_13513 = x_13511 * x_13512;
                            float res_13514 = 1.0F - x_13512;
                            float res_13515 = res_13513 * res_13514;
                            
                            ((__global float *) mem_13325)[phys_tid_13152 +
                                                           i_13510 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13515;
                        }
                        for (int32_t i_13520 = 0; i_13520 < sizze_12938;
                             i_13520++) {
                            float x_13521 = ((__global
                                              float *) mem_13293)[phys_tid_13152 *
                                                                  (sizze_12938 *
                                                                   sizze_12937) +
                                                                  (i_13506 *
                                                                   sizze_12938 +
                                                                   i_13520)];
                            float res_13522;
                            float redout_13523 = 0.0F;
                            
                            for (int32_t i_13524 = 0; i_13524 < sizze_12938;
                                 i_13524++) {
                                float x_13525 = ((__global
                                                  float *) mem_13325)[phys_tid_13152 +
                                                                      i_13524 *
                                                                      (num_groups_13146 *
                                                                       segred_group_sizze_13144)];
                                float x_13526 = ((__global
                                                  float *) mem_13298)[phys_tid_13152 +
                                                                      (i_13520 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13524 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13527 = x_13525 * x_13526;
                                float res_13528 = res_13527 + redout_13523;
                                float redout_tmp_13697 = res_13528;
                                
                                redout_13523 = redout_tmp_13697;
                            }
                            res_13522 = redout_13523;
                            
                            float res_13529 = lr_12939 * res_13522;
                            float res_13530 = x_13521 + res_13529;
                            
                            ((__global float *) mem_13328)[phys_tid_13152 +
                                                           i_13520 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13530;
                        }
                        for (int32_t i_13698 = 0; i_13698 < sizze_12938;
                             i_13698++) {
                            ((__global float *) mem_13322)[phys_tid_13152 +
                                                           i_13506 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12938) +
                                                           i_13698 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13328)[phys_tid_13152 +
                                                               i_13698 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13536 = 0; i_13536 < sizze_12934;
                         i_13536++) {
                        for (int32_t i_13540 = 0; i_13540 < sizze_12935;
                             i_13540++) {
                            float x_13541 = ((__global
                                              float *) mem_13298)[phys_tid_13152 +
                                                                  (i_13536 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935) +
                                                                   i_13540 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float res_13542 = 1.0F - x_13541;
                            
                            ((__global float *) mem_13336)[phys_tid_13152 +
                                                           i_13540 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13542;
                        }
                        for (int32_t i_13701 = 0; i_13701 < sizze_12935;
                             i_13701++) {
                            ((__global float *) mem_13333)[phys_tid_13152 +
                                                           i_13536 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12935) +
                                                           i_13701 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13336)[phys_tid_13152 +
                                                               i_13701 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13548 = 0; i_13548 < sizze_12938;
                         i_13548++) {
                        for (int32_t i_13552 = 0; i_13552 < sizze_12938;
                             i_13552++) {
                            float x_13553 = ((__global
                                              float *) mem_13298)[phys_tid_13152 +
                                                                  (i_13548 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935) +
                                                                   i_13552 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float x_13554 = ((__global
                                              float *) mem_13333)[phys_tid_13152 +
                                                                  (i_13548 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12935) +
                                                                   i_13552 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float res_13555;
                            float redout_13556 = 0.0F;
                            
                            for (int32_t i_13557 = 0; i_13557 < sizze_12937;
                                 i_13557++) {
                                float x_13558 = ((__global
                                                  float *) mem_13293)[phys_tid_13152 *
                                                                      (sizze_12938 *
                                                                       sizze_12937) +
                                                                      (i_13557 *
                                                                       sizze_12938 +
                                                                       i_13548)];
                                float x_13559 = ((__global
                                                  float *) mem_13306)[phys_tid_13152 +
                                                                      (i_13557 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12938) +
                                                                       i_13552 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13560 = x_13558 * x_13559;
                                float res_13561 = res_13560 + redout_13556;
                                float redout_tmp_13704 = res_13561;
                                
                                redout_13556 = redout_tmp_13704;
                            }
                            res_13555 = redout_13556;
                            
                            float res_13562 = x_13553 * res_13555;
                            float res_13563 = x_13554 * res_13562;
                            
                            ((__global float *) mem_13344)[phys_tid_13152 +
                                                           i_13552 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13563;
                        }
                        for (int32_t i_13568 = 0; i_13568 < sizze_12934;
                             i_13568++) {
                            float res_13569;
                            float redout_13570 = 0.0F;
                            
                            for (int32_t i_13571 = 0; i_13571 < sizze_12938;
                                 i_13571++) {
                                float x_13572 = ((__global
                                                  float *) mem_13344)[phys_tid_13152 +
                                                                      i_13571 *
                                                                      (num_groups_13146 *
                                                                       segred_group_sizze_13144)];
                                float x_13573 = ((__global
                                                  float *) mem_13285)[(phys_tid_13152 +
                                                                       num_threads_13277) *
                                                                      (sizze_12935 *
                                                                       sizze_12934) +
                                                                      (i_13568 *
                                                                       sizze_12935 +
                                                                       i_13571)];
                                float res_13574 = x_13572 * x_13573;
                                float res_13575 = res_13574 + redout_13570;
                                float redout_tmp_13706 = res_13575;
                                
                                redout_13570 = redout_tmp_13706;
                            }
                            res_13569 = redout_13570;
                            
                            float res_13576 = lr_12939 * res_13569;
                            
                            ((__global float *) mem_13347)[phys_tid_13152 +
                                                           i_13568 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13576;
                        }
                        for (int32_t i_13707 = 0; i_13707 < sizze_12934;
                             i_13707++) {
                            ((__global float *) mem_13341)[phys_tid_13152 +
                                                           i_13548 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12934) +
                                                           i_13707 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13347)[phys_tid_13152 +
                                                               i_13707 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13582 = 0; i_13582 < sizze_12934;
                         i_13582++) {
                        for (int32_t i_13586 = 0; i_13586 < sizze_12935;
                             i_13586++) {
                            float x_13587 = ((__global
                                              float *) mem_13285)[phys_tid_13152 *
                                                                  (sizze_12935 *
                                                                   sizze_12934) +
                                                                  (i_13582 *
                                                                   sizze_12935 +
                                                                   i_13586)];
                            float x_13588 = ((__global
                                              float *) mem_13341)[phys_tid_13152 +
                                                                  (i_13582 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144 *
                                                                    sizze_12934) +
                                                                   i_13586 *
                                                                   (num_groups_13146 *
                                                                    segred_group_sizze_13144))];
                            float res_13589 = x_13587 + x_13588;
                            
                            ((__global float *) mem_13355)[phys_tid_13152 +
                                                           i_13586 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                res_13589;
                        }
                        for (int32_t i_13710 = 0; i_13710 < sizze_12935;
                             i_13710++) {
                            ((__global float *) mem_13352)[phys_tid_13152 +
                                                           i_13582 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12935) +
                                                           i_13710 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144)] =
                                ((__global float *) mem_13355)[phys_tid_13152 +
                                                               i_13710 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)];
                        }
                    }
                    for (int32_t i_13711 = 0; i_13711 < sizze_12934;
                         i_13711++) {
                        for (int32_t i_13712 = 0; i_13712 < sizze_12935;
                             i_13712++) {
                            ((__global float *) mem_13285)[phys_tid_13152 *
                                                           (sizze_12935 *
                                                            sizze_12934) +
                                                           (i_13711 *
                                                            sizze_12935 +
                                                            i_13712)] =
                                ((__global float *) mem_13352)[phys_tid_13152 +
                                                               (i_13711 *
                                                                (num_groups_13146 *
                                                                 segred_group_sizze_13144 *
                                                                 sizze_12935) +
                                                                i_13712 *
                                                                (num_groups_13146 *
                                                                 segred_group_sizze_13144))];
                        }
                    }
                    for (int32_t i_13713 = 0; i_13713 < sizze_12937;
                         i_13713++) {
                        for (int32_t i_13714 = 0; i_13714 < sizze_12938;
                             i_13714++) {
                            ((__global float *) mem_13293)[phys_tid_13152 *
                                                           (sizze_12938 *
                                                            sizze_12937) +
                                                           (i_13713 *
                                                            sizze_12938 +
                                                            i_13714)] =
                                ((__global float *) mem_13322)[phys_tid_13152 +
                                                               (i_13713 *
                                                                (num_groups_13146 *
                                                                 segred_group_sizze_13144 *
                                                                 sizze_12938) +
                                                                i_13714 *
                                                                (num_groups_13146 *
                                                                 segred_group_sizze_13144))];
                        }
                    }
                }
                // write result of operation
                { }
            }
            skip_waves_13638 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_13430 == 0) {
                for (int32_t i_13715 = 0; i_13715 < sizze_12934; i_13715++) {
                    for (int32_t i_13716 = 0; i_13716 < sizze_12935;
                         i_13716++) {
                        ((__global
                          float *) group_res_arr_mem_13424)[group_tid_13431 *
                                                            (segred_group_sizze_13144 *
                                                             sizze_12935 *
                                                             sizze_12934) +
                                                            segred_group_sizze_13144 *
                                                            sizze_12935 * 0 +
                                                            segred_group_sizze_13144 *
                                                            0 +
                                                            local_tid_13430 +
                                                            (i_13715 *
                                                             (segred_group_sizze_13144 *
                                                              sizze_12935) +
                                                             i_13716 *
                                                             segred_group_sizze_13144)] =
                            ((__global float *) mem_13285)[phys_tid_13152 *
                                                           (sizze_12935 *
                                                            sizze_12934) +
                                                           (i_13715 *
                                                            sizze_12935 +
                                                            i_13716)];
                    }
                }
                for (int32_t i_13717 = 0; i_13717 < sizze_12937; i_13717++) {
                    for (int32_t i_13718 = 0; i_13718 < sizze_12938;
                         i_13718++) {
                        ((__global
                          float *) group_res_arr_mem_13426)[group_tid_13431 *
                                                            (segred_group_sizze_13144 *
                                                             sizze_12938 *
                                                             sizze_12937) +
                                                            segred_group_sizze_13144 *
                                                            sizze_12938 * 0 +
                                                            segred_group_sizze_13144 *
                                                            0 +
                                                            local_tid_13430 +
                                                            (i_13717 *
                                                             (segred_group_sizze_13144 *
                                                              sizze_12938) +
                                                             i_13718 *
                                                             segred_group_sizze_13144)] =
                            ((__global float *) mem_13293)[phys_tid_13152 *
                                                           (sizze_12938 *
                                                            sizze_12937) +
                                                           (i_13717 *
                                                            sizze_12938 +
                                                            i_13718)];
                    }
                }
            }
        }
        // first thread keeps accumulator; others reset to neutral element
        {
            if (!(local_tid_13430 == 0)) {
                for (int32_t i_13719 = 0; i_13719 < sizze_12934; i_13719++) {
                    for (int32_t i_13720 = 0; i_13720 < sizze_12935;
                         i_13720++) {
                        ((__global
                          float *) group_res_arr_mem_13424)[group_tid_13431 *
                                                            (segred_group_sizze_13144 *
                                                             sizze_12935 *
                                                             sizze_12934) +
                                                            segred_group_sizze_13144 *
                                                            sizze_12935 * 0 +
                                                            segred_group_sizze_13144 *
                                                            0 +
                                                            local_tid_13430 +
                                                            (i_13719 *
                                                             (segred_group_sizze_13144 *
                                                              sizze_12935) +
                                                             i_13720 *
                                                             segred_group_sizze_13144)] =
                            ((__global float *) wih_mem_13263)[i_13719 *
                                                               sizze_12930 +
                                                               i_13720];
                    }
                }
                for (int32_t i_13721 = 0; i_13721 < sizze_12937; i_13721++) {
                    for (int32_t i_13722 = 0; i_13722 < sizze_12938;
                         i_13722++) {
                        ((__global
                          float *) group_res_arr_mem_13426)[group_tid_13431 *
                                                            (segred_group_sizze_13144 *
                                                             sizze_12938 *
                                                             sizze_12937) +
                                                            segred_group_sizze_13144 *
                                                            sizze_12938 * 0 +
                                                            segred_group_sizze_13144 *
                                                            0 +
                                                            local_tid_13430 +
                                                            (i_13721 *
                                                             (segred_group_sizze_13144 *
                                                              sizze_12938) +
                                                             i_13722 *
                                                             segred_group_sizze_13144)] =
                            ((__global float *) who_mem_13264)[i_13721 *
                                                               sizze_12932 +
                                                               i_13722];
                    }
                }
            }
        }
    }
    for (int32_t i_13723 = 0; i_13723 < sizze_12934; i_13723++) {
        for (int32_t i_13724 = 0; i_13724 < sizze_12935; i_13724++) {
            ((__global float *) mem_13285)[phys_tid_13152 * (sizze_12935 *
                                                             sizze_12934) +
                                           (i_13723 * sizze_12935 + i_13724)] =
                ((__global float *) group_res_arr_mem_13424)[group_tid_13431 *
                                                             (segred_group_sizze_13144 *
                                                              sizze_12935 *
                                                              sizze_12934) +
                                                             segred_group_sizze_13144 *
                                                             sizze_12935 * 0 +
                                                             segred_group_sizze_13144 *
                                                             0 +
                                                             local_tid_13430 +
                                                             (i_13723 *
                                                              (segred_group_sizze_13144 *
                                                               sizze_12935) +
                                                              i_13724 *
                                                              segred_group_sizze_13144)];
        }
    }
    for (int32_t i_13725 = 0; i_13725 < sizze_12937; i_13725++) {
        for (int32_t i_13726 = 0; i_13726 < sizze_12938; i_13726++) {
            ((__global float *) mem_13293)[phys_tid_13152 * (sizze_12938 *
                                                             sizze_12937) +
                                           (i_13725 * sizze_12938 + i_13726)] =
                ((__global float *) group_res_arr_mem_13426)[group_tid_13431 *
                                                             (segred_group_sizze_13144 *
                                                              sizze_12938 *
                                                              sizze_12937) +
                                                             segred_group_sizze_13144 *
                                                             sizze_12938 * 0 +
                                                             segred_group_sizze_13144 *
                                                             0 +
                                                             local_tid_13430 +
                                                             (i_13725 *
                                                              (segred_group_sizze_13144 *
                                                               sizze_12938) +
                                                              i_13726 *
                                                              segred_group_sizze_13144)];
        }
    }
    
    int32_t old_counter_13727;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_13430 == 0) {
            for (int32_t i_13728 = 0; i_13728 < sizze_12934; i_13728++) {
                for (int32_t i_13729 = 0; i_13729 < sizze_12935; i_13729++) {
                    ((__global
                      float *) group_res_arr_mem_13424)[group_tid_13431 *
                                                        (segred_group_sizze_13144 *
                                                         sizze_12935 *
                                                         sizze_12934) +
                                                        segred_group_sizze_13144 *
                                                        sizze_12935 * 0 +
                                                        segred_group_sizze_13144 *
                                                        0 + (i_13728 *
                                                             (segred_group_sizze_13144 *
                                                              sizze_12935) +
                                                             i_13729 *
                                                             segred_group_sizze_13144)] =
                        ((__global
                          float *) group_res_arr_mem_13424)[group_tid_13431 *
                                                            (segred_group_sizze_13144 *
                                                             sizze_12935 *
                                                             sizze_12934) +
                                                            segred_group_sizze_13144 *
                                                            sizze_12935 * 0 +
                                                            segred_group_sizze_13144 *
                                                            0 +
                                                            local_tid_13430 +
                                                            (i_13728 *
                                                             (segred_group_sizze_13144 *
                                                              sizze_12935) +
                                                             i_13729 *
                                                             segred_group_sizze_13144)];
                }
            }
            for (int32_t i_13730 = 0; i_13730 < sizze_12937; i_13730++) {
                for (int32_t i_13731 = 0; i_13731 < sizze_12938; i_13731++) {
                    ((__global
                      float *) group_res_arr_mem_13426)[group_tid_13431 *
                                                        (segred_group_sizze_13144 *
                                                         sizze_12938 *
                                                         sizze_12937) +
                                                        segred_group_sizze_13144 *
                                                        sizze_12938 * 0 +
                                                        segred_group_sizze_13144 *
                                                        0 + (i_13730 *
                                                             (segred_group_sizze_13144 *
                                                              sizze_12938) +
                                                             i_13731 *
                                                             segred_group_sizze_13144)] =
                        ((__global
                          float *) group_res_arr_mem_13426)[group_tid_13431 *
                                                            (segred_group_sizze_13144 *
                                                             sizze_12938 *
                                                             sizze_12937) +
                                                            segred_group_sizze_13144 *
                                                            sizze_12938 * 0 +
                                                            segred_group_sizze_13144 *
                                                            0 +
                                                            local_tid_13430 +
                                                            (i_13730 *
                                                             (segred_group_sizze_13144 *
                                                              sizze_12938) +
                                                             i_13731 *
                                                             segred_group_sizze_13144)];
                }
            }
            mem_fence_global();
            old_counter_13727 = atomic_add(&((volatile __global
                                              int *) counter_mem_13422)[0],
                                           (int) 1);
            ((__local bool *) sync_arr_mem_13434)[0] = old_counter_13727 ==
                num_groups_13146 - 1;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    bool is_last_group_13732 = ((__local bool *) sync_arr_mem_13434)[0];
    
    if (is_last_group_13732) {
        if (local_tid_13430 == 0) {
            old_counter_13727 = atomic_add(&((volatile __global
                                              int *) counter_mem_13422)[0],
                                           (int) (0 - num_groups_13146));
        }
        // read in the per-group-results
        {
            if (slt32(local_tid_13430, num_groups_13146)) {
                for (int32_t i_13733 = 0; i_13733 < sizze_12934; i_13733++) {
                    for (int32_t i_13734 = 0; i_13734 < sizze_12935;
                         i_13734++) {
                        ((__global float *) mem_13285)[phys_tid_13152 *
                                                       (sizze_12935 *
                                                        sizze_12934) +
                                                       (i_13733 * sizze_12935 +
                                                        i_13734)] = ((__global
                                                                      float *) group_res_arr_mem_13424)[local_tid_13430 *
                                                                                                        (segred_group_sizze_13144 *
                                                                                                         sizze_12935 *
                                                                                                         sizze_12934) +
                                                                                                        segred_group_sizze_13144 *
                                                                                                        sizze_12935 *
                                                                                                        0 +
                                                                                                        segred_group_sizze_13144 *
                                                                                                        0 +
                                                                                                        (i_13733 *
                                                                                                         (segred_group_sizze_13144 *
                                                                                                          sizze_12935) +
                                                                                                         i_13734 *
                                                                                                         segred_group_sizze_13144)];
                    }
                }
            } else {
                for (int32_t i_13735 = 0; i_13735 < sizze_12934; i_13735++) {
                    for (int32_t i_13736 = 0; i_13736 < sizze_12935;
                         i_13736++) {
                        ((__global float *) mem_13285)[phys_tid_13152 *
                                                       (sizze_12935 *
                                                        sizze_12934) +
                                                       (i_13735 * sizze_12935 +
                                                        i_13736)] = ((__global
                                                                      float *) wih_mem_13263)[i_13735 *
                                                                                              sizze_12930 +
                                                                                              i_13736];
                    }
                }
            }
            if (slt32(local_tid_13430, num_groups_13146)) {
                for (int32_t i_13737 = 0; i_13737 < sizze_12937; i_13737++) {
                    for (int32_t i_13738 = 0; i_13738 < sizze_12938;
                         i_13738++) {
                        ((__global float *) mem_13293)[phys_tid_13152 *
                                                       (sizze_12938 *
                                                        sizze_12937) +
                                                       (i_13737 * sizze_12938 +
                                                        i_13738)] = ((__global
                                                                      float *) group_res_arr_mem_13426)[local_tid_13430 *
                                                                                                        (segred_group_sizze_13144 *
                                                                                                         sizze_12938 *
                                                                                                         sizze_12937) +
                                                                                                        segred_group_sizze_13144 *
                                                                                                        sizze_12938 *
                                                                                                        0 +
                                                                                                        segred_group_sizze_13144 *
                                                                                                        0 +
                                                                                                        (i_13737 *
                                                                                                         (segred_group_sizze_13144 *
                                                                                                          sizze_12938) +
                                                                                                         i_13738 *
                                                                                                         segred_group_sizze_13144)];
                    }
                }
            } else {
                for (int32_t i_13739 = 0; i_13739 < sizze_12937; i_13739++) {
                    for (int32_t i_13740 = 0; i_13740 < sizze_12938;
                         i_13740++) {
                        ((__global float *) mem_13293)[phys_tid_13152 *
                                                       (sizze_12938 *
                                                        sizze_12937) +
                                                       (i_13739 * sizze_12938 +
                                                        i_13740)] = ((__global
                                                                      float *) who_mem_13264)[i_13739 *
                                                                                              sizze_12932 +
                                                                                              i_13740];
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_13741;
            int32_t skip_waves_13742;
            
            offset_13741 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_13430, segred_group_sizze_13144)) {
                    for (int32_t i_13743 = 0; i_13743 < sizze_12934;
                         i_13743++) {
                        for (int32_t i_13744 = 0; i_13744 < sizze_12935;
                             i_13744++) {
                            ((__global float *) mem_13285)[phys_tid_13152 *
                                                           (sizze_12935 *
                                                            sizze_12934) +
                                                           (i_13743 *
                                                            sizze_12935 +
                                                            i_13744)] =
                                ((__global
                                  float *) mem_13285)[(global_tid_13429 +
                                                       offset_13741) *
                                                      (sizze_12935 *
                                                       sizze_12934) + (i_13743 *
                                                                       sizze_12935 +
                                                                       i_13744)];
                        }
                    }
                    for (int32_t i_13745 = 0; i_13745 < sizze_12937;
                         i_13745++) {
                        for (int32_t i_13746 = 0; i_13746 < sizze_12938;
                             i_13746++) {
                            ((__global float *) mem_13293)[phys_tid_13152 *
                                                           (sizze_12938 *
                                                            sizze_12937) +
                                                           (i_13745 *
                                                            sizze_12938 +
                                                            i_13746)] =
                                ((__global
                                  float *) mem_13293)[(global_tid_13429 +
                                                       offset_13741) *
                                                      (sizze_12938 *
                                                       sizze_12937) + (i_13745 *
                                                                       sizze_12938 +
                                                                       i_13746)];
                        }
                    }
                }
            }
            offset_13741 = 1;
            while (slt32(offset_13741, wave_sizze_13432)) {
                if (slt32(local_tid_13430 + offset_13741,
                          segred_group_sizze_13144) && ((local_tid_13430 -
                                                         squot32(local_tid_13430,
                                                                 wave_sizze_13432) *
                                                         wave_sizze_13432) &
                                                        (2 * offset_13741 -
                                                         1)) == 0) {
                    // read array element
                    {
                        for (int32_t i_13747 = 0; i_13747 < sizze_12934;
                             i_13747++) {
                            for (int32_t i_13748 = 0; i_13748 < sizze_12935;
                                 i_13748++) {
                                ((volatile __global
                                  float *) mem_13285)[(phys_tid_13152 +
                                                       num_threads_13277) *
                                                      (sizze_12935 *
                                                       sizze_12934) + (i_13747 *
                                                                       sizze_12935 +
                                                                       i_13748)] =
                                    ((volatile __global
                                      float *) mem_13285)[(global_tid_13429 +
                                                           offset_13741) *
                                                          (sizze_12935 *
                                                           sizze_12934) +
                                                          (i_13747 *
                                                           sizze_12935 +
                                                           i_13748)];
                            }
                        }
                        for (int32_t i_13749 = 0; i_13749 < sizze_12937;
                             i_13749++) {
                            for (int32_t i_13750 = 0; i_13750 < sizze_12938;
                                 i_13750++) {
                                ((volatile __global
                                  float *) mem_13293)[(phys_tid_13152 +
                                                       num_threads_13277) *
                                                      (sizze_12938 *
                                                       sizze_12937) + (i_13749 *
                                                                       sizze_12938 +
                                                                       i_13750)] =
                                    ((volatile __global
                                      float *) mem_13293)[(global_tid_13429 +
                                                           offset_13741) *
                                                          (sizze_12938 *
                                                           sizze_12937) +
                                                          (i_13749 *
                                                           sizze_12938 +
                                                           i_13750)];
                            }
                        }
                    }
                    // apply reduction operation
                    {
                        for (int32_t i_13452 = 0; i_13452 < sizze_12934;
                             i_13452++) {
                            for (int32_t i_13456 = 0; i_13456 < sizze_12935;
                                 i_13456++) {
                                float res_13457;
                                float redout_13458 = 0.0F;
                                
                                for (int32_t i_13459 = 0; i_13459 < sizze_12935;
                                     i_13459++) {
                                    float x_13460 = ((volatile __global
                                                      float *) mem_13285)[phys_tid_13152 *
                                                                          (sizze_12935 *
                                                                           sizze_12934) +
                                                                          (i_13452 *
                                                                           sizze_12935 +
                                                                           i_13459)];
                                    float x_13461 = ((volatile __global
                                                      float *) mem_13285)[(phys_tid_13152 +
                                                                           num_threads_13277) *
                                                                          (sizze_12935 *
                                                                           sizze_12934) +
                                                                          (i_13459 *
                                                                           sizze_12935 +
                                                                           i_13456)];
                                    float res_13462 = x_13460 * x_13461;
                                    float res_13463 = res_13462 + redout_13458;
                                    float redout_tmp_13753 = res_13463;
                                    
                                    redout_13458 = redout_tmp_13753;
                                }
                                res_13457 = redout_13458;
                                
                                float res_13464 = 0.0F - res_13457;
                                float res_13465 = fpow32(2.7182817F, res_13464);
                                float res_13466 = 1.0F + res_13465;
                                float res_13467 = 1.0F / res_13466;
                                
                                ((volatile __global
                                  float *) mem_13301)[phys_tid_13152 + i_13456 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    res_13467;
                            }
                            for (int32_t i_13754 = 0; i_13754 < sizze_12935;
                                 i_13754++) {
                                ((volatile __global
                                  float *) mem_13298)[phys_tid_13152 + i_13452 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144 *
                                                       sizze_12935) + i_13754 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    ((volatile __global
                                      float *) mem_13301)[phys_tid_13152 +
                                                          i_13754 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                        }
                        for (int32_t i_13476 = 0; i_13476 < sizze_12937;
                             i_13476++) {
                            for (int32_t i_13480 = 0; i_13480 < sizze_12935;
                                 i_13480++) {
                                float res_13481;
                                float redout_13482 = 0.0F;
                                
                                for (int32_t i_13483 = 0; i_13483 < sizze_12938;
                                     i_13483++) {
                                    float x_13484 = ((volatile __global
                                                      float *) mem_13293)[phys_tid_13152 *
                                                                          (sizze_12938 *
                                                                           sizze_12937) +
                                                                          (i_13476 *
                                                                           sizze_12938 +
                                                                           i_13483)];
                                    float x_13485 = ((volatile __global
                                                      float *) mem_13298)[phys_tid_13152 +
                                                                          (i_13483 *
                                                                           (num_groups_13146 *
                                                                            segred_group_sizze_13144 *
                                                                            sizze_12935) +
                                                                           i_13480 *
                                                                           (num_groups_13146 *
                                                                            segred_group_sizze_13144))];
                                    float res_13486 = x_13484 * x_13485;
                                    float res_13487 = res_13486 + redout_13482;
                                    float redout_tmp_13758 = res_13487;
                                    
                                    redout_13482 = redout_tmp_13758;
                                }
                                res_13481 = redout_13482;
                                
                                float res_13488 = 0.0F - res_13481;
                                float res_13489 = fpow32(2.7182817F, res_13488);
                                float res_13490 = 1.0F + res_13489;
                                float res_13491 = 1.0F / res_13490;
                                
                                ((volatile __global
                                  float *) mem_13314)[phys_tid_13152 + i_13480 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    res_13491;
                            }
                            for (int32_t i_13496 = 0; i_13496 < sizze_12938;
                                 i_13496++) {
                                float x_13497 = ((volatile __global
                                                  float *) mem_13293)[(phys_tid_13152 +
                                                                       num_threads_13277) *
                                                                      (sizze_12938 *
                                                                       sizze_12937) +
                                                                      (i_13476 *
                                                                       sizze_12938 +
                                                                       i_13496)];
                                float x_13498 = ((volatile __global
                                                  float *) mem_13314)[phys_tid_13152 +
                                                                      i_13496 *
                                                                      (num_groups_13146 *
                                                                       segred_group_sizze_13144)];
                                float res_13499 = x_13497 - x_13498;
                                
                                ((volatile __global
                                  float *) mem_13317)[phys_tid_13152 + i_13496 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    res_13499;
                            }
                            for (int32_t i_13760 = 0; i_13760 < sizze_12938;
                                 i_13760++) {
                                ((volatile __global
                                  float *) mem_13306)[phys_tid_13152 + i_13476 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144 *
                                                       sizze_12938) + i_13760 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    ((volatile __global
                                      float *) mem_13317)[phys_tid_13152 +
                                                          i_13760 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                            for (int32_t i_13761 = 0; i_13761 < sizze_12935;
                                 i_13761++) {
                                ((volatile __global
                                  float *) mem_13311)[phys_tid_13152 + i_13476 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144 *
                                                       sizze_12935) + i_13761 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    ((volatile __global
                                      float *) mem_13314)[phys_tid_13152 +
                                                          i_13761 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                        }
                        for (int32_t i_13506 = 0; i_13506 < sizze_12937;
                             i_13506++) {
                            for (int32_t i_13510 = 0; i_13510 < sizze_12938;
                                 i_13510++) {
                                float x_13511 = ((volatile __global
                                                  float *) mem_13306)[phys_tid_13152 +
                                                                      (i_13506 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12938) +
                                                                       i_13510 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float x_13512 = ((volatile __global
                                                  float *) mem_13311)[phys_tid_13152 +
                                                                      (i_13506 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13510 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13513 = x_13511 * x_13512;
                                float res_13514 = 1.0F - x_13512;
                                float res_13515 = res_13513 * res_13514;
                                
                                ((volatile __global
                                  float *) mem_13325)[phys_tid_13152 + i_13510 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    res_13515;
                            }
                            for (int32_t i_13520 = 0; i_13520 < sizze_12938;
                                 i_13520++) {
                                float x_13521 = ((volatile __global
                                                  float *) mem_13293)[phys_tid_13152 *
                                                                      (sizze_12938 *
                                                                       sizze_12937) +
                                                                      (i_13506 *
                                                                       sizze_12938 +
                                                                       i_13520)];
                                float res_13522;
                                float redout_13523 = 0.0F;
                                
                                for (int32_t i_13524 = 0; i_13524 < sizze_12938;
                                     i_13524++) {
                                    float x_13525 = ((volatile __global
                                                      float *) mem_13325)[phys_tid_13152 +
                                                                          i_13524 *
                                                                          (num_groups_13146 *
                                                                           segred_group_sizze_13144)];
                                    float x_13526 = ((volatile __global
                                                      float *) mem_13298)[phys_tid_13152 +
                                                                          (i_13520 *
                                                                           (num_groups_13146 *
                                                                            segred_group_sizze_13144 *
                                                                            sizze_12935) +
                                                                           i_13524 *
                                                                           (num_groups_13146 *
                                                                            segred_group_sizze_13144))];
                                    float res_13527 = x_13525 * x_13526;
                                    float res_13528 = res_13527 + redout_13523;
                                    float redout_tmp_13765 = res_13528;
                                    
                                    redout_13523 = redout_tmp_13765;
                                }
                                res_13522 = redout_13523;
                                
                                float res_13529 = lr_12939 * res_13522;
                                float res_13530 = x_13521 + res_13529;
                                
                                ((volatile __global
                                  float *) mem_13328)[phys_tid_13152 + i_13520 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    res_13530;
                            }
                            for (int32_t i_13766 = 0; i_13766 < sizze_12938;
                                 i_13766++) {
                                ((volatile __global
                                  float *) mem_13322)[phys_tid_13152 + i_13506 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144 *
                                                       sizze_12938) + i_13766 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    ((volatile __global
                                      float *) mem_13328)[phys_tid_13152 +
                                                          i_13766 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                        }
                        for (int32_t i_13536 = 0; i_13536 < sizze_12934;
                             i_13536++) {
                            for (int32_t i_13540 = 0; i_13540 < sizze_12935;
                                 i_13540++) {
                                float x_13541 = ((volatile __global
                                                  float *) mem_13298)[phys_tid_13152 +
                                                                      (i_13536 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13540 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13542 = 1.0F - x_13541;
                                
                                ((volatile __global
                                  float *) mem_13336)[phys_tid_13152 + i_13540 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    res_13542;
                            }
                            for (int32_t i_13769 = 0; i_13769 < sizze_12935;
                                 i_13769++) {
                                ((volatile __global
                                  float *) mem_13333)[phys_tid_13152 + i_13536 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144 *
                                                       sizze_12935) + i_13769 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    ((volatile __global
                                      float *) mem_13336)[phys_tid_13152 +
                                                          i_13769 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                        }
                        for (int32_t i_13548 = 0; i_13548 < sizze_12938;
                             i_13548++) {
                            for (int32_t i_13552 = 0; i_13552 < sizze_12938;
                                 i_13552++) {
                                float x_13553 = ((volatile __global
                                                  float *) mem_13298)[phys_tid_13152 +
                                                                      (i_13548 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13552 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float x_13554 = ((volatile __global
                                                  float *) mem_13333)[phys_tid_13152 +
                                                                      (i_13548 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13552 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13555;
                                float redout_13556 = 0.0F;
                                
                                for (int32_t i_13557 = 0; i_13557 < sizze_12937;
                                     i_13557++) {
                                    float x_13558 = ((volatile __global
                                                      float *) mem_13293)[phys_tid_13152 *
                                                                          (sizze_12938 *
                                                                           sizze_12937) +
                                                                          (i_13557 *
                                                                           sizze_12938 +
                                                                           i_13548)];
                                    float x_13559 = ((volatile __global
                                                      float *) mem_13306)[phys_tid_13152 +
                                                                          (i_13557 *
                                                                           (num_groups_13146 *
                                                                            segred_group_sizze_13144 *
                                                                            sizze_12938) +
                                                                           i_13552 *
                                                                           (num_groups_13146 *
                                                                            segred_group_sizze_13144))];
                                    float res_13560 = x_13558 * x_13559;
                                    float res_13561 = res_13560 + redout_13556;
                                    float redout_tmp_13772 = res_13561;
                                    
                                    redout_13556 = redout_tmp_13772;
                                }
                                res_13555 = redout_13556;
                                
                                float res_13562 = x_13553 * res_13555;
                                float res_13563 = x_13554 * res_13562;
                                
                                ((volatile __global
                                  float *) mem_13344)[phys_tid_13152 + i_13552 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    res_13563;
                            }
                            for (int32_t i_13568 = 0; i_13568 < sizze_12934;
                                 i_13568++) {
                                float res_13569;
                                float redout_13570 = 0.0F;
                                
                                for (int32_t i_13571 = 0; i_13571 < sizze_12938;
                                     i_13571++) {
                                    float x_13572 = ((volatile __global
                                                      float *) mem_13344)[phys_tid_13152 +
                                                                          i_13571 *
                                                                          (num_groups_13146 *
                                                                           segred_group_sizze_13144)];
                                    float x_13573 = ((volatile __global
                                                      float *) mem_13285)[(phys_tid_13152 +
                                                                           num_threads_13277) *
                                                                          (sizze_12935 *
                                                                           sizze_12934) +
                                                                          (i_13568 *
                                                                           sizze_12935 +
                                                                           i_13571)];
                                    float res_13574 = x_13572 * x_13573;
                                    float res_13575 = res_13574 + redout_13570;
                                    float redout_tmp_13774 = res_13575;
                                    
                                    redout_13570 = redout_tmp_13774;
                                }
                                res_13569 = redout_13570;
                                
                                float res_13576 = lr_12939 * res_13569;
                                
                                ((volatile __global
                                  float *) mem_13347)[phys_tid_13152 + i_13568 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    res_13576;
                            }
                            for (int32_t i_13775 = 0; i_13775 < sizze_12934;
                                 i_13775++) {
                                ((volatile __global
                                  float *) mem_13341)[phys_tid_13152 + i_13548 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144 *
                                                       sizze_12934) + i_13775 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    ((volatile __global
                                      float *) mem_13347)[phys_tid_13152 +
                                                          i_13775 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                        }
                        for (int32_t i_13582 = 0; i_13582 < sizze_12934;
                             i_13582++) {
                            for (int32_t i_13586 = 0; i_13586 < sizze_12935;
                                 i_13586++) {
                                float x_13587 = ((volatile __global
                                                  float *) mem_13285)[phys_tid_13152 *
                                                                      (sizze_12935 *
                                                                       sizze_12934) +
                                                                      (i_13582 *
                                                                       sizze_12935 +
                                                                       i_13586)];
                                float x_13588 = ((volatile __global
                                                  float *) mem_13341)[phys_tid_13152 +
                                                                      (i_13582 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12934) +
                                                                       i_13586 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13589 = x_13587 + x_13588;
                                
                                ((volatile __global
                                  float *) mem_13355)[phys_tid_13152 + i_13586 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    res_13589;
                            }
                            for (int32_t i_13778 = 0; i_13778 < sizze_12935;
                                 i_13778++) {
                                ((volatile __global
                                  float *) mem_13352)[phys_tid_13152 + i_13582 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144 *
                                                       sizze_12935) + i_13778 *
                                                      (num_groups_13146 *
                                                       segred_group_sizze_13144)] =
                                    ((volatile __global
                                      float *) mem_13355)[phys_tid_13152 +
                                                          i_13778 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                        }
                        for (int32_t i_13779 = 0; i_13779 < sizze_12934;
                             i_13779++) {
                            for (int32_t i_13780 = 0; i_13780 < sizze_12935;
                                 i_13780++) {
                                ((volatile __global
                                  float *) mem_13285)[phys_tid_13152 *
                                                      (sizze_12935 *
                                                       sizze_12934) + (i_13779 *
                                                                       sizze_12935 +
                                                                       i_13780)] =
                                    ((volatile __global
                                      float *) mem_13352)[phys_tid_13152 +
                                                          (i_13779 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12935) +
                                                           i_13780 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144))];
                            }
                        }
                        for (int32_t i_13781 = 0; i_13781 < sizze_12937;
                             i_13781++) {
                            for (int32_t i_13782 = 0; i_13782 < sizze_12938;
                                 i_13782++) {
                                ((volatile __global
                                  float *) mem_13293)[phys_tid_13152 *
                                                      (sizze_12938 *
                                                       sizze_12937) + (i_13781 *
                                                                       sizze_12938 +
                                                                       i_13782)] =
                                    ((volatile __global
                                      float *) mem_13322)[phys_tid_13152 +
                                                          (i_13781 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12938) +
                                                           i_13782 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144))];
                            }
                        }
                    }
                    // write result of operation
                    { }
                }
                offset_13741 *= 2;
            }
            skip_waves_13742 = 1;
            while (slt32(skip_waves_13742, squot32(segred_group_sizze_13144 +
                                                   wave_sizze_13432 - 1,
                                                   wave_sizze_13432))) {
                barrier(CLK_GLOBAL_MEM_FENCE);
                offset_13741 = skip_waves_13742 * wave_sizze_13432;
                if (slt32(local_tid_13430 + offset_13741,
                          segred_group_sizze_13144) && ((local_tid_13430 -
                                                         squot32(local_tid_13430,
                                                                 wave_sizze_13432) *
                                                         wave_sizze_13432) ==
                                                        0 &&
                                                        (squot32(local_tid_13430,
                                                                 wave_sizze_13432) &
                                                         (2 * skip_waves_13742 -
                                                          1)) == 0)) {
                    // read array element
                    {
                        for (int32_t i_13783 = 0; i_13783 < sizze_12934;
                             i_13783++) {
                            for (int32_t i_13784 = 0; i_13784 < sizze_12935;
                                 i_13784++) {
                                ((__global float *) mem_13285)[(phys_tid_13152 +
                                                                num_threads_13277) *
                                                               (sizze_12935 *
                                                                sizze_12934) +
                                                               (i_13783 *
                                                                sizze_12935 +
                                                                i_13784)] =
                                    ((__global
                                      float *) mem_13285)[(global_tid_13429 +
                                                           offset_13741) *
                                                          (sizze_12935 *
                                                           sizze_12934) +
                                                          (i_13783 *
                                                           sizze_12935 +
                                                           i_13784)];
                            }
                        }
                        for (int32_t i_13785 = 0; i_13785 < sizze_12937;
                             i_13785++) {
                            for (int32_t i_13786 = 0; i_13786 < sizze_12938;
                                 i_13786++) {
                                ((__global float *) mem_13293)[(phys_tid_13152 +
                                                                num_threads_13277) *
                                                               (sizze_12938 *
                                                                sizze_12937) +
                                                               (i_13785 *
                                                                sizze_12938 +
                                                                i_13786)] =
                                    ((__global
                                      float *) mem_13293)[(global_tid_13429 +
                                                           offset_13741) *
                                                          (sizze_12938 *
                                                           sizze_12937) +
                                                          (i_13785 *
                                                           sizze_12938 +
                                                           i_13786)];
                            }
                        }
                    }
                    // apply reduction operation
                    {
                        for (int32_t i_13452 = 0; i_13452 < sizze_12934;
                             i_13452++) {
                            for (int32_t i_13456 = 0; i_13456 < sizze_12935;
                                 i_13456++) {
                                float res_13457;
                                float redout_13458 = 0.0F;
                                
                                for (int32_t i_13459 = 0; i_13459 < sizze_12935;
                                     i_13459++) {
                                    float x_13460 = ((__global
                                                      float *) mem_13285)[phys_tid_13152 *
                                                                          (sizze_12935 *
                                                                           sizze_12934) +
                                                                          (i_13452 *
                                                                           sizze_12935 +
                                                                           i_13459)];
                                    float x_13461 = ((__global
                                                      float *) mem_13285)[(phys_tid_13152 +
                                                                           num_threads_13277) *
                                                                          (sizze_12935 *
                                                                           sizze_12934) +
                                                                          (i_13459 *
                                                                           sizze_12935 +
                                                                           i_13456)];
                                    float res_13462 = x_13460 * x_13461;
                                    float res_13463 = res_13462 + redout_13458;
                                    float redout_tmp_13789 = res_13463;
                                    
                                    redout_13458 = redout_tmp_13789;
                                }
                                res_13457 = redout_13458;
                                
                                float res_13464 = 0.0F - res_13457;
                                float res_13465 = fpow32(2.7182817F, res_13464);
                                float res_13466 = 1.0F + res_13465;
                                float res_13467 = 1.0F / res_13466;
                                
                                ((__global float *) mem_13301)[phys_tid_13152 +
                                                               i_13456 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    res_13467;
                            }
                            for (int32_t i_13790 = 0; i_13790 < sizze_12935;
                                 i_13790++) {
                                ((__global float *) mem_13298)[phys_tid_13152 +
                                                               i_13452 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144 *
                                                                sizze_12935) +
                                                               i_13790 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    ((__global
                                      float *) mem_13301)[phys_tid_13152 +
                                                          i_13790 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                        }
                        for (int32_t i_13476 = 0; i_13476 < sizze_12937;
                             i_13476++) {
                            for (int32_t i_13480 = 0; i_13480 < sizze_12935;
                                 i_13480++) {
                                float res_13481;
                                float redout_13482 = 0.0F;
                                
                                for (int32_t i_13483 = 0; i_13483 < sizze_12938;
                                     i_13483++) {
                                    float x_13484 = ((__global
                                                      float *) mem_13293)[phys_tid_13152 *
                                                                          (sizze_12938 *
                                                                           sizze_12937) +
                                                                          (i_13476 *
                                                                           sizze_12938 +
                                                                           i_13483)];
                                    float x_13485 = ((__global
                                                      float *) mem_13298)[phys_tid_13152 +
                                                                          (i_13483 *
                                                                           (num_groups_13146 *
                                                                            segred_group_sizze_13144 *
                                                                            sizze_12935) +
                                                                           i_13480 *
                                                                           (num_groups_13146 *
                                                                            segred_group_sizze_13144))];
                                    float res_13486 = x_13484 * x_13485;
                                    float res_13487 = res_13486 + redout_13482;
                                    float redout_tmp_13794 = res_13487;
                                    
                                    redout_13482 = redout_tmp_13794;
                                }
                                res_13481 = redout_13482;
                                
                                float res_13488 = 0.0F - res_13481;
                                float res_13489 = fpow32(2.7182817F, res_13488);
                                float res_13490 = 1.0F + res_13489;
                                float res_13491 = 1.0F / res_13490;
                                
                                ((__global float *) mem_13314)[phys_tid_13152 +
                                                               i_13480 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    res_13491;
                            }
                            for (int32_t i_13496 = 0; i_13496 < sizze_12938;
                                 i_13496++) {
                                float x_13497 = ((__global
                                                  float *) mem_13293)[(phys_tid_13152 +
                                                                       num_threads_13277) *
                                                                      (sizze_12938 *
                                                                       sizze_12937) +
                                                                      (i_13476 *
                                                                       sizze_12938 +
                                                                       i_13496)];
                                float x_13498 = ((__global
                                                  float *) mem_13314)[phys_tid_13152 +
                                                                      i_13496 *
                                                                      (num_groups_13146 *
                                                                       segred_group_sizze_13144)];
                                float res_13499 = x_13497 - x_13498;
                                
                                ((__global float *) mem_13317)[phys_tid_13152 +
                                                               i_13496 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    res_13499;
                            }
                            for (int32_t i_13796 = 0; i_13796 < sizze_12938;
                                 i_13796++) {
                                ((__global float *) mem_13306)[phys_tid_13152 +
                                                               i_13476 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144 *
                                                                sizze_12938) +
                                                               i_13796 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    ((__global
                                      float *) mem_13317)[phys_tid_13152 +
                                                          i_13796 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                            for (int32_t i_13797 = 0; i_13797 < sizze_12935;
                                 i_13797++) {
                                ((__global float *) mem_13311)[phys_tid_13152 +
                                                               i_13476 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144 *
                                                                sizze_12935) +
                                                               i_13797 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    ((__global
                                      float *) mem_13314)[phys_tid_13152 +
                                                          i_13797 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                        }
                        for (int32_t i_13506 = 0; i_13506 < sizze_12937;
                             i_13506++) {
                            for (int32_t i_13510 = 0; i_13510 < sizze_12938;
                                 i_13510++) {
                                float x_13511 = ((__global
                                                  float *) mem_13306)[phys_tid_13152 +
                                                                      (i_13506 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12938) +
                                                                       i_13510 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float x_13512 = ((__global
                                                  float *) mem_13311)[phys_tid_13152 +
                                                                      (i_13506 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13510 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13513 = x_13511 * x_13512;
                                float res_13514 = 1.0F - x_13512;
                                float res_13515 = res_13513 * res_13514;
                                
                                ((__global float *) mem_13325)[phys_tid_13152 +
                                                               i_13510 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    res_13515;
                            }
                            for (int32_t i_13520 = 0; i_13520 < sizze_12938;
                                 i_13520++) {
                                float x_13521 = ((__global
                                                  float *) mem_13293)[phys_tid_13152 *
                                                                      (sizze_12938 *
                                                                       sizze_12937) +
                                                                      (i_13506 *
                                                                       sizze_12938 +
                                                                       i_13520)];
                                float res_13522;
                                float redout_13523 = 0.0F;
                                
                                for (int32_t i_13524 = 0; i_13524 < sizze_12938;
                                     i_13524++) {
                                    float x_13525 = ((__global
                                                      float *) mem_13325)[phys_tid_13152 +
                                                                          i_13524 *
                                                                          (num_groups_13146 *
                                                                           segred_group_sizze_13144)];
                                    float x_13526 = ((__global
                                                      float *) mem_13298)[phys_tid_13152 +
                                                                          (i_13520 *
                                                                           (num_groups_13146 *
                                                                            segred_group_sizze_13144 *
                                                                            sizze_12935) +
                                                                           i_13524 *
                                                                           (num_groups_13146 *
                                                                            segred_group_sizze_13144))];
                                    float res_13527 = x_13525 * x_13526;
                                    float res_13528 = res_13527 + redout_13523;
                                    float redout_tmp_13801 = res_13528;
                                    
                                    redout_13523 = redout_tmp_13801;
                                }
                                res_13522 = redout_13523;
                                
                                float res_13529 = lr_12939 * res_13522;
                                float res_13530 = x_13521 + res_13529;
                                
                                ((__global float *) mem_13328)[phys_tid_13152 +
                                                               i_13520 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    res_13530;
                            }
                            for (int32_t i_13802 = 0; i_13802 < sizze_12938;
                                 i_13802++) {
                                ((__global float *) mem_13322)[phys_tid_13152 +
                                                               i_13506 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144 *
                                                                sizze_12938) +
                                                               i_13802 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    ((__global
                                      float *) mem_13328)[phys_tid_13152 +
                                                          i_13802 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                        }
                        for (int32_t i_13536 = 0; i_13536 < sizze_12934;
                             i_13536++) {
                            for (int32_t i_13540 = 0; i_13540 < sizze_12935;
                                 i_13540++) {
                                float x_13541 = ((__global
                                                  float *) mem_13298)[phys_tid_13152 +
                                                                      (i_13536 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13540 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13542 = 1.0F - x_13541;
                                
                                ((__global float *) mem_13336)[phys_tid_13152 +
                                                               i_13540 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    res_13542;
                            }
                            for (int32_t i_13805 = 0; i_13805 < sizze_12935;
                                 i_13805++) {
                                ((__global float *) mem_13333)[phys_tid_13152 +
                                                               i_13536 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144 *
                                                                sizze_12935) +
                                                               i_13805 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    ((__global
                                      float *) mem_13336)[phys_tid_13152 +
                                                          i_13805 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                        }
                        for (int32_t i_13548 = 0; i_13548 < sizze_12938;
                             i_13548++) {
                            for (int32_t i_13552 = 0; i_13552 < sizze_12938;
                                 i_13552++) {
                                float x_13553 = ((__global
                                                  float *) mem_13298)[phys_tid_13152 +
                                                                      (i_13548 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13552 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float x_13554 = ((__global
                                                  float *) mem_13333)[phys_tid_13152 +
                                                                      (i_13548 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12935) +
                                                                       i_13552 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13555;
                                float redout_13556 = 0.0F;
                                
                                for (int32_t i_13557 = 0; i_13557 < sizze_12937;
                                     i_13557++) {
                                    float x_13558 = ((__global
                                                      float *) mem_13293)[phys_tid_13152 *
                                                                          (sizze_12938 *
                                                                           sizze_12937) +
                                                                          (i_13557 *
                                                                           sizze_12938 +
                                                                           i_13548)];
                                    float x_13559 = ((__global
                                                      float *) mem_13306)[phys_tid_13152 +
                                                                          (i_13557 *
                                                                           (num_groups_13146 *
                                                                            segred_group_sizze_13144 *
                                                                            sizze_12938) +
                                                                           i_13552 *
                                                                           (num_groups_13146 *
                                                                            segred_group_sizze_13144))];
                                    float res_13560 = x_13558 * x_13559;
                                    float res_13561 = res_13560 + redout_13556;
                                    float redout_tmp_13808 = res_13561;
                                    
                                    redout_13556 = redout_tmp_13808;
                                }
                                res_13555 = redout_13556;
                                
                                float res_13562 = x_13553 * res_13555;
                                float res_13563 = x_13554 * res_13562;
                                
                                ((__global float *) mem_13344)[phys_tid_13152 +
                                                               i_13552 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    res_13563;
                            }
                            for (int32_t i_13568 = 0; i_13568 < sizze_12934;
                                 i_13568++) {
                                float res_13569;
                                float redout_13570 = 0.0F;
                                
                                for (int32_t i_13571 = 0; i_13571 < sizze_12938;
                                     i_13571++) {
                                    float x_13572 = ((__global
                                                      float *) mem_13344)[phys_tid_13152 +
                                                                          i_13571 *
                                                                          (num_groups_13146 *
                                                                           segred_group_sizze_13144)];
                                    float x_13573 = ((__global
                                                      float *) mem_13285)[(phys_tid_13152 +
                                                                           num_threads_13277) *
                                                                          (sizze_12935 *
                                                                           sizze_12934) +
                                                                          (i_13568 *
                                                                           sizze_12935 +
                                                                           i_13571)];
                                    float res_13574 = x_13572 * x_13573;
                                    float res_13575 = res_13574 + redout_13570;
                                    float redout_tmp_13810 = res_13575;
                                    
                                    redout_13570 = redout_tmp_13810;
                                }
                                res_13569 = redout_13570;
                                
                                float res_13576 = lr_12939 * res_13569;
                                
                                ((__global float *) mem_13347)[phys_tid_13152 +
                                                               i_13568 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    res_13576;
                            }
                            for (int32_t i_13811 = 0; i_13811 < sizze_12934;
                                 i_13811++) {
                                ((__global float *) mem_13341)[phys_tid_13152 +
                                                               i_13548 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144 *
                                                                sizze_12934) +
                                                               i_13811 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    ((__global
                                      float *) mem_13347)[phys_tid_13152 +
                                                          i_13811 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                        }
                        for (int32_t i_13582 = 0; i_13582 < sizze_12934;
                             i_13582++) {
                            for (int32_t i_13586 = 0; i_13586 < sizze_12935;
                                 i_13586++) {
                                float x_13587 = ((__global
                                                  float *) mem_13285)[phys_tid_13152 *
                                                                      (sizze_12935 *
                                                                       sizze_12934) +
                                                                      (i_13582 *
                                                                       sizze_12935 +
                                                                       i_13586)];
                                float x_13588 = ((__global
                                                  float *) mem_13341)[phys_tid_13152 +
                                                                      (i_13582 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144 *
                                                                        sizze_12934) +
                                                                       i_13586 *
                                                                       (num_groups_13146 *
                                                                        segred_group_sizze_13144))];
                                float res_13589 = x_13587 + x_13588;
                                
                                ((__global float *) mem_13355)[phys_tid_13152 +
                                                               i_13586 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    res_13589;
                            }
                            for (int32_t i_13814 = 0; i_13814 < sizze_12935;
                                 i_13814++) {
                                ((__global float *) mem_13352)[phys_tid_13152 +
                                                               i_13582 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144 *
                                                                sizze_12935) +
                                                               i_13814 *
                                                               (num_groups_13146 *
                                                                segred_group_sizze_13144)] =
                                    ((__global
                                      float *) mem_13355)[phys_tid_13152 +
                                                          i_13814 *
                                                          (num_groups_13146 *
                                                           segred_group_sizze_13144)];
                            }
                        }
                        for (int32_t i_13815 = 0; i_13815 < sizze_12934;
                             i_13815++) {
                            for (int32_t i_13816 = 0; i_13816 < sizze_12935;
                                 i_13816++) {
                                ((__global float *) mem_13285)[phys_tid_13152 *
                                                               (sizze_12935 *
                                                                sizze_12934) +
                                                               (i_13815 *
                                                                sizze_12935 +
                                                                i_13816)] =
                                    ((__global
                                      float *) mem_13352)[phys_tid_13152 +
                                                          (i_13815 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12935) +
                                                           i_13816 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144))];
                            }
                        }
                        for (int32_t i_13817 = 0; i_13817 < sizze_12937;
                             i_13817++) {
                            for (int32_t i_13818 = 0; i_13818 < sizze_12938;
                                 i_13818++) {
                                ((__global float *) mem_13293)[phys_tid_13152 *
                                                               (sizze_12938 *
                                                                sizze_12937) +
                                                               (i_13817 *
                                                                sizze_12938 +
                                                                i_13818)] =
                                    ((__global
                                      float *) mem_13322)[phys_tid_13152 +
                                                          (i_13817 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144 *
                                                            sizze_12938) +
                                                           i_13818 *
                                                           (num_groups_13146 *
                                                            segred_group_sizze_13144))];
                            }
                        }
                    }
                    // write result of operation
                    { }
                }
                skip_waves_13742 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_13430 == 0) {
                    for (int32_t i_13819 = 0; i_13819 < sizze_12934;
                         i_13819++) {
                        for (int32_t i_13820 = 0; i_13820 < sizze_12935;
                             i_13820++) {
                            ((__global float *) mem_13362)[0 * (sizze_12935 *
                                                                sizze_12934) +
                                                           (i_13819 *
                                                            sizze_12935 +
                                                            i_13820)] =
                                ((__global float *) mem_13285)[phys_tid_13152 *
                                                               (sizze_12935 *
                                                                sizze_12934) +
                                                               (i_13819 *
                                                                sizze_12935 +
                                                                i_13820)];
                        }
                    }
                    for (int32_t i_13821 = 0; i_13821 < sizze_12937;
                         i_13821++) {
                        for (int32_t i_13822 = 0; i_13822 < sizze_12938;
                             i_13822++) {
                            ((__global float *) mem_13369)[0 * (sizze_12938 *
                                                                sizze_12937) +
                                                           (i_13821 *
                                                            sizze_12938 +
                                                            i_13822)] =
                                ((__global float *) mem_13293)[phys_tid_13152 *
                                                               (sizze_12938 *
                                                                sizze_12937) +
                                                               (i_13821 *
                                                                sizze_12938 +
                                                                i_13822)];
                        }
                    }
                }
            }
        }
    }
}
"""
# Start of values.py.

# Hacky parser/reader/writer for values written in Futhark syntax.
# Used for reading stdin when compiling standalone programs with the
# Python code generator.

import numpy as np
import string
import struct
import sys

class ReaderInput:
    def __init__(self, f):
        self.f = f
        self.lookahead_buffer = []

    def get_char(self):
        if len(self.lookahead_buffer) == 0:
            return self.f.read(1)
        else:
            c = self.lookahead_buffer[0]
            self.lookahead_buffer = self.lookahead_buffer[1:]
            return c

    def unget_char(self, c):
        self.lookahead_buffer = [c] + self.lookahead_buffer

    def get_chars(self, n):
        n1 = min(n, len(self.lookahead_buffer))
        s = b''.join(self.lookahead_buffer[:n1])
        self.lookahead_buffer = self.lookahead_buffer[n1:]
        n2 = n - n1
        if n2 > 0:
            s += self.f.read(n2)
        return s

    def peek_char(self):
        c = self.get_char()
        if c:
            self.unget_char(c)
        return c

def skip_spaces(f):
    c = f.get_char()
    while c != None:
        if c.isspace():
            c = f.get_char()
        elif c == b'-':
          # May be line comment.
          if f.peek_char() == b'-':
            # Yes, line comment. Skip to end of line.
            while (c != b'\n' and c != None):
              c = f.get_char()
          else:
            break
        else:
          break
    if c:
        f.unget_char(c)

def parse_specific_char(f, expected):
    got = f.get_char()
    if got != expected:
        f.unget_char(got)
        raise ValueError
    return True

def parse_specific_string(f, s):
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    read = []
    try:
        for c in bs:
            parse_specific_char(f, c)
            read.append(c)
        return True
    except ValueError:
        for c in read[::-1]:
            f.unget_char(c)
        raise

def optional(p, *args):
    try:
        return p(*args)
    except ValueError:
        return None

def optional_specific_string(f, s):
    c = f.peek_char()
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    if c == bs[0]:
        return parse_specific_string(f, s)
    else:
        return False

def sepBy(p, sep, *args):
    elems = []
    x = optional(p, *args)
    if x != None:
        elems += [x]
        while optional(sep, *args) != None:
            x = p(*args)
            elems += [x]
    return elems

# Assumes '0x' has already been read
def parse_hex_int(f):
    s = b''
    c = f.get_char()
    while c != None:
        if c in b'01234556789ABCDEFabcdef':
            s += c
            c = f.get_char()
        elif c == b'_':
            c = f.get_char() # skip _
        else:
            f.unget_char(c)
            break
    return str(int(s, 16)).encode('utf8') # ugh

def parse_int(f):
    s = b''
    c = f.get_char()
    if c == b'0' and f.peek_char() in b'xX':
        c = f.get_char() # skip X
        return parse_hex_int(f)
    else:
        while c != None:
            if c.isdigit():
                s += c
                c = f.get_char()
            elif c == b'_':
                c = f.get_char() # skip _
            else:
                f.unget_char(c)
                break
        if len(s) == 0:
            raise ValueError
        return s

def parse_int_signed(f):
    s = b''
    c = f.get_char()

    if c == b'-' and f.peek_char().isdigit():
      return c + parse_int(f)
    else:
      if c != b'+':
          f.unget_char(c)
      return parse_int(f)

def read_str_comma(f):
    skip_spaces(f)
    parse_specific_char(f, b',')
    return b','

def read_str_int(f, s):
    skip_spaces(f)
    x = int(parse_int_signed(f))
    optional_specific_string(f, s)
    return x

def read_str_uint(f, s):
    skip_spaces(f)
    x = int(parse_int(f))
    optional_specific_string(f, s)
    return x

def read_str_i8(f):
    return np.int8(read_str_int(f, 'i8'))
def read_str_i16(f):
    return np.int16(read_str_int(f, 'i16'))
def read_str_i32(f):
    return np.int32(read_str_int(f, 'i32'))
def read_str_i64(f):
    return np.int64(read_str_int(f, 'i64'))

def read_str_u8(f):
    return np.uint8(read_str_int(f, 'u8'))
def read_str_u16(f):
    return np.uint16(read_str_int(f, 'u16'))
def read_str_u32(f):
    return np.uint32(read_str_int(f, 'u32'))
def read_str_u64(f):
    return np.uint64(read_str_int(f, 'u64'))

def read_char(f):
    skip_spaces(f)
    parse_specific_char(f, b'\'')
    c = f.get_char()
    parse_specific_char(f, b'\'')
    return c

def read_str_hex_float(f, sign):
    int_part = parse_hex_int(f)
    parse_specific_char(f, b'.')
    frac_part = parse_hex_int(f)
    parse_specific_char(f, b'p')
    exponent = parse_int(f)

    int_val = int(int_part, 16)
    frac_val = float(int(frac_part, 16)) / (16 ** len(frac_part))
    exp_val = int(exponent)

    total_val = (int_val + frac_val) * (2.0 ** exp_val)
    if sign == b'-':
        total_val = -1 * total_val

    return float(total_val)


def read_str_decimal(f):
    skip_spaces(f)
    c = f.get_char()
    if (c == b'-'):
      sign = b'-'
    else:
      f.unget_char(c)
      sign = b''

    # Check for hexadecimal float
    c = f.get_char()
    if (c == '0' and (f.peek_char() in ['x', 'X'])):
        f.get_char()
        return read_str_hex_float(f, sign)
    else:
        f.unget_char(c)

    bef = optional(parse_int, f)
    if bef == None:
        bef = b'0'
        parse_specific_char(f, b'.')
        aft = parse_int(f)
    elif optional(parse_specific_char, f, b'.'):
        aft = parse_int(f)
    else:
        aft = b'0'
    if (optional(parse_specific_char, f, b'E') or
        optional(parse_specific_char, f, b'e')):
        expt = parse_int_signed(f)
    else:
        expt = b'0'
    return float(sign + bef + b'.' + aft + b'E' + expt)

def read_str_f32(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f32.nan')
        return np.float32(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f32.inf')
            return np.float32(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f32.inf')
               return np.float32(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f32')
               return x

def read_str_f64(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f64.nan')
        return np.float64(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f64.inf')
            return np.float64(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f64.inf')
               return np.float64(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f64')
               return x

def read_str_bool(f):
    skip_spaces(f)
    if f.peek_char() == b't':
        parse_specific_string(f, 'true')
        return True
    elif f.peek_char() == b'f':
        parse_specific_string(f, 'false')
        return False
    else:
        raise ValueError

def read_str_empty_array(f, type_name, rank):
    parse_specific_string(f, 'empty')
    parse_specific_char(f, b'(')
    dims = []
    for i in range(rank):
        parse_specific_string(f, '[')
        dims += [int(parse_int(f))]
        parse_specific_string(f, ']')
    if np.product(dims) != 0:
        raise ValueError
    parse_specific_string(f, type_name)
    parse_specific_char(f, b')')

    return tuple(dims)

def read_str_array_elems(f, elem_reader, type_name, rank):
    skip_spaces(f)
    try:
        parse_specific_char(f, b'[')
    except ValueError:
        return read_str_empty_array(f, type_name, rank)
    else:
        xs = sepBy(elem_reader, read_str_comma, f)
        skip_spaces(f)
        parse_specific_char(f, b']')
        return xs

def read_str_array_helper(f, elem_reader, type_name, rank):
    def nested_row_reader(_):
        return read_str_array_helper(f, elem_reader, type_name, rank-1)
    if rank == 1:
        row_reader = elem_reader
    else:
        row_reader = nested_row_reader
    return read_str_array_elems(f, row_reader, type_name, rank)

def expected_array_dims(l, rank):
  if rank > 1:
      n = len(l)
      if n == 0:
          elem = []
      else:
          elem = l[0]
      return [n] + expected_array_dims(elem, rank-1)
  else:
      return [len(l)]

def verify_array_dims(l, dims):
    if dims[0] != len(l):
        raise ValueError
    if len(dims) > 1:
        for x in l:
            verify_array_dims(x, dims[1:])

def read_str_array(f, elem_reader, type_name, rank, bt):
    elems = read_str_array_helper(f, elem_reader, type_name, rank)
    if type(elems) == tuple:
        # Empty array
        return np.empty(elems, dtype=bt)
    else:
        dims = expected_array_dims(elems, rank)
        verify_array_dims(elems, dims)
        return np.array(elems, dtype=bt)

################################################################################

READ_BINARY_VERSION = 2

# struct format specified at
# https://docs.python.org/2/library/struct.html#format-characters

def mk_bin_scalar_reader(t):
    def bin_reader(f):
        fmt = FUTHARK_PRIMTYPES[t]['bin_format']
        size = FUTHARK_PRIMTYPES[t]['size']
        return struct.unpack('<' + fmt, f.get_chars(size))[0]
    return bin_reader

read_bin_i8 = mk_bin_scalar_reader('i8')
read_bin_i16 = mk_bin_scalar_reader('i16')
read_bin_i32 = mk_bin_scalar_reader('i32')
read_bin_i64 = mk_bin_scalar_reader('i64')

read_bin_u8 = mk_bin_scalar_reader('u8')
read_bin_u16 = mk_bin_scalar_reader('u16')
read_bin_u32 = mk_bin_scalar_reader('u32')
read_bin_u64 = mk_bin_scalar_reader('u64')

read_bin_f32 = mk_bin_scalar_reader('f32')
read_bin_f64 = mk_bin_scalar_reader('f64')

read_bin_bool = mk_bin_scalar_reader('bool')

def read_is_binary(f):
    skip_spaces(f)
    c = f.get_char()
    if c == b'b':
        bin_version = read_bin_u8(f)
        if bin_version != READ_BINARY_VERSION:
            panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
                  bin_version, READ_BINARY_VERSION)
        return True
    else:
        f.unget_char(c)
        return False

FUTHARK_PRIMTYPES = {
    'i8':  {'binname' : b"  i8",
            'size' : 1,
            'bin_reader': read_bin_i8,
            'str_reader': read_str_i8,
            'bin_format': 'b',
            'numpy_type': np.int8 },

    'i16': {'binname' : b" i16",
            'size' : 2,
            'bin_reader': read_bin_i16,
            'str_reader': read_str_i16,
            'bin_format': 'h',
            'numpy_type': np.int16 },

    'i32': {'binname' : b" i32",
            'size' : 4,
            'bin_reader': read_bin_i32,
            'str_reader': read_str_i32,
            'bin_format': 'i',
            'numpy_type': np.int32 },

    'i64': {'binname' : b" i64",
            'size' : 8,
            'bin_reader': read_bin_i64,
            'str_reader': read_str_i64,
            'bin_format': 'q',
            'numpy_type': np.int64},

    'u8':  {'binname' : b"  u8",
            'size' : 1,
            'bin_reader': read_bin_u8,
            'str_reader': read_str_u8,
            'bin_format': 'B',
            'numpy_type': np.uint8 },

    'u16': {'binname' : b" u16",
            'size' : 2,
            'bin_reader': read_bin_u16,
            'str_reader': read_str_u16,
            'bin_format': 'H',
            'numpy_type': np.uint16 },

    'u32': {'binname' : b" u32",
            'size' : 4,
            'bin_reader': read_bin_u32,
            'str_reader': read_str_u32,
            'bin_format': 'I',
            'numpy_type': np.uint32 },

    'u64': {'binname' : b" u64",
            'size' : 8,
            'bin_reader': read_bin_u64,
            'str_reader': read_str_u64,
            'bin_format': 'Q',
            'numpy_type': np.uint64 },

    'f32': {'binname' : b" f32",
            'size' : 4,
            'bin_reader': read_bin_f32,
            'str_reader': read_str_f32,
            'bin_format': 'f',
            'numpy_type': np.float32 },

    'f64': {'binname' : b" f64",
            'size' : 8,
            'bin_reader': read_bin_f64,
            'str_reader': read_str_f64,
            'bin_format': 'd',
            'numpy_type': np.float64 },

    'bool': {'binname' : b"bool",
             'size' : 1,
             'bin_reader': read_bin_bool,
             'str_reader': read_str_bool,
             'bin_format': 'b',
             'numpy_type': np.bool }
}

def read_bin_read_type(f):
    read_binname = f.get_chars(4)

    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['binname'] == read_binname:
            return k
    panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname)

def numpy_type_to_type_name(t):
    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['numpy_type'] == t:
            return k
    raise Exception('Unknown Numpy type: {}'.format(t))

def read_bin_ensure_scalar(f, expected_type):
  dims = read_bin_i8(f)

  if dims != 0:
      panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n", dims)

  bin_type = read_bin_read_type(f)
  if bin_type != expected_type:
      panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
            expected_type, bin_type)

# ------------------------------------------------------------------------------
# General interface for reading Primitive Futhark Values
# ------------------------------------------------------------------------------

def read_scalar(f, ty):
    if read_is_binary(f):
        read_bin_ensure_scalar(f, ty)
        return FUTHARK_PRIMTYPES[ty]['bin_reader'](f)
    return FUTHARK_PRIMTYPES[ty]['str_reader'](f)

def read_array(f, expected_type, rank):
    if not read_is_binary(f):
        str_reader = FUTHARK_PRIMTYPES[expected_type]['str_reader']
        return read_str_array(f, str_reader, expected_type, rank,
                              FUTHARK_PRIMTYPES[expected_type]['numpy_type'])

    bin_rank = read_bin_u8(f)

    if bin_rank != rank:
        panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
              rank, bin_rank)

    bin_type_enum = read_bin_read_type(f)
    if expected_type != bin_type_enum:
        panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
              rank, expected_type, bin_rank, bin_type_enum)

    shape = []
    elem_count = 1
    for i in range(rank):
        bin_size = read_bin_u64(f)
        elem_count *= bin_size
        shape.append(bin_size)

    bin_fmt = FUTHARK_PRIMTYPES[bin_type_enum]['bin_format']

    # We first read the expected number of types into a bytestring,
    # then use np.fromstring.  This is because np.fromfile does not
    # work on things that are insufficiently file-like, like a network
    # stream.
    bytes = f.get_chars(elem_count * FUTHARK_PRIMTYPES[expected_type]['size'])
    arr = np.fromstring(bytes, dtype=FUTHARK_PRIMTYPES[bin_type_enum]['numpy_type'])
    arr.shape = shape

    return arr

if sys.version_info >= (3,0):
    input_reader = ReaderInput(sys.stdin.buffer)
else:
    input_reader = ReaderInput(sys.stdin)

import re

def read_value(type_desc, reader=input_reader):
    """Read a value of the given type.  The type is a string
representation of the Futhark type."""
    m = re.match(r'((?:\[\])*)([a-z0-9]+)$', type_desc)
    if m:
        dims = int(len(m.group(1))/2)
        basetype = m.group(2)
        assert basetype in FUTHARK_PRIMTYPES, "Unknown type: {}".format(type_desc)
        if dims > 0:
            return read_array(reader, basetype, dims)
        else:
            return read_scalar(reader, basetype)
        return (dims, basetype)

def write_value_text(v, out=sys.stdout):
    if type(v) == np.uint8:
        out.write("%uu8" % v)
    elif type(v) == np.uint16:
        out.write("%uu16" % v)
    elif type(v) == np.uint32:
        out.write("%uu32" % v)
    elif type(v) == np.uint64:
        out.write("%uu64" % v)
    elif type(v) == np.int8:
        out.write("%di8" % v)
    elif type(v) == np.int16:
        out.write("%di16" % v)
    elif type(v) == np.int32:
        out.write("%di32" % v)
    elif type(v) == np.int64:
        out.write("%di64" % v)
    elif type(v) in [np.bool, np.bool_]:
        if v:
            out.write("true")
        else:
            out.write("false")
    elif type(v) == np.float32:
        if np.isnan(v):
            out.write('f32.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f32.inf')
            else:
                out.write('-f32.inf')
        else:
            out.write("%.6ff32" % v)
    elif type(v) == np.float64:
        if np.isnan(v):
            out.write('f64.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f64.inf')
            else:
                out.write('-f64.inf')
        else:
            out.write("%.6ff64" % v)
    elif type(v) == np.ndarray:
        if np.product(v.shape) == 0:
            tname = numpy_type_to_type_name(v.dtype)
            out.write('empty({}{})'.format(''.join(['[{}]'.format(d)
                                                    for d in v.shape]), tname))
        else:
            first = True
            out.write('[')
            for x in v:
                if not first: out.write(', ')
                first = False
                write_value(x, out=out)
            out.write(']')
    else:
        raise Exception("Cannot print value of type {}: {}".format(type(v), v))

type_strs = { np.dtype('int8'): b'  i8',
              np.dtype('int16'): b' i16',
              np.dtype('int32'): b' i32',
              np.dtype('int64'): b' i64',
              np.dtype('uint8'): b'  u8',
              np.dtype('uint16'): b' u16',
              np.dtype('uint32'): b' u32',
              np.dtype('uint64'): b' u64',
              np.dtype('float32'): b' f32',
              np.dtype('float64'): b' f64',
              np.dtype('bool'): b'bool'}

def construct_binary_value(v):
    t = v.dtype
    shape = v.shape

    elems = 1
    for d in shape:
        elems *= d

    num_bytes = 1 + 1 + 1 + 4 + len(shape) * 8 + elems * t.itemsize
    bytes = bytearray(num_bytes)
    bytes[0] = np.int8(ord('b'))
    bytes[1] = 2
    bytes[2] = np.int8(len(shape))
    bytes[3:7] = type_strs[t]

    for i in range(len(shape)):
        bytes[7+i*8:7+(i+1)*8] = np.int64(shape[i]).tostring()

    bytes[7+len(shape)*8:] = np.ascontiguousarray(v).tostring()

    return bytes

def write_value_binary(v, out=sys.stdout):
    if sys.version_info >= (3,0):
        out = out.buffer
    out.write(construct_binary_value(v))

def write_value(v, out=sys.stdout, binary=False):
    if binary:
        return write_value_binary(v, out=out)
    else:
        return write_value_text(v, out=out)

# End of values.py.
# Start of memory.py.

import ctypes as ct

def addressOffset(x, offset, bt):
  return ct.cast(ct.addressof(x.contents)+int(offset), ct.POINTER(bt))

def allocateMem(size):
  return ct.cast((ct.c_byte * max(0,size))(), ct.POINTER(ct.c_byte))

# Copy an array if its is not-None.  This is important for treating
# Numpy arrays as flat memory, but has some overhead.
def normaliseArray(x):
  if (x.base is x) or (x.base is None):
    return x
  else:
    return x.copy()

def unwrapArray(x):
  return normaliseArray(x).ctypes.data_as(ct.POINTER(ct.c_byte))

def createArray(x, shape):
  # HACK: np.ctypeslib.as_array may fail if the shape contains zeroes,
  # for some reason.
  if any(map(lambda x: x == 0, shape)):
      return np.ndarray(shape, dtype=x._type_)
  else:
      return np.ctypeslib.as_array(x, shape=shape)

def indexArray(x, offset, bt, nptype):
  return nptype(addressOffset(x, offset*ct.sizeof(bt), bt)[0])

def writeScalarArray(x, offset, v):
  ct.memmove(ct.addressof(x.contents)+int(offset)*ct.sizeof(v), ct.addressof(v), ct.sizeof(v))

# An opaque Futhark value.
class opaque(object):
  def __init__(self, desc, *payload):
    self.data = payload
    self.desc = desc

  def __repr__(self):
    return "<opaque Futhark value of type {}>".format(self.desc)

# End of memory.py.
# Start of panic.py.

def panic(exitcode, fmt, *args):
    sys.stderr.write('%s: ' % sys.argv[0])
    sys.stderr.write(fmt % args)
    sys.exit(exitcode)

# End of panic.py.
# Start of tuning.py

def read_tuning_file(kvs, f):
    for line in f.read().splitlines():
        size, value = line.split('=')
        kvs[size] = int(value)
    return kvs

# End of tuning.py.
# Start of scalar.py.

import numpy as np
import math
import struct

def signed(x):
  if type(x) == np.uint8:
    return np.int8(x)
  elif type(x) == np.uint16:
    return np.int16(x)
  elif type(x) == np.uint32:
    return np.int32(x)
  else:
    return np.int64(x)

def unsigned(x):
  if type(x) == np.int8:
    return np.uint8(x)
  elif type(x) == np.int16:
    return np.uint16(x)
  elif type(x) == np.int32:
    return np.uint32(x)
  else:
    return np.uint64(x)

def shlN(x,y):
  return x << y

def ashrN(x,y):
  return x >> y

def sdivN(x,y):
  return x // y

def smodN(x,y):
  return x % y

def udivN(x,y):
  return signed(unsigned(x) // unsigned(y))

def umodN(x,y):
  return signed(unsigned(x) % unsigned(y))

def squotN(x,y):
  return np.floor_divide(np.abs(x), np.abs(y)) * np.sign(x) * np.sign(y)

def sremN(x,y):
  return np.remainder(np.abs(x), np.abs(y)) * np.sign(x)

def sminN(x,y):
  return min(x,y)

def smaxN(x,y):
  return max(x,y)

def uminN(x,y):
  return signed(min(unsigned(x),unsigned(y)))

def umaxN(x,y):
  return signed(max(unsigned(x),unsigned(y)))

def fminN(x,y):
  return min(x,y)

def fmaxN(x,y):
  return max(x,y)

def powN(x,y):
  return x ** y

def fpowN(x,y):
  return x ** y

def sleN(x,y):
  return x <= y

def sltN(x,y):
  return x < y

def uleN(x,y):
  return unsigned(x) <= unsigned(y)

def ultN(x,y):
  return unsigned(x) < unsigned(y)

def lshr8(x,y):
  return np.int8(np.uint8(x) >> np.uint8(y))

def lshr16(x,y):
  return np.int16(np.uint16(x) >> np.uint16(y))

def lshr32(x,y):
  return np.int32(np.uint32(x) >> np.uint32(y))

def lshr64(x,y):
  return np.int64(np.uint64(x) >> np.uint64(y))

def sext_T_i8(x):
  return np.int8(x)

def sext_T_i16(x):
  return np.int16(x)

def sext_T_i32(x):
  return np.int32(x)

def sext_T_i64(x):
  return np.int64(x)

def itob_T_bool(x):
  return np.bool(x)

def btoi_bool_i8(x):
  return np.int8(x)

def btoi_bool_i16(x):
  return np.int8(x)

def btoi_bool_i32(x):
  return np.int8(x)

def btoi_bool_i64(x):
  return np.int8(x)

def zext_i8_i8(x):
  return np.int8(np.uint8(x))

def zext_i8_i16(x):
  return np.int16(np.uint8(x))

def zext_i8_i32(x):
  return np.int32(np.uint8(x))

def zext_i8_i64(x):
  return np.int64(np.uint8(x))

def zext_i16_i8(x):
  return np.int8(np.uint16(x))

def zext_i16_i16(x):
  return np.int16(np.uint16(x))

def zext_i16_i32(x):
  return np.int32(np.uint16(x))

def zext_i16_i64(x):
  return np.int64(np.uint16(x))

def zext_i32_i8(x):
  return np.int8(np.uint32(x))

def zext_i32_i16(x):
  return np.int16(np.uint32(x))

def zext_i32_i32(x):
  return np.int32(np.uint32(x))

def zext_i32_i64(x):
  return np.int64(np.uint32(x))

def zext_i64_i8(x):
  return np.int8(np.uint64(x))

def zext_i64_i16(x):
  return np.int16(np.uint64(x))

def zext_i64_i32(x):
  return np.int32(np.uint64(x))

def zext_i64_i64(x):
  return np.int64(np.uint64(x))

shl8 = shl16 = shl32 = shl64 = shlN
ashr8 = ashr16 = ashr32 = ashr64 = ashrN
sdiv8 = sdiv16 = sdiv32 = sdiv64 = sdivN
smod8 = smod16 = smod32 = smod64 = smodN
udiv8 = udiv16 = udiv32 = udiv64 = udivN
umod8 = umod16 = umod32 = umod64 = umodN
squot8 = squot16 = squot32 = squot64 = squotN
srem8 = srem16 = srem32 = srem64 = sremN
smax8 = smax16 = smax32 = smax64 = smaxN
smin8 = smin16 = smin32 = smin64 = sminN
umax8 = umax16 = umax32 = umax64 = umaxN
umin8 = umin16 = umin32 = umin64 = uminN
pow8 = pow16 = pow32 = pow64 = powN
fpow32 = fpow64 = fpowN
fmax32 = fmax64 = fmaxN
fmin32 = fmin64 = fminN
sle8 = sle16 = sle32 = sle64 = sleN
slt8 = slt16 = slt32 = slt64 = sltN
ule8 = ule16 = ule32 = ule64 = uleN
ult8 = ult16 = ult32 = ult64 = ultN
sext_i8_i8 = sext_i16_i8 = sext_i32_i8 = sext_i64_i8 = sext_T_i8
sext_i8_i16 = sext_i16_i16 = sext_i32_i16 = sext_i64_i16 = sext_T_i16
sext_i8_i32 = sext_i16_i32 = sext_i32_i32 = sext_i64_i32 = sext_T_i32
sext_i8_i64 = sext_i16_i64 = sext_i32_i64 = sext_i64_i64 = sext_T_i64
itob_i8_bool = itob_i16_bool = itob_i32_bool = itob_i64_bool = itob_T_bool

def clz_T(x):
  n = np.int32(0)
  bits = x.itemsize * 8
  for i in range(bits):
    if x < 0:
      break
    n += 1
    x <<= np.int8(1)
  return n

def popc_T(x):
  c = np.int32(0)
  while x != 0:
    x &= x - np.int8(1)
    c += np.int8(1)
  return c

futhark_popc8 = futhark_popc16 = futhark_popc32 = futhark_popc64 = popc_T
futhark_clzz8 = futhark_clzz16 = futhark_clzz32 = futhark_clzz64 = clz_T

def ssignum(x):
  return np.sign(x)

def usignum(x):
  if x < 0:
    return ssignum(-x)
  else:
    return ssignum(x)

def sitofp_T_f32(x):
  return np.float32(x)
sitofp_i8_f32 = sitofp_i16_f32 = sitofp_i32_f32 = sitofp_i64_f32 = sitofp_T_f32

def sitofp_T_f64(x):
  return np.float64(x)
sitofp_i8_f64 = sitofp_i16_f64 = sitofp_i32_f64 = sitofp_i64_f64 = sitofp_T_f64

def uitofp_T_f32(x):
  return np.float32(unsigned(x))
uitofp_i8_f32 = uitofp_i16_f32 = uitofp_i32_f32 = uitofp_i64_f32 = uitofp_T_f32

def uitofp_T_f64(x):
  return np.float64(unsigned(x))
uitofp_i8_f64 = uitofp_i16_f64 = uitofp_i32_f64 = uitofp_i64_f64 = uitofp_T_f64

def fptosi_T_i8(x):
  return np.int8(np.trunc(x))
fptosi_f32_i8 = fptosi_f64_i8 = fptosi_T_i8

def fptosi_T_i16(x):
  return np.int16(np.trunc(x))
fptosi_f32_i16 = fptosi_f64_i16 = fptosi_T_i16

def fptosi_T_i32(x):
  return np.int32(np.trunc(x))
fptosi_f32_i32 = fptosi_f64_i32 = fptosi_T_i32

def fptosi_T_i64(x):
  return np.int64(np.trunc(x))
fptosi_f32_i64 = fptosi_f64_i64 = fptosi_T_i64

def fptoui_T_i8(x):
  return np.uint8(np.trunc(x))
fptoui_f32_i8 = fptoui_f64_i8 = fptoui_T_i8

def fptoui_T_i16(x):
  return np.uint16(np.trunc(x))
fptoui_f32_i16 = fptoui_f64_i16 = fptoui_T_i16

def fptoui_T_i32(x):
  return np.uint32(np.trunc(x))
fptoui_f32_i32 = fptoui_f64_i32 = fptoui_T_i32

def fptoui_T_i64(x):
  return np.uint64(np.trunc(x))
fptoui_f32_i64 = fptoui_f64_i64 = fptoui_T_i64

def fpconv_f32_f64(x):
  return np.float64(x)

def fpconv_f64_f32(x):
  return np.float32(x)

def futhark_log64(x):
  return np.float64(np.log(x))

def futhark_log2_64(x):
  return np.float64(np.log2(x))

def futhark_log10_64(x):
  return np.float64(np.log10(x))

def futhark_sqrt64(x):
  return np.sqrt(x)

def futhark_exp64(x):
  return np.exp(x)

def futhark_cos64(x):
  return np.cos(x)

def futhark_sin64(x):
  return np.sin(x)

def futhark_tan64(x):
  return np.tan(x)

def futhark_acos64(x):
  return np.arccos(x)

def futhark_asin64(x):
  return np.arcsin(x)

def futhark_atan64(x):
  return np.arctan(x)

def futhark_atan2_64(x, y):
  return np.arctan2(x, y)

def futhark_gamma64(x):
  return np.float64(math.gamma(x))

def futhark_lgamma64(x):
  return np.float64(math.lgamma(x))

def futhark_round64(x):
  return np.round(x)

def futhark_ceil64(x):
  return np.ceil(x)

def futhark_floor64(x):
  return np.floor(x)

def futhark_isnan64(x):
  return np.isnan(x)

def futhark_isinf64(x):
  return np.isinf(x)

def futhark_to_bits64(x):
  s = struct.pack('>d', x)
  return np.int64(struct.unpack('>q', s)[0])

def futhark_from_bits64(x):
  s = struct.pack('>q', x)
  return np.float64(struct.unpack('>d', s)[0])

def futhark_log32(x):
  return np.float32(np.log(x))

def futhark_log2_32(x):
  return np.float32(np.log2(x))

def futhark_log10_32(x):
  return np.float32(np.log10(x))

def futhark_sqrt32(x):
  return np.float32(np.sqrt(x))

def futhark_exp32(x):
  return np.exp(x)

def futhark_cos32(x):
  return np.cos(x)

def futhark_sin32(x):
  return np.sin(x)

def futhark_tan32(x):
  return np.tan(x)

def futhark_acos32(x):
  return np.arccos(x)

def futhark_asin32(x):
  return np.arcsin(x)

def futhark_atan32(x):
  return np.arctan(x)

def futhark_atan2_32(x, y):
  return np.arctan2(x, y)

def futhark_gamma32(x):
  return np.float32(math.gamma(x))

def futhark_lgamma32(x):
  return np.float32(math.lgamma(x))

def futhark_round32(x):
  return np.round(x)

def futhark_ceil32(x):
  return np.ceil(x)

def futhark_floor32(x):
  return np.floor(x)

def futhark_isnan32(x):
  return np.isnan(x)

def futhark_isinf32(x):
  return np.isinf(x)

def futhark_to_bits32(x):
  s = struct.pack('>f', x)
  return np.int32(struct.unpack('>l', s)[0])

def futhark_from_bits32(x):
  s = struct.pack('>l', x)
  return np.float32(struct.unpack('>f', s)[0])

def futhark_lerp32(v0, v1, t):
  return v0 + (v1-v0)*t

def futhark_lerp64(v0, v1, t):
  return v0 + (v1-v0)*t

# End of scalar.py.
class GPUTraining:
  entry_points = {"main": (["f32", "[][]f32", "[][]f32", "[][][]f32",
                            "[][][]f32"], ["[][]f32", "[][]f32"])}
  def __init__(self, command_queue=None, interactive=False,
               platform_pref=preferred_platform, device_pref=preferred_device,
               default_group_size=default_group_size,
               default_num_groups=default_num_groups,
               default_tile_size=default_tile_size,
               default_threshold=default_threshold, sizes=sizes):
    size_heuristics=[("NVIDIA CUDA", cl.device_type.GPU, "lockstep_width", 32),
     ("AMD Accelerated Parallel Processing", cl.device_type.GPU, "lockstep_width",
      32), ("", cl.device_type.GPU, "lockstep_width", 1), ("", cl.device_type.GPU,
                                                           "num_groups", 256), ("",
                                                                                cl.device_type.GPU,
                                                                                "group_size",
                                                                                256),
     ("", cl.device_type.GPU, "tile_size", 32), ("", cl.device_type.GPU,
                                                 "threshold", 32768), ("",
                                                                       cl.device_type.CPU,
                                                                       "lockstep_width",
                                                                       1), ("",
                                                                            cl.device_type.CPU,
                                                                            "num_groups",
                                                                            "MAX_COMPUTE_UNITS"),
     ("", cl.device_type.CPU, "group_size", 32), ("", cl.device_type.CPU,
                                                  "tile_size", 4), ("",
                                                                    cl.device_type.CPU,
                                                                    "threshold",
                                                                    "MAX_COMPUTE_UNITS")]
    program = initialise_opencl_object(self,
                                       program_src=fut_opencl_src,
                                       command_queue=command_queue,
                                       interactive=interactive,
                                       platform_pref=platform_pref,
                                       device_pref=device_pref,
                                       default_group_size=default_group_size,
                                       default_num_groups=default_num_groups,
                                       default_tile_size=default_tile_size,
                                       default_threshold=default_threshold,
                                       size_heuristics=size_heuristics,
                                       required_types=["i32", "f32", "bool"],
                                       user_sizes=sizes,
                                       all_sizes={"main.segred_group_size_13143": {"class": "group_size", "value": None},
                                        "main.segred_num_groups_13145": {"class": "num_groups", "value": None}})
    self.map_transpose_f32_var = program.map_transpose_f32
    self.map_transpose_f32_low_height_var = program.map_transpose_f32_low_height
    self.map_transpose_f32_low_width_var = program.map_transpose_f32_low_width
    self.map_transpose_f32_small_var = program.map_transpose_f32_small
    self.segred_nonseg_13152_var = program.segred_nonseg_13152
    counter_mem_13422 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                  np.int32(0), np.int32(0), np.int32(0),
                                  np.int32(0), np.int32(0), np.int32(0),
                                  np.int32(0)], dtype=np.int32)
    static_mem_13823 = opencl_alloc(self, 40, "static_mem_13823")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_13823,
                      normaliseArray(counter_mem_13422),
                      is_blocking=synchronous)
    self.counter_mem_13422 = static_mem_13823
  def futhark_main(self, wih_mem_13263, who_mem_13264, inputs_mem_13265,
                   targets_mem_13266, sizze_12929, sizze_12930, sizze_12931,
                   sizze_12932, sizze_12933, sizze_12934, sizze_12935,
                   sizze_12936, sizze_12937, sizze_12938, lr_12939):
    dim_zzero_12944 = (np.int32(0) == sizze_12936)
    dim_zzero_12945 = (np.int32(0) == sizze_12937)
    dim_zzero_12946 = (np.int32(0) == sizze_12938)
    y_12947 = (dim_zzero_12945 or dim_zzero_12946)
    old_empty_12948 = (dim_zzero_12944 or y_12947)
    dim_zzero_12949 = (np.int32(0) == sizze_12933)
    new_empty_12950 = (y_12947 or dim_zzero_12949)
    both_empty_12951 = (old_empty_12948 and new_empty_12950)
    dim_match_12952 = (sizze_12933 == sizze_12936)
    empty_or_match_12953 = (both_empty_12951 or dim_match_12952)
    empty_or_match_cert_12954 = True
    assert empty_or_match_12953, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:181:98-115\n   #1  GPUTraining.fut:180:1-181:116\n" % ("function arguments of wrong shape",))
    dim_zzero_12956 = (np.int32(0) == sizze_12929)
    dim_zzero_12957 = (np.int32(0) == sizze_12930)
    old_empty_12958 = (dim_zzero_12956 or dim_zzero_12957)
    dim_zzero_12959 = (np.int32(0) == sizze_12934)
    dim_zzero_12960 = (np.int32(0) == sizze_12935)
    new_empty_12961 = (dim_zzero_12959 or dim_zzero_12960)
    both_empty_12962 = (old_empty_12958 and new_empty_12961)
    dim_match_12963 = (sizze_12934 == sizze_12929)
    dim_match_12964 = (sizze_12935 == sizze_12930)
    match_12965 = (dim_match_12963 and dim_match_12964)
    empty_or_match_12966 = (both_empty_12962 or match_12965)
    empty_or_match_cert_12967 = True
    assert empty_or_match_12966, ("Error: %s\n\nBacktrace:\n-> #0  /futlib/soacs.fut:88:3-32\n   #1  GPUTraining.fut:181:3-116\n   #2  GPUTraining.fut:180:1-181:116\n" % ("Row shape of input array does not match shape of neutral element",))
    dim_zzero_12969 = (np.int32(0) == sizze_12931)
    dim_zzero_12970 = (np.int32(0) == sizze_12932)
    old_empty_12971 = (dim_zzero_12969 or dim_zzero_12970)
    both_empty_12972 = (y_12947 and old_empty_12971)
    dim_match_12973 = (sizze_12937 == sizze_12931)
    dim_match_12974 = (sizze_12938 == sizze_12932)
    match_12975 = (dim_match_12973 and dim_match_12974)
    empty_or_match_12976 = (both_empty_12972 or match_12975)
    empty_or_match_cert_12977 = True
    assert empty_or_match_12976, ("Error: %s\n\nBacktrace:\n-> #0  /futlib/soacs.fut:88:3-32\n   #1  GPUTraining.fut:181:3-116\n   #2  GPUTraining.fut:180:1-181:116\n" % ("Row shape of input array does not match shape of neutral element",))
    both_empty_12979 = (dim_zzero_12959 and dim_zzero_12960)
    dim_match_12980 = (sizze_12935 == sizze_12934)
    empty_or_match_12981 = (both_empty_12979 or dim_match_12980)
    empty_or_match_cert_12982 = True
    assert empty_or_match_12981, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:3:19-30\n   #1  GPUTraining.fut:8:21-38\n   #2  GPUTraining.fut:8:3-9:17\n   #3  GPUTraining.fut:17:19-34\n   #4  GPUTraining.fut:17:3-37\n   #5  GPUTraining.fut:159:23-36\n   #6  GPUTraining.fut:181:48-82\n   #7  GPUTraining.fut:181:3-116\n   #8  GPUTraining.fut:180:1-181:116\n" % ("function arguments of wrong shape",))
    both_empty_12983 = (dim_zzero_12946 and dim_zzero_12959)
    dim_match_12984 = (sizze_12938 == sizze_12934)
    empty_or_match_12985 = (both_empty_12983 or dim_match_12984)
    empty_or_match_cert_12986 = True
    assert empty_or_match_12985, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:3:19-30\n   #1  GPUTraining.fut:8:21-38\n   #2  GPUTraining.fut:8:3-9:17\n   #3  GPUTraining.fut:17:19-34\n   #4  GPUTraining.fut:17:3-37\n   #5  GPUTraining.fut:163:22-43\n   #6  GPUTraining.fut:181:48-82\n   #7  GPUTraining.fut:181:3-116\n   #8  GPUTraining.fut:180:1-181:116\n" % ("function arguments of wrong shape",))
    both_empty_12987 = (dim_zzero_12946 and dim_zzero_12960)
    dim_match_12988 = (sizze_12938 == sizze_12935)
    empty_or_match_12989 = (both_empty_12987 or dim_match_12988)
    empty_or_match_cert_12990 = True
    assert empty_or_match_12989, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:122:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPUTraining.fut:130:3-21\n   #4  GPUTraining.fut:167:23-56\n   #5  GPUTraining.fut:181:48-82\n   #6  GPUTraining.fut:181:3-116\n   #7  GPUTraining.fut:180:1-181:116\n" % ("function arguments of wrong shape",))
    empty_or_match_cert_12991 = True
    assert empty_or_match_12989, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:3:19-30\n   #1  GPUTraining.fut:8:21-38\n   #2  GPUTraining.fut:8:3-9:17\n   #3  GPUTraining.fut:17:19-34\n   #4  GPUTraining.fut:17:3-37\n   #5  GPUTraining.fut:173:50-85\n   #6  GPUTraining.fut:181:48-82\n   #7  GPUTraining.fut:181:3-116\n   #8  GPUTraining.fut:180:1-181:116\n" % ("function arguments of wrong shape",))
    empty_or_match_cert_12992 = True
    assert empty_or_match_12985, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:90:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPUTraining.fut:98:3-15\n   #4  GPUTraining.fut:173:21-87\n   #5  GPUTraining.fut:181:48-82\n   #6  GPUTraining.fut:181:3-116\n   #7  GPUTraining.fut:180:1-181:116\n" % ("function arguments of wrong shape",))
    new_empty_12993 = (dim_zzero_12946 or dim_zzero_12960)
    both_empty_12994 = (new_empty_12961 and new_empty_12993)
    empty_or_match_12995 = (dim_match_12984 or both_empty_12994)
    empty_or_match_cert_12996 = True
    assert empty_or_match_12995, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:66:3-20\n   #1  GPUTraining.fut:175:28-67\n   #2  GPUTraining.fut:181:48-82\n   #3  GPUTraining.fut:181:3-116\n   #4  GPUTraining.fut:180:1-181:116\n" % ("function arguments of wrong shape",))
    old_empty_12997 = (dim_zzero_12946 or dim_zzero_12959)
    new_empty_12998 = (dim_zzero_12959 or dim_zzero_12959)
    both_empty_12999 = (old_empty_12997 and new_empty_12998)
    dim_match_13000 = (sizze_12934 == sizze_12938)
    empty_or_match_13001 = (both_empty_12999 or dim_match_13000)
    empty_or_match_cert_13002 = True
    assert empty_or_match_13001, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:98:3-15\n   #1  GPUTraining.fut:177:21-79\n   #2  GPUTraining.fut:181:48-82\n   #3  GPUTraining.fut:181:3-116\n   #4  GPUTraining.fut:180:1-181:116\n" % ("function arguments of wrong shape",))
    empty_or_match_cert_13003 = True
    assert empty_or_match_12981, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:90:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPUTraining.fut:98:3-15\n   #4  GPUTraining.fut:177:21-79\n   #5  GPUTraining.fut:181:48-82\n   #6  GPUTraining.fut:181:3-116\n   #7  GPUTraining.fut:180:1-181:116\n" % ("function arguments of wrong shape",))
    sizze_13141 = sext_i32_i64(sizze_12933)
    segred_group_sizze_13144 = self.sizes["main.segred_group_size_13143"]
    max_num_groups_13421 = self.sizes["main.segred_num_groups_13145"]
    num_groups_13146 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(squot64(((sizze_13141 + sext_i32_i64(segred_group_sizze_13144)) - np.int64(1)),
                                                          sext_i32_i64(segred_group_sizze_13144)),
                                                  sext_i32_i64(max_num_groups_13421))))
    binop_x_13268 = (sizze_12934 * sizze_12935)
    convop_x_13269 = (sizze_12933 * binop_x_13268)
    binop_x_13270 = sext_i32_i64(convop_x_13269)
    bytes_13267 = (np.int64(4) * binop_x_13270)
    mem_13271 = opencl_alloc(self, bytes_13267, "mem_13271")
    self.futhark__map_transpose_f32(mem_13271, np.int32(0), inputs_mem_13265,
                                    np.int32(0), np.int32(1),
                                    (sizze_12934 * sizze_12935), sizze_12933,
                                    ((sizze_12933 * sizze_12934) * sizze_12935),
                                    ((sizze_12933 * sizze_12934) * sizze_12935))
    binop_x_13273 = (sizze_12937 * sizze_12938)
    convop_x_13274 = (sizze_12936 * binop_x_13273)
    binop_x_13275 = sext_i32_i64(convop_x_13274)
    bytes_13272 = (np.int64(4) * binop_x_13275)
    mem_13276 = opencl_alloc(self, bytes_13272, "mem_13276")
    self.futhark__map_transpose_f32(mem_13276, np.int32(0), targets_mem_13266,
                                    np.int32(0), np.int32(1),
                                    (sizze_12937 * sizze_12938), sizze_12936,
                                    ((sizze_12936 * sizze_12937) * sizze_12938),
                                    ((sizze_12936 * sizze_12937) * sizze_12938))
    num_threads_13277 = (segred_group_sizze_13144 * num_groups_13146)
    twice_num_threads_13278 = (np.int32(2) * num_threads_13277)
    binop_x_13280 = sext_i32_i64(twice_num_threads_13278)
    binop_y_13281 = sext_i32_i64(sizze_12934)
    binop_x_13282 = (binop_x_13280 * binop_y_13281)
    binop_y_13283 = sext_i32_i64(sizze_12935)
    binop_x_13284 = (binop_x_13282 * binop_y_13283)
    bytes_13279 = (np.int64(4) * binop_x_13284)
    mem_13285 = opencl_alloc(self, bytes_13279, "mem_13285")
    binop_y_13289 = sext_i32_i64(sizze_12937)
    binop_x_13290 = (binop_x_13280 * binop_y_13289)
    binop_y_13291 = sext_i32_i64(sizze_12938)
    binop_x_13292 = (binop_x_13290 * binop_y_13291)
    bytes_13287 = (np.int64(4) * binop_x_13292)
    mem_13293 = opencl_alloc(self, bytes_13287, "mem_13293")
    binop_x_13361 = (binop_y_13281 * binop_y_13283)
    bytes_13356 = (np.int64(4) * binop_x_13361)
    mem_13362 = opencl_alloc(self, bytes_13356, "mem_13362")
    binop_x_13368 = (binop_y_13289 * binop_y_13291)
    bytes_13363 = (np.int64(4) * binop_x_13368)
    mem_13369 = opencl_alloc(self, bytes_13363, "mem_13369")
    bytes_13299 = (np.int64(4) * binop_y_13283)
    binop_x_13310 = (binop_y_13283 * binop_y_13289)
    bytes_13307 = (np.int64(4) * binop_x_13310)
    bytes_13315 = (np.int64(4) * binop_y_13291)
    binop_x_13340 = (binop_y_13281 * binop_y_13291)
    bytes_13337 = (np.int64(4) * binop_x_13340)
    bytes_13345 = (np.int64(4) * binop_y_13281)
    num_threads_13397 = (segred_group_sizze_13144 * num_groups_13146)
    num_threads64_13398 = sext_i32_i64(num_threads_13397)
    total_sizze_13399 = (bytes_13356 * num_threads64_13398)
    mem_13298 = opencl_alloc(self, total_sizze_13399, "mem_13298")
    total_sizze_13400 = (bytes_13299 * num_threads64_13398)
    mem_13301 = opencl_alloc(self, total_sizze_13400, "mem_13301")
    total_sizze_13401 = (bytes_13363 * num_threads64_13398)
    mem_13306 = opencl_alloc(self, total_sizze_13401, "mem_13306")
    total_sizze_13402 = (bytes_13307 * num_threads64_13398)
    mem_13311 = opencl_alloc(self, total_sizze_13402, "mem_13311")
    total_sizze_13403 = (bytes_13299 * num_threads64_13398)
    mem_13314 = opencl_alloc(self, total_sizze_13403, "mem_13314")
    total_sizze_13404 = (bytes_13315 * num_threads64_13398)
    mem_13317 = opencl_alloc(self, total_sizze_13404, "mem_13317")
    total_sizze_13405 = (bytes_13363 * num_threads64_13398)
    mem_13322 = opencl_alloc(self, total_sizze_13405, "mem_13322")
    total_sizze_13406 = (bytes_13315 * num_threads64_13398)
    mem_13325 = opencl_alloc(self, total_sizze_13406, "mem_13325")
    total_sizze_13407 = (bytes_13315 * num_threads64_13398)
    mem_13328 = opencl_alloc(self, total_sizze_13407, "mem_13328")
    total_sizze_13408 = (bytes_13356 * num_threads64_13398)
    mem_13333 = opencl_alloc(self, total_sizze_13408, "mem_13333")
    total_sizze_13409 = (bytes_13299 * num_threads64_13398)
    mem_13336 = opencl_alloc(self, total_sizze_13409, "mem_13336")
    total_sizze_13410 = (bytes_13337 * num_threads64_13398)
    mem_13341 = opencl_alloc(self, total_sizze_13410, "mem_13341")
    total_sizze_13411 = (bytes_13315 * num_threads64_13398)
    mem_13344 = opencl_alloc(self, total_sizze_13411, "mem_13344")
    total_sizze_13412 = (bytes_13345 * num_threads64_13398)
    mem_13347 = opencl_alloc(self, total_sizze_13412, "mem_13347")
    total_sizze_13413 = (bytes_13356 * num_threads64_13398)
    mem_13352 = opencl_alloc(self, total_sizze_13413, "mem_13352")
    total_sizze_13414 = (bytes_13299 * num_threads64_13398)
    mem_13355 = opencl_alloc(self, total_sizze_13414, "mem_13355")
    counter_mem_13422 = self.counter_mem_13422
    group_res_arr_mem_13424 = opencl_alloc(self,
                                           (np.int32(4) * (((segred_group_sizze_13144 * num_groups_13146) * sizze_12934) * sizze_12935)),
                                           "group_res_arr_mem_13424")
    group_res_arr_mem_13426 = opencl_alloc(self,
                                           (np.int32(4) * (((segred_group_sizze_13144 * num_groups_13146) * sizze_12937) * sizze_12938)),
                                           "group_res_arr_mem_13426")
    num_threads_13428 = (num_groups_13146 * segred_group_sizze_13144)
    if ((1 * (np.long(num_groups_13146) * np.long(segred_group_sizze_13144))) != 0):
      self.segred_nonseg_13152_var.set_args(cl.LocalMemory(np.long(np.int32(1))),
                                            np.int32(sizze_12930),
                                            np.int32(sizze_12932),
                                            np.int32(sizze_12933),
                                            np.int32(sizze_12934),
                                            np.int32(sizze_12935),
                                            np.int32(sizze_12936),
                                            np.int32(sizze_12937),
                                            np.int32(sizze_12938),
                                            np.float32(lr_12939),
                                            np.int32(num_groups_13146),
                                            wih_mem_13263, who_mem_13264,
                                            mem_13271, mem_13276,
                                            np.int32(num_threads_13277),
                                            mem_13285, mem_13293, mem_13298,
                                            mem_13301, mem_13306, mem_13311,
                                            mem_13314, mem_13317, mem_13322,
                                            mem_13325, mem_13328, mem_13333,
                                            mem_13336, mem_13341, mem_13344,
                                            mem_13347, mem_13352, mem_13355,
                                            mem_13362, mem_13369,
                                            counter_mem_13422,
                                            group_res_arr_mem_13424,
                                            group_res_arr_mem_13426)
      cl.enqueue_nd_range_kernel(self.queue, self.segred_nonseg_13152_var,
                                 ((np.long(num_groups_13146) * np.long(segred_group_sizze_13144)),),
                                 (np.long(segred_group_sizze_13144),))
      if synchronous:
        self.queue.finish()
    mem_13271 = None
    mem_13276 = None
    mem_13285 = None
    mem_13293 = None
    mem_13298 = None
    mem_13301 = None
    mem_13306 = None
    mem_13311 = None
    mem_13314 = None
    mem_13317 = None
    mem_13322 = None
    mem_13325 = None
    mem_13328 = None
    mem_13333 = None
    mem_13336 = None
    mem_13341 = None
    mem_13344 = None
    mem_13347 = None
    mem_13352 = None
    mem_13355 = None
    mem_13374 = opencl_alloc(self, bytes_13356, "mem_13374")
    if ((sext_i32_i64((sizze_12934 * sizze_12935)) * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_13374, mem_13362,
                      dest_offset=np.long(np.int32(0)),
                      src_offset=np.long(((np.int32(0) * (sizze_12935 * sizze_12934)) * np.int32(4))),
                      byte_count=np.long((sext_i32_i64((sizze_12934 * sizze_12935)) * np.int32(4))))
    if synchronous:
      self.queue.finish()
    mem_13362 = None
    mem_13380 = opencl_alloc(self, bytes_13363, "mem_13380")
    if ((sext_i32_i64((sizze_12937 * sizze_12938)) * np.int32(4)) != 0):
      cl.enqueue_copy(self.queue, mem_13380, mem_13369,
                      dest_offset=np.long(np.int32(0)),
                      src_offset=np.long(((np.int32(0) * (sizze_12938 * sizze_12937)) * np.int32(4))),
                      byte_count=np.long((sext_i32_i64((sizze_12937 * sizze_12938)) * np.int32(4))))
    if synchronous:
      self.queue.finish()
    mem_13369 = None
    out_arrsizze_13416 = sizze_12934
    out_arrsizze_13417 = sizze_12935
    out_arrsizze_13419 = sizze_12937
    out_arrsizze_13420 = sizze_12938
    out_mem_13415 = mem_13374
    out_mem_13418 = mem_13380
    return (out_mem_13415, out_arrsizze_13416, out_arrsizze_13417,
            out_mem_13418, out_arrsizze_13419, out_arrsizze_13420)
  def futhark__map_transpose_f32(self, destmem_0, destoffset_1, srcmem_2,
                                 srcoffset_3, num_arrays_4, x_elems_5,
                                 y_elems_6, in_elems_7, out_elems_8):
    if ((num_arrays_4 == np.int32(0)) or ((x_elems_5 == np.int32(0)) or (y_elems_6 == np.int32(0)))):
      pass
    else:
      muly_10 = squot32(np.int32(16), x_elems_5)
      mulx_9 = squot32(np.int32(16), y_elems_6)
      if ((in_elems_7 == out_elems_8) and (((num_arrays_4 == np.int32(1)) or ((x_elems_5 * y_elems_6) == in_elems_7)) and ((x_elems_5 == np.int32(1)) or (y_elems_6 == np.int32(1))))):
        if ((in_elems_7 * np.int32(4)) != 0):
          cl.enqueue_copy(self.queue, destmem_0, srcmem_2,
                          dest_offset=np.long(destoffset_1),
                          src_offset=np.long(srcoffset_3),
                          byte_count=np.long((in_elems_7 * np.int32(4))))
        if synchronous:
          self.queue.finish()
      else:
        if (sle32(x_elems_5, np.int32(8)) and slt32(np.int32(16), y_elems_6)):
          if ((((1 * (np.long(squot32(((x_elems_5 + np.int32(16)) - np.int32(1)),
                                      np.int32(16))) * np.long(np.int32(16)))) * (np.long(squot32(((squot32(((y_elems_6 + muly_10) - np.int32(1)),
                                                                                                            muly_10) + np.int32(16)) - np.int32(1)),
                                                                                                  np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
            self.map_transpose_f32_low_width_var.set_args(cl.LocalMemory(np.long(np.int32(1088))),
                                                          np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(in_elems_7),
                                                          np.int32(out_elems_8),
                                                          np.int32(mulx_9),
                                                          np.int32(muly_10),
                                                          destmem_0, srcmem_2)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.map_transpose_f32_low_width_var,
                                       ((np.long(squot32(((x_elems_5 + np.int32(16)) - np.int32(1)),
                                                         np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(squot32(((squot32(((y_elems_6 + muly_10) - np.int32(1)),
                                                                   muly_10) + np.int32(16)) - np.int32(1)),
                                                         np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                       (np.long(np.int32(16)),
                                        np.long(np.int32(16)),
                                        np.long(np.int32(1))))
            if synchronous:
              self.queue.finish()
        else:
          if (sle32(y_elems_6, np.int32(8)) and slt32(np.int32(16), x_elems_5)):
            if ((((1 * (np.long(squot32(((squot32(((x_elems_5 + mulx_9) - np.int32(1)),
                                                  mulx_9) + np.int32(16)) - np.int32(1)),
                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(squot32(((y_elems_6 + np.int32(16)) - np.int32(1)),
                                                                                                    np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
              self.map_transpose_f32_low_height_var.set_args(cl.LocalMemory(np.long(np.int32(1088))),
                                                             np.int32(destoffset_1),
                                                             np.int32(srcoffset_3),
                                                             np.int32(num_arrays_4),
                                                             np.int32(x_elems_5),
                                                             np.int32(y_elems_6),
                                                             np.int32(in_elems_7),
                                                             np.int32(out_elems_8),
                                                             np.int32(mulx_9),
                                                             np.int32(muly_10),
                                                             destmem_0,
                                                             srcmem_2)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.map_transpose_f32_low_height_var,
                                         ((np.long(squot32(((squot32(((x_elems_5 + mulx_9) - np.int32(1)),
                                                                     mulx_9) + np.int32(16)) - np.int32(1)),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(squot32(((y_elems_6 + np.int32(16)) - np.int32(1)),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                         (np.long(np.int32(16)),
                                          np.long(np.int32(16)),
                                          np.long(np.int32(1))))
              if synchronous:
                self.queue.finish()
          else:
            if (sle32(x_elems_5, np.int32(8)) and sle32(y_elems_6,
                                                        np.int32(8))):
              if ((1 * (np.long(squot32(((((num_arrays_4 * x_elems_5) * y_elems_6) + np.int32(256)) - np.int32(1)),
                                        np.int32(256))) * np.long(np.int32(256)))) != 0):
                self.map_transpose_f32_small_var.set_args(cl.LocalMemory(np.long(np.int32(1))),
                                                          np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(in_elems_7),
                                                          np.int32(out_elems_8),
                                                          np.int32(mulx_9),
                                                          np.int32(muly_10),
                                                          destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_f32_small_var,
                                           ((np.long(squot32(((((num_arrays_4 * x_elems_5) * y_elems_6) + np.int32(256)) - np.int32(1)),
                                                             np.int32(256))) * np.long(np.int32(256))),),
                                           (np.long(np.int32(256)),))
                if synchronous:
                  self.queue.finish()
            else:
              if ((((1 * (np.long(squot32(((x_elems_5 + np.int32(32)) - np.int32(1)),
                                          np.int32(32))) * np.long(np.int32(32)))) * (np.long(squot32(((y_elems_6 + np.int32(32)) - np.int32(1)),
                                                                                                      np.int32(32))) * np.long(np.int32(8)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
                self.map_transpose_f32_var.set_args(cl.LocalMemory(np.long(np.int32(4224))),
                                                    np.int32(destoffset_1),
                                                    np.int32(srcoffset_3),
                                                    np.int32(num_arrays_4),
                                                    np.int32(x_elems_5),
                                                    np.int32(y_elems_6),
                                                    np.int32(in_elems_7),
                                                    np.int32(out_elems_8),
                                                    np.int32(mulx_9),
                                                    np.int32(muly_10),
                                                    destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_f32_var,
                                           ((np.long(squot32(((x_elems_5 + np.int32(32)) - np.int32(1)),
                                                             np.int32(32))) * np.long(np.int32(32))),
                                            (np.long(squot32(((y_elems_6 + np.int32(32)) - np.int32(1)),
                                                             np.int32(32))) * np.long(np.int32(8))),
                                            (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                           (np.long(np.int32(32)),
                                            np.long(np.int32(8)),
                                            np.long(np.int32(1))))
                if synchronous:
                  self.queue.finish()
    return ()
  def main(self, lr_12939_ext, wih_mem_13263_ext, who_mem_13264_ext,
           inputs_mem_13265_ext, targets_mem_13266_ext):
    try:
      lr_12939 = np.float32(ct.c_float(lr_12939_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(lr_12939_ext),
                                                                                                                            lr_12939_ext))
    try:
      assert ((type(wih_mem_13263_ext) in [np.ndarray,
                                           cl.array.Array]) and (wih_mem_13263_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_12929 = np.int32(wih_mem_13263_ext.shape[0])
      sizze_12930 = np.int32(wih_mem_13263_ext.shape[1])
      if (type(wih_mem_13263_ext) == cl.array.Array):
        wih_mem_13263 = wih_mem_13263_ext.data
      else:
        wih_mem_13263 = opencl_alloc(self, np.int64(wih_mem_13263_ext.nbytes),
                                     "wih_mem_13263")
        if (np.int64(wih_mem_13263_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, wih_mem_13263,
                          normaliseArray(wih_mem_13263_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(wih_mem_13263_ext),
                                                                                                                            wih_mem_13263_ext))
    try:
      assert ((type(who_mem_13264_ext) in [np.ndarray,
                                           cl.array.Array]) and (who_mem_13264_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_12931 = np.int32(who_mem_13264_ext.shape[0])
      sizze_12932 = np.int32(who_mem_13264_ext.shape[1])
      if (type(who_mem_13264_ext) == cl.array.Array):
        who_mem_13264 = who_mem_13264_ext.data
      else:
        who_mem_13264 = opencl_alloc(self, np.int64(who_mem_13264_ext.nbytes),
                                     "who_mem_13264")
        if (np.int64(who_mem_13264_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, who_mem_13264,
                          normaliseArray(who_mem_13264_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(who_mem_13264_ext),
                                                                                                                            who_mem_13264_ext))
    try:
      assert ((type(inputs_mem_13265_ext) in [np.ndarray,
                                              cl.array.Array]) and (inputs_mem_13265_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_12933 = np.int32(inputs_mem_13265_ext.shape[0])
      sizze_12934 = np.int32(inputs_mem_13265_ext.shape[1])
      sizze_12935 = np.int32(inputs_mem_13265_ext.shape[2])
      if (type(inputs_mem_13265_ext) == cl.array.Array):
        inputs_mem_13265 = inputs_mem_13265_ext.data
      else:
        inputs_mem_13265 = opencl_alloc(self,
                                        np.int64(inputs_mem_13265_ext.nbytes),
                                        "inputs_mem_13265")
        if (np.int64(inputs_mem_13265_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, inputs_mem_13265,
                          normaliseArray(inputs_mem_13265_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][][]f32",
                                                                                                                            type(inputs_mem_13265_ext),
                                                                                                                            inputs_mem_13265_ext))
    try:
      assert ((type(targets_mem_13266_ext) in [np.ndarray,
                                               cl.array.Array]) and (targets_mem_13266_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_12936 = np.int32(targets_mem_13266_ext.shape[0])
      sizze_12937 = np.int32(targets_mem_13266_ext.shape[1])
      sizze_12938 = np.int32(targets_mem_13266_ext.shape[2])
      if (type(targets_mem_13266_ext) == cl.array.Array):
        targets_mem_13266 = targets_mem_13266_ext.data
      else:
        targets_mem_13266 = opencl_alloc(self,
                                         np.int64(targets_mem_13266_ext.nbytes),
                                         "targets_mem_13266")
        if (np.int64(targets_mem_13266_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, targets_mem_13266,
                          normaliseArray(targets_mem_13266_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][][]f32",
                                                                                                                            type(targets_mem_13266_ext),
                                                                                                                            targets_mem_13266_ext))
    (out_mem_13415, out_arrsizze_13416, out_arrsizze_13417, out_mem_13418,
     out_arrsizze_13419, out_arrsizze_13420) = self.futhark_main(wih_mem_13263,
                                                                 who_mem_13264,
                                                                 inputs_mem_13265,
                                                                 targets_mem_13266,
                                                                 sizze_12929,
                                                                 sizze_12930,
                                                                 sizze_12931,
                                                                 sizze_12932,
                                                                 sizze_12933,
                                                                 sizze_12934,
                                                                 sizze_12935,
                                                                 sizze_12936,
                                                                 sizze_12937,
                                                                 sizze_12938,
                                                                 lr_12939)
    return (cl.array.Array(self.queue, (out_arrsizze_13416, out_arrsizze_13417),
                           ct.c_float, data=out_mem_13415),
            cl.array.Array(self.queue, (out_arrsizze_13419, out_arrsizze_13420),
                           ct.c_float, data=out_mem_13418))