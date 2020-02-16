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
__kernel void segmap_11374(int32_t sizze_11066, int32_t sizze_11067,
                           int32_t sizze_11068, __global
                           unsigned char *targets_mem_13443, __global
                           unsigned char *mem_13529, __global
                           unsigned char *mem_13534)
{
    const int32_t segmap_group_sizze_11380 = mainzisegmap_group_sizze_11379;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_13825;
    int32_t local_tid_13826;
    int32_t group_sizze_13829;
    int32_t wave_sizze_13828;
    int32_t group_tid_13827;
    
    global_tid_13825 = get_global_id(0);
    local_tid_13826 = get_local_id(0);
    group_sizze_13829 = get_local_size(0);
    wave_sizze_13828 = LOCKSTEP_WIDTH;
    group_tid_13827 = get_group_id(0);
    
    int32_t phys_tid_11374 = global_tid_13825;
    int32_t gtid_11372 = squot32(group_tid_13827 * segmap_group_sizze_11380 +
                                 local_tid_13826, sizze_11068);
    int32_t gtid_11373;
    
    gtid_11373 = group_tid_13827 * segmap_group_sizze_11380 + local_tid_13826 -
        squot32(group_tid_13827 * segmap_group_sizze_11380 + local_tid_13826,
                sizze_11068) * sizze_11068;
    if (slt32(gtid_11372, sizze_11067) && slt32(gtid_11373, sizze_11068)) {
        float x_11387 = ((__global float *) targets_mem_13443)[gtid_11372 *
                                                               sizze_11068 +
                                                               gtid_11373];
        int32_t binop_x_11657 = sizze_11068 * gtid_11372;
        int32_t binop_x_11658 = gtid_11373 + binop_x_11657;
        int32_t new_index_11659 = squot32(binop_x_11658, sizze_11066);
        int32_t binop_y_11665 = sizze_11066 * new_index_11659;
        int32_t new_index_11666 = binop_x_11658 - binop_y_11665;
        float x_11388 = ((__global float *) mem_13529)[new_index_11659 *
                                                       sizze_11066 +
                                                       new_index_11666];
        float res_11389 = x_11387 - x_11388;
        
        ((__global float *) mem_13534)[gtid_11372 * sizze_11068 + gtid_11373] =
            res_11389;
    }
}
__kernel void segmap_11392(int32_t sizze_11063, int32_t sizze_11066, __global
                           unsigned char *mem_13529, __global
                           unsigned char *mem_13539)
{
    const int32_t segmap_group_sizze_11398 = mainzisegmap_group_sizze_11397;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_13830;
    int32_t local_tid_13831;
    int32_t group_sizze_13834;
    int32_t wave_sizze_13833;
    int32_t group_tid_13832;
    
    global_tid_13830 = get_global_id(0);
    local_tid_13831 = get_local_id(0);
    group_sizze_13834 = get_local_size(0);
    wave_sizze_13833 = LOCKSTEP_WIDTH;
    group_tid_13832 = get_group_id(0);
    
    int32_t phys_tid_11392 = global_tid_13830;
    int32_t gtid_11390 = squot32(group_tid_13832 * segmap_group_sizze_11398 +
                                 local_tid_13831, sizze_11066);
    int32_t gtid_11391;
    
    gtid_11391 = group_tid_13832 * segmap_group_sizze_11398 + local_tid_13831 -
        squot32(group_tid_13832 * segmap_group_sizze_11398 + local_tid_13831,
                sizze_11066) * sizze_11066;
    if (slt32(gtid_11390, sizze_11063) && slt32(gtid_11391, sizze_11066)) {
        float x_11405 = ((__global float *) mem_13529)[gtid_11390 *
                                                       sizze_11066 +
                                                       gtid_11391];
        float res_11406 = 1.0F - x_11405;
        
        ((__global float *) mem_13539)[gtid_11390 * sizze_11066 + gtid_11391] =
            res_11406;
    }
}
__kernel void segmap_11466(int32_t sizze_11066, int32_t sizze_11067,
                           int32_t sizze_11068, __global
                           unsigned char *mem_13529, __global
                           unsigned char *mem_13534, __global
                           unsigned char *mem_13539, __global
                           unsigned char *mem_13544)
{
    const int32_t segmap_group_sizze_11472 = mainzisegmap_group_sizze_11471;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_13835;
    int32_t local_tid_13836;
    int32_t group_sizze_13839;
    int32_t wave_sizze_13838;
    int32_t group_tid_13837;
    
    global_tid_13835 = get_global_id(0);
    local_tid_13836 = get_local_id(0);
    group_sizze_13839 = get_local_size(0);
    wave_sizze_13838 = LOCKSTEP_WIDTH;
    group_tid_13837 = get_group_id(0);
    
    int32_t phys_tid_11466 = global_tid_13835;
    int32_t gtid_11464 = squot32(group_tid_13837 * segmap_group_sizze_11472 +
                                 local_tid_13836, sizze_11068);
    int32_t gtid_11465;
    
    gtid_11465 = group_tid_13837 * segmap_group_sizze_11472 + local_tid_13836 -
        squot32(group_tid_13837 * segmap_group_sizze_11472 + local_tid_13836,
                sizze_11068) * sizze_11068;
    if (slt32(gtid_11464, sizze_11067) && slt32(gtid_11465, sizze_11068)) {
        float x_11479 = ((__global float *) mem_13534)[gtid_11464 *
                                                       sizze_11068 +
                                                       gtid_11465];
        int32_t binop_x_11667 = sizze_11068 * gtid_11464;
        int32_t binop_x_11668 = gtid_11465 + binop_x_11667;
        int32_t new_index_11669 = squot32(binop_x_11668, sizze_11066);
        int32_t binop_y_11675 = sizze_11066 * new_index_11669;
        int32_t new_index_11676 = binop_x_11668 - binop_y_11675;
        float x_11480 = ((__global float *) mem_13529)[new_index_11669 *
                                                       sizze_11066 +
                                                       new_index_11676];
        float x_11481 = ((__global float *) mem_13539)[new_index_11669 *
                                                       sizze_11066 +
                                                       new_index_11676];
        float res_11482 = x_11479 * x_11480;
        float res_11483 = x_11481 * res_11482;
        
        ((__global float *) mem_13544)[gtid_11464 * sizze_11068 + gtid_11465] =
            res_11483;
    }
}
__kernel void segmap_11487(int32_t sizze_11061, int32_t sizze_11063,
                           int32_t sizze_11064, __global
                           unsigned char *who_mem_13441, __global
                           unsigned char *mem_13591, __global
                           unsigned char *mem_13596)
{
    const int32_t segmap_group_sizze_11493 = mainzisegmap_group_sizze_11492;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_13853;
    int32_t local_tid_13854;
    int32_t group_sizze_13857;
    int32_t wave_sizze_13856;
    int32_t group_tid_13855;
    
    global_tid_13853 = get_global_id(0);
    local_tid_13854 = get_local_id(0);
    group_sizze_13857 = get_local_size(0);
    wave_sizze_13856 = LOCKSTEP_WIDTH;
    group_tid_13855 = get_group_id(0);
    
    int32_t phys_tid_11487 = global_tid_13853;
    int32_t gtid_11485 = squot32(group_tid_13855 * segmap_group_sizze_11493 +
                                 local_tid_13854, sizze_11064);
    int32_t gtid_11486;
    
    gtid_11486 = group_tid_13855 * segmap_group_sizze_11493 + local_tid_13854 -
        squot32(group_tid_13855 * segmap_group_sizze_11493 + local_tid_13854,
                sizze_11064) * sizze_11064;
    if (slt32(gtid_11485, sizze_11063) && slt32(gtid_11486, sizze_11064)) {
        float x_11500 = ((__global float *) who_mem_13441)[gtid_11485 *
                                                           sizze_11064 +
                                                           gtid_11486];
        int32_t binop_x_11687 = sizze_11064 * gtid_11485;
        int32_t binop_x_11688 = gtid_11486 + binop_x_11687;
        int32_t new_index_11689 = squot32(binop_x_11688, sizze_11061);
        int32_t binop_y_11695 = sizze_11061 * new_index_11689;
        int32_t new_index_11696 = binop_x_11688 - binop_y_11695;
        float x_11501 = ((__global float *) mem_13591)[new_index_11689 *
                                                       sizze_11061 +
                                                       new_index_11696];
        float res_11502 = x_11500 + x_11501;
        
        ((__global float *) mem_13596)[gtid_11485 * sizze_11064 + gtid_11486] =
            res_11502;
    }
}
__kernel void segmap_11505(int32_t sizze_11061, int32_t sizze_11066, __global
                           unsigned char *mem_13486, __global
                           unsigned char *mem_13601)
{
    const int32_t segmap_group_sizze_11511 = mainzisegmap_group_sizze_11510;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_13858;
    int32_t local_tid_13859;
    int32_t group_sizze_13862;
    int32_t wave_sizze_13861;
    int32_t group_tid_13860;
    
    global_tid_13858 = get_global_id(0);
    local_tid_13859 = get_local_id(0);
    group_sizze_13862 = get_local_size(0);
    wave_sizze_13861 = LOCKSTEP_WIDTH;
    group_tid_13860 = get_group_id(0);
    
    int32_t phys_tid_11505 = global_tid_13858;
    int32_t gtid_11503 = squot32(group_tid_13860 * segmap_group_sizze_11511 +
                                 local_tid_13859, sizze_11066);
    int32_t gtid_11504;
    
    gtid_11504 = group_tid_13860 * segmap_group_sizze_11511 + local_tid_13859 -
        squot32(group_tid_13860 * segmap_group_sizze_11511 + local_tid_13859,
                sizze_11066) * sizze_11066;
    if (slt32(gtid_11503, sizze_11061) && slt32(gtid_11504, sizze_11066)) {
        float x_11518 = ((__global float *) mem_13486)[gtid_11503 *
                                                       sizze_11066 +
                                                       gtid_11504];
        float res_11519 = 1.0F - x_11518;
        
        ((__global float *) mem_13601)[gtid_11503 * sizze_11066 + gtid_11504] =
            res_11519;
    }
}
__kernel void segmap_11641(int32_t sizze_11061, int32_t sizze_11062,
                           int32_t sizze_11065, __global
                           unsigned char *wih_mem_13440, __global
                           unsigned char *mem_13695, __global
                           unsigned char *mem_13700)
{
    const int32_t segmap_group_sizze_11647 = mainzisegmap_group_sizze_11646;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_13889;
    int32_t local_tid_13890;
    int32_t group_sizze_13893;
    int32_t wave_sizze_13892;
    int32_t group_tid_13891;
    
    global_tid_13889 = get_global_id(0);
    local_tid_13890 = get_local_id(0);
    group_sizze_13893 = get_local_size(0);
    wave_sizze_13892 = LOCKSTEP_WIDTH;
    group_tid_13891 = get_group_id(0);
    
    int32_t phys_tid_11641 = global_tid_13889;
    int32_t gtid_11639 = squot32(group_tid_13891 * segmap_group_sizze_11647 +
                                 local_tid_13890, sizze_11062);
    int32_t gtid_11640;
    
    gtid_11640 = group_tid_13891 * segmap_group_sizze_11647 + local_tid_13890 -
        squot32(group_tid_13891 * segmap_group_sizze_11647 + local_tid_13890,
                sizze_11062) * sizze_11062;
    if (slt32(gtid_11639, sizze_11061) && slt32(gtid_11640, sizze_11062)) {
        float x_11654 = ((__global float *) wih_mem_13440)[gtid_11639 *
                                                           sizze_11062 +
                                                           gtid_11640];
        int32_t binop_x_11717 = sizze_11062 * gtid_11639;
        int32_t binop_x_11718 = gtid_11640 + binop_x_11717;
        int32_t new_index_11719 = squot32(binop_x_11718, sizze_11065);
        int32_t binop_y_11725 = sizze_11065 * new_index_11719;
        int32_t new_index_11726 = binop_x_11718 - binop_y_11725;
        float x_11655 = ((__global float *) mem_13695)[new_index_11719 *
                                                       sizze_11065 +
                                                       new_index_11726];
        float res_11656 = x_11654 + x_11655;
        
        ((__global float *) mem_13700)[gtid_11639 * sizze_11062 + gtid_11640] =
            res_11656;
    }
}
__kernel void segmap_intragroup_11753(__local volatile
                                      int64_t *mem_13452_backing_aligned_0,
                                      __local volatile
                                      int64_t *mem_13457_backing_aligned_1,
                                      __local volatile
                                      int64_t *mem_13467_backing_aligned_2,
                                      __local volatile
                                      int64_t *mem_13472_backing_aligned_3,
                                      int32_t sizze_11061, int32_t sizze_11062,
                                      int32_t sizze_11066,
                                      int32_t num_groups_y_11751,
                                      int32_t num_whole_tiles_11754,
                                      int32_t residual_input_11887,
                                      unsigned char cond_11888, __global
                                      unsigned char *wih_mem_13440, __global
                                      unsigned char *inputs_mem_13442, __global
                                      unsigned char *mem_13486)
{
    const int32_t tile_sizze_11744 = mainzitile_sizze_11743;
    const int32_t group_sizze_11745 = mainzitile_sizze_11743 *
                  mainzitile_sizze_11743;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_13452_backing_0 = (__local volatile
                                                           char *) mem_13452_backing_aligned_0;
    __local volatile char *restrict mem_13457_backing_1 = (__local volatile
                                                           char *) mem_13457_backing_aligned_1;
    __local volatile char *restrict mem_13467_backing_2 = (__local volatile
                                                           char *) mem_13467_backing_aligned_2;
    __local volatile char *restrict mem_13472_backing_3 = (__local volatile
                                                           char *) mem_13472_backing_aligned_3;
    int32_t global_tid_13799;
    int32_t local_tid_13800;
    int32_t group_sizze_13803;
    int32_t wave_sizze_13802;
    int32_t group_tid_13801;
    
    global_tid_13799 = get_global_id(0);
    local_tid_13800 = get_local_id(0);
    group_sizze_13803 = get_local_size(0);
    wave_sizze_13802 = LOCKSTEP_WIDTH;
    group_tid_13801 = get_group_id(0);
    
    int32_t gid_flat_11753 = group_tid_13801;
    int32_t gid_x_11741 = squot32(group_tid_13801, num_groups_y_11751);
    int32_t gid_y_11742;
    
    gid_y_11742 = group_tid_13801 - squot32(group_tid_13801,
                                            num_groups_y_11751) *
        num_groups_y_11751;
    
    float mem_13447;
    int32_t ltid_x_11770 = squot32(local_tid_13800, tile_sizze_11744);
    int32_t ltid_y_11771;
    
    ltid_y_11771 = local_tid_13800 - squot32(local_tid_13800,
                                             tile_sizze_11744) *
        tile_sizze_11744;
    
    int32_t ltid_flat_11772;
    
    ltid_flat_11772 = local_tid_13800;
    if (slt32(ltid_x_11770, tile_sizze_11744) && slt32(ltid_y_11771,
                                                       tile_sizze_11744)) {
        mem_13447 = 0.0F;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t binop_x_11849 = gid_x_11741 * tile_sizze_11744;
    int32_t binop_x_11851 = gid_y_11742 * tile_sizze_11744;
    __local char *mem_13452;
    
    mem_13452 = (__local char *) mem_13452_backing_0;
    
    __local char *mem_13457;
    
    mem_13457 = (__local char *) mem_13457_backing_1;
    
    float mem_13461;
    
    for (int32_t tile_id_11781 = 0; tile_id_11781 < num_whole_tiles_11754;
         tile_id_11781++) {
        int32_t binop_x_11845 = tile_sizze_11744 * tile_id_11781;
        int32_t ltid_x_11782 = squot32(local_tid_13800, tile_sizze_11744);
        int32_t ltid_y_11783;
        
        ltid_y_11783 = local_tid_13800 - squot32(local_tid_13800,
                                                 tile_sizze_11744) *
            tile_sizze_11744;
        
        int32_t ltid_flat_11784;
        
        ltid_flat_11784 = local_tid_13800;
        if (slt32(ltid_x_11782, tile_sizze_11744) && slt32(ltid_y_11783,
                                                           tile_sizze_11744)) {
            int32_t i_11846 = ltid_x_11782 + binop_x_11845;
            int32_t j_11848 = ltid_y_11783 + binop_x_11845;
            int32_t gtid_11850 = ltid_x_11782 + binop_x_11849;
            int32_t gtid_11852 = ltid_y_11783 + binop_x_11851;
            float tile_elem_11856 = ((__global
                                      float *) wih_mem_13440)[gtid_11850 *
                                                              sizze_11062 +
                                                              j_11848];
            float tile_elem_11857 = ((__global
                                      float *) inputs_mem_13442)[i_11846 *
                                                                 sizze_11066 +
                                                                 gtid_11852];
            
            ((__local float *) mem_13452)[ltid_x_11782 * tile_sizze_11744 +
                                          ltid_y_11783] = tile_elem_11856;
            ((__local float *) mem_13457)[ltid_x_11782 * tile_sizze_11744 +
                                          ltid_y_11783] = tile_elem_11857;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_11808 = squot32(local_tid_13800, tile_sizze_11744);
        int32_t ltid_y_11809;
        
        ltid_y_11809 = local_tid_13800 - squot32(local_tid_13800,
                                                 tile_sizze_11744) *
            tile_sizze_11744;
        
        int32_t ltid_flat_11810;
        
        ltid_flat_11810 = local_tid_13800;
        if (slt32(ltid_x_11808, tile_sizze_11744) && slt32(ltid_y_11809,
                                                           tile_sizze_11744)) {
            int32_t gtid_11860 = ltid_x_11808 + binop_x_11849;
            int32_t gtid_11862 = ltid_y_11809 + binop_x_11851;
            float acc_11866 = mem_13447;
            bool binop_x_11869 = slt32(gtid_11860, sizze_11061);
            bool binop_y_11870 = slt32(gtid_11862, sizze_11066);
            bool cond_11871 = binop_x_11869 && binop_y_11870;
            float acc_11872;
            
            if (cond_11871) {
                float x_11873;
                float redout_13300 = acc_11866;
                
                for (int32_t i_13301 = 0; i_13301 < tile_sizze_11744;
                     i_13301++) {
                    float x_11877 = ((__local float *) mem_13452)[ltid_x_11808 *
                                                                  tile_sizze_11744 +
                                                                  i_13301];
                    float x_11878 = ((__local float *) mem_13457)[i_13301 *
                                                                  tile_sizze_11744 +
                                                                  ltid_y_11809];
                    float res_11879 = x_11877 * x_11878;
                    float res_11876 = res_11879 + redout_13300;
                    float redout_tmp_13805 = res_11876;
                    
                    redout_13300 = redout_tmp_13805;
                }
                x_11873 = redout_13300;
                acc_11872 = x_11873;
            } else {
                acc_11872 = acc_11866;
            }
            mem_13461 = acc_11872;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_13806 = 0; i_13806 < squot32(tile_sizze_11744 *
                                                    tile_sizze_11744 -
                                                    local_tid_13800 +
                                                    group_sizze_11745 - 1,
                                                    group_sizze_11745);
             i_13806++) {
            mem_13447 = mem_13461;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    __local char *mem_13467;
    
    mem_13467 = (__local char *) mem_13467_backing_2;
    
    __local char *mem_13472;
    
    mem_13472 = (__local char *) mem_13472_backing_3;
    
    float mem_13476;
    float mem_13712;
    
    if (cond_11888) {
        for (int32_t i_13807 = 0; i_13807 < squot32(tile_sizze_11744 *
                                                    tile_sizze_11744 -
                                                    local_tid_13800 +
                                                    group_sizze_11745 - 1,
                                                    group_sizze_11745);
             i_13807++) {
            mem_13712 = mem_13447;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
        int32_t binop_x_11974 = tile_sizze_11744 * num_whole_tiles_11754;
        int32_t ltid_x_11889 = squot32(local_tid_13800, tile_sizze_11744);
        int32_t ltid_y_11890;
        
        ltid_y_11890 = local_tid_13800 - squot32(local_tid_13800,
                                                 tile_sizze_11744) *
            tile_sizze_11744;
        
        int32_t ltid_flat_11891;
        
        ltid_flat_11891 = local_tid_13800;
        if (slt32(ltid_x_11889, tile_sizze_11744) && slt32(ltid_y_11890,
                                                           tile_sizze_11744)) {
            int32_t i_11975 = ltid_x_11889 + binop_x_11974;
            int32_t j_11977 = ltid_y_11890 + binop_x_11974;
            int32_t gtid_11979 = binop_x_11849 + ltid_x_11889;
            int32_t gtid_11981 = binop_x_11851 + ltid_y_11890;
            bool binop_x_11985 = slt32(j_11977, sizze_11062);
            bool binop_y_11986 = slt32(gtid_11979, sizze_11061);
            bool cond_11987 = binop_x_11985 && binop_y_11986;
            float pre_11988;
            
            if (cond_11987) {
                float x_11989 = ((__global float *) wih_mem_13440)[gtid_11979 *
                                                                   sizze_11062 +
                                                                   j_11977];
                
                pre_11988 = x_11989;
            } else {
                pre_11988 = 0.0F;
            }
            
            bool binop_x_11991 = slt32(i_11975, sizze_11062);
            bool binop_y_11992 = slt32(gtid_11981, sizze_11066);
            bool cond_11993 = binop_x_11991 && binop_y_11992;
            float pre_11994;
            
            if (cond_11993) {
                float x_11995 = ((__global float *) inputs_mem_13442)[i_11975 *
                                                                      sizze_11066 +
                                                                      gtid_11981];
                
                pre_11994 = x_11995;
            } else {
                pre_11994 = 0.0F;
            }
            ((__local float *) mem_13467)[ltid_x_11889 * tile_sizze_11744 +
                                          ltid_y_11890] = pre_11988;
            ((__local float *) mem_13472)[ltid_x_11889 * tile_sizze_11744 +
                                          ltid_y_11890] = pre_11994;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_11937 = squot32(local_tid_13800, tile_sizze_11744);
        int32_t ltid_y_11938;
        
        ltid_y_11938 = local_tid_13800 - squot32(local_tid_13800,
                                                 tile_sizze_11744) *
            tile_sizze_11744;
        
        int32_t ltid_flat_11939;
        
        ltid_flat_11939 = local_tid_13800;
        if (slt32(ltid_x_11937, tile_sizze_11744) && slt32(ltid_y_11938,
                                                           tile_sizze_11744)) {
            int32_t gtid_12001 = binop_x_11849 + ltid_x_11937;
            int32_t gtid_12003 = binop_x_11851 + ltid_y_11938;
            float acc_12007 = mem_13447;
            bool binop_x_12010 = slt32(gtid_12001, sizze_11061);
            bool binop_y_12011 = slt32(gtid_12003, sizze_11066);
            bool cond_12012 = binop_x_12010 && binop_y_12011;
            float acc_12013;
            
            if (cond_12012) {
                float x_12014;
                float redout_13302 = acc_12007;
                
                for (int32_t i_13303 = 0; i_13303 < residual_input_11887;
                     i_13303++) {
                    float x_12018 = ((__local float *) mem_13467)[ltid_x_11937 *
                                                                  tile_sizze_11744 +
                                                                  i_13303];
                    float x_12019 = ((__local float *) mem_13472)[i_13303 *
                                                                  tile_sizze_11744 +
                                                                  ltid_y_11938];
                    float res_12020 = x_12018 * x_12019;
                    float res_12017 = res_12020 + redout_13302;
                    float redout_tmp_13808 = res_12017;
                    
                    redout_13302 = redout_tmp_13808;
                }
                x_12014 = redout_13302;
                acc_12013 = x_12014;
            } else {
                acc_12013 = acc_12007;
            }
            mem_13476 = acc_12013;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_13809 = 0; i_13809 < squot32(tile_sizze_11744 *
                                                    tile_sizze_11744 -
                                                    local_tid_13800 +
                                                    group_sizze_11745 - 1,
                                                    group_sizze_11745);
             i_13809++) {
            mem_13712 = mem_13476;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    float mem_13481;
    int32_t ltid_x_12022 = squot32(local_tid_13800, tile_sizze_11744);
    int32_t ltid_y_12023;
    
    ltid_y_12023 = local_tid_13800 - squot32(local_tid_13800,
                                             tile_sizze_11744) *
        tile_sizze_11744;
    
    int32_t ltid_flat_12024;
    
    ltid_flat_12024 = local_tid_13800;
    if (slt32(ltid_x_12022, tile_sizze_11744) && slt32(ltid_y_12023,
                                                       tile_sizze_11744)) {
        int32_t gtid_12033 = binop_x_11849 + ltid_x_12022;
        int32_t gtid_12035 = binop_x_11851 + ltid_y_12023;
        bool binop_x_12037 = slt32(gtid_12033, sizze_11061);
        bool binop_y_12038 = slt32(gtid_12035, sizze_11066);
        bool cond_12039 = binop_x_12037 && binop_y_12038;
        float postlude_12040;
        
        if (cond_12039) {
            float res_12036 = mem_13712;
            float res_12044 = 0.0F - res_12036;
            float res_12045 = fpow32(2.7182817F, res_12044);
            float res_12046 = 1.0F + res_12045;
            float res_12047 = 1.0F / res_12046;
            
            postlude_12040 = res_12047;
        } else {
            postlude_12040 = 0.0F;
        }
        mem_13481 = postlude_12040;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t thread_out_index_13810 = gid_x_11741 * tile_sizze_11744 +
            squot32(local_tid_13800, tile_sizze_11744);
    int32_t thread_out_index_13811;
    
    thread_out_index_13811 = gid_y_11742 * tile_sizze_11744 + (local_tid_13800 -
                                                               squot32(local_tid_13800,
                                                                       tile_sizze_11744) *
                                                               tile_sizze_11744);
    if (slt32(thread_out_index_13810, sizze_11061) &&
        slt32(thread_out_index_13811, sizze_11066)) {
        ((__global float *) mem_13486)[thread_out_index_13810 * sizze_11066 +
                                       thread_out_index_13811] = mem_13481;
    }
}
__kernel void segmap_intragroup_12062(__local volatile
                                      int64_t *mem_13495_backing_aligned_0,
                                      __local volatile
                                      int64_t *mem_13500_backing_aligned_1,
                                      __local volatile
                                      int64_t *mem_13510_backing_aligned_2,
                                      __local volatile
                                      int64_t *mem_13515_backing_aligned_3,
                                      int32_t sizze_11063, int32_t sizze_11064,
                                      int32_t sizze_11066,
                                      int32_t num_groups_y_12060,
                                      int32_t num_whole_tiles_12063,
                                      int32_t residual_input_12196,
                                      unsigned char cond_12197, __global
                                      unsigned char *who_mem_13441, __global
                                      unsigned char *mem_13486, __global
                                      unsigned char *mem_13529)
{
    const int32_t tile_sizze_12053 = mainzitile_sizze_12052;
    const int32_t group_sizze_12054 = mainzitile_sizze_12052 *
                  mainzitile_sizze_12052;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_13495_backing_0 = (__local volatile
                                                           char *) mem_13495_backing_aligned_0;
    __local volatile char *restrict mem_13500_backing_1 = (__local volatile
                                                           char *) mem_13500_backing_aligned_1;
    __local volatile char *restrict mem_13510_backing_2 = (__local volatile
                                                           char *) mem_13510_backing_aligned_2;
    __local volatile char *restrict mem_13515_backing_3 = (__local volatile
                                                           char *) mem_13515_backing_aligned_3;
    int32_t global_tid_13812;
    int32_t local_tid_13813;
    int32_t group_sizze_13816;
    int32_t wave_sizze_13815;
    int32_t group_tid_13814;
    
    global_tid_13812 = get_global_id(0);
    local_tid_13813 = get_local_id(0);
    group_sizze_13816 = get_local_size(0);
    wave_sizze_13815 = LOCKSTEP_WIDTH;
    group_tid_13814 = get_group_id(0);
    
    int32_t gid_flat_12062 = group_tid_13814;
    int32_t gid_x_12050 = squot32(group_tid_13814, num_groups_y_12060);
    int32_t gid_y_12051;
    
    gid_y_12051 = group_tid_13814 - squot32(group_tid_13814,
                                            num_groups_y_12060) *
        num_groups_y_12060;
    
    float mem_13490;
    int32_t ltid_x_12079 = squot32(local_tid_13813, tile_sizze_12053);
    int32_t ltid_y_12080;
    
    ltid_y_12080 = local_tid_13813 - squot32(local_tid_13813,
                                             tile_sizze_12053) *
        tile_sizze_12053;
    
    int32_t ltid_flat_12081;
    
    ltid_flat_12081 = local_tid_13813;
    if (slt32(ltid_x_12079, tile_sizze_12053) && slt32(ltid_y_12080,
                                                       tile_sizze_12053)) {
        mem_13490 = 0.0F;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t binop_x_12158 = gid_x_12050 * tile_sizze_12053;
    int32_t binop_x_12160 = gid_y_12051 * tile_sizze_12053;
    __local char *mem_13495;
    
    mem_13495 = (__local char *) mem_13495_backing_0;
    
    __local char *mem_13500;
    
    mem_13500 = (__local char *) mem_13500_backing_1;
    
    float mem_13504;
    
    for (int32_t tile_id_12090 = 0; tile_id_12090 < num_whole_tiles_12063;
         tile_id_12090++) {
        int32_t binop_x_12154 = tile_sizze_12053 * tile_id_12090;
        int32_t ltid_x_12091 = squot32(local_tid_13813, tile_sizze_12053);
        int32_t ltid_y_12092;
        
        ltid_y_12092 = local_tid_13813 - squot32(local_tid_13813,
                                                 tile_sizze_12053) *
            tile_sizze_12053;
        
        int32_t ltid_flat_12093;
        
        ltid_flat_12093 = local_tid_13813;
        if (slt32(ltid_x_12091, tile_sizze_12053) && slt32(ltid_y_12092,
                                                           tile_sizze_12053)) {
            int32_t i_12155 = ltid_x_12091 + binop_x_12154;
            int32_t j_12157 = ltid_y_12092 + binop_x_12154;
            int32_t gtid_12159 = ltid_x_12091 + binop_x_12158;
            int32_t gtid_12161 = ltid_y_12092 + binop_x_12160;
            float tile_elem_12165 = ((__global
                                      float *) who_mem_13441)[gtid_12159 *
                                                              sizze_11064 +
                                                              j_12157];
            float tile_elem_12166 = ((__global float *) mem_13486)[i_12155 *
                                                                   sizze_11066 +
                                                                   gtid_12161];
            
            ((__local float *) mem_13495)[ltid_x_12091 * tile_sizze_12053 +
                                          ltid_y_12092] = tile_elem_12165;
            ((__local float *) mem_13500)[ltid_x_12091 * tile_sizze_12053 +
                                          ltid_y_12092] = tile_elem_12166;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_12117 = squot32(local_tid_13813, tile_sizze_12053);
        int32_t ltid_y_12118;
        
        ltid_y_12118 = local_tid_13813 - squot32(local_tid_13813,
                                                 tile_sizze_12053) *
            tile_sizze_12053;
        
        int32_t ltid_flat_12119;
        
        ltid_flat_12119 = local_tid_13813;
        if (slt32(ltid_x_12117, tile_sizze_12053) && slt32(ltid_y_12118,
                                                           tile_sizze_12053)) {
            int32_t gtid_12169 = ltid_x_12117 + binop_x_12158;
            int32_t gtid_12171 = ltid_y_12118 + binop_x_12160;
            float acc_12175 = mem_13490;
            bool binop_x_12178 = slt32(gtid_12169, sizze_11063);
            bool binop_y_12179 = slt32(gtid_12171, sizze_11066);
            bool cond_12180 = binop_x_12178 && binop_y_12179;
            float acc_12181;
            
            if (cond_12180) {
                float x_12182;
                float redout_13304 = acc_12175;
                
                for (int32_t i_13305 = 0; i_13305 < tile_sizze_12053;
                     i_13305++) {
                    float x_12186 = ((__local float *) mem_13495)[ltid_x_12117 *
                                                                  tile_sizze_12053 +
                                                                  i_13305];
                    float x_12187 = ((__local float *) mem_13500)[i_13305 *
                                                                  tile_sizze_12053 +
                                                                  ltid_y_12118];
                    float res_12188 = x_12186 * x_12187;
                    float res_12185 = res_12188 + redout_13304;
                    float redout_tmp_13818 = res_12185;
                    
                    redout_13304 = redout_tmp_13818;
                }
                x_12182 = redout_13304;
                acc_12181 = x_12182;
            } else {
                acc_12181 = acc_12175;
            }
            mem_13504 = acc_12181;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_13819 = 0; i_13819 < squot32(tile_sizze_12053 *
                                                    tile_sizze_12053 -
                                                    local_tid_13813 +
                                                    group_sizze_12054 - 1,
                                                    group_sizze_12054);
             i_13819++) {
            mem_13490 = mem_13504;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    __local char *mem_13510;
    
    mem_13510 = (__local char *) mem_13510_backing_2;
    
    __local char *mem_13515;
    
    mem_13515 = (__local char *) mem_13515_backing_3;
    
    float mem_13519;
    float mem_13726;
    
    if (cond_12197) {
        for (int32_t i_13820 = 0; i_13820 < squot32(tile_sizze_12053 *
                                                    tile_sizze_12053 -
                                                    local_tid_13813 +
                                                    group_sizze_12054 - 1,
                                                    group_sizze_12054);
             i_13820++) {
            mem_13726 = mem_13490;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
        int32_t binop_x_12283 = tile_sizze_12053 * num_whole_tiles_12063;
        int32_t ltid_x_12198 = squot32(local_tid_13813, tile_sizze_12053);
        int32_t ltid_y_12199;
        
        ltid_y_12199 = local_tid_13813 - squot32(local_tid_13813,
                                                 tile_sizze_12053) *
            tile_sizze_12053;
        
        int32_t ltid_flat_12200;
        
        ltid_flat_12200 = local_tid_13813;
        if (slt32(ltid_x_12198, tile_sizze_12053) && slt32(ltid_y_12199,
                                                           tile_sizze_12053)) {
            int32_t i_12284 = ltid_x_12198 + binop_x_12283;
            int32_t j_12286 = ltid_y_12199 + binop_x_12283;
            int32_t gtid_12288 = binop_x_12158 + ltid_x_12198;
            int32_t gtid_12290 = binop_x_12160 + ltid_y_12199;
            bool binop_x_12294 = slt32(j_12286, sizze_11064);
            bool binop_y_12295 = slt32(gtid_12288, sizze_11063);
            bool cond_12296 = binop_x_12294 && binop_y_12295;
            float pre_12297;
            
            if (cond_12296) {
                float x_12298 = ((__global float *) who_mem_13441)[gtid_12288 *
                                                                   sizze_11064 +
                                                                   j_12286];
                
                pre_12297 = x_12298;
            } else {
                pre_12297 = 0.0F;
            }
            
            bool binop_x_12300 = slt32(i_12284, sizze_11064);
            bool binop_y_12301 = slt32(gtid_12290, sizze_11066);
            bool cond_12302 = binop_x_12300 && binop_y_12301;
            float pre_12303;
            
            if (cond_12302) {
                float x_12304 = ((__global float *) mem_13486)[i_12284 *
                                                               sizze_11066 +
                                                               gtid_12290];
                
                pre_12303 = x_12304;
            } else {
                pre_12303 = 0.0F;
            }
            ((__local float *) mem_13510)[ltid_x_12198 * tile_sizze_12053 +
                                          ltid_y_12199] = pre_12297;
            ((__local float *) mem_13515)[ltid_x_12198 * tile_sizze_12053 +
                                          ltid_y_12199] = pre_12303;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_12246 = squot32(local_tid_13813, tile_sizze_12053);
        int32_t ltid_y_12247;
        
        ltid_y_12247 = local_tid_13813 - squot32(local_tid_13813,
                                                 tile_sizze_12053) *
            tile_sizze_12053;
        
        int32_t ltid_flat_12248;
        
        ltid_flat_12248 = local_tid_13813;
        if (slt32(ltid_x_12246, tile_sizze_12053) && slt32(ltid_y_12247,
                                                           tile_sizze_12053)) {
            int32_t gtid_12310 = binop_x_12158 + ltid_x_12246;
            int32_t gtid_12312 = binop_x_12160 + ltid_y_12247;
            float acc_12316 = mem_13490;
            bool binop_x_12319 = slt32(gtid_12310, sizze_11063);
            bool binop_y_12320 = slt32(gtid_12312, sizze_11066);
            bool cond_12321 = binop_x_12319 && binop_y_12320;
            float acc_12322;
            
            if (cond_12321) {
                float x_12323;
                float redout_13306 = acc_12316;
                
                for (int32_t i_13307 = 0; i_13307 < residual_input_12196;
                     i_13307++) {
                    float x_12327 = ((__local float *) mem_13510)[ltid_x_12246 *
                                                                  tile_sizze_12053 +
                                                                  i_13307];
                    float x_12328 = ((__local float *) mem_13515)[i_13307 *
                                                                  tile_sizze_12053 +
                                                                  ltid_y_12247];
                    float res_12329 = x_12327 * x_12328;
                    float res_12326 = res_12329 + redout_13306;
                    float redout_tmp_13821 = res_12326;
                    
                    redout_13306 = redout_tmp_13821;
                }
                x_12323 = redout_13306;
                acc_12322 = x_12323;
            } else {
                acc_12322 = acc_12316;
            }
            mem_13519 = acc_12322;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_13822 = 0; i_13822 < squot32(tile_sizze_12053 *
                                                    tile_sizze_12053 -
                                                    local_tid_13813 +
                                                    group_sizze_12054 - 1,
                                                    group_sizze_12054);
             i_13822++) {
            mem_13726 = mem_13519;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    float mem_13524;
    int32_t ltid_x_12331 = squot32(local_tid_13813, tile_sizze_12053);
    int32_t ltid_y_12332;
    
    ltid_y_12332 = local_tid_13813 - squot32(local_tid_13813,
                                             tile_sizze_12053) *
        tile_sizze_12053;
    
    int32_t ltid_flat_12333;
    
    ltid_flat_12333 = local_tid_13813;
    if (slt32(ltid_x_12331, tile_sizze_12053) && slt32(ltid_y_12332,
                                                       tile_sizze_12053)) {
        int32_t gtid_12342 = binop_x_12158 + ltid_x_12331;
        int32_t gtid_12344 = binop_x_12160 + ltid_y_12332;
        bool binop_x_12346 = slt32(gtid_12342, sizze_11063);
        bool binop_y_12347 = slt32(gtid_12344, sizze_11066);
        bool cond_12348 = binop_x_12346 && binop_y_12347;
        float postlude_12349;
        
        if (cond_12348) {
            float res_12345 = mem_13726;
            float res_12353 = 0.0F - res_12345;
            float res_12354 = fpow32(2.7182817F, res_12353);
            float res_12355 = 1.0F + res_12354;
            float res_12356 = 1.0F / res_12355;
            
            postlude_12349 = res_12356;
        } else {
            postlude_12349 = 0.0F;
        }
        mem_13524 = postlude_12349;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t thread_out_index_13823 = gid_x_12050 * tile_sizze_12053 +
            squot32(local_tid_13813, tile_sizze_12053);
    int32_t thread_out_index_13824;
    
    thread_out_index_13824 = gid_y_12051 * tile_sizze_12053 + (local_tid_13813 -
                                                               squot32(local_tid_13813,
                                                                       tile_sizze_12053) *
                                                               tile_sizze_12053);
    if (slt32(thread_out_index_13823, sizze_11063) &&
        slt32(thread_out_index_13824, sizze_11066)) {
        ((__global float *) mem_13529)[thread_out_index_13823 * sizze_11066 +
                                       thread_out_index_13824] = mem_13524;
    }
}
__kernel void segmap_intragroup_12371(__local volatile
                                      int64_t *mem_13557_backing_aligned_0,
                                      __local volatile
                                      int64_t *mem_13562_backing_aligned_1,
                                      __local volatile
                                      int64_t *mem_13572_backing_aligned_2,
                                      __local volatile
                                      int64_t *mem_13577_backing_aligned_3,
                                      int32_t sizze_11061, int32_t sizze_11067,
                                      int32_t sizze_11068, float lr_11069,
                                      int32_t num_groups_y_12369,
                                      int32_t num_whole_tiles_12372,
                                      int32_t residual_input_12505,
                                      unsigned char cond_12506, __global
                                      unsigned char *mem_13544, __global
                                      unsigned char *mem_13548, __global
                                      unsigned char *mem_13591)
{
    const int32_t tile_sizze_12362 = mainzitile_sizze_12361;
    const int32_t group_sizze_12363 = mainzitile_sizze_12361 *
                  mainzitile_sizze_12361;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_13557_backing_0 = (__local volatile
                                                           char *) mem_13557_backing_aligned_0;
    __local volatile char *restrict mem_13562_backing_1 = (__local volatile
                                                           char *) mem_13562_backing_aligned_1;
    __local volatile char *restrict mem_13572_backing_2 = (__local volatile
                                                           char *) mem_13572_backing_aligned_2;
    __local volatile char *restrict mem_13577_backing_3 = (__local volatile
                                                           char *) mem_13577_backing_aligned_3;
    int32_t global_tid_13840;
    int32_t local_tid_13841;
    int32_t group_sizze_13844;
    int32_t wave_sizze_13843;
    int32_t group_tid_13842;
    
    global_tid_13840 = get_global_id(0);
    local_tid_13841 = get_local_id(0);
    group_sizze_13844 = get_local_size(0);
    wave_sizze_13843 = LOCKSTEP_WIDTH;
    group_tid_13842 = get_group_id(0);
    
    int32_t gid_flat_12371 = group_tid_13842;
    int32_t gid_x_12359 = squot32(group_tid_13842, num_groups_y_12369);
    int32_t gid_y_12360;
    
    gid_y_12360 = group_tid_13842 - squot32(group_tid_13842,
                                            num_groups_y_12369) *
        num_groups_y_12369;
    
    float mem_13552;
    int32_t ltid_x_12388 = squot32(local_tid_13841, tile_sizze_12362);
    int32_t ltid_y_12389;
    
    ltid_y_12389 = local_tid_13841 - squot32(local_tid_13841,
                                             tile_sizze_12362) *
        tile_sizze_12362;
    
    int32_t ltid_flat_12390;
    
    ltid_flat_12390 = local_tid_13841;
    if (slt32(ltid_x_12388, tile_sizze_12362) && slt32(ltid_y_12389,
                                                       tile_sizze_12362)) {
        mem_13552 = 0.0F;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t binop_x_12467 = gid_x_12359 * tile_sizze_12362;
    int32_t binop_x_12469 = gid_y_12360 * tile_sizze_12362;
    __local char *mem_13557;
    
    mem_13557 = (__local char *) mem_13557_backing_0;
    
    __local char *mem_13562;
    
    mem_13562 = (__local char *) mem_13562_backing_1;
    
    float mem_13566;
    
    for (int32_t tile_id_12399 = 0; tile_id_12399 < num_whole_tiles_12372;
         tile_id_12399++) {
        int32_t binop_x_12463 = tile_sizze_12362 * tile_id_12399;
        int32_t ltid_x_12400 = squot32(local_tid_13841, tile_sizze_12362);
        int32_t ltid_y_12401;
        
        ltid_y_12401 = local_tid_13841 - squot32(local_tid_13841,
                                                 tile_sizze_12362) *
            tile_sizze_12362;
        
        int32_t ltid_flat_12402;
        
        ltid_flat_12402 = local_tid_13841;
        if (slt32(ltid_x_12400, tile_sizze_12362) && slt32(ltid_y_12401,
                                                           tile_sizze_12362)) {
            int32_t i_12464 = ltid_x_12400 + binop_x_12463;
            int32_t j_12466 = ltid_y_12401 + binop_x_12463;
            int32_t gtid_12468 = ltid_x_12400 + binop_x_12467;
            int32_t gtid_12470 = ltid_y_12401 + binop_x_12469;
            float tile_elem_12474 = ((__global float *) mem_13544)[gtid_12468 *
                                                                   sizze_11068 +
                                                                   j_12466];
            float tile_elem_12475 = ((__global float *) mem_13548)[i_12464 *
                                                                   sizze_11061 +
                                                                   gtid_12470];
            
            ((__local float *) mem_13557)[ltid_x_12400 * tile_sizze_12362 +
                                          ltid_y_12401] = tile_elem_12474;
            ((__local float *) mem_13562)[ltid_x_12400 * tile_sizze_12362 +
                                          ltid_y_12401] = tile_elem_12475;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_12426 = squot32(local_tid_13841, tile_sizze_12362);
        int32_t ltid_y_12427;
        
        ltid_y_12427 = local_tid_13841 - squot32(local_tid_13841,
                                                 tile_sizze_12362) *
            tile_sizze_12362;
        
        int32_t ltid_flat_12428;
        
        ltid_flat_12428 = local_tid_13841;
        if (slt32(ltid_x_12426, tile_sizze_12362) && slt32(ltid_y_12427,
                                                           tile_sizze_12362)) {
            int32_t gtid_12478 = ltid_x_12426 + binop_x_12467;
            int32_t gtid_12480 = ltid_y_12427 + binop_x_12469;
            float acc_12484 = mem_13552;
            bool binop_x_12487 = slt32(gtid_12478, sizze_11067);
            bool binop_y_12488 = slt32(gtid_12480, sizze_11061);
            bool cond_12489 = binop_x_12487 && binop_y_12488;
            float acc_12490;
            
            if (cond_12489) {
                float x_12491;
                float redout_13308 = acc_12484;
                
                for (int32_t i_13309 = 0; i_13309 < tile_sizze_12362;
                     i_13309++) {
                    float x_12495 = ((__local float *) mem_13557)[ltid_x_12426 *
                                                                  tile_sizze_12362 +
                                                                  i_13309];
                    float x_12496 = ((__local float *) mem_13562)[i_13309 *
                                                                  tile_sizze_12362 +
                                                                  ltid_y_12427];
                    float res_12497 = x_12495 * x_12496;
                    float res_12494 = res_12497 + redout_13308;
                    float redout_tmp_13846 = res_12494;
                    
                    redout_13308 = redout_tmp_13846;
                }
                x_12491 = redout_13308;
                acc_12490 = x_12491;
            } else {
                acc_12490 = acc_12484;
            }
            mem_13566 = acc_12490;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_13847 = 0; i_13847 < squot32(tile_sizze_12362 *
                                                    tile_sizze_12362 -
                                                    local_tid_13841 +
                                                    group_sizze_12363 - 1,
                                                    group_sizze_12363);
             i_13847++) {
            mem_13552 = mem_13566;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    __local char *mem_13572;
    
    mem_13572 = (__local char *) mem_13572_backing_2;
    
    __local char *mem_13577;
    
    mem_13577 = (__local char *) mem_13577_backing_3;
    
    float mem_13581;
    float mem_13740;
    
    if (cond_12506) {
        for (int32_t i_13848 = 0; i_13848 < squot32(tile_sizze_12362 *
                                                    tile_sizze_12362 -
                                                    local_tid_13841 +
                                                    group_sizze_12363 - 1,
                                                    group_sizze_12363);
             i_13848++) {
            mem_13740 = mem_13552;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
        int32_t binop_x_12592 = tile_sizze_12362 * num_whole_tiles_12372;
        int32_t ltid_x_12507 = squot32(local_tid_13841, tile_sizze_12362);
        int32_t ltid_y_12508;
        
        ltid_y_12508 = local_tid_13841 - squot32(local_tid_13841,
                                                 tile_sizze_12362) *
            tile_sizze_12362;
        
        int32_t ltid_flat_12509;
        
        ltid_flat_12509 = local_tid_13841;
        if (slt32(ltid_x_12507, tile_sizze_12362) && slt32(ltid_y_12508,
                                                           tile_sizze_12362)) {
            int32_t i_12593 = ltid_x_12507 + binop_x_12592;
            int32_t j_12595 = ltid_y_12508 + binop_x_12592;
            int32_t gtid_12597 = binop_x_12467 + ltid_x_12507;
            int32_t gtid_12599 = binop_x_12469 + ltid_y_12508;
            bool binop_x_12603 = slt32(j_12595, sizze_11068);
            bool binop_y_12604 = slt32(gtid_12597, sizze_11067);
            bool cond_12605 = binop_x_12603 && binop_y_12604;
            float pre_12606;
            
            if (cond_12605) {
                float x_12607 = ((__global float *) mem_13544)[gtid_12597 *
                                                               sizze_11068 +
                                                               j_12595];
                
                pre_12606 = x_12607;
            } else {
                pre_12606 = 0.0F;
            }
            
            bool binop_x_12609 = slt32(i_12593, sizze_11068);
            bool binop_y_12610 = slt32(gtid_12599, sizze_11061);
            bool cond_12611 = binop_x_12609 && binop_y_12610;
            float pre_12612;
            
            if (cond_12611) {
                float x_12613 = ((__global float *) mem_13548)[i_12593 *
                                                               sizze_11061 +
                                                               gtid_12599];
                
                pre_12612 = x_12613;
            } else {
                pre_12612 = 0.0F;
            }
            ((__local float *) mem_13572)[ltid_x_12507 * tile_sizze_12362 +
                                          ltid_y_12508] = pre_12606;
            ((__local float *) mem_13577)[ltid_x_12507 * tile_sizze_12362 +
                                          ltid_y_12508] = pre_12612;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_12555 = squot32(local_tid_13841, tile_sizze_12362);
        int32_t ltid_y_12556;
        
        ltid_y_12556 = local_tid_13841 - squot32(local_tid_13841,
                                                 tile_sizze_12362) *
            tile_sizze_12362;
        
        int32_t ltid_flat_12557;
        
        ltid_flat_12557 = local_tid_13841;
        if (slt32(ltid_x_12555, tile_sizze_12362) && slt32(ltid_y_12556,
                                                           tile_sizze_12362)) {
            int32_t gtid_12619 = binop_x_12467 + ltid_x_12555;
            int32_t gtid_12621 = binop_x_12469 + ltid_y_12556;
            float acc_12625 = mem_13552;
            bool binop_x_12628 = slt32(gtid_12619, sizze_11067);
            bool binop_y_12629 = slt32(gtid_12621, sizze_11061);
            bool cond_12630 = binop_x_12628 && binop_y_12629;
            float acc_12631;
            
            if (cond_12630) {
                float x_12632;
                float redout_13310 = acc_12625;
                
                for (int32_t i_13311 = 0; i_13311 < residual_input_12505;
                     i_13311++) {
                    float x_12636 = ((__local float *) mem_13572)[ltid_x_12555 *
                                                                  tile_sizze_12362 +
                                                                  i_13311];
                    float x_12637 = ((__local float *) mem_13577)[i_13311 *
                                                                  tile_sizze_12362 +
                                                                  ltid_y_12556];
                    float res_12638 = x_12636 * x_12637;
                    float res_12635 = res_12638 + redout_13310;
                    float redout_tmp_13849 = res_12635;
                    
                    redout_13310 = redout_tmp_13849;
                }
                x_12632 = redout_13310;
                acc_12631 = x_12632;
            } else {
                acc_12631 = acc_12625;
            }
            mem_13581 = acc_12631;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_13850 = 0; i_13850 < squot32(tile_sizze_12362 *
                                                    tile_sizze_12362 -
                                                    local_tid_13841 +
                                                    group_sizze_12363 - 1,
                                                    group_sizze_12363);
             i_13850++) {
            mem_13740 = mem_13581;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    float mem_13586;
    int32_t ltid_x_12640 = squot32(local_tid_13841, tile_sizze_12362);
    int32_t ltid_y_12641;
    
    ltid_y_12641 = local_tid_13841 - squot32(local_tid_13841,
                                             tile_sizze_12362) *
        tile_sizze_12362;
    
    int32_t ltid_flat_12642;
    
    ltid_flat_12642 = local_tid_13841;
    if (slt32(ltid_x_12640, tile_sizze_12362) && slt32(ltid_y_12641,
                                                       tile_sizze_12362)) {
        int32_t gtid_12651 = binop_x_12467 + ltid_x_12640;
        int32_t gtid_12653 = binop_x_12469 + ltid_y_12641;
        bool binop_x_12655 = slt32(gtid_12651, sizze_11067);
        bool binop_y_12656 = slt32(gtid_12653, sizze_11061);
        bool cond_12657 = binop_x_12655 && binop_y_12656;
        float postlude_12658;
        
        if (cond_12657) {
            float res_12654 = mem_13740;
            float res_12662 = lr_11069 * res_12654;
            
            postlude_12658 = res_12662;
        } else {
            postlude_12658 = 0.0F;
        }
        mem_13586 = postlude_12658;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t thread_out_index_13851 = gid_x_12359 * tile_sizze_12362 +
            squot32(local_tid_13841, tile_sizze_12362);
    int32_t thread_out_index_13852;
    
    thread_out_index_13852 = gid_y_12360 * tile_sizze_12362 + (local_tid_13841 -
                                                               squot32(local_tid_13841,
                                                                       tile_sizze_12362) *
                                                               tile_sizze_12362);
    if (slt32(thread_out_index_13851, sizze_11067) &&
        slt32(thread_out_index_13852, sizze_11061)) {
        ((__global float *) mem_13591)[thread_out_index_13851 * sizze_11061 +
                                       thread_out_index_13852] = mem_13586;
    }
}
__kernel void segmap_intragroup_12677(__local volatile
                                      int64_t *mem_13614_backing_aligned_0,
                                      __local volatile
                                      int64_t *mem_13619_backing_aligned_1,
                                      __local volatile
                                      int64_t *mem_13629_backing_aligned_2,
                                      __local volatile
                                      int64_t *mem_13634_backing_aligned_3,
                                      int32_t sizze_11063, int32_t sizze_11064,
                                      int32_t sizze_11066, int32_t sizze_11068,
                                      int32_t num_groups_y_12675,
                                      int32_t num_whole_tiles_12678,
                                      int32_t residual_input_12811,
                                      unsigned char cond_12812, __global
                                      unsigned char *mem_13486, __global
                                      unsigned char *mem_13534, __global
                                      unsigned char *mem_13601, __global
                                      unsigned char *mem_13605, __global
                                      unsigned char *mem_13648)
{
    const int32_t tile_sizze_12668 = mainzitile_sizze_12667;
    const int32_t group_sizze_12669 = mainzitile_sizze_12667 *
                  mainzitile_sizze_12667;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_13614_backing_0 = (__local volatile
                                                           char *) mem_13614_backing_aligned_0;
    __local volatile char *restrict mem_13619_backing_1 = (__local volatile
                                                           char *) mem_13619_backing_aligned_1;
    __local volatile char *restrict mem_13629_backing_2 = (__local volatile
                                                           char *) mem_13629_backing_aligned_2;
    __local volatile char *restrict mem_13634_backing_3 = (__local volatile
                                                           char *) mem_13634_backing_aligned_3;
    int32_t global_tid_13863;
    int32_t local_tid_13864;
    int32_t group_sizze_13867;
    int32_t wave_sizze_13866;
    int32_t group_tid_13865;
    
    global_tid_13863 = get_global_id(0);
    local_tid_13864 = get_local_id(0);
    group_sizze_13867 = get_local_size(0);
    wave_sizze_13866 = LOCKSTEP_WIDTH;
    group_tid_13865 = get_group_id(0);
    
    int32_t gid_flat_12677 = group_tid_13865;
    int32_t gid_x_12665 = squot32(group_tid_13865, num_groups_y_12675);
    int32_t gid_y_12666;
    
    gid_y_12666 = group_tid_13865 - squot32(group_tid_13865,
                                            num_groups_y_12675) *
        num_groups_y_12675;
    
    float mem_13609;
    int32_t ltid_x_12694 = squot32(local_tid_13864, tile_sizze_12668);
    int32_t ltid_y_12695;
    
    ltid_y_12695 = local_tid_13864 - squot32(local_tid_13864,
                                             tile_sizze_12668) *
        tile_sizze_12668;
    
    int32_t ltid_flat_12696;
    
    ltid_flat_12696 = local_tid_13864;
    if (slt32(ltid_x_12694, tile_sizze_12668) && slt32(ltid_y_12695,
                                                       tile_sizze_12668)) {
        mem_13609 = 0.0F;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t binop_x_12773 = gid_x_12665 * tile_sizze_12668;
    int32_t binop_x_12775 = gid_y_12666 * tile_sizze_12668;
    __local char *mem_13614;
    
    mem_13614 = (__local char *) mem_13614_backing_0;
    
    __local char *mem_13619;
    
    mem_13619 = (__local char *) mem_13619_backing_1;
    
    float mem_13623;
    
    for (int32_t tile_id_12705 = 0; tile_id_12705 < num_whole_tiles_12678;
         tile_id_12705++) {
        int32_t binop_x_12769 = tile_sizze_12668 * tile_id_12705;
        int32_t ltid_x_12706 = squot32(local_tid_13864, tile_sizze_12668);
        int32_t ltid_y_12707;
        
        ltid_y_12707 = local_tid_13864 - squot32(local_tid_13864,
                                                 tile_sizze_12668) *
            tile_sizze_12668;
        
        int32_t ltid_flat_12708;
        
        ltid_flat_12708 = local_tid_13864;
        if (slt32(ltid_x_12706, tile_sizze_12668) && slt32(ltid_y_12707,
                                                           tile_sizze_12668)) {
            int32_t i_12770 = ltid_x_12706 + binop_x_12769;
            int32_t j_12772 = ltid_y_12707 + binop_x_12769;
            int32_t gtid_12774 = ltid_x_12706 + binop_x_12773;
            int32_t gtid_12776 = ltid_y_12707 + binop_x_12775;
            float tile_elem_12780 = ((__global float *) mem_13605)[gtid_12774 *
                                                                   sizze_11063 +
                                                                   j_12772];
            float tile_elem_12781 = ((__global float *) mem_13534)[i_12770 *
                                                                   sizze_11068 +
                                                                   gtid_12776];
            
            ((__local float *) mem_13614)[ltid_x_12706 * tile_sizze_12668 +
                                          ltid_y_12707] = tile_elem_12780;
            ((__local float *) mem_13619)[ltid_x_12706 * tile_sizze_12668 +
                                          ltid_y_12707] = tile_elem_12781;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_12732 = squot32(local_tid_13864, tile_sizze_12668);
        int32_t ltid_y_12733;
        
        ltid_y_12733 = local_tid_13864 - squot32(local_tid_13864,
                                                 tile_sizze_12668) *
            tile_sizze_12668;
        
        int32_t ltid_flat_12734;
        
        ltid_flat_12734 = local_tid_13864;
        if (slt32(ltid_x_12732, tile_sizze_12668) && slt32(ltid_y_12733,
                                                           tile_sizze_12668)) {
            int32_t gtid_12784 = ltid_x_12732 + binop_x_12773;
            int32_t gtid_12786 = ltid_y_12733 + binop_x_12775;
            float acc_12790 = mem_13609;
            bool binop_x_12793 = slt32(gtid_12784, sizze_11064);
            bool binop_y_12794 = slt32(gtid_12786, sizze_11068);
            bool cond_12795 = binop_x_12793 && binop_y_12794;
            float acc_12796;
            
            if (cond_12795) {
                float x_12797;
                float redout_13312 = acc_12790;
                
                for (int32_t i_13313 = 0; i_13313 < tile_sizze_12668;
                     i_13313++) {
                    float x_12801 = ((__local float *) mem_13614)[ltid_x_12732 *
                                                                  tile_sizze_12668 +
                                                                  i_13313];
                    float x_12802 = ((__local float *) mem_13619)[i_13313 *
                                                                  tile_sizze_12668 +
                                                                  ltid_y_12733];
                    float res_12803 = x_12801 * x_12802;
                    float res_12800 = res_12803 + redout_13312;
                    float redout_tmp_13869 = res_12800;
                    
                    redout_13312 = redout_tmp_13869;
                }
                x_12797 = redout_13312;
                acc_12796 = x_12797;
            } else {
                acc_12796 = acc_12790;
            }
            mem_13623 = acc_12796;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_13870 = 0; i_13870 < squot32(tile_sizze_12668 *
                                                    tile_sizze_12668 -
                                                    local_tid_13864 +
                                                    group_sizze_12669 - 1,
                                                    group_sizze_12669);
             i_13870++) {
            mem_13609 = mem_13623;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    __local char *mem_13629;
    
    mem_13629 = (__local char *) mem_13629_backing_2;
    
    __local char *mem_13634;
    
    mem_13634 = (__local char *) mem_13634_backing_3;
    
    float mem_13638;
    float mem_13754;
    
    if (cond_12812) {
        for (int32_t i_13871 = 0; i_13871 < squot32(tile_sizze_12668 *
                                                    tile_sizze_12668 -
                                                    local_tid_13864 +
                                                    group_sizze_12669 - 1,
                                                    group_sizze_12669);
             i_13871++) {
            mem_13754 = mem_13609;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
        int32_t binop_x_12898 = tile_sizze_12668 * num_whole_tiles_12678;
        int32_t ltid_x_12813 = squot32(local_tid_13864, tile_sizze_12668);
        int32_t ltid_y_12814;
        
        ltid_y_12814 = local_tid_13864 - squot32(local_tid_13864,
                                                 tile_sizze_12668) *
            tile_sizze_12668;
        
        int32_t ltid_flat_12815;
        
        ltid_flat_12815 = local_tid_13864;
        if (slt32(ltid_x_12813, tile_sizze_12668) && slt32(ltid_y_12814,
                                                           tile_sizze_12668)) {
            int32_t i_12899 = ltid_x_12813 + binop_x_12898;
            int32_t j_12901 = ltid_y_12814 + binop_x_12898;
            int32_t gtid_12903 = binop_x_12773 + ltid_x_12813;
            int32_t gtid_12905 = binop_x_12775 + ltid_y_12814;
            bool binop_x_12909 = slt32(j_12901, sizze_11063);
            bool binop_y_12910 = slt32(gtid_12903, sizze_11064);
            bool cond_12911 = binop_x_12909 && binop_y_12910;
            float pre_12912;
            
            if (cond_12911) {
                float x_12913 = ((__global float *) mem_13605)[gtid_12903 *
                                                               sizze_11063 +
                                                               j_12901];
                
                pre_12912 = x_12913;
            } else {
                pre_12912 = 0.0F;
            }
            
            bool binop_x_12915 = slt32(i_12899, sizze_11063);
            bool binop_y_12916 = slt32(gtid_12905, sizze_11068);
            bool cond_12917 = binop_x_12915 && binop_y_12916;
            float pre_12918;
            
            if (cond_12917) {
                float x_12919 = ((__global float *) mem_13534)[i_12899 *
                                                               sizze_11068 +
                                                               gtid_12905];
                
                pre_12918 = x_12919;
            } else {
                pre_12918 = 0.0F;
            }
            ((__local float *) mem_13629)[ltid_x_12813 * tile_sizze_12668 +
                                          ltid_y_12814] = pre_12912;
            ((__local float *) mem_13634)[ltid_x_12813 * tile_sizze_12668 +
                                          ltid_y_12814] = pre_12918;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_12861 = squot32(local_tid_13864, tile_sizze_12668);
        int32_t ltid_y_12862;
        
        ltid_y_12862 = local_tid_13864 - squot32(local_tid_13864,
                                                 tile_sizze_12668) *
            tile_sizze_12668;
        
        int32_t ltid_flat_12863;
        
        ltid_flat_12863 = local_tid_13864;
        if (slt32(ltid_x_12861, tile_sizze_12668) && slt32(ltid_y_12862,
                                                           tile_sizze_12668)) {
            int32_t gtid_12925 = binop_x_12773 + ltid_x_12861;
            int32_t gtid_12927 = binop_x_12775 + ltid_y_12862;
            float acc_12931 = mem_13609;
            bool binop_x_12934 = slt32(gtid_12925, sizze_11064);
            bool binop_y_12935 = slt32(gtid_12927, sizze_11068);
            bool cond_12936 = binop_x_12934 && binop_y_12935;
            float acc_12937;
            
            if (cond_12936) {
                float x_12938;
                float redout_13314 = acc_12931;
                
                for (int32_t i_13315 = 0; i_13315 < residual_input_12811;
                     i_13315++) {
                    float x_12942 = ((__local float *) mem_13629)[ltid_x_12861 *
                                                                  tile_sizze_12668 +
                                                                  i_13315];
                    float x_12943 = ((__local float *) mem_13634)[i_13315 *
                                                                  tile_sizze_12668 +
                                                                  ltid_y_12862];
                    float res_12944 = x_12942 * x_12943;
                    float res_12941 = res_12944 + redout_13314;
                    float redout_tmp_13872 = res_12941;
                    
                    redout_13314 = redout_tmp_13872;
                }
                x_12938 = redout_13314;
                acc_12937 = x_12938;
            } else {
                acc_12937 = acc_12931;
            }
            mem_13638 = acc_12937;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_13873 = 0; i_13873 < squot32(tile_sizze_12668 *
                                                    tile_sizze_12668 -
                                                    local_tid_13864 +
                                                    group_sizze_12669 - 1,
                                                    group_sizze_12669);
             i_13873++) {
            mem_13754 = mem_13638;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    float mem_13643;
    int32_t ltid_x_12946 = squot32(local_tid_13864, tile_sizze_12668);
    int32_t ltid_y_12947;
    
    ltid_y_12947 = local_tid_13864 - squot32(local_tid_13864,
                                             tile_sizze_12668) *
        tile_sizze_12668;
    
    int32_t ltid_flat_12948;
    
    ltid_flat_12948 = local_tid_13864;
    if (slt32(ltid_x_12946, tile_sizze_12668) && slt32(ltid_y_12947,
                                                       tile_sizze_12668)) {
        int32_t gtid_12957 = binop_x_12773 + ltid_x_12946;
        int32_t gtid_12959 = binop_x_12775 + ltid_y_12947;
        bool binop_x_12961 = slt32(gtid_12957, sizze_11064);
        bool binop_y_12962 = slt32(gtid_12959, sizze_11068);
        bool cond_12963 = binop_x_12961 && binop_y_12962;
        float postlude_12964;
        
        if (cond_12963) {
            float res_12960 = mem_13754;
            int32_t binop_x_12968 = sizze_11068 * gtid_12957;
            int32_t binop_x_12969 = gtid_12959 + binop_x_12968;
            int32_t new_index_12970 = squot32(binop_x_12969, sizze_11066);
            int32_t binop_y_12976 = sizze_11066 * new_index_12970;
            int32_t new_index_12977 = binop_x_12969 - binop_y_12976;
            float x_12978 = ((__global float *) mem_13486)[new_index_12970 *
                                                           sizze_11066 +
                                                           new_index_12977];
            float x_12989 = ((__global float *) mem_13601)[new_index_12970 *
                                                           sizze_11066 +
                                                           new_index_12977];
            float res_12990 = res_12960 * x_12978;
            float res_12991 = x_12989 * res_12990;
            
            postlude_12964 = res_12991;
        } else {
            postlude_12964 = 0.0F;
        }
        mem_13643 = postlude_12964;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t thread_out_index_13874 = gid_x_12665 * tile_sizze_12668 +
            squot32(local_tid_13864, tile_sizze_12668);
    int32_t thread_out_index_13875;
    
    thread_out_index_13875 = gid_y_12666 * tile_sizze_12668 + (local_tid_13864 -
                                                               squot32(local_tid_13864,
                                                                       tile_sizze_12668) *
                                                               tile_sizze_12668);
    if (slt32(thread_out_index_13874, sizze_11064) &&
        slt32(thread_out_index_13875, sizze_11068)) {
        ((__global float *) mem_13648)[thread_out_index_13874 * sizze_11068 +
                                       thread_out_index_13875] = mem_13643;
    }
}
__kernel void segmap_intragroup_13006(__local volatile
                                      int64_t *mem_13661_backing_aligned_0,
                                      __local volatile
                                      int64_t *mem_13666_backing_aligned_1,
                                      __local volatile
                                      int64_t *mem_13676_backing_aligned_2,
                                      __local volatile
                                      int64_t *mem_13681_backing_aligned_3,
                                      int32_t sizze_11064, int32_t sizze_11065,
                                      int32_t sizze_11068, float lr_11069,
                                      int32_t num_groups_y_13004,
                                      int32_t num_whole_tiles_13007,
                                      int32_t residual_input_13140,
                                      unsigned char cond_13141, __global
                                      unsigned char *mem_13648, __global
                                      unsigned char *mem_13652, __global
                                      unsigned char *mem_13695)
{
    const int32_t tile_sizze_12997 = mainzitile_sizze_12996;
    const int32_t group_sizze_12998 = mainzitile_sizze_12996 *
                  mainzitile_sizze_12996;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_13661_backing_0 = (__local volatile
                                                           char *) mem_13661_backing_aligned_0;
    __local volatile char *restrict mem_13666_backing_1 = (__local volatile
                                                           char *) mem_13666_backing_aligned_1;
    __local volatile char *restrict mem_13676_backing_2 = (__local volatile
                                                           char *) mem_13676_backing_aligned_2;
    __local volatile char *restrict mem_13681_backing_3 = (__local volatile
                                                           char *) mem_13681_backing_aligned_3;
    int32_t global_tid_13876;
    int32_t local_tid_13877;
    int32_t group_sizze_13880;
    int32_t wave_sizze_13879;
    int32_t group_tid_13878;
    
    global_tid_13876 = get_global_id(0);
    local_tid_13877 = get_local_id(0);
    group_sizze_13880 = get_local_size(0);
    wave_sizze_13879 = LOCKSTEP_WIDTH;
    group_tid_13878 = get_group_id(0);
    
    int32_t gid_flat_13006 = group_tid_13878;
    int32_t gid_x_12994 = squot32(group_tid_13878, num_groups_y_13004);
    int32_t gid_y_12995;
    
    gid_y_12995 = group_tid_13878 - squot32(group_tid_13878,
                                            num_groups_y_13004) *
        num_groups_y_13004;
    
    float mem_13656;
    int32_t ltid_x_13023 = squot32(local_tid_13877, tile_sizze_12997);
    int32_t ltid_y_13024;
    
    ltid_y_13024 = local_tid_13877 - squot32(local_tid_13877,
                                             tile_sizze_12997) *
        tile_sizze_12997;
    
    int32_t ltid_flat_13025;
    
    ltid_flat_13025 = local_tid_13877;
    if (slt32(ltid_x_13023, tile_sizze_12997) && slt32(ltid_y_13024,
                                                       tile_sizze_12997)) {
        mem_13656 = 0.0F;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t binop_x_13102 = gid_x_12994 * tile_sizze_12997;
    int32_t binop_x_13104 = gid_y_12995 * tile_sizze_12997;
    __local char *mem_13661;
    
    mem_13661 = (__local char *) mem_13661_backing_0;
    
    __local char *mem_13666;
    
    mem_13666 = (__local char *) mem_13666_backing_1;
    
    float mem_13670;
    
    for (int32_t tile_id_13034 = 0; tile_id_13034 < num_whole_tiles_13007;
         tile_id_13034++) {
        int32_t binop_x_13098 = tile_sizze_12997 * tile_id_13034;
        int32_t ltid_x_13035 = squot32(local_tid_13877, tile_sizze_12997);
        int32_t ltid_y_13036;
        
        ltid_y_13036 = local_tid_13877 - squot32(local_tid_13877,
                                                 tile_sizze_12997) *
            tile_sizze_12997;
        
        int32_t ltid_flat_13037;
        
        ltid_flat_13037 = local_tid_13877;
        if (slt32(ltid_x_13035, tile_sizze_12997) && slt32(ltid_y_13036,
                                                           tile_sizze_12997)) {
            int32_t i_13099 = ltid_x_13035 + binop_x_13098;
            int32_t j_13101 = ltid_y_13036 + binop_x_13098;
            int32_t gtid_13103 = ltid_x_13035 + binop_x_13102;
            int32_t gtid_13105 = ltid_y_13036 + binop_x_13104;
            float tile_elem_13109 = ((__global float *) mem_13648)[gtid_13103 *
                                                                   sizze_11068 +
                                                                   j_13101];
            float tile_elem_13110 = ((__global float *) mem_13652)[i_13099 *
                                                                   sizze_11065 +
                                                                   gtid_13105];
            
            ((__local float *) mem_13661)[ltid_x_13035 * tile_sizze_12997 +
                                          ltid_y_13036] = tile_elem_13109;
            ((__local float *) mem_13666)[ltid_x_13035 * tile_sizze_12997 +
                                          ltid_y_13036] = tile_elem_13110;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_13061 = squot32(local_tid_13877, tile_sizze_12997);
        int32_t ltid_y_13062;
        
        ltid_y_13062 = local_tid_13877 - squot32(local_tid_13877,
                                                 tile_sizze_12997) *
            tile_sizze_12997;
        
        int32_t ltid_flat_13063;
        
        ltid_flat_13063 = local_tid_13877;
        if (slt32(ltid_x_13061, tile_sizze_12997) && slt32(ltid_y_13062,
                                                           tile_sizze_12997)) {
            int32_t gtid_13113 = ltid_x_13061 + binop_x_13102;
            int32_t gtid_13115 = ltid_y_13062 + binop_x_13104;
            float acc_13119 = mem_13656;
            bool binop_x_13122 = slt32(gtid_13113, sizze_11064);
            bool binop_y_13123 = slt32(gtid_13115, sizze_11065);
            bool cond_13124 = binop_x_13122 && binop_y_13123;
            float acc_13125;
            
            if (cond_13124) {
                float x_13126;
                float redout_13316 = acc_13119;
                
                for (int32_t i_13317 = 0; i_13317 < tile_sizze_12997;
                     i_13317++) {
                    float x_13130 = ((__local float *) mem_13661)[ltid_x_13061 *
                                                                  tile_sizze_12997 +
                                                                  i_13317];
                    float x_13131 = ((__local float *) mem_13666)[i_13317 *
                                                                  tile_sizze_12997 +
                                                                  ltid_y_13062];
                    float res_13132 = x_13130 * x_13131;
                    float res_13129 = res_13132 + redout_13316;
                    float redout_tmp_13882 = res_13129;
                    
                    redout_13316 = redout_tmp_13882;
                }
                x_13126 = redout_13316;
                acc_13125 = x_13126;
            } else {
                acc_13125 = acc_13119;
            }
            mem_13670 = acc_13125;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_13883 = 0; i_13883 < squot32(tile_sizze_12997 *
                                                    tile_sizze_12997 -
                                                    local_tid_13877 +
                                                    group_sizze_12998 - 1,
                                                    group_sizze_12998);
             i_13883++) {
            mem_13656 = mem_13670;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    __local char *mem_13676;
    
    mem_13676 = (__local char *) mem_13676_backing_2;
    
    __local char *mem_13681;
    
    mem_13681 = (__local char *) mem_13681_backing_3;
    
    float mem_13685;
    float mem_13768;
    
    if (cond_13141) {
        for (int32_t i_13884 = 0; i_13884 < squot32(tile_sizze_12997 *
                                                    tile_sizze_12997 -
                                                    local_tid_13877 +
                                                    group_sizze_12998 - 1,
                                                    group_sizze_12998);
             i_13884++) {
            mem_13768 = mem_13656;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
        int32_t binop_x_13227 = tile_sizze_12997 * num_whole_tiles_13007;
        int32_t ltid_x_13142 = squot32(local_tid_13877, tile_sizze_12997);
        int32_t ltid_y_13143;
        
        ltid_y_13143 = local_tid_13877 - squot32(local_tid_13877,
                                                 tile_sizze_12997) *
            tile_sizze_12997;
        
        int32_t ltid_flat_13144;
        
        ltid_flat_13144 = local_tid_13877;
        if (slt32(ltid_x_13142, tile_sizze_12997) && slt32(ltid_y_13143,
                                                           tile_sizze_12997)) {
            int32_t i_13228 = ltid_x_13142 + binop_x_13227;
            int32_t j_13230 = ltid_y_13143 + binop_x_13227;
            int32_t gtid_13232 = binop_x_13102 + ltid_x_13142;
            int32_t gtid_13234 = binop_x_13104 + ltid_y_13143;
            bool binop_x_13238 = slt32(j_13230, sizze_11068);
            bool binop_y_13239 = slt32(gtid_13232, sizze_11064);
            bool cond_13240 = binop_x_13238 && binop_y_13239;
            float pre_13241;
            
            if (cond_13240) {
                float x_13242 = ((__global float *) mem_13648)[gtid_13232 *
                                                               sizze_11068 +
                                                               j_13230];
                
                pre_13241 = x_13242;
            } else {
                pre_13241 = 0.0F;
            }
            
            bool binop_x_13244 = slt32(i_13228, sizze_11068);
            bool binop_y_13245 = slt32(gtid_13234, sizze_11065);
            bool cond_13246 = binop_x_13244 && binop_y_13245;
            float pre_13247;
            
            if (cond_13246) {
                float x_13248 = ((__global float *) mem_13652)[i_13228 *
                                                               sizze_11065 +
                                                               gtid_13234];
                
                pre_13247 = x_13248;
            } else {
                pre_13247 = 0.0F;
            }
            ((__local float *) mem_13676)[ltid_x_13142 * tile_sizze_12997 +
                                          ltid_y_13143] = pre_13241;
            ((__local float *) mem_13681)[ltid_x_13142 * tile_sizze_12997 +
                                          ltid_y_13143] = pre_13247;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_13190 = squot32(local_tid_13877, tile_sizze_12997);
        int32_t ltid_y_13191;
        
        ltid_y_13191 = local_tid_13877 - squot32(local_tid_13877,
                                                 tile_sizze_12997) *
            tile_sizze_12997;
        
        int32_t ltid_flat_13192;
        
        ltid_flat_13192 = local_tid_13877;
        if (slt32(ltid_x_13190, tile_sizze_12997) && slt32(ltid_y_13191,
                                                           tile_sizze_12997)) {
            int32_t gtid_13254 = binop_x_13102 + ltid_x_13190;
            int32_t gtid_13256 = binop_x_13104 + ltid_y_13191;
            float acc_13260 = mem_13656;
            bool binop_x_13263 = slt32(gtid_13254, sizze_11064);
            bool binop_y_13264 = slt32(gtid_13256, sizze_11065);
            bool cond_13265 = binop_x_13263 && binop_y_13264;
            float acc_13266;
            
            if (cond_13265) {
                float x_13267;
                float redout_13318 = acc_13260;
                
                for (int32_t i_13319 = 0; i_13319 < residual_input_13140;
                     i_13319++) {
                    float x_13271 = ((__local float *) mem_13676)[ltid_x_13190 *
                                                                  tile_sizze_12997 +
                                                                  i_13319];
                    float x_13272 = ((__local float *) mem_13681)[i_13319 *
                                                                  tile_sizze_12997 +
                                                                  ltid_y_13191];
                    float res_13273 = x_13271 * x_13272;
                    float res_13270 = res_13273 + redout_13318;
                    float redout_tmp_13885 = res_13270;
                    
                    redout_13318 = redout_tmp_13885;
                }
                x_13267 = redout_13318;
                acc_13266 = x_13267;
            } else {
                acc_13266 = acc_13260;
            }
            mem_13685 = acc_13266;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_13886 = 0; i_13886 < squot32(tile_sizze_12997 *
                                                    tile_sizze_12997 -
                                                    local_tid_13877 +
                                                    group_sizze_12998 - 1,
                                                    group_sizze_12998);
             i_13886++) {
            mem_13768 = mem_13685;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    float mem_13690;
    int32_t ltid_x_13275 = squot32(local_tid_13877, tile_sizze_12997);
    int32_t ltid_y_13276;
    
    ltid_y_13276 = local_tid_13877 - squot32(local_tid_13877,
                                             tile_sizze_12997) *
        tile_sizze_12997;
    
    int32_t ltid_flat_13277;
    
    ltid_flat_13277 = local_tid_13877;
    if (slt32(ltid_x_13275, tile_sizze_12997) && slt32(ltid_y_13276,
                                                       tile_sizze_12997)) {
        int32_t gtid_13286 = binop_x_13102 + ltid_x_13275;
        int32_t gtid_13288 = binop_x_13104 + ltid_y_13276;
        bool binop_x_13290 = slt32(gtid_13286, sizze_11064);
        bool binop_y_13291 = slt32(gtid_13288, sizze_11065);
        bool cond_13292 = binop_x_13290 && binop_y_13291;
        float postlude_13293;
        
        if (cond_13292) {
            float res_13289 = mem_13768;
            float res_13297 = lr_11069 * res_13289;
            
            postlude_13293 = res_13297;
        } else {
            postlude_13293 = 0.0F;
        }
        mem_13690 = postlude_13293;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t thread_out_index_13887 = gid_x_12994 * tile_sizze_12997 +
            squot32(local_tid_13877, tile_sizze_12997);
    int32_t thread_out_index_13888;
    
    thread_out_index_13888 = gid_y_12995 * tile_sizze_12997 + (local_tid_13877 -
                                                               squot32(local_tid_13877,
                                                                       tile_sizze_12997) *
                                                               tile_sizze_12997);
    if (slt32(thread_out_index_13887, sizze_11064) &&
        slt32(thread_out_index_13888, sizze_11065)) {
        ((__global float *) mem_13695)[thread_out_index_13887 * sizze_11065 +
                                       thread_out_index_13888] = mem_13690;
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
  entry_points = {"main": (["f32", "[][]f32", "[][]f32", "[][]f32", "[][]f32"],
                           ["[][]f32", "[][]f32"])}
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
                                       all_sizes={"main.segmap_group_size_11379": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_11397": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_11471": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_11492": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_11510": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_11646": {"class": "group_size", "value": None},
                                        "main.tile_size_11743": {"class": "tile_size", "value": None},
                                        "main.tile_size_12052": {"class": "tile_size", "value": None},
                                        "main.tile_size_12361": {"class": "tile_size", "value": None},
                                        "main.tile_size_12667": {"class": "tile_size", "value": None},
                                        "main.tile_size_12996": {"class": "tile_size", "value": None}})
    self.map_transpose_f32_var = program.map_transpose_f32
    self.map_transpose_f32_low_height_var = program.map_transpose_f32_low_height
    self.map_transpose_f32_low_width_var = program.map_transpose_f32_low_width
    self.map_transpose_f32_small_var = program.map_transpose_f32_small
    self.segmap_11374_var = program.segmap_11374
    self.segmap_11392_var = program.segmap_11392
    self.segmap_11466_var = program.segmap_11466
    self.segmap_11487_var = program.segmap_11487
    self.segmap_11505_var = program.segmap_11505
    self.segmap_11641_var = program.segmap_11641
    self.segmap_intragroup_11753_var = program.segmap_intragroup_11753
    self.segmap_intragroup_12062_var = program.segmap_intragroup_12062
    self.segmap_intragroup_12371_var = program.segmap_intragroup_12371
    self.segmap_intragroup_12677_var = program.segmap_intragroup_12677
    self.segmap_intragroup_13006_var = program.segmap_intragroup_13006
  def futhark_main(self, wih_mem_13440, who_mem_13441, inputs_mem_13442,
                   targets_mem_13443, sizze_11061, sizze_11062, sizze_11063,
                   sizze_11064, sizze_11065, sizze_11066, sizze_11067,
                   sizze_11068, lr_11069):
    dim_zzero_11075 = (np.int32(0) == sizze_11065)
    dim_zzero_11076 = (np.int32(0) == sizze_11062)
    both_empty_11077 = (dim_zzero_11075 and dim_zzero_11076)
    dim_match_11078 = (sizze_11062 == sizze_11065)
    empty_or_match_11079 = (both_empty_11077 or dim_match_11078)
    empty_or_match_cert_11080 = True
    assert empty_or_match_11079, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:3:19-30\n   #1  GPUTraining.fut:8:21-38\n   #2  GPUTraining.fut:8:3-9:17\n   #3  GPUTraining.fut:17:19-34\n   #4  GPUTraining.fut:17:3-37\n   #5  GPUTraining.fut:159:25-38\n   #6  GPUTraining.fut:157:1-178:33\n" % ("function arguments of wrong shape",))
    sizze_11286 = sext_i32_i64(sizze_11061)
    sizze_11287 = sext_i32_i64(sizze_11066)
    nest_sizze_11289 = (sizze_11286 * sizze_11287)
    tile_sizze_11744 = self.sizes["main.tile_size_11743"]
    group_sizze_11745 = (tile_sizze_11744 * tile_sizze_11744)
    y_11746 = (tile_sizze_11744 - np.int32(1))
    x_11747 = (sizze_11061 + y_11746)
    num_groups_x_11748 = squot32(x_11747, tile_sizze_11744)
    x_11750 = (sizze_11066 + y_11746)
    num_groups_y_11751 = squot32(x_11750, tile_sizze_11744)
    num_groups_top_11752 = (num_groups_x_11748 * num_groups_y_11751)
    num_whole_tiles_11754 = squot32(sizze_11062, tile_sizze_11744)
    residual_input_11887 = srem32(sizze_11062, tile_sizze_11744)
    cond_11888 = (residual_input_11887 == np.int32(0))
    bytes_13482 = (np.int64(4) * nest_sizze_11289)
    mem_13486 = opencl_alloc(self, bytes_13482, "mem_13486")
    binop_x_13446 = sext_i32_i64(group_sizze_11745)
    bytes_13444 = (np.int64(4) * binop_x_13446)
    binop_x_13449 = sext_i32_i64(tile_sizze_11744)
    binop_x_13451 = (binop_x_13449 * binop_x_13449)
    bytes_13448 = (np.int64(4) * binop_x_13451)
    if ((1 * (np.long(num_groups_top_11752) * np.long(group_sizze_11745))) != 0):
      self.segmap_intragroup_11753_var.set_args(cl.LocalMemory(np.long(bytes_13448)),
                                                cl.LocalMemory(np.long(bytes_13448)),
                                                cl.LocalMemory(np.long(bytes_13448)),
                                                cl.LocalMemory(np.long(bytes_13448)),
                                                np.int32(sizze_11061),
                                                np.int32(sizze_11062),
                                                np.int32(sizze_11066),
                                                np.int32(num_groups_y_11751),
                                                np.int32(num_whole_tiles_11754),
                                                np.int32(residual_input_11887),
                                                np.byte(cond_11888),
                                                wih_mem_13440, inputs_mem_13442,
                                                mem_13486)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_intragroup_11753_var,
                                 ((np.long(num_groups_top_11752) * np.long(group_sizze_11745)),),
                                 (np.long(group_sizze_11745),))
      if synchronous:
        self.queue.finish()
    dim_zzero_11098 = (np.int32(0) == sizze_11061)
    dim_zzero_11099 = (np.int32(0) == sizze_11064)
    both_empty_11100 = (dim_zzero_11098 and dim_zzero_11099)
    dim_match_11101 = (sizze_11064 == sizze_11061)
    empty_or_match_11102 = (both_empty_11100 or dim_match_11101)
    empty_or_match_cert_11103 = True
    assert empty_or_match_11102, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:3:19-30\n   #1  GPUTraining.fut:8:21-38\n   #2  GPUTraining.fut:8:3-9:17\n   #3  GPUTraining.fut:17:19-34\n   #4  GPUTraining.fut:17:3-37\n   #5  GPUTraining.fut:163:24-45\n   #6  GPUTraining.fut:157:1-178:33\n" % ("function arguments of wrong shape",))
    sizze_11345 = sext_i32_i64(sizze_11063)
    nest_sizze_11348 = (sizze_11287 * sizze_11345)
    tile_sizze_12053 = self.sizes["main.tile_size_12052"]
    group_sizze_12054 = (tile_sizze_12053 * tile_sizze_12053)
    y_12055 = (tile_sizze_12053 - np.int32(1))
    x_12056 = (sizze_11063 + y_12055)
    num_groups_x_12057 = squot32(x_12056, tile_sizze_12053)
    x_12059 = (sizze_11066 + y_12055)
    num_groups_y_12060 = squot32(x_12059, tile_sizze_12053)
    num_groups_top_12061 = (num_groups_x_12057 * num_groups_y_12060)
    num_whole_tiles_12063 = squot32(sizze_11064, tile_sizze_12053)
    residual_input_12196 = srem32(sizze_11064, tile_sizze_12053)
    cond_12197 = (residual_input_12196 == np.int32(0))
    binop_x_13528 = (sizze_11287 * sizze_11345)
    bytes_13525 = (np.int64(4) * binop_x_13528)
    mem_13529 = opencl_alloc(self, bytes_13525, "mem_13529")
    binop_x_13489 = sext_i32_i64(group_sizze_12054)
    bytes_13487 = (np.int64(4) * binop_x_13489)
    binop_x_13492 = sext_i32_i64(tile_sizze_12053)
    binop_x_13494 = (binop_x_13492 * binop_x_13492)
    bytes_13491 = (np.int64(4) * binop_x_13494)
    if ((1 * (np.long(num_groups_top_12061) * np.long(group_sizze_12054))) != 0):
      self.segmap_intragroup_12062_var.set_args(cl.LocalMemory(np.long(bytes_13491)),
                                                cl.LocalMemory(np.long(bytes_13491)),
                                                cl.LocalMemory(np.long(bytes_13491)),
                                                cl.LocalMemory(np.long(bytes_13491)),
                                                np.int32(sizze_11063),
                                                np.int32(sizze_11064),
                                                np.int32(sizze_11066),
                                                np.int32(num_groups_y_12060),
                                                np.int32(num_whole_tiles_12063),
                                                np.int32(residual_input_12196),
                                                np.byte(cond_12197),
                                                who_mem_13441, mem_13486,
                                                mem_13529)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_intragroup_12062_var,
                                 ((np.long(num_groups_top_12061) * np.long(group_sizze_12054)),),
                                 (np.long(group_sizze_12054),))
      if synchronous:
        self.queue.finish()
    dim_zzero_11120 = (np.int32(0) == sizze_11063)
    dim_zzero_11121 = (np.int32(0) == sizze_11066)
    old_empty_11122 = (dim_zzero_11120 or dim_zzero_11121)
    dim_zzero_11123 = (np.int32(0) == sizze_11067)
    new_empty_11124 = (dim_zzero_11121 or dim_zzero_11123)
    both_empty_11125 = (old_empty_11122 and new_empty_11124)
    dim_match_11126 = (sizze_11067 == sizze_11063)
    empty_or_match_11127 = (both_empty_11125 or dim_match_11126)
    empty_or_match_cert_11128 = True
    assert empty_or_match_11127, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:130:3-21\n   #1  GPUTraining.fut:167:25-58\n   #2  GPUTraining.fut:157:1-178:33\n" % ("function arguments of wrong shape",))
    dim_zzero_11130 = (np.int32(0) == sizze_11068)
    both_empty_11131 = (dim_zzero_11121 and dim_zzero_11130)
    dim_match_11132 = (sizze_11068 == sizze_11066)
    empty_or_match_11133 = (both_empty_11131 or dim_match_11132)
    empty_or_match_cert_11134 = True
    assert empty_or_match_11133, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:122:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPUTraining.fut:130:3-21\n   #4  GPUTraining.fut:167:25-58\n   #5  GPUTraining.fut:157:1-178:33\n" % ("function arguments of wrong shape",))
    sizze_11375 = sext_i32_i64(sizze_11067)
    sizze_11376 = sext_i32_i64(sizze_11068)
    nest_sizze_11378 = (sizze_11375 * sizze_11376)
    segmap_group_sizze_11380 = self.sizes["main.segmap_group_size_11379"]
    segmap_group_sizze_11381 = sext_i32_i64(segmap_group_sizze_11380)
    y_11382 = (segmap_group_sizze_11381 - np.int64(1))
    x_11383 = (nest_sizze_11378 + y_11382)
    segmap_usable_groups_64_11385 = squot64(x_11383, segmap_group_sizze_11381)
    segmap_usable_groups_11386 = sext_i64_i32(segmap_usable_groups_64_11385)
    bytes_13530 = (np.int64(4) * nest_sizze_11378)
    mem_13534 = opencl_alloc(self, bytes_13530, "mem_13534")
    if ((1 * (np.long(segmap_usable_groups_11386) * np.long(segmap_group_sizze_11380))) != 0):
      self.segmap_11374_var.set_args(np.int32(sizze_11066),
                                     np.int32(sizze_11067),
                                     np.int32(sizze_11068), targets_mem_13443,
                                     mem_13529, mem_13534)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_11374_var,
                                 ((np.long(segmap_usable_groups_11386) * np.long(segmap_group_sizze_11380)),),
                                 (np.long(segmap_group_sizze_11380),))
      if synchronous:
        self.queue.finish()
    both_empty_11144 = (dim_zzero_11120 and dim_zzero_11123)
    dim_match_11145 = (sizze_11063 == sizze_11067)
    empty_or_match_11146 = (both_empty_11144 or dim_match_11145)
    empty_or_match_cert_11147 = True
    assert empty_or_match_11146, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:3:19-30\n   #1  GPUTraining.fut:8:21-38\n   #2  GPUTraining.fut:8:3-9:17\n   #3  GPUTraining.fut:17:19-34\n   #4  GPUTraining.fut:17:3-37\n   #5  GPUTraining.fut:169:25-57\n   #6  GPUTraining.fut:157:1-178:33\n" % ("function arguments of wrong shape",))
    segmap_group_sizze_11398 = self.sizes["main.segmap_group_size_11397"]
    segmap_group_sizze_11399 = sext_i32_i64(segmap_group_sizze_11398)
    y_11400 = (segmap_group_sizze_11399 - np.int64(1))
    x_11401 = (nest_sizze_11348 + y_11400)
    segmap_usable_groups_64_11403 = squot64(x_11401, segmap_group_sizze_11399)
    segmap_usable_groups_11404 = sext_i64_i32(segmap_usable_groups_64_11403)
    mem_13539 = opencl_alloc(self, bytes_13525, "mem_13539")
    if ((1 * (np.long(segmap_usable_groups_11404) * np.long(segmap_group_sizze_11398))) != 0):
      self.segmap_11392_var.set_args(np.int32(sizze_11063),
                                     np.int32(sizze_11066), mem_13529,
                                     mem_13539)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_11392_var,
                                 ((np.long(segmap_usable_groups_11404) * np.long(segmap_group_sizze_11398)),),
                                 (np.long(segmap_group_sizze_11398),))
      if synchronous:
        self.queue.finish()
    empty_or_match_cert_11153 = True
    assert empty_or_match_11133, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:3:19-30\n   #1  GPUTraining.fut:8:21-38\n   #2  GPUTraining.fut:8:3-9:17\n   #3  GPUTraining.fut:17:19-34\n   #4  GPUTraining.fut:17:3-37\n   #5  GPUTraining.fut:173:52-87\n   #6  GPUTraining.fut:157:1-178:33\n" % ("function arguments of wrong shape",))
    segmap_group_sizze_11472 = self.sizes["main.segmap_group_size_11471"]
    segmap_group_sizze_11473 = sext_i32_i64(segmap_group_sizze_11472)
    y_11474 = (segmap_group_sizze_11473 - np.int64(1))
    x_11475 = (nest_sizze_11378 + y_11474)
    segmap_usable_groups_64_11477 = squot64(x_11475, segmap_group_sizze_11473)
    segmap_usable_groups_11478 = sext_i64_i32(segmap_usable_groups_64_11477)
    mem_13544 = opencl_alloc(self, bytes_13530, "mem_13544")
    if ((1 * (np.long(segmap_usable_groups_11478) * np.long(segmap_group_sizze_11472))) != 0):
      self.segmap_11466_var.set_args(np.int32(sizze_11066),
                                     np.int32(sizze_11067),
                                     np.int32(sizze_11068), mem_13529,
                                     mem_13534, mem_13539, mem_13544)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_11466_var,
                                 ((np.long(segmap_usable_groups_11478) * np.long(segmap_group_sizze_11472)),),
                                 (np.long(segmap_group_sizze_11472),))
      if synchronous:
        self.queue.finish()
    mem_13529 = None
    mem_13539 = None
    convop_x_13546 = (sizze_11061 * sizze_11066)
    binop_x_13547 = sext_i32_i64(convop_x_13546)
    bytes_13545 = (np.int64(4) * binop_x_13547)
    mem_13548 = opencl_alloc(self, bytes_13545, "mem_13548")
    self.futhark__map_transpose_f32(mem_13548, np.int32(0), mem_13486,
                                    np.int32(0), np.int32(1), sizze_11066,
                                    sizze_11061, (sizze_11061 * sizze_11066),
                                    (sizze_11061 * sizze_11066))
    tile_sizze_12362 = self.sizes["main.tile_size_12361"]
    group_sizze_12363 = (tile_sizze_12362 * tile_sizze_12362)
    y_12364 = (tile_sizze_12362 - np.int32(1))
    x_12365 = (sizze_11067 + y_12364)
    num_groups_x_12366 = squot32(x_12365, tile_sizze_12362)
    x_12368 = (sizze_11061 + y_12364)
    num_groups_y_12369 = squot32(x_12368, tile_sizze_12362)
    num_groups_top_12370 = (num_groups_x_12366 * num_groups_y_12369)
    num_whole_tiles_12372 = squot32(sizze_11068, tile_sizze_12362)
    residual_input_12505 = srem32(sizze_11068, tile_sizze_12362)
    cond_12506 = (residual_input_12505 == np.int32(0))
    binop_x_13590 = (sizze_11286 * sizze_11375)
    bytes_13587 = (np.int64(4) * binop_x_13590)
    mem_13591 = opencl_alloc(self, bytes_13587, "mem_13591")
    binop_x_13551 = sext_i32_i64(group_sizze_12363)
    bytes_13549 = (np.int64(4) * binop_x_13551)
    binop_x_13554 = sext_i32_i64(tile_sizze_12362)
    binop_x_13556 = (binop_x_13554 * binop_x_13554)
    bytes_13553 = (np.int64(4) * binop_x_13556)
    if ((1 * (np.long(num_groups_top_12370) * np.long(group_sizze_12363))) != 0):
      self.segmap_intragroup_12371_var.set_args(cl.LocalMemory(np.long(bytes_13553)),
                                                cl.LocalMemory(np.long(bytes_13553)),
                                                cl.LocalMemory(np.long(bytes_13553)),
                                                cl.LocalMemory(np.long(bytes_13553)),
                                                np.int32(sizze_11061),
                                                np.int32(sizze_11067),
                                                np.int32(sizze_11068),
                                                np.float32(lr_11069),
                                                np.int32(num_groups_y_12369),
                                                np.int32(num_whole_tiles_12372),
                                                np.int32(residual_input_12505),
                                                np.byte(cond_12506), mem_13544,
                                                mem_13548, mem_13591)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_intragroup_12371_var,
                                 ((np.long(num_groups_top_12370) * np.long(group_sizze_12363)),),
                                 (np.long(group_sizze_12363),))
      if synchronous:
        self.queue.finish()
    mem_13544 = None
    mem_13548 = None
    old_empty_11178 = (dim_zzero_11098 or dim_zzero_11123)
    new_empty_11179 = (dim_zzero_11098 or dim_zzero_11120)
    both_empty_11180 = (old_empty_11178 and new_empty_11179)
    empty_or_match_11181 = (dim_match_11145 or both_empty_11180)
    empty_or_match_cert_11182 = True
    assert empty_or_match_11181, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:98:3-15\n   #1  GPUTraining.fut:173:23-89\n   #2  GPUTraining.fut:157:1-178:33\n" % ("function arguments of wrong shape",))
    empty_or_match_cert_11184 = True
    assert empty_or_match_11102, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:90:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPUTraining.fut:98:3-15\n   #4  GPUTraining.fut:173:23-89\n   #5  GPUTraining.fut:157:1-178:33\n" % ("function arguments of wrong shape",))
    sizze_11489 = sext_i32_i64(sizze_11064)
    nest_sizze_11491 = (sizze_11345 * sizze_11489)
    segmap_group_sizze_11493 = self.sizes["main.segmap_group_size_11492"]
    segmap_group_sizze_11494 = sext_i32_i64(segmap_group_sizze_11493)
    y_11495 = (segmap_group_sizze_11494 - np.int64(1))
    x_11496 = (nest_sizze_11491 + y_11495)
    segmap_usable_groups_64_11498 = squot64(x_11496, segmap_group_sizze_11494)
    segmap_usable_groups_11499 = sext_i64_i32(segmap_usable_groups_64_11498)
    bytes_13592 = (np.int64(4) * nest_sizze_11491)
    mem_13596 = opencl_alloc(self, bytes_13592, "mem_13596")
    if ((1 * (np.long(segmap_usable_groups_11499) * np.long(segmap_group_sizze_11493))) != 0):
      self.segmap_11487_var.set_args(np.int32(sizze_11061),
                                     np.int32(sizze_11063),
                                     np.int32(sizze_11064), who_mem_13441,
                                     mem_13591, mem_13596)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_11487_var,
                                 ((np.long(segmap_usable_groups_11499) * np.long(segmap_group_sizze_11493)),),
                                 (np.long(segmap_group_sizze_11493),))
      if synchronous:
        self.queue.finish()
    mem_13591 = None
    old_empty_11193 = (dim_zzero_11098 or dim_zzero_11121)
    new_empty_11194 = (dim_zzero_11099 or dim_zzero_11121)
    both_empty_11195 = (old_empty_11193 and new_empty_11194)
    empty_or_match_11196 = (dim_match_11101 or both_empty_11195)
    empty_or_match_cert_11197 = True
    assert empty_or_match_11196, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:66:3-20\n   #1  GPUTraining.fut:175:30-69\n   #2  GPUTraining.fut:157:1-178:33\n" % ("function arguments of wrong shape",))
    segmap_group_sizze_11511 = self.sizes["main.segmap_group_size_11510"]
    segmap_group_sizze_11512 = sext_i32_i64(segmap_group_sizze_11511)
    y_11513 = (segmap_group_sizze_11512 - np.int64(1))
    x_11514 = (nest_sizze_11289 + y_11513)
    segmap_usable_groups_64_11516 = squot64(x_11514, segmap_group_sizze_11512)
    segmap_usable_groups_11517 = sext_i64_i32(segmap_usable_groups_64_11516)
    mem_13601 = opencl_alloc(self, bytes_13482, "mem_13601")
    if ((1 * (np.long(segmap_usable_groups_11517) * np.long(segmap_group_sizze_11511))) != 0):
      self.segmap_11505_var.set_args(np.int32(sizze_11061),
                                     np.int32(sizze_11066), mem_13486,
                                     mem_13601)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_11505_var,
                                 ((np.long(segmap_usable_groups_11517) * np.long(segmap_group_sizze_11511)),),
                                 (np.long(segmap_group_sizze_11511),))
      if synchronous:
        self.queue.finish()
    convop_x_13603 = (sizze_11063 * sizze_11064)
    binop_x_13604 = sext_i32_i64(convop_x_13603)
    bytes_13602 = (np.int64(4) * binop_x_13604)
    mem_13605 = opencl_alloc(self, bytes_13602, "mem_13605")
    self.futhark__map_transpose_f32(mem_13605, np.int32(0), who_mem_13441,
                                    np.int32(0), np.int32(1), sizze_11064,
                                    sizze_11063, (sizze_11064 * sizze_11063),
                                    (sizze_11064 * sizze_11063))
    tile_sizze_12668 = self.sizes["main.tile_size_12667"]
    group_sizze_12669 = (tile_sizze_12668 * tile_sizze_12668)
    y_12670 = (tile_sizze_12668 - np.int32(1))
    x_12671 = (sizze_11064 + y_12670)
    num_groups_x_12672 = squot32(x_12671, tile_sizze_12668)
    x_12674 = (sizze_11068 + y_12670)
    num_groups_y_12675 = squot32(x_12674, tile_sizze_12668)
    num_groups_top_12676 = (num_groups_x_12672 * num_groups_y_12675)
    num_whole_tiles_12678 = squot32(sizze_11063, tile_sizze_12668)
    residual_input_12811 = srem32(sizze_11063, tile_sizze_12668)
    cond_12812 = (residual_input_12811 == np.int32(0))
    binop_x_13647 = (sizze_11376 * sizze_11489)
    bytes_13644 = (np.int64(4) * binop_x_13647)
    mem_13648 = opencl_alloc(self, bytes_13644, "mem_13648")
    binop_x_13608 = sext_i32_i64(group_sizze_12669)
    bytes_13606 = (np.int64(4) * binop_x_13608)
    binop_x_13611 = sext_i32_i64(tile_sizze_12668)
    binop_x_13613 = (binop_x_13611 * binop_x_13611)
    bytes_13610 = (np.int64(4) * binop_x_13613)
    if ((1 * (np.long(num_groups_top_12676) * np.long(group_sizze_12669))) != 0):
      self.segmap_intragroup_12677_var.set_args(cl.LocalMemory(np.long(bytes_13610)),
                                                cl.LocalMemory(np.long(bytes_13610)),
                                                cl.LocalMemory(np.long(bytes_13610)),
                                                cl.LocalMemory(np.long(bytes_13610)),
                                                np.int32(sizze_11063),
                                                np.int32(sizze_11064),
                                                np.int32(sizze_11066),
                                                np.int32(sizze_11068),
                                                np.int32(num_groups_y_12675),
                                                np.int32(num_whole_tiles_12678),
                                                np.int32(residual_input_12811),
                                                np.byte(cond_12812), mem_13486,
                                                mem_13534, mem_13601, mem_13605,
                                                mem_13648)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_intragroup_12677_var,
                                 ((np.long(num_groups_top_12676) * np.long(group_sizze_12669)),),
                                 (np.long(group_sizze_12669),))
      if synchronous:
        self.queue.finish()
    mem_13486 = None
    mem_13534 = None
    mem_13601 = None
    mem_13605 = None
    convop_x_13650 = (sizze_11065 * sizze_11066)
    binop_x_13651 = sext_i32_i64(convop_x_13650)
    bytes_13649 = (np.int64(4) * binop_x_13651)
    mem_13652 = opencl_alloc(self, bytes_13649, "mem_13652")
    self.futhark__map_transpose_f32(mem_13652, np.int32(0), inputs_mem_13442,
                                    np.int32(0), np.int32(1), sizze_11066,
                                    sizze_11065, (sizze_11065 * sizze_11066),
                                    (sizze_11065 * sizze_11066))
    tile_sizze_12997 = self.sizes["main.tile_size_12996"]
    group_sizze_12998 = (tile_sizze_12997 * tile_sizze_12997)
    y_12999 = (tile_sizze_12997 - np.int32(1))
    x_13000 = (sizze_11064 + y_12999)
    num_groups_x_13001 = squot32(x_13000, tile_sizze_12997)
    x_13003 = (sizze_11065 + y_12999)
    num_groups_y_13004 = squot32(x_13003, tile_sizze_12997)
    num_groups_top_13005 = (num_groups_x_13001 * num_groups_y_13004)
    num_whole_tiles_13007 = squot32(sizze_11068, tile_sizze_12997)
    residual_input_13140 = srem32(sizze_11068, tile_sizze_12997)
    cond_13141 = (residual_input_13140 == np.int32(0))
    binop_y_13693 = sext_i32_i64(sizze_11065)
    binop_x_13694 = (sizze_11489 * binop_y_13693)
    bytes_13691 = (np.int64(4) * binop_x_13694)
    mem_13695 = opencl_alloc(self, bytes_13691, "mem_13695")
    binop_x_13655 = sext_i32_i64(group_sizze_12998)
    bytes_13653 = (np.int64(4) * binop_x_13655)
    binop_x_13658 = sext_i32_i64(tile_sizze_12997)
    binop_x_13660 = (binop_x_13658 * binop_x_13658)
    bytes_13657 = (np.int64(4) * binop_x_13660)
    if ((1 * (np.long(num_groups_top_13005) * np.long(group_sizze_12998))) != 0):
      self.segmap_intragroup_13006_var.set_args(cl.LocalMemory(np.long(bytes_13657)),
                                                cl.LocalMemory(np.long(bytes_13657)),
                                                cl.LocalMemory(np.long(bytes_13657)),
                                                cl.LocalMemory(np.long(bytes_13657)),
                                                np.int32(sizze_11064),
                                                np.int32(sizze_11065),
                                                np.int32(sizze_11068),
                                                np.float32(lr_11069),
                                                np.int32(num_groups_y_13004),
                                                np.int32(num_whole_tiles_13007),
                                                np.int32(residual_input_13140),
                                                np.byte(cond_13141), mem_13648,
                                                mem_13652, mem_13695)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_intragroup_13006_var,
                                 ((np.long(num_groups_top_13005) * np.long(group_sizze_12998)),),
                                 (np.long(group_sizze_12998),))
      if synchronous:
        self.queue.finish()
    mem_13648 = None
    mem_13652 = None
    old_empty_11237 = (dim_zzero_11075 or dim_zzero_11099)
    new_empty_11238 = (dim_zzero_11075 or dim_zzero_11098)
    both_empty_11239 = (old_empty_11237 and new_empty_11238)
    dim_match_11240 = (sizze_11061 == sizze_11064)
    empty_or_match_11241 = (both_empty_11239 or dim_match_11240)
    empty_or_match_cert_11242 = True
    assert empty_or_match_11241, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:98:3-15\n   #1  GPUTraining.fut:177:23-81\n   #2  GPUTraining.fut:157:1-178:33\n" % ("function arguments of wrong shape",))
    empty_or_match_cert_11244 = True
    assert empty_or_match_11079, ("Error: %s\n\nBacktrace:\n-> #0  GPUTraining.fut:90:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPUTraining.fut:98:3-15\n   #4  GPUTraining.fut:177:23-81\n   #5  GPUTraining.fut:157:1-178:33\n" % ("function arguments of wrong shape",))
    sizze_11643 = sext_i32_i64(sizze_11062)
    nest_sizze_11645 = (sizze_11286 * sizze_11643)
    segmap_group_sizze_11647 = self.sizes["main.segmap_group_size_11646"]
    segmap_group_sizze_11648 = sext_i32_i64(segmap_group_sizze_11647)
    y_11649 = (segmap_group_sizze_11648 - np.int64(1))
    x_11650 = (nest_sizze_11645 + y_11649)
    segmap_usable_groups_64_11652 = squot64(x_11650, segmap_group_sizze_11648)
    segmap_usable_groups_11653 = sext_i64_i32(segmap_usable_groups_64_11652)
    bytes_13696 = (np.int64(4) * nest_sizze_11645)
    mem_13700 = opencl_alloc(self, bytes_13696, "mem_13700")
    if ((1 * (np.long(segmap_usable_groups_11653) * np.long(segmap_group_sizze_11647))) != 0):
      self.segmap_11641_var.set_args(np.int32(sizze_11061),
                                     np.int32(sizze_11062),
                                     np.int32(sizze_11065), wih_mem_13440,
                                     mem_13695, mem_13700)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_11641_var,
                                 ((np.long(segmap_usable_groups_11653) * np.long(segmap_group_sizze_11647)),),
                                 (np.long(segmap_group_sizze_11647),))
      if synchronous:
        self.queue.finish()
    mem_13695 = None
    out_arrsizze_13794 = sizze_11063
    out_arrsizze_13795 = sizze_11064
    out_arrsizze_13797 = sizze_11061
    out_arrsizze_13798 = sizze_11062
    out_mem_13793 = mem_13596
    out_mem_13796 = mem_13700
    return (out_mem_13793, out_arrsizze_13794, out_arrsizze_13795,
            out_mem_13796, out_arrsizze_13797, out_arrsizze_13798)
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
  def main(self, lr_11069_ext, wih_mem_13440_ext, who_mem_13441_ext,
           inputs_mem_13442_ext, targets_mem_13443_ext):
    try:
      lr_11069 = np.float32(ct.c_float(lr_11069_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(lr_11069_ext),
                                                                                                                            lr_11069_ext))
    try:
      assert ((type(wih_mem_13440_ext) in [np.ndarray,
                                           cl.array.Array]) and (wih_mem_13440_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_11061 = np.int32(wih_mem_13440_ext.shape[0])
      sizze_11062 = np.int32(wih_mem_13440_ext.shape[1])
      if (type(wih_mem_13440_ext) == cl.array.Array):
        wih_mem_13440 = wih_mem_13440_ext.data
      else:
        wih_mem_13440 = opencl_alloc(self, np.int64(wih_mem_13440_ext.nbytes),
                                     "wih_mem_13440")
        if (np.int64(wih_mem_13440_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, wih_mem_13440,
                          normaliseArray(wih_mem_13440_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(wih_mem_13440_ext),
                                                                                                                            wih_mem_13440_ext))
    try:
      assert ((type(who_mem_13441_ext) in [np.ndarray,
                                           cl.array.Array]) and (who_mem_13441_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_11063 = np.int32(who_mem_13441_ext.shape[0])
      sizze_11064 = np.int32(who_mem_13441_ext.shape[1])
      if (type(who_mem_13441_ext) == cl.array.Array):
        who_mem_13441 = who_mem_13441_ext.data
      else:
        who_mem_13441 = opencl_alloc(self, np.int64(who_mem_13441_ext.nbytes),
                                     "who_mem_13441")
        if (np.int64(who_mem_13441_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, who_mem_13441,
                          normaliseArray(who_mem_13441_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(who_mem_13441_ext),
                                                                                                                            who_mem_13441_ext))
    try:
      assert ((type(inputs_mem_13442_ext) in [np.ndarray,
                                              cl.array.Array]) and (inputs_mem_13442_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_11065 = np.int32(inputs_mem_13442_ext.shape[0])
      sizze_11066 = np.int32(inputs_mem_13442_ext.shape[1])
      if (type(inputs_mem_13442_ext) == cl.array.Array):
        inputs_mem_13442 = inputs_mem_13442_ext.data
      else:
        inputs_mem_13442 = opencl_alloc(self,
                                        np.int64(inputs_mem_13442_ext.nbytes),
                                        "inputs_mem_13442")
        if (np.int64(inputs_mem_13442_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, inputs_mem_13442,
                          normaliseArray(inputs_mem_13442_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(inputs_mem_13442_ext),
                                                                                                                            inputs_mem_13442_ext))
    try:
      assert ((type(targets_mem_13443_ext) in [np.ndarray,
                                               cl.array.Array]) and (targets_mem_13443_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_11067 = np.int32(targets_mem_13443_ext.shape[0])
      sizze_11068 = np.int32(targets_mem_13443_ext.shape[1])
      if (type(targets_mem_13443_ext) == cl.array.Array):
        targets_mem_13443 = targets_mem_13443_ext.data
      else:
        targets_mem_13443 = opencl_alloc(self,
                                         np.int64(targets_mem_13443_ext.nbytes),
                                         "targets_mem_13443")
        if (np.int64(targets_mem_13443_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, targets_mem_13443,
                          normaliseArray(targets_mem_13443_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(targets_mem_13443_ext),
                                                                                                                            targets_mem_13443_ext))
    (out_mem_13793, out_arrsizze_13794, out_arrsizze_13795, out_mem_13796,
     out_arrsizze_13797, out_arrsizze_13798) = self.futhark_main(wih_mem_13440,
                                                                 who_mem_13441,
                                                                 inputs_mem_13442,
                                                                 targets_mem_13443,
                                                                 sizze_11061,
                                                                 sizze_11062,
                                                                 sizze_11063,
                                                                 sizze_11064,
                                                                 sizze_11065,
                                                                 sizze_11066,
                                                                 sizze_11067,
                                                                 sizze_11068,
                                                                 lr_11069)
    return (cl.array.Array(self.queue, (out_arrsizze_13794, out_arrsizze_13795),
                           ct.c_float, data=out_mem_13793),
            cl.array.Array(self.queue, (out_arrsizze_13797, out_arrsizze_13798),
                           ct.c_float, data=out_mem_13796))