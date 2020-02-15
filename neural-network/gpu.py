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
__kernel void segmap_10003(int32_t sizze_9751, __global
                           unsigned char *x_mem_10662, __global
                           unsigned char *mem_10665)
{
    const int32_t segmap_group_sizze_10007 = expzisegmap_group_sizze_10006;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10761;
    int32_t local_tid_10762;
    int32_t group_sizze_10765;
    int32_t wave_sizze_10764;
    int32_t group_tid_10763;
    
    global_tid_10761 = get_global_id(0);
    local_tid_10762 = get_local_id(0);
    group_sizze_10765 = get_local_size(0);
    wave_sizze_10764 = LOCKSTEP_WIDTH;
    group_tid_10763 = get_group_id(0);
    
    int32_t phys_tid_10003 = global_tid_10761;
    int32_t gtid_10002 = group_tid_10763 * segmap_group_sizze_10007 +
            local_tid_10762;
    
    if (slt32(gtid_10002, sizze_9751)) {
        float x_10014 = ((__global float *) x_mem_10662)[gtid_10002];
        float res_10015 = fpow32(2.7182817F, x_10014);
        
        ((__global float *) mem_10665)[gtid_10002] = res_10015;
    }
}
__kernel void segmap_10017(int32_t sizze_9756, __global
                           unsigned char *x_mem_10662, __global
                           unsigned char *mem_10665)
{
    const int32_t segmap_group_sizze_10021 = negationzisegmap_group_sizze_10020;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10768;
    int32_t local_tid_10769;
    int32_t group_sizze_10772;
    int32_t wave_sizze_10771;
    int32_t group_tid_10770;
    
    global_tid_10768 = get_global_id(0);
    local_tid_10769 = get_local_id(0);
    group_sizze_10772 = get_local_size(0);
    wave_sizze_10771 = LOCKSTEP_WIDTH;
    group_tid_10770 = get_group_id(0);
    
    int32_t phys_tid_10017 = global_tid_10768;
    int32_t gtid_10016 = group_tid_10770 * segmap_group_sizze_10021 +
            local_tid_10769;
    
    if (slt32(gtid_10016, sizze_9756)) {
        float x_10028 = ((__global float *) x_mem_10662)[gtid_10016];
        float res_10029 = 0.0F - x_10028;
        
        ((__global float *) mem_10665)[gtid_10016] = res_10029;
    }
}
__kernel void segmap_10031(int32_t sizze_9761, float d_9762, __global
                           unsigned char *x_mem_10662, __global
                           unsigned char *mem_10665)
{
    const int32_t segmap_group_sizze_10035 = dividezisegmap_group_sizze_10034;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10775;
    int32_t local_tid_10776;
    int32_t group_sizze_10779;
    int32_t wave_sizze_10778;
    int32_t group_tid_10777;
    
    global_tid_10775 = get_global_id(0);
    local_tid_10776 = get_local_id(0);
    group_sizze_10779 = get_local_size(0);
    wave_sizze_10778 = LOCKSTEP_WIDTH;
    group_tid_10777 = get_group_id(0);
    
    int32_t phys_tid_10031 = global_tid_10775;
    int32_t gtid_10030 = group_tid_10777 * segmap_group_sizze_10035 +
            local_tid_10776;
    
    if (slt32(gtid_10030, sizze_9761)) {
        float x_10042 = ((__global float *) x_mem_10662)[gtid_10030];
        float res_10043 = d_9762 / x_10042;
        
        ((__global float *) mem_10665)[gtid_10030] = res_10043;
    }
}
__kernel void segmap_10045(int32_t sizze_9767, float m_9768, __global
                           unsigned char *x_mem_10662, __global
                           unsigned char *mem_10665)
{
    const int32_t segmap_group_sizze_10049 = multiplyzisegmap_group_sizze_10048;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10782;
    int32_t local_tid_10783;
    int32_t group_sizze_10786;
    int32_t wave_sizze_10785;
    int32_t group_tid_10784;
    
    global_tid_10782 = get_global_id(0);
    local_tid_10783 = get_local_id(0);
    group_sizze_10786 = get_local_size(0);
    wave_sizze_10785 = LOCKSTEP_WIDTH;
    group_tid_10784 = get_group_id(0);
    
    int32_t phys_tid_10045 = global_tid_10782;
    int32_t gtid_10044 = group_tid_10784 * segmap_group_sizze_10049 +
            local_tid_10783;
    
    if (slt32(gtid_10044, sizze_9767)) {
        float x_10056 = ((__global float *) x_mem_10662)[gtid_10044];
        float res_10057 = m_9768 * x_10056;
        
        ((__global float *) mem_10665)[gtid_10044] = res_10057;
    }
}
__kernel void segmap_10059(int32_t sizze_9773, float s_9774, __global
                           unsigned char *x_mem_10662, __global
                           unsigned char *mem_10665)
{
    const int32_t segmap_group_sizze_10063 = addzisegmap_group_sizze_10062;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10789;
    int32_t local_tid_10790;
    int32_t group_sizze_10793;
    int32_t wave_sizze_10792;
    int32_t group_tid_10791;
    
    global_tid_10789 = get_global_id(0);
    local_tid_10790 = get_local_id(0);
    group_sizze_10793 = get_local_size(0);
    wave_sizze_10792 = LOCKSTEP_WIDTH;
    group_tid_10791 = get_group_id(0);
    
    int32_t phys_tid_10059 = global_tid_10789;
    int32_t gtid_10058 = group_tid_10791 * segmap_group_sizze_10063 +
            local_tid_10790;
    
    if (slt32(gtid_10058, sizze_9773)) {
        float x_10070 = ((__global float *) x_mem_10662)[gtid_10058];
        float res_10071 = s_9774 + x_10070;
        
        ((__global float *) mem_10665)[gtid_10058] = res_10071;
    }
}
__kernel void segmap_10073(int32_t sizze_9779, float d_9780, __global
                           unsigned char *x_mem_10662, __global
                           unsigned char *mem_10665)
{
    const int32_t segmap_group_sizze_10077 =
                  substractzisegmap_group_sizze_10076;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10796;
    int32_t local_tid_10797;
    int32_t group_sizze_10800;
    int32_t wave_sizze_10799;
    int32_t group_tid_10798;
    
    global_tid_10796 = get_global_id(0);
    local_tid_10797 = get_local_id(0);
    group_sizze_10800 = get_local_size(0);
    wave_sizze_10799 = LOCKSTEP_WIDTH;
    group_tid_10798 = get_group_id(0);
    
    int32_t phys_tid_10073 = global_tid_10796;
    int32_t gtid_10072 = group_tid_10798 * segmap_group_sizze_10077 +
            local_tid_10797;
    
    if (slt32(gtid_10072, sizze_9779)) {
        float x_10084 = ((__global float *) x_mem_10662)[gtid_10072];
        float res_10085 = d_9780 - x_10084;
        
        ((__global float *) mem_10665)[gtid_10072] = res_10085;
    }
}
__kernel void segmap_10087(int32_t sizze_9785, __global
                           unsigned char *x_mem_10662, __global
                           unsigned char *y_mem_10663, __global
                           unsigned char *mem_10666)
{
    const int32_t segmap_group_sizze_10091 =
                  multiply2zisegmap_group_sizze_10090;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10803;
    int32_t local_tid_10804;
    int32_t group_sizze_10807;
    int32_t wave_sizze_10806;
    int32_t group_tid_10805;
    
    global_tid_10803 = get_global_id(0);
    local_tid_10804 = get_local_id(0);
    group_sizze_10807 = get_local_size(0);
    wave_sizze_10806 = LOCKSTEP_WIDTH;
    group_tid_10805 = get_group_id(0);
    
    int32_t phys_tid_10087 = global_tid_10803;
    int32_t gtid_10086 = group_tid_10805 * segmap_group_sizze_10091 +
            local_tid_10804;
    
    if (slt32(gtid_10086, sizze_9785)) {
        float x_10098 = ((__global float *) x_mem_10662)[gtid_10086];
        float x_10099 = ((__global float *) y_mem_10663)[gtid_10086];
        float res_10100 = x_10098 * x_10099;
        
        ((__global float *) mem_10666)[gtid_10086] = res_10100;
    }
}
__kernel void segmap_10103(int32_t sizze_9800, int32_t sizze_9801, float p_9802,
                           __global unsigned char *x_mem_10662, __global
                           unsigned char *mem_10667)
{
    const int32_t segmap_group_sizze_10109 =
                  lmatmultiplyzisegmap_group_sizze_10108;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10811;
    int32_t local_tid_10812;
    int32_t group_sizze_10815;
    int32_t wave_sizze_10814;
    int32_t group_tid_10813;
    
    global_tid_10811 = get_global_id(0);
    local_tid_10812 = get_local_id(0);
    group_sizze_10815 = get_local_size(0);
    wave_sizze_10814 = LOCKSTEP_WIDTH;
    group_tid_10813 = get_group_id(0);
    
    int32_t phys_tid_10103 = global_tid_10811;
    int32_t gtid_10101 = squot32(group_tid_10813 * segmap_group_sizze_10109 +
                                 local_tid_10812, sizze_9801);
    int32_t gtid_10102;
    
    gtid_10102 = group_tid_10813 * segmap_group_sizze_10109 + local_tid_10812 -
        squot32(group_tid_10813 * segmap_group_sizze_10109 + local_tid_10812,
                sizze_9801) * sizze_9801;
    if (slt32(gtid_10101, sizze_9800) && slt32(gtid_10102, sizze_9801)) {
        float x_10116 = ((__global float *) x_mem_10662)[gtid_10101 *
                                                         sizze_9801 +
                                                         gtid_10102];
        float res_10117 = p_9802 * x_10116;
        
        ((__global float *) mem_10667)[gtid_10101 * sizze_9801 + gtid_10102] =
            res_10117;
    }
}
__kernel void segmap_10119(int32_t sizze_9809, __global
                           unsigned char *x_mem_10662, __global
                           unsigned char *y_mem_10663, __global
                           unsigned char *mem_10666)
{
    const int32_t segmap_group_sizze_10123 = add2zisegmap_group_sizze_10122;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10818;
    int32_t local_tid_10819;
    int32_t group_sizze_10822;
    int32_t wave_sizze_10821;
    int32_t group_tid_10820;
    
    global_tid_10818 = get_global_id(0);
    local_tid_10819 = get_local_id(0);
    group_sizze_10822 = get_local_size(0);
    wave_sizze_10821 = LOCKSTEP_WIDTH;
    group_tid_10820 = get_group_id(0);
    
    int32_t phys_tid_10119 = global_tid_10818;
    int32_t gtid_10118 = group_tid_10820 * segmap_group_sizze_10123 +
            local_tid_10819;
    
    if (slt32(gtid_10118, sizze_9809)) {
        float x_10130 = ((__global float *) x_mem_10662)[gtid_10118];
        float x_10131 = ((__global float *) y_mem_10663)[gtid_10118];
        float res_10132 = x_10130 + x_10131;
        
        ((__global float *) mem_10666)[gtid_10118] = res_10132;
    }
}
__kernel void segmap_10134(int32_t sizze_9824, __global
                           unsigned char *x_mem_10662, __global
                           unsigned char *y_mem_10663, __global
                           unsigned char *mem_10666)
{
    const int32_t segmap_group_sizze_10138 =
                  substract2zisegmap_group_sizze_10137;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10825;
    int32_t local_tid_10826;
    int32_t group_sizze_10829;
    int32_t wave_sizze_10828;
    int32_t group_tid_10827;
    
    global_tid_10825 = get_global_id(0);
    local_tid_10826 = get_local_id(0);
    group_sizze_10829 = get_local_size(0);
    wave_sizze_10828 = LOCKSTEP_WIDTH;
    group_tid_10827 = get_group_id(0);
    
    int32_t phys_tid_10134 = global_tid_10825;
    int32_t gtid_10133 = group_tid_10827 * segmap_group_sizze_10138 +
            local_tid_10826;
    
    if (slt32(gtid_10133, sizze_9824)) {
        float x_10145 = ((__global float *) x_mem_10662)[gtid_10133];
        float x_10146 = ((__global float *) y_mem_10663)[gtid_10133];
        float res_10147 = x_10145 - x_10146;
        
        ((__global float *) mem_10666)[gtid_10133] = res_10147;
    }
}
__kernel void segmap_10150(int32_t sizze_9839, int32_t sizze_9840, float s_9841,
                           __global unsigned char *x_mem_10662, __global
                           unsigned char *mem_10667)
{
    const int32_t segmap_group_sizze_10156 = lmataddzisegmap_group_sizze_10155;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10833;
    int32_t local_tid_10834;
    int32_t group_sizze_10837;
    int32_t wave_sizze_10836;
    int32_t group_tid_10835;
    
    global_tid_10833 = get_global_id(0);
    local_tid_10834 = get_local_id(0);
    group_sizze_10837 = get_local_size(0);
    wave_sizze_10836 = LOCKSTEP_WIDTH;
    group_tid_10835 = get_group_id(0);
    
    int32_t phys_tid_10150 = global_tid_10833;
    int32_t gtid_10148 = squot32(group_tid_10835 * segmap_group_sizze_10156 +
                                 local_tid_10834, sizze_9840);
    int32_t gtid_10149;
    
    gtid_10149 = group_tid_10835 * segmap_group_sizze_10156 + local_tid_10834 -
        squot32(group_tid_10835 * segmap_group_sizze_10156 + local_tid_10834,
                sizze_9840) * sizze_9840;
    if (slt32(gtid_10148, sizze_9839) && slt32(gtid_10149, sizze_9840)) {
        float x_10163 = ((__global float *) x_mem_10662)[gtid_10148 *
                                                         sizze_9840 +
                                                         gtid_10149];
        float res_10164 = s_9841 + x_10163;
        
        ((__global float *) mem_10667)[gtid_10148 * sizze_9840 + gtid_10149] =
            res_10164;
    }
}
__kernel void segmap_10167(int32_t sizze_9848, int32_t sizze_9849, float d_9850,
                           __global unsigned char *x_mem_10662, __global
                           unsigned char *mem_10667)
{
    const int32_t segmap_group_sizze_10173 =
                  lmatsubstractzisegmap_group_sizze_10172;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10841;
    int32_t local_tid_10842;
    int32_t group_sizze_10845;
    int32_t wave_sizze_10844;
    int32_t group_tid_10843;
    
    global_tid_10841 = get_global_id(0);
    local_tid_10842 = get_local_id(0);
    group_sizze_10845 = get_local_size(0);
    wave_sizze_10844 = LOCKSTEP_WIDTH;
    group_tid_10843 = get_group_id(0);
    
    int32_t phys_tid_10167 = global_tid_10841;
    int32_t gtid_10165 = squot32(group_tid_10843 * segmap_group_sizze_10173 +
                                 local_tid_10842, sizze_9849);
    int32_t gtid_10166;
    
    gtid_10166 = group_tid_10843 * segmap_group_sizze_10173 + local_tid_10842 -
        squot32(group_tid_10843 * segmap_group_sizze_10173 + local_tid_10842,
                sizze_9849) * sizze_9849;
    if (slt32(gtid_10165, sizze_9848) && slt32(gtid_10166, sizze_9849)) {
        float x_10180 = ((__global float *) x_mem_10662)[gtid_10165 *
                                                         sizze_9849 +
                                                         gtid_10166];
        float res_10181 = d_9850 - x_10180;
        
        ((__global float *) mem_10667)[gtid_10165 * sizze_9849 + gtid_10166] =
            res_10181;
    }
}
__kernel void segmap_10184(int32_t sizze_9857, int32_t sizze_9858, __global
                           unsigned char *x_mem_10662, __global
                           unsigned char *mem_10667)
{
    const int32_t segmap_group_sizze_10190 = sigmoidzisegmap_group_sizze_10189;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10849;
    int32_t local_tid_10850;
    int32_t group_sizze_10853;
    int32_t wave_sizze_10852;
    int32_t group_tid_10851;
    
    global_tid_10849 = get_global_id(0);
    local_tid_10850 = get_local_id(0);
    group_sizze_10853 = get_local_size(0);
    wave_sizze_10852 = LOCKSTEP_WIDTH;
    group_tid_10851 = get_group_id(0);
    
    int32_t phys_tid_10184 = global_tid_10849;
    int32_t gtid_10182 = squot32(group_tid_10851 * segmap_group_sizze_10190 +
                                 local_tid_10850, sizze_9858);
    int32_t gtid_10183;
    
    gtid_10183 = group_tid_10851 * segmap_group_sizze_10190 + local_tid_10850 -
        squot32(group_tid_10851 * segmap_group_sizze_10190 + local_tid_10850,
                sizze_9858) * sizze_9858;
    if (slt32(gtid_10182, sizze_9857) && slt32(gtid_10183, sizze_9858)) {
        float x_10197 = ((__global float *) x_mem_10662)[gtid_10182 *
                                                         sizze_9858 +
                                                         gtid_10183];
        float res_10198 = 0.0F - x_10197;
        float res_10199 = fpow32(2.7182817F, res_10198);
        float res_10200 = 1.0F + res_10199;
        float res_10201 = 1.0F / res_10200;
        
        ((__global float *) mem_10667)[gtid_10182 * sizze_9858 + gtid_10183] =
            res_10201;
    }
}
__kernel void segmap_10204(int32_t sizze_9868, int32_t sizze_9869,
                           int32_t sizze_9870, __global
                           unsigned char *u_mem_10662, __global
                           unsigned char *b_mem_10663, __global
                           unsigned char *mem_10666)
{
    const int32_t segmap_group_sizze_10208 = lvecmulzisegmap_group_sizze_10207;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10856;
    int32_t local_tid_10857;
    int32_t group_sizze_10860;
    int32_t wave_sizze_10859;
    int32_t group_tid_10858;
    
    global_tid_10856 = get_global_id(0);
    local_tid_10857 = get_local_id(0);
    group_sizze_10860 = get_local_size(0);
    wave_sizze_10859 = LOCKSTEP_WIDTH;
    group_tid_10858 = get_group_id(0);
    
    int32_t phys_tid_10204 = global_tid_10856;
    int32_t gtid_10203 = group_tid_10858 * segmap_group_sizze_10208 +
            local_tid_10857;
    
    if (slt32(gtid_10203, sizze_9870)) {
        int32_t binop_x_10640 = sizze_9868 * gtid_10203;
        float res_10216;
        float redout_10634 = 0.0F;
        
        for (int32_t i_10635 = 0; i_10635 < sizze_9868; i_10635++) {
            float x_10220 = ((__global float *) u_mem_10662)[i_10635];
            int32_t binop_x_10641 = i_10635 + binop_x_10640;
            int32_t new_index_10642 = squot32(binop_x_10641, sizze_9869);
            int32_t binop_y_10648 = sizze_9869 * new_index_10642;
            int32_t new_index_10649 = binop_x_10641 - binop_y_10648;
            float x_10221 = ((__global float *) b_mem_10663)[new_index_10649 *
                                                             sizze_9870 +
                                                             new_index_10642];
            float res_10222 = x_10220 * x_10221;
            float res_10219 = res_10222 + redout_10634;
            float redout_tmp_10861 = res_10219;
            
            redout_10634 = redout_tmp_10861;
        }
        res_10216 = redout_10634;
        ((__global float *) mem_10666)[gtid_10203] = res_10216;
    }
}
__kernel void segmap_10226(int32_t sizze_9890, int32_t sizze_9891,
                           int32_t sizze_9893, __global
                           unsigned char *x_mem_10662, __global
                           unsigned char *y_mem_10663, __global
                           unsigned char *mem_10668)
{
    const int32_t segmap_group_sizze_10232 =
                  matmultiplyzisegmap_group_sizze_10231;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10865;
    int32_t local_tid_10866;
    int32_t group_sizze_10869;
    int32_t wave_sizze_10868;
    int32_t group_tid_10867;
    
    global_tid_10865 = get_global_id(0);
    local_tid_10866 = get_local_id(0);
    group_sizze_10869 = get_local_size(0);
    wave_sizze_10868 = LOCKSTEP_WIDTH;
    group_tid_10867 = get_group_id(0);
    
    int32_t phys_tid_10226 = global_tid_10865;
    int32_t gtid_10224 = squot32(group_tid_10867 * segmap_group_sizze_10232 +
                                 local_tid_10866, sizze_9891);
    int32_t gtid_10225;
    
    gtid_10225 = group_tid_10867 * segmap_group_sizze_10232 + local_tid_10866 -
        squot32(group_tid_10867 * segmap_group_sizze_10232 + local_tid_10866,
                sizze_9891) * sizze_9891;
    if (slt32(gtid_10224, sizze_9890) && slt32(gtid_10225, sizze_9891)) {
        float x_10239 = ((__global float *) x_mem_10662)[gtid_10224 *
                                                         sizze_9891 +
                                                         gtid_10225];
        int32_t binop_x_10331 = sizze_9891 * gtid_10224;
        int32_t binop_x_10332 = gtid_10225 + binop_x_10331;
        int32_t new_index_10333 = squot32(binop_x_10332, sizze_9893);
        int32_t binop_y_10339 = sizze_9893 * new_index_10333;
        int32_t new_index_10340 = binop_x_10332 - binop_y_10339;
        float x_10240 = ((__global float *) y_mem_10663)[new_index_10333 *
                                                         sizze_9893 +
                                                         new_index_10340];
        float res_10241 = x_10239 * x_10240;
        
        ((__global float *) mem_10668)[gtid_10224 * sizze_9891 + gtid_10225] =
            res_10241;
    }
}
__kernel void segmap_10245(int32_t sizze_9919, int32_t sizze_9920,
                           int32_t sizze_9922, __global
                           unsigned char *x_mem_10662, __global
                           unsigned char *y_mem_10663, __global
                           unsigned char *mem_10668)
{
    const int32_t segmap_group_sizze_10251 = mataddzisegmap_group_sizze_10250;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10873;
    int32_t local_tid_10874;
    int32_t group_sizze_10877;
    int32_t wave_sizze_10876;
    int32_t group_tid_10875;
    
    global_tid_10873 = get_global_id(0);
    local_tid_10874 = get_local_id(0);
    group_sizze_10877 = get_local_size(0);
    wave_sizze_10876 = LOCKSTEP_WIDTH;
    group_tid_10875 = get_group_id(0);
    
    int32_t phys_tid_10245 = global_tid_10873;
    int32_t gtid_10243 = squot32(group_tid_10875 * segmap_group_sizze_10251 +
                                 local_tid_10874, sizze_9920);
    int32_t gtid_10244;
    
    gtid_10244 = group_tid_10875 * segmap_group_sizze_10251 + local_tid_10874 -
        squot32(group_tid_10875 * segmap_group_sizze_10251 + local_tid_10874,
                sizze_9920) * sizze_9920;
    if (slt32(gtid_10243, sizze_9919) && slt32(gtid_10244, sizze_9920)) {
        float x_10258 = ((__global float *) x_mem_10662)[gtid_10243 *
                                                         sizze_9920 +
                                                         gtid_10244];
        int32_t binop_x_10331 = sizze_9920 * gtid_10243;
        int32_t binop_x_10332 = gtid_10244 + binop_x_10331;
        int32_t new_index_10333 = squot32(binop_x_10332, sizze_9922);
        int32_t binop_y_10339 = sizze_9922 * new_index_10333;
        int32_t new_index_10340 = binop_x_10332 - binop_y_10339;
        float x_10259 = ((__global float *) y_mem_10663)[new_index_10333 *
                                                         sizze_9922 +
                                                         new_index_10340];
        float res_10260 = x_10258 + x_10259;
        
        ((__global float *) mem_10668)[gtid_10243 * sizze_9920 + gtid_10244] =
            res_10260;
    }
}
__kernel void segmap_10264(int32_t sizze_9948, int32_t sizze_9949,
                           int32_t sizze_9951, __global
                           unsigned char *x_mem_10662, __global
                           unsigned char *y_mem_10663, __global
                           unsigned char *mem_10668)
{
    const int32_t segmap_group_sizze_10270 =
                  matsubstractzisegmap_group_sizze_10269;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10881;
    int32_t local_tid_10882;
    int32_t group_sizze_10885;
    int32_t wave_sizze_10884;
    int32_t group_tid_10883;
    
    global_tid_10881 = get_global_id(0);
    local_tid_10882 = get_local_id(0);
    group_sizze_10885 = get_local_size(0);
    wave_sizze_10884 = LOCKSTEP_WIDTH;
    group_tid_10883 = get_group_id(0);
    
    int32_t phys_tid_10264 = global_tid_10881;
    int32_t gtid_10262 = squot32(group_tid_10883 * segmap_group_sizze_10270 +
                                 local_tid_10882, sizze_9949);
    int32_t gtid_10263;
    
    gtid_10263 = group_tid_10883 * segmap_group_sizze_10270 + local_tid_10882 -
        squot32(group_tid_10883 * segmap_group_sizze_10270 + local_tid_10882,
                sizze_9949) * sizze_9949;
    if (slt32(gtid_10262, sizze_9948) && slt32(gtid_10263, sizze_9949)) {
        float x_10277 = ((__global float *) x_mem_10662)[gtid_10262 *
                                                         sizze_9949 +
                                                         gtid_10263];
        int32_t binop_x_10331 = sizze_9949 * gtid_10262;
        int32_t binop_x_10332 = gtid_10263 + binop_x_10331;
        int32_t new_index_10333 = squot32(binop_x_10332, sizze_9951);
        int32_t binop_y_10339 = sizze_9951 * new_index_10333;
        int32_t new_index_10340 = binop_x_10332 - binop_y_10339;
        float x_10278 = ((__global float *) y_mem_10663)[new_index_10333 *
                                                         sizze_9951 +
                                                         new_index_10340];
        float res_10279 = x_10277 - x_10278;
        
        ((__global float *) mem_10668)[gtid_10262 * sizze_9949 + gtid_10263] =
            res_10279;
    }
}
__kernel void segmap_intragroup_10354(__local volatile
                                      int64_t *mem_10672_backing_aligned_0,
                                      __local volatile
                                      int64_t *mem_10677_backing_aligned_1,
                                      __local volatile
                                      int64_t *mem_10687_backing_aligned_2,
                                      __local volatile
                                      int64_t *mem_10692_backing_aligned_3,
                                      int32_t sizze_9977, int32_t sizze_9978,
                                      int32_t sizze_9980,
                                      int32_t num_groups_y_10352,
                                      int32_t num_whole_tiles_10355,
                                      int32_t residual_input_10488,
                                      unsigned char cond_10489, __global
                                      unsigned char *a_mem_10662, __global
                                      unsigned char *b_mem_10663, __global
                                      unsigned char *mem_10702)
{
    const int32_t tile_sizze_10345 = dotzitile_sizze_10344;
    const int32_t group_sizze_10346 = dotzitile_sizze_10344 *
                  dotzitile_sizze_10344;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_10672_backing_0 = (__local volatile
                                                           char *) mem_10672_backing_aligned_0;
    __local volatile char *restrict mem_10677_backing_1 = (__local volatile
                                                           char *) mem_10677_backing_aligned_1;
    __local volatile char *restrict mem_10687_backing_2 = (__local volatile
                                                           char *) mem_10687_backing_aligned_2;
    __local volatile char *restrict mem_10692_backing_3 = (__local volatile
                                                           char *) mem_10692_backing_aligned_3;
    int32_t global_tid_10889;
    int32_t local_tid_10890;
    int32_t group_sizze_10893;
    int32_t wave_sizze_10892;
    int32_t group_tid_10891;
    
    global_tid_10889 = get_global_id(0);
    local_tid_10890 = get_local_id(0);
    group_sizze_10893 = get_local_size(0);
    wave_sizze_10892 = LOCKSTEP_WIDTH;
    group_tid_10891 = get_group_id(0);
    
    int32_t gid_flat_10354 = group_tid_10891;
    int32_t gid_x_10342 = squot32(group_tid_10891, num_groups_y_10352);
    int32_t gid_y_10343;
    
    gid_y_10343 = group_tid_10891 - squot32(group_tid_10891,
                                            num_groups_y_10352) *
        num_groups_y_10352;
    
    float mem_10667;
    int32_t ltid_x_10371 = squot32(local_tid_10890, tile_sizze_10345);
    int32_t ltid_y_10372;
    
    ltid_y_10372 = local_tid_10890 - squot32(local_tid_10890,
                                             tile_sizze_10345) *
        tile_sizze_10345;
    
    int32_t ltid_flat_10373;
    
    ltid_flat_10373 = local_tid_10890;
    if (slt32(ltid_x_10371, tile_sizze_10345) && slt32(ltid_y_10372,
                                                       tile_sizze_10345)) {
        mem_10667 = 0.0F;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t binop_x_10450 = gid_x_10342 * tile_sizze_10345;
    int32_t binop_x_10452 = gid_y_10343 * tile_sizze_10345;
    __local char *mem_10672;
    
    mem_10672 = (__local char *) mem_10672_backing_0;
    
    __local char *mem_10677;
    
    mem_10677 = (__local char *) mem_10677_backing_1;
    
    float mem_10681;
    
    for (int32_t tile_id_10382 = 0; tile_id_10382 < num_whole_tiles_10355;
         tile_id_10382++) {
        int32_t binop_x_10446 = tile_sizze_10345 * tile_id_10382;
        int32_t ltid_x_10383 = squot32(local_tid_10890, tile_sizze_10345);
        int32_t ltid_y_10384;
        
        ltid_y_10384 = local_tid_10890 - squot32(local_tid_10890,
                                                 tile_sizze_10345) *
            tile_sizze_10345;
        
        int32_t ltid_flat_10385;
        
        ltid_flat_10385 = local_tid_10890;
        if (slt32(ltid_x_10383, tile_sizze_10345) && slt32(ltid_y_10384,
                                                           tile_sizze_10345)) {
            int32_t i_10447 = ltid_x_10383 + binop_x_10446;
            int32_t j_10449 = ltid_y_10384 + binop_x_10446;
            int32_t gtid_10451 = ltid_x_10383 + binop_x_10450;
            int32_t gtid_10453 = ltid_y_10384 + binop_x_10452;
            float tile_elem_10457 = ((__global
                                      float *) a_mem_10662)[gtid_10451 *
                                                            sizze_9978 +
                                                            j_10449];
            float tile_elem_10458 = ((__global float *) b_mem_10663)[i_10447 *
                                                                     sizze_9980 +
                                                                     gtid_10453];
            
            ((__local float *) mem_10672)[ltid_x_10383 * tile_sizze_10345 +
                                          ltid_y_10384] = tile_elem_10457;
            ((__local float *) mem_10677)[ltid_x_10383 * tile_sizze_10345 +
                                          ltid_y_10384] = tile_elem_10458;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_10409 = squot32(local_tid_10890, tile_sizze_10345);
        int32_t ltid_y_10410;
        
        ltid_y_10410 = local_tid_10890 - squot32(local_tid_10890,
                                                 tile_sizze_10345) *
            tile_sizze_10345;
        
        int32_t ltid_flat_10411;
        
        ltid_flat_10411 = local_tid_10890;
        if (slt32(ltid_x_10409, tile_sizze_10345) && slt32(ltid_y_10410,
                                                           tile_sizze_10345)) {
            int32_t gtid_10461 = ltid_x_10409 + binop_x_10450;
            int32_t gtid_10463 = ltid_y_10410 + binop_x_10452;
            float acc_10467 = mem_10667;
            bool binop_x_10470 = slt32(gtid_10461, sizze_9977);
            bool binop_y_10471 = slt32(gtid_10463, sizze_9980);
            bool cond_10472 = binop_x_10470 && binop_y_10471;
            float acc_10473;
            
            if (cond_10472) {
                float x_10474;
                float redout_10634 = acc_10467;
                
                for (int32_t i_10635 = 0; i_10635 < tile_sizze_10345;
                     i_10635++) {
                    float x_10478 = ((__local float *) mem_10672)[ltid_x_10409 *
                                                                  tile_sizze_10345 +
                                                                  i_10635];
                    float x_10479 = ((__local float *) mem_10677)[i_10635 *
                                                                  tile_sizze_10345 +
                                                                  ltid_y_10410];
                    float res_10480 = x_10478 * x_10479;
                    float res_10477 = res_10480 + redout_10634;
                    float redout_tmp_10895 = res_10477;
                    
                    redout_10634 = redout_tmp_10895;
                }
                x_10474 = redout_10634;
                acc_10473 = x_10474;
            } else {
                acc_10473 = acc_10467;
            }
            mem_10681 = acc_10473;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_10896 = 0; i_10896 < squot32(tile_sizze_10345 *
                                                    tile_sizze_10345 -
                                                    local_tid_10890 +
                                                    group_sizze_10346 - 1,
                                                    group_sizze_10346);
             i_10896++) {
            mem_10667 = mem_10681;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    __local char *mem_10687;
    
    mem_10687 = (__local char *) mem_10687_backing_2;
    
    __local char *mem_10692;
    
    mem_10692 = (__local char *) mem_10692_backing_3;
    
    float mem_10696;
    float mem_10714;
    
    if (cond_10489) {
        for (int32_t i_10897 = 0; i_10897 < squot32(tile_sizze_10345 *
                                                    tile_sizze_10345 -
                                                    local_tid_10890 +
                                                    group_sizze_10346 - 1,
                                                    group_sizze_10346);
             i_10897++) {
            mem_10714 = mem_10667;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
        int32_t binop_x_10575 = tile_sizze_10345 * num_whole_tiles_10355;
        int32_t ltid_x_10490 = squot32(local_tid_10890, tile_sizze_10345);
        int32_t ltid_y_10491;
        
        ltid_y_10491 = local_tid_10890 - squot32(local_tid_10890,
                                                 tile_sizze_10345) *
            tile_sizze_10345;
        
        int32_t ltid_flat_10492;
        
        ltid_flat_10492 = local_tid_10890;
        if (slt32(ltid_x_10490, tile_sizze_10345) && slt32(ltid_y_10491,
                                                           tile_sizze_10345)) {
            int32_t i_10576 = ltid_x_10490 + binop_x_10575;
            int32_t j_10578 = ltid_y_10491 + binop_x_10575;
            int32_t gtid_10580 = binop_x_10450 + ltid_x_10490;
            int32_t gtid_10582 = binop_x_10452 + ltid_y_10491;
            bool binop_x_10586 = slt32(j_10578, sizze_9978);
            bool binop_y_10587 = slt32(gtid_10580, sizze_9977);
            bool cond_10588 = binop_x_10586 && binop_y_10587;
            float pre_10589;
            
            if (cond_10588) {
                float x_10590 = ((__global float *) a_mem_10662)[gtid_10580 *
                                                                 sizze_9978 +
                                                                 j_10578];
                
                pre_10589 = x_10590;
            } else {
                pre_10589 = 0.0F;
            }
            
            bool binop_x_10592 = slt32(i_10576, sizze_9978);
            bool binop_y_10593 = slt32(gtid_10582, sizze_9980);
            bool cond_10594 = binop_x_10592 && binop_y_10593;
            float pre_10595;
            
            if (cond_10594) {
                float x_10596 = ((__global float *) b_mem_10663)[i_10576 *
                                                                 sizze_9980 +
                                                                 gtid_10582];
                
                pre_10595 = x_10596;
            } else {
                pre_10595 = 0.0F;
            }
            ((__local float *) mem_10687)[ltid_x_10490 * tile_sizze_10345 +
                                          ltid_y_10491] = pre_10589;
            ((__local float *) mem_10692)[ltid_x_10490 * tile_sizze_10345 +
                                          ltid_y_10491] = pre_10595;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_10538 = squot32(local_tid_10890, tile_sizze_10345);
        int32_t ltid_y_10539;
        
        ltid_y_10539 = local_tid_10890 - squot32(local_tid_10890,
                                                 tile_sizze_10345) *
            tile_sizze_10345;
        
        int32_t ltid_flat_10540;
        
        ltid_flat_10540 = local_tid_10890;
        if (slt32(ltid_x_10538, tile_sizze_10345) && slt32(ltid_y_10539,
                                                           tile_sizze_10345)) {
            int32_t gtid_10602 = binop_x_10450 + ltid_x_10538;
            int32_t gtid_10604 = binop_x_10452 + ltid_y_10539;
            float acc_10608 = mem_10667;
            bool binop_x_10611 = slt32(gtid_10602, sizze_9977);
            bool binop_y_10612 = slt32(gtid_10604, sizze_9980);
            bool cond_10613 = binop_x_10611 && binop_y_10612;
            float acc_10614;
            
            if (cond_10613) {
                float x_10615;
                float redout_10636 = acc_10608;
                
                for (int32_t i_10637 = 0; i_10637 < residual_input_10488;
                     i_10637++) {
                    float x_10619 = ((__local float *) mem_10687)[ltid_x_10538 *
                                                                  tile_sizze_10345 +
                                                                  i_10637];
                    float x_10620 = ((__local float *) mem_10692)[i_10637 *
                                                                  tile_sizze_10345 +
                                                                  ltid_y_10539];
                    float res_10621 = x_10619 * x_10620;
                    float res_10618 = res_10621 + redout_10636;
                    float redout_tmp_10898 = res_10618;
                    
                    redout_10636 = redout_tmp_10898;
                }
                x_10615 = redout_10636;
                acc_10614 = x_10615;
            } else {
                acc_10614 = acc_10608;
            }
            mem_10696 = acc_10614;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_10899 = 0; i_10899 < squot32(tile_sizze_10345 *
                                                    tile_sizze_10345 -
                                                    local_tid_10890 +
                                                    group_sizze_10346 - 1,
                                                    group_sizze_10346);
             i_10899++) {
            mem_10714 = mem_10696;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    int32_t thread_out_index_10900 = gid_x_10342 * tile_sizze_10345 +
            squot32(local_tid_10890, tile_sizze_10345);
    int32_t thread_out_index_10901;
    
    thread_out_index_10901 = gid_y_10343 * tile_sizze_10345 + (local_tid_10890 -
                                                               squot32(local_tid_10890,
                                                                       tile_sizze_10345) *
                                                               tile_sizze_10345);
    if (slt32(thread_out_index_10900, sizze_9977) &&
        slt32(thread_out_index_10901, sizze_9980)) {
        ((__global float *) mem_10702)[thread_out_index_10900 * sizze_9980 +
                                       thread_out_index_10901] = mem_10714;
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
class GPU:
  entry_points = {"dot": (["[][]f32", "[][]f32"], ["[][]f32"]),
                  "matsubstract": (["[][]f32", "[][]f32"], ["[][]f32"]),
                  "matadd": (["[][]f32", "[][]f32"], ["[][]f32"]),
                  "matmultiply": (["[][]f32", "[][]f32"], ["[][]f32"]),
                  "lvecmul": (["[]f32", "[][]f32"], ["[]f32"]),
                  "sigmoid": (["[][]f32"], ["[][]f32"]),
                  "lmatsubstract": (["f32", "[][]f32"], ["[][]f32"]),
                  "lmatadd": (["f32", "[][]f32"], ["[][]f32"]),
                  "substract2": (["[]f32", "[]f32"], ["[]f32"]),
                  "add2": (["[]f32", "[]f32"], ["[]f32"]),
                  "lmatmultiply": (["f32", "[][]f32"], ["[][]f32"]),
                  "multiply2": (["[]f32", "[]f32"], ["[]f32"]),
                  "substract": (["f32", "[]f32"], ["[]f32"]), "add": (["f32",
                                                                       "[]f32"],
                                                                      ["[]f32"]),
                  "multiply": (["f32", "[]f32"], ["[]f32"]), "divide": (["f32",
                                                                         "[]f32"],
                                                                        ["[]f32"]),
                  "negation": (["[]f32"], ["[]f32"]), "exp": (["[]f32"],
                                                              ["[]f32"]),
                  "transp": (["[][]f32"], ["[][]f32"]), "arr": (["[][]f32"],
                                                                ["[][]f32"])}
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
                                       all_sizes={"add.segmap_group_size_10062": {"class": "group_size", "value": None},
                                        "add2.segmap_group_size_10122": {"class": "group_size", "value": None},
                                        "divide.segmap_group_size_10034": {"class": "group_size", "value": None},
                                        "dot.tile_size_10344": {"class": "tile_size", "value": None},
                                        "exp.segmap_group_size_10006": {"class": "group_size", "value": None},
                                        "lmatadd.segmap_group_size_10155": {"class": "group_size", "value": None},
                                        "lmatmultiply.segmap_group_size_10108": {"class": "group_size", "value": None},
                                        "lmatsubstract.segmap_group_size_10172": {"class": "group_size",
                                                                                  "value": None},
                                        "lvecmul.segmap_group_size_10207": {"class": "group_size", "value": None},
                                        "matadd.segmap_group_size_10250": {"class": "group_size", "value": None},
                                        "matmultiply.segmap_group_size_10231": {"class": "group_size", "value": None},
                                        "matsubstract.segmap_group_size_10269": {"class": "group_size", "value": None},
                                        "multiply.segmap_group_size_10048": {"class": "group_size", "value": None},
                                        "multiply2.segmap_group_size_10090": {"class": "group_size", "value": None},
                                        "negation.segmap_group_size_10020": {"class": "group_size", "value": None},
                                        "sigmoid.segmap_group_size_10189": {"class": "group_size", "value": None},
                                        "substract.segmap_group_size_10076": {"class": "group_size", "value": None},
                                        "substract2.segmap_group_size_10137": {"class": "group_size", "value": None}})
    self.map_transpose_f32_var = program.map_transpose_f32
    self.map_transpose_f32_low_height_var = program.map_transpose_f32_low_height
    self.map_transpose_f32_low_width_var = program.map_transpose_f32_low_width
    self.map_transpose_f32_small_var = program.map_transpose_f32_small
    self.segmap_10003_var = program.segmap_10003
    self.segmap_10017_var = program.segmap_10017
    self.segmap_10031_var = program.segmap_10031
    self.segmap_10045_var = program.segmap_10045
    self.segmap_10059_var = program.segmap_10059
    self.segmap_10073_var = program.segmap_10073
    self.segmap_10087_var = program.segmap_10087
    self.segmap_10103_var = program.segmap_10103
    self.segmap_10119_var = program.segmap_10119
    self.segmap_10134_var = program.segmap_10134
    self.segmap_10150_var = program.segmap_10150
    self.segmap_10167_var = program.segmap_10167
    self.segmap_10184_var = program.segmap_10184
    self.segmap_10204_var = program.segmap_10204
    self.segmap_10226_var = program.segmap_10226
    self.segmap_10245_var = program.segmap_10245
    self.segmap_10264_var = program.segmap_10264
    self.segmap_intragroup_10354_var = program.segmap_intragroup_10354
  def futhark_dot(self, a_mem_10662, b_mem_10663, sizze_9977, sizze_9978,
                  sizze_9979, sizze_9980):
    dim_zzero_9984 = (np.int32(0) == sizze_9979)
    dim_zzero_9985 = (np.int32(0) == sizze_9978)
    both_empty_9986 = (dim_zzero_9984 and dim_zzero_9985)
    dim_match_9987 = (sizze_9978 == sizze_9979)
    empty_or_match_9988 = (both_empty_9986 or dim_match_9987)
    empty_or_match_cert_9989 = True
    assert empty_or_match_9988, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:10:19-30\n   #1  GPU.fut:15:21-38\n   #2  GPU.fut:15:3-16:17\n   #3  GPU.fut:25:18-33\n   #4  GPU.fut:25:3-36\n   #5  GPU.fut:24:1-25:36\n" % ("function arguments of wrong shape",))
    tile_sizze_10345 = self.sizes["dot.tile_size_10344"]
    group_sizze_10346 = (tile_sizze_10345 * tile_sizze_10345)
    y_10347 = (tile_sizze_10345 - np.int32(1))
    x_10348 = (sizze_9977 + y_10347)
    num_groups_x_10349 = squot32(x_10348, tile_sizze_10345)
    x_10351 = (sizze_9980 + y_10347)
    num_groups_y_10352 = squot32(x_10351, tile_sizze_10345)
    num_groups_top_10353 = (num_groups_x_10349 * num_groups_y_10352)
    num_whole_tiles_10355 = squot32(sizze_9978, tile_sizze_10345)
    residual_input_10488 = srem32(sizze_9978, tile_sizze_10345)
    cond_10489 = (residual_input_10488 == np.int32(0))
    binop_x_10699 = sext_i32_i64(sizze_9977)
    binop_y_10700 = sext_i32_i64(sizze_9980)
    binop_x_10701 = (binop_x_10699 * binop_y_10700)
    bytes_10698 = (np.int64(4) * binop_x_10701)
    mem_10702 = opencl_alloc(self, bytes_10698, "mem_10702")
    binop_x_10666 = sext_i32_i64(group_sizze_10346)
    bytes_10664 = (np.int64(4) * binop_x_10666)
    binop_x_10669 = sext_i32_i64(tile_sizze_10345)
    binop_x_10671 = (binop_x_10669 * binop_x_10669)
    bytes_10668 = (np.int64(4) * binop_x_10671)
    if ((1 * (np.long(num_groups_top_10353) * np.long(group_sizze_10346))) != 0):
      self.segmap_intragroup_10354_var.set_args(cl.LocalMemory(np.long(bytes_10668)),
                                                cl.LocalMemory(np.long(bytes_10668)),
                                                cl.LocalMemory(np.long(bytes_10668)),
                                                cl.LocalMemory(np.long(bytes_10668)),
                                                np.int32(sizze_9977),
                                                np.int32(sizze_9978),
                                                np.int32(sizze_9980),
                                                np.int32(num_groups_y_10352),
                                                np.int32(num_whole_tiles_10355),
                                                np.int32(residual_input_10488),
                                                np.byte(cond_10489),
                                                a_mem_10662, b_mem_10663,
                                                mem_10702)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_intragroup_10354_var,
                                 ((np.long(num_groups_top_10353) * np.long(group_sizze_10346)),),
                                 (np.long(group_sizze_10346),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10887 = sizze_9977
    out_arrsizze_10888 = sizze_9980
    out_mem_10886 = mem_10702
    return (out_mem_10886, out_arrsizze_10887, out_arrsizze_10888)
  def futhark_matsubstract(self, x_mem_10662, y_mem_10663, sizze_9948,
                           sizze_9949, sizze_9950, sizze_9951):
    dim_zzero_9954 = (np.int32(0) == sizze_9950)
    dim_zzero_9955 = (np.int32(0) == sizze_9951)
    old_empty_9956 = (dim_zzero_9954 or dim_zzero_9955)
    dim_zzero_9957 = (np.int32(0) == sizze_9948)
    new_empty_9958 = (dim_zzero_9955 or dim_zzero_9957)
    both_empty_9959 = (old_empty_9956 and new_empty_9958)
    dim_match_9960 = (sizze_9948 == sizze_9950)
    empty_or_match_9961 = (both_empty_9959 or dim_match_9960)
    empty_or_match_cert_9962 = True
    assert empty_or_match_9961, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:150:3-21\n   #1  GPU.fut:149:1-150:21\n" % ("function arguments of wrong shape",))
    dim_zzero_9964 = (np.int32(0) == sizze_9949)
    both_empty_9965 = (dim_zzero_9955 and dim_zzero_9964)
    dim_match_9966 = (sizze_9949 == sizze_9951)
    empty_or_match_9967 = (both_empty_9965 or dim_match_9966)
    empty_or_match_cert_9968 = True
    assert empty_or_match_9967, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:141:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPU.fut:150:3-21\n   #4  GPU.fut:149:1-150:21\n" % ("function arguments of wrong shape",))
    sizze_10265 = sext_i32_i64(sizze_9948)
    sizze_10266 = sext_i32_i64(sizze_9949)
    nest_sizze_10268 = (sizze_10265 * sizze_10266)
    segmap_group_sizze_10270 = self.sizes["matsubstract.segmap_group_size_10269"]
    segmap_group_sizze_10271 = sext_i32_i64(segmap_group_sizze_10270)
    y_10272 = (segmap_group_sizze_10271 - np.int64(1))
    x_10273 = (nest_sizze_10268 + y_10272)
    segmap_usable_groups_64_10275 = squot64(x_10273, segmap_group_sizze_10271)
    segmap_usable_groups_10276 = sext_i64_i32(segmap_usable_groups_64_10275)
    bytes_10664 = (np.int64(4) * nest_sizze_10268)
    mem_10668 = opencl_alloc(self, bytes_10664, "mem_10668")
    if ((1 * (np.long(segmap_usable_groups_10276) * np.long(segmap_group_sizze_10270))) != 0):
      self.segmap_10264_var.set_args(np.int32(sizze_9948), np.int32(sizze_9949),
                                     np.int32(sizze_9951), x_mem_10662,
                                     y_mem_10663, mem_10668)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10264_var,
                                 ((np.long(segmap_usable_groups_10276) * np.long(segmap_group_sizze_10270)),),
                                 (np.long(segmap_group_sizze_10270),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10879 = sizze_9948
    out_arrsizze_10880 = sizze_9949
    out_mem_10878 = mem_10668
    return (out_mem_10878, out_arrsizze_10879, out_arrsizze_10880)
  def futhark_matadd(self, x_mem_10662, y_mem_10663, sizze_9919, sizze_9920,
                     sizze_9921, sizze_9922):
    dim_zzero_9925 = (np.int32(0) == sizze_9921)
    dim_zzero_9926 = (np.int32(0) == sizze_9922)
    old_empty_9927 = (dim_zzero_9925 or dim_zzero_9926)
    dim_zzero_9928 = (np.int32(0) == sizze_9919)
    new_empty_9929 = (dim_zzero_9926 or dim_zzero_9928)
    both_empty_9930 = (old_empty_9927 and new_empty_9929)
    dim_match_9931 = (sizze_9919 == sizze_9921)
    empty_or_match_9932 = (both_empty_9930 or dim_match_9931)
    empty_or_match_cert_9933 = True
    assert empty_or_match_9932, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:114:3-15\n   #1  GPU.fut:113:1-114:15\n" % ("function arguments of wrong shape",))
    dim_zzero_9935 = (np.int32(0) == sizze_9920)
    both_empty_9936 = (dim_zzero_9926 and dim_zzero_9935)
    dim_match_9937 = (sizze_9920 == sizze_9922)
    empty_or_match_9938 = (both_empty_9936 or dim_match_9937)
    empty_or_match_cert_9939 = True
    assert empty_or_match_9938, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:105:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPU.fut:114:3-15\n   #4  GPU.fut:113:1-114:15\n" % ("function arguments of wrong shape",))
    sizze_10246 = sext_i32_i64(sizze_9919)
    sizze_10247 = sext_i32_i64(sizze_9920)
    nest_sizze_10249 = (sizze_10246 * sizze_10247)
    segmap_group_sizze_10251 = self.sizes["matadd.segmap_group_size_10250"]
    segmap_group_sizze_10252 = sext_i32_i64(segmap_group_sizze_10251)
    y_10253 = (segmap_group_sizze_10252 - np.int64(1))
    x_10254 = (nest_sizze_10249 + y_10253)
    segmap_usable_groups_64_10256 = squot64(x_10254, segmap_group_sizze_10252)
    segmap_usable_groups_10257 = sext_i64_i32(segmap_usable_groups_64_10256)
    bytes_10664 = (np.int64(4) * nest_sizze_10249)
    mem_10668 = opencl_alloc(self, bytes_10664, "mem_10668")
    if ((1 * (np.long(segmap_usable_groups_10257) * np.long(segmap_group_sizze_10251))) != 0):
      self.segmap_10245_var.set_args(np.int32(sizze_9919), np.int32(sizze_9920),
                                     np.int32(sizze_9922), x_mem_10662,
                                     y_mem_10663, mem_10668)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10245_var,
                                 ((np.long(segmap_usable_groups_10257) * np.long(segmap_group_sizze_10251)),),
                                 (np.long(segmap_group_sizze_10251),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10871 = sizze_9919
    out_arrsizze_10872 = sizze_9920
    out_mem_10870 = mem_10668
    return (out_mem_10870, out_arrsizze_10871, out_arrsizze_10872)
  def futhark_matmultiply(self, x_mem_10662, y_mem_10663, sizze_9890,
                          sizze_9891, sizze_9892, sizze_9893):
    dim_zzero_9896 = (np.int32(0) == sizze_9892)
    dim_zzero_9897 = (np.int32(0) == sizze_9893)
    old_empty_9898 = (dim_zzero_9896 or dim_zzero_9897)
    dim_zzero_9899 = (np.int32(0) == sizze_9890)
    new_empty_9900 = (dim_zzero_9897 or dim_zzero_9899)
    both_empty_9901 = (old_empty_9898 and new_empty_9900)
    dim_match_9902 = (sizze_9890 == sizze_9892)
    empty_or_match_9903 = (both_empty_9901 or dim_match_9902)
    empty_or_match_cert_9904 = True
    assert empty_or_match_9903, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:80:3-20\n   #1  GPU.fut:79:1-80:20\n" % ("function arguments of wrong shape",))
    dim_zzero_9906 = (np.int32(0) == sizze_9891)
    both_empty_9907 = (dim_zzero_9897 and dim_zzero_9906)
    dim_match_9908 = (sizze_9891 == sizze_9893)
    empty_or_match_9909 = (both_empty_9907 or dim_match_9908)
    empty_or_match_cert_9910 = True
    assert empty_or_match_9909, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:71:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPU.fut:80:3-20\n   #4  GPU.fut:79:1-80:20\n" % ("function arguments of wrong shape",))
    sizze_10227 = sext_i32_i64(sizze_9890)
    sizze_10228 = sext_i32_i64(sizze_9891)
    nest_sizze_10230 = (sizze_10227 * sizze_10228)
    segmap_group_sizze_10232 = self.sizes["matmultiply.segmap_group_size_10231"]
    segmap_group_sizze_10233 = sext_i32_i64(segmap_group_sizze_10232)
    y_10234 = (segmap_group_sizze_10233 - np.int64(1))
    x_10235 = (nest_sizze_10230 + y_10234)
    segmap_usable_groups_64_10237 = squot64(x_10235, segmap_group_sizze_10233)
    segmap_usable_groups_10238 = sext_i64_i32(segmap_usable_groups_64_10237)
    bytes_10664 = (np.int64(4) * nest_sizze_10230)
    mem_10668 = opencl_alloc(self, bytes_10664, "mem_10668")
    if ((1 * (np.long(segmap_usable_groups_10238) * np.long(segmap_group_sizze_10232))) != 0):
      self.segmap_10226_var.set_args(np.int32(sizze_9890), np.int32(sizze_9891),
                                     np.int32(sizze_9893), x_mem_10662,
                                     y_mem_10663, mem_10668)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10226_var,
                                 ((np.long(segmap_usable_groups_10238) * np.long(segmap_group_sizze_10232)),),
                                 (np.long(segmap_group_sizze_10232),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10863 = sizze_9890
    out_arrsizze_10864 = sizze_9891
    out_mem_10862 = mem_10668
    return (out_mem_10862, out_arrsizze_10863, out_arrsizze_10864)
  def futhark_lvecmul(self, u_mem_10662, b_mem_10663, sizze_9868, sizze_9869,
                      sizze_9870):
    dim_zzero_9874 = (np.int32(0) == sizze_9869)
    dim_zzero_9875 = (np.int32(0) == sizze_9868)
    both_empty_9876 = (dim_zzero_9874 and dim_zzero_9875)
    dim_match_9877 = (sizze_9868 == sizze_9869)
    empty_or_match_9878 = (both_empty_9876 or dim_match_9877)
    empty_or_match_cert_9879 = True
    assert empty_or_match_9878, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:10:19-30\n   #1  GPU.fut:15:21-38\n   #2  GPU.fut:15:3-16:17\n   #3  GPU.fut:14:1-16:17\n" % ("function arguments of wrong shape",))
    sizze_10205 = sext_i32_i64(sizze_9870)
    segmap_group_sizze_10208 = self.sizes["lvecmul.segmap_group_size_10207"]
    segmap_group_sizze_10209 = sext_i32_i64(segmap_group_sizze_10208)
    y_10210 = (segmap_group_sizze_10209 - np.int64(1))
    x_10211 = (sizze_10205 + y_10210)
    segmap_usable_groups_64_10213 = squot64(x_10211, segmap_group_sizze_10209)
    segmap_usable_groups_10214 = sext_i64_i32(segmap_usable_groups_64_10213)
    bytes_10664 = (np.int64(4) * sizze_10205)
    mem_10666 = opencl_alloc(self, bytes_10664, "mem_10666")
    if ((1 * (np.long(segmap_usable_groups_10214) * np.long(segmap_group_sizze_10208))) != 0):
      self.segmap_10204_var.set_args(np.int32(sizze_9868), np.int32(sizze_9869),
                                     np.int32(sizze_9870), u_mem_10662,
                                     b_mem_10663, mem_10666)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10204_var,
                                 ((np.long(segmap_usable_groups_10214) * np.long(segmap_group_sizze_10208)),),
                                 (np.long(segmap_group_sizze_10208),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10855 = sizze_9870
    out_mem_10854 = mem_10666
    return (out_mem_10854, out_arrsizze_10855)
  def futhark_sigmoid(self, x_mem_10662, sizze_9857, sizze_9858):
    sizze_10185 = sext_i32_i64(sizze_9857)
    sizze_10186 = sext_i32_i64(sizze_9858)
    nest_sizze_10188 = (sizze_10185 * sizze_10186)
    segmap_group_sizze_10190 = self.sizes["sigmoid.segmap_group_size_10189"]
    segmap_group_sizze_10191 = sext_i32_i64(segmap_group_sizze_10190)
    y_10192 = (segmap_group_sizze_10191 - np.int64(1))
    x_10193 = (nest_sizze_10188 + y_10192)
    segmap_usable_groups_64_10195 = squot64(x_10193, segmap_group_sizze_10191)
    segmap_usable_groups_10196 = sext_i64_i32(segmap_usable_groups_64_10195)
    bytes_10663 = (np.int64(4) * nest_sizze_10188)
    mem_10667 = opencl_alloc(self, bytes_10663, "mem_10667")
    if ((1 * (np.long(segmap_usable_groups_10196) * np.long(segmap_group_sizze_10190))) != 0):
      self.segmap_10184_var.set_args(np.int32(sizze_9857), np.int32(sizze_9858),
                                     x_mem_10662, mem_10667)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10184_var,
                                 ((np.long(segmap_usable_groups_10196) * np.long(segmap_group_sizze_10190)),),
                                 (np.long(segmap_group_sizze_10190),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10847 = sizze_9857
    out_arrsizze_10848 = sizze_9858
    out_mem_10846 = mem_10667
    return (out_mem_10846, out_arrsizze_10847, out_arrsizze_10848)
  def futhark_lmatsubstract(self, x_mem_10662, sizze_9848, sizze_9849, d_9850):
    sizze_10168 = sext_i32_i64(sizze_9848)
    sizze_10169 = sext_i32_i64(sizze_9849)
    nest_sizze_10171 = (sizze_10168 * sizze_10169)
    segmap_group_sizze_10173 = self.sizes["lmatsubstract.segmap_group_size_10172"]
    segmap_group_sizze_10174 = sext_i32_i64(segmap_group_sizze_10173)
    y_10175 = (segmap_group_sizze_10174 - np.int64(1))
    x_10176 = (nest_sizze_10171 + y_10175)
    segmap_usable_groups_64_10178 = squot64(x_10176, segmap_group_sizze_10174)
    segmap_usable_groups_10179 = sext_i64_i32(segmap_usable_groups_64_10178)
    bytes_10663 = (np.int64(4) * nest_sizze_10171)
    mem_10667 = opencl_alloc(self, bytes_10663, "mem_10667")
    if ((1 * (np.long(segmap_usable_groups_10179) * np.long(segmap_group_sizze_10173))) != 0):
      self.segmap_10167_var.set_args(np.int32(sizze_9848), np.int32(sizze_9849),
                                     np.float32(d_9850), x_mem_10662, mem_10667)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10167_var,
                                 ((np.long(segmap_usable_groups_10179) * np.long(segmap_group_sizze_10173)),),
                                 (np.long(segmap_group_sizze_10173),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10839 = sizze_9848
    out_arrsizze_10840 = sizze_9849
    out_mem_10838 = mem_10667
    return (out_mem_10838, out_arrsizze_10839, out_arrsizze_10840)
  def futhark_lmatadd(self, x_mem_10662, sizze_9839, sizze_9840, s_9841):
    sizze_10151 = sext_i32_i64(sizze_9839)
    sizze_10152 = sext_i32_i64(sizze_9840)
    nest_sizze_10154 = (sizze_10151 * sizze_10152)
    segmap_group_sizze_10156 = self.sizes["lmatadd.segmap_group_size_10155"]
    segmap_group_sizze_10157 = sext_i32_i64(segmap_group_sizze_10156)
    y_10158 = (segmap_group_sizze_10157 - np.int64(1))
    x_10159 = (nest_sizze_10154 + y_10158)
    segmap_usable_groups_64_10161 = squot64(x_10159, segmap_group_sizze_10157)
    segmap_usable_groups_10162 = sext_i64_i32(segmap_usable_groups_64_10161)
    bytes_10663 = (np.int64(4) * nest_sizze_10154)
    mem_10667 = opencl_alloc(self, bytes_10663, "mem_10667")
    if ((1 * (np.long(segmap_usable_groups_10162) * np.long(segmap_group_sizze_10156))) != 0):
      self.segmap_10150_var.set_args(np.int32(sizze_9839), np.int32(sizze_9840),
                                     np.float32(s_9841), x_mem_10662, mem_10667)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10150_var,
                                 ((np.long(segmap_usable_groups_10162) * np.long(segmap_group_sizze_10156)),),
                                 (np.long(segmap_group_sizze_10156),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10831 = sizze_9839
    out_arrsizze_10832 = sizze_9840
    out_mem_10830 = mem_10667
    return (out_mem_10830, out_arrsizze_10831, out_arrsizze_10832)
  def futhark_substract2(self, x_mem_10662, y_mem_10663, sizze_9824,
                         sizze_9825):
    dim_zzero_9828 = (np.int32(0) == sizze_9825)
    dim_zzero_9829 = (np.int32(0) == sizze_9824)
    both_empty_9830 = (dim_zzero_9828 and dim_zzero_9829)
    dim_match_9831 = (sizze_9824 == sizze_9825)
    empty_or_match_9832 = (both_empty_9830 or dim_match_9831)
    empty_or_match_cert_9833 = True
    assert empty_or_match_9832, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:141:3-14\n   #1  GPU.fut:140:1-141:14\n" % ("function arguments of wrong shape",))
    sizze_10135 = sext_i32_i64(sizze_9824)
    segmap_group_sizze_10138 = self.sizes["substract2.segmap_group_size_10137"]
    segmap_group_sizze_10139 = sext_i32_i64(segmap_group_sizze_10138)
    y_10140 = (segmap_group_sizze_10139 - np.int64(1))
    x_10141 = (sizze_10135 + y_10140)
    segmap_usable_groups_64_10143 = squot64(x_10141, segmap_group_sizze_10139)
    segmap_usable_groups_10144 = sext_i64_i32(segmap_usable_groups_64_10143)
    bytes_10664 = (np.int64(4) * sizze_10135)
    mem_10666 = opencl_alloc(self, bytes_10664, "mem_10666")
    if ((1 * (np.long(segmap_usable_groups_10144) * np.long(segmap_group_sizze_10138))) != 0):
      self.segmap_10134_var.set_args(np.int32(sizze_9824), x_mem_10662,
                                     y_mem_10663, mem_10666)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10134_var,
                                 ((np.long(segmap_usable_groups_10144) * np.long(segmap_group_sizze_10138)),),
                                 (np.long(segmap_group_sizze_10138),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10824 = sizze_9824
    out_mem_10823 = mem_10666
    return (out_mem_10823, out_arrsizze_10824)
  def futhark_add2(self, x_mem_10662, y_mem_10663, sizze_9809, sizze_9810):
    dim_zzero_9813 = (np.int32(0) == sizze_9810)
    dim_zzero_9814 = (np.int32(0) == sizze_9809)
    both_empty_9815 = (dim_zzero_9813 and dim_zzero_9814)
    dim_match_9816 = (sizze_9809 == sizze_9810)
    empty_or_match_9817 = (both_empty_9815 or dim_match_9816)
    empty_or_match_cert_9818 = True
    assert empty_or_match_9817, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:105:3-14\n   #1  GPU.fut:104:1-105:14\n" % ("function arguments of wrong shape",))
    sizze_10120 = sext_i32_i64(sizze_9809)
    segmap_group_sizze_10123 = self.sizes["add2.segmap_group_size_10122"]
    segmap_group_sizze_10124 = sext_i32_i64(segmap_group_sizze_10123)
    y_10125 = (segmap_group_sizze_10124 - np.int64(1))
    x_10126 = (sizze_10120 + y_10125)
    segmap_usable_groups_64_10128 = squot64(x_10126, segmap_group_sizze_10124)
    segmap_usable_groups_10129 = sext_i64_i32(segmap_usable_groups_64_10128)
    bytes_10664 = (np.int64(4) * sizze_10120)
    mem_10666 = opencl_alloc(self, bytes_10664, "mem_10666")
    if ((1 * (np.long(segmap_usable_groups_10129) * np.long(segmap_group_sizze_10123))) != 0):
      self.segmap_10119_var.set_args(np.int32(sizze_9809), x_mem_10662,
                                     y_mem_10663, mem_10666)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10119_var,
                                 ((np.long(segmap_usable_groups_10129) * np.long(segmap_group_sizze_10123)),),
                                 (np.long(segmap_group_sizze_10123),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10817 = sizze_9809
    out_mem_10816 = mem_10666
    return (out_mem_10816, out_arrsizze_10817)
  def futhark_lmatmultiply(self, x_mem_10662, sizze_9800, sizze_9801, p_9802):
    sizze_10104 = sext_i32_i64(sizze_9800)
    sizze_10105 = sext_i32_i64(sizze_9801)
    nest_sizze_10107 = (sizze_10104 * sizze_10105)
    segmap_group_sizze_10109 = self.sizes["lmatmultiply.segmap_group_size_10108"]
    segmap_group_sizze_10110 = sext_i32_i64(segmap_group_sizze_10109)
    y_10111 = (segmap_group_sizze_10110 - np.int64(1))
    x_10112 = (nest_sizze_10107 + y_10111)
    segmap_usable_groups_64_10114 = squot64(x_10112, segmap_group_sizze_10110)
    segmap_usable_groups_10115 = sext_i64_i32(segmap_usable_groups_64_10114)
    bytes_10663 = (np.int64(4) * nest_sizze_10107)
    mem_10667 = opencl_alloc(self, bytes_10663, "mem_10667")
    if ((1 * (np.long(segmap_usable_groups_10115) * np.long(segmap_group_sizze_10109))) != 0):
      self.segmap_10103_var.set_args(np.int32(sizze_9800), np.int32(sizze_9801),
                                     np.float32(p_9802), x_mem_10662, mem_10667)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10103_var,
                                 ((np.long(segmap_usable_groups_10115) * np.long(segmap_group_sizze_10109)),),
                                 (np.long(segmap_group_sizze_10109),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10809 = sizze_9800
    out_arrsizze_10810 = sizze_9801
    out_mem_10808 = mem_10667
    return (out_mem_10808, out_arrsizze_10809, out_arrsizze_10810)
  def futhark_multiply2(self, x_mem_10662, y_mem_10663, sizze_9785, sizze_9786):
    dim_zzero_9789 = (np.int32(0) == sizze_9786)
    dim_zzero_9790 = (np.int32(0) == sizze_9785)
    both_empty_9791 = (dim_zzero_9789 and dim_zzero_9790)
    dim_match_9792 = (sizze_9785 == sizze_9786)
    empty_or_match_9793 = (both_empty_9791 or dim_match_9792)
    empty_or_match_cert_9794 = True
    assert empty_or_match_9793, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:71:3-14\n   #1  GPU.fut:70:1-71:14\n" % ("function arguments of wrong shape",))
    sizze_10088 = sext_i32_i64(sizze_9785)
    segmap_group_sizze_10091 = self.sizes["multiply2.segmap_group_size_10090"]
    segmap_group_sizze_10092 = sext_i32_i64(segmap_group_sizze_10091)
    y_10093 = (segmap_group_sizze_10092 - np.int64(1))
    x_10094 = (sizze_10088 + y_10093)
    segmap_usable_groups_64_10096 = squot64(x_10094, segmap_group_sizze_10092)
    segmap_usable_groups_10097 = sext_i64_i32(segmap_usable_groups_64_10096)
    bytes_10664 = (np.int64(4) * sizze_10088)
    mem_10666 = opencl_alloc(self, bytes_10664, "mem_10666")
    if ((1 * (np.long(segmap_usable_groups_10097) * np.long(segmap_group_sizze_10091))) != 0):
      self.segmap_10087_var.set_args(np.int32(sizze_9785), x_mem_10662,
                                     y_mem_10663, mem_10666)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10087_var,
                                 ((np.long(segmap_usable_groups_10097) * np.long(segmap_group_sizze_10091)),),
                                 (np.long(segmap_group_sizze_10091),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10802 = sizze_9785
    out_mem_10801 = mem_10666
    return (out_mem_10801, out_arrsizze_10802)
  def futhark_substract(self, x_mem_10662, sizze_9779, d_9780):
    sizze_10074 = sext_i32_i64(sizze_9779)
    segmap_group_sizze_10077 = self.sizes["substract.segmap_group_size_10076"]
    segmap_group_sizze_10078 = sext_i32_i64(segmap_group_sizze_10077)
    y_10079 = (segmap_group_sizze_10078 - np.int64(1))
    x_10080 = (sizze_10074 + y_10079)
    segmap_usable_groups_64_10082 = squot64(x_10080, segmap_group_sizze_10078)
    segmap_usable_groups_10083 = sext_i64_i32(segmap_usable_groups_64_10082)
    bytes_10663 = (np.int64(4) * sizze_10074)
    mem_10665 = opencl_alloc(self, bytes_10663, "mem_10665")
    if ((1 * (np.long(segmap_usable_groups_10083) * np.long(segmap_group_sizze_10077))) != 0):
      self.segmap_10073_var.set_args(np.int32(sizze_9779), np.float32(d_9780),
                                     x_mem_10662, mem_10665)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10073_var,
                                 ((np.long(segmap_usable_groups_10083) * np.long(segmap_group_sizze_10077)),),
                                 (np.long(segmap_group_sizze_10077),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10795 = sizze_9779
    out_mem_10794 = mem_10665
    return (out_mem_10794, out_arrsizze_10795)
  def futhark_add(self, x_mem_10662, sizze_9773, s_9774):
    sizze_10060 = sext_i32_i64(sizze_9773)
    segmap_group_sizze_10063 = self.sizes["add.segmap_group_size_10062"]
    segmap_group_sizze_10064 = sext_i32_i64(segmap_group_sizze_10063)
    y_10065 = (segmap_group_sizze_10064 - np.int64(1))
    x_10066 = (sizze_10060 + y_10065)
    segmap_usable_groups_64_10068 = squot64(x_10066, segmap_group_sizze_10064)
    segmap_usable_groups_10069 = sext_i64_i32(segmap_usable_groups_64_10068)
    bytes_10663 = (np.int64(4) * sizze_10060)
    mem_10665 = opencl_alloc(self, bytes_10663, "mem_10665")
    if ((1 * (np.long(segmap_usable_groups_10069) * np.long(segmap_group_sizze_10063))) != 0):
      self.segmap_10059_var.set_args(np.int32(sizze_9773), np.float32(s_9774),
                                     x_mem_10662, mem_10665)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10059_var,
                                 ((np.long(segmap_usable_groups_10069) * np.long(segmap_group_sizze_10063)),),
                                 (np.long(segmap_group_sizze_10063),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10788 = sizze_9773
    out_mem_10787 = mem_10665
    return (out_mem_10787, out_arrsizze_10788)
  def futhark_multiply(self, x_mem_10662, sizze_9767, m_9768):
    sizze_10046 = sext_i32_i64(sizze_9767)
    segmap_group_sizze_10049 = self.sizes["multiply.segmap_group_size_10048"]
    segmap_group_sizze_10050 = sext_i32_i64(segmap_group_sizze_10049)
    y_10051 = (segmap_group_sizze_10050 - np.int64(1))
    x_10052 = (sizze_10046 + y_10051)
    segmap_usable_groups_64_10054 = squot64(x_10052, segmap_group_sizze_10050)
    segmap_usable_groups_10055 = sext_i64_i32(segmap_usable_groups_64_10054)
    bytes_10663 = (np.int64(4) * sizze_10046)
    mem_10665 = opencl_alloc(self, bytes_10663, "mem_10665")
    if ((1 * (np.long(segmap_usable_groups_10055) * np.long(segmap_group_sizze_10049))) != 0):
      self.segmap_10045_var.set_args(np.int32(sizze_9767), np.float32(m_9768),
                                     x_mem_10662, mem_10665)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10045_var,
                                 ((np.long(segmap_usable_groups_10055) * np.long(segmap_group_sizze_10049)),),
                                 (np.long(segmap_group_sizze_10049),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10781 = sizze_9767
    out_mem_10780 = mem_10665
    return (out_mem_10780, out_arrsizze_10781)
  def futhark_divide(self, x_mem_10662, sizze_9761, d_9762):
    sizze_10032 = sext_i32_i64(sizze_9761)
    segmap_group_sizze_10035 = self.sizes["divide.segmap_group_size_10034"]
    segmap_group_sizze_10036 = sext_i32_i64(segmap_group_sizze_10035)
    y_10037 = (segmap_group_sizze_10036 - np.int64(1))
    x_10038 = (sizze_10032 + y_10037)
    segmap_usable_groups_64_10040 = squot64(x_10038, segmap_group_sizze_10036)
    segmap_usable_groups_10041 = sext_i64_i32(segmap_usable_groups_64_10040)
    bytes_10663 = (np.int64(4) * sizze_10032)
    mem_10665 = opencl_alloc(self, bytes_10663, "mem_10665")
    if ((1 * (np.long(segmap_usable_groups_10041) * np.long(segmap_group_sizze_10035))) != 0):
      self.segmap_10031_var.set_args(np.int32(sizze_9761), np.float32(d_9762),
                                     x_mem_10662, mem_10665)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10031_var,
                                 ((np.long(segmap_usable_groups_10041) * np.long(segmap_group_sizze_10035)),),
                                 (np.long(segmap_group_sizze_10035),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10774 = sizze_9761
    out_mem_10773 = mem_10665
    return (out_mem_10773, out_arrsizze_10774)
  def futhark_negation(self, x_mem_10662, sizze_9756):
    sizze_10018 = sext_i32_i64(sizze_9756)
    segmap_group_sizze_10021 = self.sizes["negation.segmap_group_size_10020"]
    segmap_group_sizze_10022 = sext_i32_i64(segmap_group_sizze_10021)
    y_10023 = (segmap_group_sizze_10022 - np.int64(1))
    x_10024 = (sizze_10018 + y_10023)
    segmap_usable_groups_64_10026 = squot64(x_10024, segmap_group_sizze_10022)
    segmap_usable_groups_10027 = sext_i64_i32(segmap_usable_groups_64_10026)
    bytes_10663 = (np.int64(4) * sizze_10018)
    mem_10665 = opencl_alloc(self, bytes_10663, "mem_10665")
    if ((1 * (np.long(segmap_usable_groups_10027) * np.long(segmap_group_sizze_10021))) != 0):
      self.segmap_10017_var.set_args(np.int32(sizze_9756), x_mem_10662,
                                     mem_10665)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10017_var,
                                 ((np.long(segmap_usable_groups_10027) * np.long(segmap_group_sizze_10021)),),
                                 (np.long(segmap_group_sizze_10021),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10767 = sizze_9756
    out_mem_10766 = mem_10665
    return (out_mem_10766, out_arrsizze_10767)
  def futhark_exp(self, x_mem_10662, sizze_9751):
    sizze_10004 = sext_i32_i64(sizze_9751)
    segmap_group_sizze_10007 = self.sizes["exp.segmap_group_size_10006"]
    segmap_group_sizze_10008 = sext_i32_i64(segmap_group_sizze_10007)
    y_10009 = (segmap_group_sizze_10008 - np.int64(1))
    x_10010 = (sizze_10004 + y_10009)
    segmap_usable_groups_64_10012 = squot64(x_10010, segmap_group_sizze_10008)
    segmap_usable_groups_10013 = sext_i64_i32(segmap_usable_groups_64_10012)
    bytes_10663 = (np.int64(4) * sizze_10004)
    mem_10665 = opencl_alloc(self, bytes_10663, "mem_10665")
    if ((1 * (np.long(segmap_usable_groups_10013) * np.long(segmap_group_sizze_10007))) != 0):
      self.segmap_10003_var.set_args(np.int32(sizze_9751), x_mem_10662,
                                     mem_10665)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10003_var,
                                 ((np.long(segmap_usable_groups_10013) * np.long(segmap_group_sizze_10007)),),
                                 (np.long(segmap_group_sizze_10007),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10760 = sizze_9751
    out_mem_10759 = mem_10665
    return (out_mem_10759, out_arrsizze_10760)
  def futhark_transp(self, x_mem_10662, sizze_9747, sizze_9748):
    binop_x_10664 = sext_i32_i64(sizze_9748)
    binop_y_10665 = sext_i32_i64(sizze_9747)
    binop_x_10666 = (binop_x_10664 * binop_y_10665)
    bytes_10663 = (np.int64(4) * binop_x_10666)
    mem_10667 = opencl_alloc(self, bytes_10663, "mem_10667")
    self.futhark__map_transpose_f32(mem_10667, np.int32(0), x_mem_10662,
                                    np.int32(0), np.int32(1), sizze_9748,
                                    sizze_9747, (sizze_9748 * sizze_9747),
                                    (sizze_9748 * sizze_9747))
    out_arrsizze_10757 = sizze_9748
    out_arrsizze_10758 = sizze_9747
    out_mem_10756 = mem_10667
    return (out_mem_10756, out_arrsizze_10757, out_arrsizze_10758)
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
  def futhark_arr(self, x_mem_10662, sizze_9744, sizze_9745):
    out_arrsizze_10754 = sizze_9744
    out_arrsizze_10755 = sizze_9745
    out_mem_10753 = x_mem_10662
    return (out_mem_10753, out_arrsizze_10754, out_arrsizze_10755)
  def dot(self, a_mem_10662_ext, b_mem_10663_ext):
    try:
      assert ((type(a_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (a_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9977 = np.int32(a_mem_10662_ext.shape[0])
      sizze_9978 = np.int32(a_mem_10662_ext.shape[1])
      if (type(a_mem_10662_ext) == cl.array.Array):
        a_mem_10662 = a_mem_10662_ext.data
      else:
        a_mem_10662 = opencl_alloc(self, np.int64(a_mem_10662_ext.nbytes),
                                   "a_mem_10662")
        if (np.int64(a_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, a_mem_10662,
                          normaliseArray(a_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(a_mem_10662_ext),
                                                                                                                            a_mem_10662_ext))
    try:
      assert ((type(b_mem_10663_ext) in [np.ndarray,
                                         cl.array.Array]) and (b_mem_10663_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9979 = np.int32(b_mem_10663_ext.shape[0])
      sizze_9980 = np.int32(b_mem_10663_ext.shape[1])
      if (type(b_mem_10663_ext) == cl.array.Array):
        b_mem_10663 = b_mem_10663_ext.data
      else:
        b_mem_10663 = opencl_alloc(self, np.int64(b_mem_10663_ext.nbytes),
                                   "b_mem_10663")
        if (np.int64(b_mem_10663_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, b_mem_10663,
                          normaliseArray(b_mem_10663_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(b_mem_10663_ext),
                                                                                                                            b_mem_10663_ext))
    (out_mem_10886, out_arrsizze_10887,
     out_arrsizze_10888) = self.futhark_dot(a_mem_10662, b_mem_10663,
                                            sizze_9977, sizze_9978, sizze_9979,
                                            sizze_9980)
    return cl.array.Array(self.queue, (out_arrsizze_10887, out_arrsizze_10888),
                          ct.c_float, data=out_mem_10886)
  def matsubstract(self, x_mem_10662_ext, y_mem_10663_ext):
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9948 = np.int32(x_mem_10662_ext.shape[0])
      sizze_9949 = np.int32(x_mem_10662_ext.shape[1])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    try:
      assert ((type(y_mem_10663_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_10663_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9950 = np.int32(y_mem_10663_ext.shape[0])
      sizze_9951 = np.int32(y_mem_10663_ext.shape[1])
      if (type(y_mem_10663_ext) == cl.array.Array):
        y_mem_10663 = y_mem_10663_ext.data
      else:
        y_mem_10663 = opencl_alloc(self, np.int64(y_mem_10663_ext.nbytes),
                                   "y_mem_10663")
        if (np.int64(y_mem_10663_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_10663,
                          normaliseArray(y_mem_10663_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(y_mem_10663_ext),
                                                                                                                            y_mem_10663_ext))
    (out_mem_10878, out_arrsizze_10879,
     out_arrsizze_10880) = self.futhark_matsubstract(x_mem_10662, y_mem_10663,
                                                     sizze_9948, sizze_9949,
                                                     sizze_9950, sizze_9951)
    return cl.array.Array(self.queue, (out_arrsizze_10879, out_arrsizze_10880),
                          ct.c_float, data=out_mem_10878)
  def matadd(self, x_mem_10662_ext, y_mem_10663_ext):
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9919 = np.int32(x_mem_10662_ext.shape[0])
      sizze_9920 = np.int32(x_mem_10662_ext.shape[1])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    try:
      assert ((type(y_mem_10663_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_10663_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9921 = np.int32(y_mem_10663_ext.shape[0])
      sizze_9922 = np.int32(y_mem_10663_ext.shape[1])
      if (type(y_mem_10663_ext) == cl.array.Array):
        y_mem_10663 = y_mem_10663_ext.data
      else:
        y_mem_10663 = opencl_alloc(self, np.int64(y_mem_10663_ext.nbytes),
                                   "y_mem_10663")
        if (np.int64(y_mem_10663_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_10663,
                          normaliseArray(y_mem_10663_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(y_mem_10663_ext),
                                                                                                                            y_mem_10663_ext))
    (out_mem_10870, out_arrsizze_10871,
     out_arrsizze_10872) = self.futhark_matadd(x_mem_10662, y_mem_10663,
                                               sizze_9919, sizze_9920,
                                               sizze_9921, sizze_9922)
    return cl.array.Array(self.queue, (out_arrsizze_10871, out_arrsizze_10872),
                          ct.c_float, data=out_mem_10870)
  def matmultiply(self, x_mem_10662_ext, y_mem_10663_ext):
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9890 = np.int32(x_mem_10662_ext.shape[0])
      sizze_9891 = np.int32(x_mem_10662_ext.shape[1])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    try:
      assert ((type(y_mem_10663_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_10663_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9892 = np.int32(y_mem_10663_ext.shape[0])
      sizze_9893 = np.int32(y_mem_10663_ext.shape[1])
      if (type(y_mem_10663_ext) == cl.array.Array):
        y_mem_10663 = y_mem_10663_ext.data
      else:
        y_mem_10663 = opencl_alloc(self, np.int64(y_mem_10663_ext.nbytes),
                                   "y_mem_10663")
        if (np.int64(y_mem_10663_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_10663,
                          normaliseArray(y_mem_10663_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(y_mem_10663_ext),
                                                                                                                            y_mem_10663_ext))
    (out_mem_10862, out_arrsizze_10863,
     out_arrsizze_10864) = self.futhark_matmultiply(x_mem_10662, y_mem_10663,
                                                    sizze_9890, sizze_9891,
                                                    sizze_9892, sizze_9893)
    return cl.array.Array(self.queue, (out_arrsizze_10863, out_arrsizze_10864),
                          ct.c_float, data=out_mem_10862)
  def lvecmul(self, u_mem_10662_ext, b_mem_10663_ext):
    try:
      assert ((type(u_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (u_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9868 = np.int32(u_mem_10662_ext.shape[0])
      if (type(u_mem_10662_ext) == cl.array.Array):
        u_mem_10662 = u_mem_10662_ext.data
      else:
        u_mem_10662 = opencl_alloc(self, np.int64(u_mem_10662_ext.nbytes),
                                   "u_mem_10662")
        if (np.int64(u_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, u_mem_10662,
                          normaliseArray(u_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(u_mem_10662_ext),
                                                                                                                            u_mem_10662_ext))
    try:
      assert ((type(b_mem_10663_ext) in [np.ndarray,
                                         cl.array.Array]) and (b_mem_10663_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9869 = np.int32(b_mem_10663_ext.shape[0])
      sizze_9870 = np.int32(b_mem_10663_ext.shape[1])
      if (type(b_mem_10663_ext) == cl.array.Array):
        b_mem_10663 = b_mem_10663_ext.data
      else:
        b_mem_10663 = opencl_alloc(self, np.int64(b_mem_10663_ext.nbytes),
                                   "b_mem_10663")
        if (np.int64(b_mem_10663_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, b_mem_10663,
                          normaliseArray(b_mem_10663_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(b_mem_10663_ext),
                                                                                                                            b_mem_10663_ext))
    (out_mem_10854, out_arrsizze_10855) = self.futhark_lvecmul(u_mem_10662,
                                                               b_mem_10663,
                                                               sizze_9868,
                                                               sizze_9869,
                                                               sizze_9870)
    return cl.array.Array(self.queue, (out_arrsizze_10855,), ct.c_float,
                          data=out_mem_10854)
  def sigmoid(self, x_mem_10662_ext):
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9857 = np.int32(x_mem_10662_ext.shape[0])
      sizze_9858 = np.int32(x_mem_10662_ext.shape[1])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    (out_mem_10846, out_arrsizze_10847,
     out_arrsizze_10848) = self.futhark_sigmoid(x_mem_10662, sizze_9857,
                                                sizze_9858)
    return cl.array.Array(self.queue, (out_arrsizze_10847, out_arrsizze_10848),
                          ct.c_float, data=out_mem_10846)
  def lmatsubstract(self, d_9850_ext, x_mem_10662_ext):
    try:
      d_9850 = np.float32(ct.c_float(d_9850_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(d_9850_ext),
                                                                                                                            d_9850_ext))
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9848 = np.int32(x_mem_10662_ext.shape[0])
      sizze_9849 = np.int32(x_mem_10662_ext.shape[1])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    (out_mem_10838, out_arrsizze_10839,
     out_arrsizze_10840) = self.futhark_lmatsubstract(x_mem_10662, sizze_9848,
                                                      sizze_9849, d_9850)
    return cl.array.Array(self.queue, (out_arrsizze_10839, out_arrsizze_10840),
                          ct.c_float, data=out_mem_10838)
  def lmatadd(self, s_9841_ext, x_mem_10662_ext):
    try:
      s_9841 = np.float32(ct.c_float(s_9841_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(s_9841_ext),
                                                                                                                            s_9841_ext))
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9839 = np.int32(x_mem_10662_ext.shape[0])
      sizze_9840 = np.int32(x_mem_10662_ext.shape[1])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    (out_mem_10830, out_arrsizze_10831,
     out_arrsizze_10832) = self.futhark_lmatadd(x_mem_10662, sizze_9839,
                                                sizze_9840, s_9841)
    return cl.array.Array(self.queue, (out_arrsizze_10831, out_arrsizze_10832),
                          ct.c_float, data=out_mem_10830)
  def substract2(self, x_mem_10662_ext, y_mem_10663_ext):
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9824 = np.int32(x_mem_10662_ext.shape[0])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    try:
      assert ((type(y_mem_10663_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_10663_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9825 = np.int32(y_mem_10663_ext.shape[0])
      if (type(y_mem_10663_ext) == cl.array.Array):
        y_mem_10663 = y_mem_10663_ext.data
      else:
        y_mem_10663 = opencl_alloc(self, np.int64(y_mem_10663_ext.nbytes),
                                   "y_mem_10663")
        if (np.int64(y_mem_10663_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_10663,
                          normaliseArray(y_mem_10663_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(y_mem_10663_ext),
                                                                                                                            y_mem_10663_ext))
    (out_mem_10823, out_arrsizze_10824) = self.futhark_substract2(x_mem_10662,
                                                                  y_mem_10663,
                                                                  sizze_9824,
                                                                  sizze_9825)
    return cl.array.Array(self.queue, (out_arrsizze_10824,), ct.c_float,
                          data=out_mem_10823)
  def add2(self, x_mem_10662_ext, y_mem_10663_ext):
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9809 = np.int32(x_mem_10662_ext.shape[0])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    try:
      assert ((type(y_mem_10663_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_10663_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9810 = np.int32(y_mem_10663_ext.shape[0])
      if (type(y_mem_10663_ext) == cl.array.Array):
        y_mem_10663 = y_mem_10663_ext.data
      else:
        y_mem_10663 = opencl_alloc(self, np.int64(y_mem_10663_ext.nbytes),
                                   "y_mem_10663")
        if (np.int64(y_mem_10663_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_10663,
                          normaliseArray(y_mem_10663_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(y_mem_10663_ext),
                                                                                                                            y_mem_10663_ext))
    (out_mem_10816, out_arrsizze_10817) = self.futhark_add2(x_mem_10662,
                                                            y_mem_10663,
                                                            sizze_9809,
                                                            sizze_9810)
    return cl.array.Array(self.queue, (out_arrsizze_10817,), ct.c_float,
                          data=out_mem_10816)
  def lmatmultiply(self, p_9802_ext, x_mem_10662_ext):
    try:
      p_9802 = np.float32(ct.c_float(p_9802_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(p_9802_ext),
                                                                                                                            p_9802_ext))
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9800 = np.int32(x_mem_10662_ext.shape[0])
      sizze_9801 = np.int32(x_mem_10662_ext.shape[1])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    (out_mem_10808, out_arrsizze_10809,
     out_arrsizze_10810) = self.futhark_lmatmultiply(x_mem_10662, sizze_9800,
                                                     sizze_9801, p_9802)
    return cl.array.Array(self.queue, (out_arrsizze_10809, out_arrsizze_10810),
                          ct.c_float, data=out_mem_10808)
  def multiply2(self, x_mem_10662_ext, y_mem_10663_ext):
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9785 = np.int32(x_mem_10662_ext.shape[0])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    try:
      assert ((type(y_mem_10663_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_10663_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9786 = np.int32(y_mem_10663_ext.shape[0])
      if (type(y_mem_10663_ext) == cl.array.Array):
        y_mem_10663 = y_mem_10663_ext.data
      else:
        y_mem_10663 = opencl_alloc(self, np.int64(y_mem_10663_ext.nbytes),
                                   "y_mem_10663")
        if (np.int64(y_mem_10663_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_10663,
                          normaliseArray(y_mem_10663_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(y_mem_10663_ext),
                                                                                                                            y_mem_10663_ext))
    (out_mem_10801, out_arrsizze_10802) = self.futhark_multiply2(x_mem_10662,
                                                                 y_mem_10663,
                                                                 sizze_9785,
                                                                 sizze_9786)
    return cl.array.Array(self.queue, (out_arrsizze_10802,), ct.c_float,
                          data=out_mem_10801)
  def substract(self, d_9780_ext, x_mem_10662_ext):
    try:
      d_9780 = np.float32(ct.c_float(d_9780_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(d_9780_ext),
                                                                                                                            d_9780_ext))
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9779 = np.int32(x_mem_10662_ext.shape[0])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    (out_mem_10794, out_arrsizze_10795) = self.futhark_substract(x_mem_10662,
                                                                 sizze_9779,
                                                                 d_9780)
    return cl.array.Array(self.queue, (out_arrsizze_10795,), ct.c_float,
                          data=out_mem_10794)
  def add(self, s_9774_ext, x_mem_10662_ext):
    try:
      s_9774 = np.float32(ct.c_float(s_9774_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(s_9774_ext),
                                                                                                                            s_9774_ext))
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9773 = np.int32(x_mem_10662_ext.shape[0])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    (out_mem_10787, out_arrsizze_10788) = self.futhark_add(x_mem_10662,
                                                           sizze_9773, s_9774)
    return cl.array.Array(self.queue, (out_arrsizze_10788,), ct.c_float,
                          data=out_mem_10787)
  def multiply(self, m_9768_ext, x_mem_10662_ext):
    try:
      m_9768 = np.float32(ct.c_float(m_9768_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(m_9768_ext),
                                                                                                                            m_9768_ext))
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9767 = np.int32(x_mem_10662_ext.shape[0])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    (out_mem_10780, out_arrsizze_10781) = self.futhark_multiply(x_mem_10662,
                                                                sizze_9767,
                                                                m_9768)
    return cl.array.Array(self.queue, (out_arrsizze_10781,), ct.c_float,
                          data=out_mem_10780)
  def divide(self, d_9762_ext, x_mem_10662_ext):
    try:
      d_9762 = np.float32(ct.c_float(d_9762_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(d_9762_ext),
                                                                                                                            d_9762_ext))
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9761 = np.int32(x_mem_10662_ext.shape[0])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    (out_mem_10773, out_arrsizze_10774) = self.futhark_divide(x_mem_10662,
                                                              sizze_9761,
                                                              d_9762)
    return cl.array.Array(self.queue, (out_arrsizze_10774,), ct.c_float,
                          data=out_mem_10773)
  def negation(self, x_mem_10662_ext):
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9756 = np.int32(x_mem_10662_ext.shape[0])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    (out_mem_10766, out_arrsizze_10767) = self.futhark_negation(x_mem_10662,
                                                                sizze_9756)
    return cl.array.Array(self.queue, (out_arrsizze_10767,), ct.c_float,
                          data=out_mem_10766)
  def exp(self, x_mem_10662_ext):
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9751 = np.int32(x_mem_10662_ext.shape[0])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    (out_mem_10759, out_arrsizze_10760) = self.futhark_exp(x_mem_10662,
                                                           sizze_9751)
    return cl.array.Array(self.queue, (out_arrsizze_10760,), ct.c_float,
                          data=out_mem_10759)
  def transp(self, x_mem_10662_ext):
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9747 = np.int32(x_mem_10662_ext.shape[0])
      sizze_9748 = np.int32(x_mem_10662_ext.shape[1])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    (out_mem_10756, out_arrsizze_10757,
     out_arrsizze_10758) = self.futhark_transp(x_mem_10662, sizze_9747,
                                               sizze_9748)
    return cl.array.Array(self.queue, (out_arrsizze_10757, out_arrsizze_10758),
                          ct.c_float, data=out_mem_10756)
  def arr(self, x_mem_10662_ext):
    try:
      assert ((type(x_mem_10662_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10662_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9744 = np.int32(x_mem_10662_ext.shape[0])
      sizze_9745 = np.int32(x_mem_10662_ext.shape[1])
      if (type(x_mem_10662_ext) == cl.array.Array):
        x_mem_10662 = x_mem_10662_ext.data
      else:
        x_mem_10662 = opencl_alloc(self, np.int64(x_mem_10662_ext.nbytes),
                                   "x_mem_10662")
        if (np.int64(x_mem_10662_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10662,
                          normaliseArray(x_mem_10662_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10662_ext),
                                                                                                                            x_mem_10662_ext))
    (out_mem_10753, out_arrsizze_10754,
     out_arrsizze_10755) = self.futhark_arr(x_mem_10662, sizze_9744, sizze_9745)
    return cl.array.Array(self.queue, (out_arrsizze_10754, out_arrsizze_10755),
                          ct.c_float, data=out_mem_10753)