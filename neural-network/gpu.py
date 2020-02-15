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
__kernel void segmap_7697(int32_t sizze_7521, __global
                          unsigned char *x_mem_8284, __global
                          unsigned char *mem_8287)
{
    const int32_t segmap_group_sizze_7701 = expzisegmap_group_sizze_7700;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_8372;
    int32_t local_tid_8373;
    int32_t group_sizze_8376;
    int32_t wave_sizze_8375;
    int32_t group_tid_8374;
    
    global_tid_8372 = get_global_id(0);
    local_tid_8373 = get_local_id(0);
    group_sizze_8376 = get_local_size(0);
    wave_sizze_8375 = LOCKSTEP_WIDTH;
    group_tid_8374 = get_group_id(0);
    
    int32_t phys_tid_7697 = global_tid_8372;
    int32_t gtid_7696 = group_tid_8374 * segmap_group_sizze_7701 +
            local_tid_8373;
    
    if (slt32(gtid_7696, sizze_7521)) {
        float x_7708 = ((__global float *) x_mem_8284)[gtid_7696];
        float res_7709 = fpow32(2.7182817F, x_7708);
        
        ((__global float *) mem_8287)[gtid_7696] = res_7709;
    }
}
__kernel void segmap_7711(int32_t sizze_7526, __global
                          unsigned char *x_mem_8284, __global
                          unsigned char *mem_8287)
{
    const int32_t segmap_group_sizze_7715 = negationzisegmap_group_sizze_7714;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_8379;
    int32_t local_tid_8380;
    int32_t group_sizze_8383;
    int32_t wave_sizze_8382;
    int32_t group_tid_8381;
    
    global_tid_8379 = get_global_id(0);
    local_tid_8380 = get_local_id(0);
    group_sizze_8383 = get_local_size(0);
    wave_sizze_8382 = LOCKSTEP_WIDTH;
    group_tid_8381 = get_group_id(0);
    
    int32_t phys_tid_7711 = global_tid_8379;
    int32_t gtid_7710 = group_tid_8381 * segmap_group_sizze_7715 +
            local_tid_8380;
    
    if (slt32(gtid_7710, sizze_7526)) {
        float x_7722 = ((__global float *) x_mem_8284)[gtid_7710];
        float res_7723 = 0.0F - x_7722;
        
        ((__global float *) mem_8287)[gtid_7710] = res_7723;
    }
}
__kernel void segmap_7725(int32_t sizze_7531, float div_7532, __global
                          unsigned char *x_mem_8284, __global
                          unsigned char *mem_8287)
{
    const int32_t segmap_group_sizze_7729 = dividezisegmap_group_sizze_7728;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_8386;
    int32_t local_tid_8387;
    int32_t group_sizze_8390;
    int32_t wave_sizze_8389;
    int32_t group_tid_8388;
    
    global_tid_8386 = get_global_id(0);
    local_tid_8387 = get_local_id(0);
    group_sizze_8390 = get_local_size(0);
    wave_sizze_8389 = LOCKSTEP_WIDTH;
    group_tid_8388 = get_group_id(0);
    
    int32_t phys_tid_7725 = global_tid_8386;
    int32_t gtid_7724 = group_tid_8388 * segmap_group_sizze_7729 +
            local_tid_8387;
    
    if (slt32(gtid_7724, sizze_7531)) {
        float x_7736 = ((__global float *) x_mem_8284)[gtid_7724];
        float res_7737 = div_7532 / x_7736;
        
        ((__global float *) mem_8287)[gtid_7724] = res_7737;
    }
}
__kernel void segmap_7739(int32_t sizze_7537, float m_7539, __global
                          unsigned char *x_mem_8284, __global
                          unsigned char *mem_8287)
{
    const int32_t segmap_group_sizze_7743 = multiplyzisegmap_group_sizze_7742;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_8393;
    int32_t local_tid_8394;
    int32_t group_sizze_8397;
    int32_t wave_sizze_8396;
    int32_t group_tid_8395;
    
    global_tid_8393 = get_global_id(0);
    local_tid_8394 = get_local_id(0);
    group_sizze_8397 = get_local_size(0);
    wave_sizze_8396 = LOCKSTEP_WIDTH;
    group_tid_8395 = get_group_id(0);
    
    int32_t phys_tid_7739 = global_tid_8393;
    int32_t gtid_7738 = group_tid_8395 * segmap_group_sizze_7743 +
            local_tid_8394;
    
    if (slt32(gtid_7738, sizze_7537)) {
        float x_7750 = ((__global float *) x_mem_8284)[gtid_7738];
        float res_7751 = m_7539 * x_7750;
        
        ((__global float *) mem_8287)[gtid_7738] = res_7751;
    }
}
__kernel void segmap_7753(int32_t sizze_7543, float s_7545, __global
                          unsigned char *x_mem_8284, __global
                          unsigned char *mem_8287)
{
    const int32_t segmap_group_sizze_7757 = addzisegmap_group_sizze_7756;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_8400;
    int32_t local_tid_8401;
    int32_t group_sizze_8404;
    int32_t wave_sizze_8403;
    int32_t group_tid_8402;
    
    global_tid_8400 = get_global_id(0);
    local_tid_8401 = get_local_id(0);
    group_sizze_8404 = get_local_size(0);
    wave_sizze_8403 = LOCKSTEP_WIDTH;
    group_tid_8402 = get_group_id(0);
    
    int32_t phys_tid_7753 = global_tid_8400;
    int32_t gtid_7752 = group_tid_8402 * segmap_group_sizze_7757 +
            local_tid_8401;
    
    if (slt32(gtid_7752, sizze_7543)) {
        float x_7764 = ((__global float *) x_mem_8284)[gtid_7752];
        float res_7765 = s_7545 + x_7764;
        
        ((__global float *) mem_8287)[gtid_7752] = res_7765;
    }
}
__kernel void segmap_7767(int32_t sizze_7549, float d_7550, __global
                          unsigned char *x_mem_8284, __global
                          unsigned char *mem_8287)
{
    const int32_t segmap_group_sizze_7771 = substractzisegmap_group_sizze_7770;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_8407;
    int32_t local_tid_8408;
    int32_t group_sizze_8411;
    int32_t wave_sizze_8410;
    int32_t group_tid_8409;
    
    global_tid_8407 = get_global_id(0);
    local_tid_8408 = get_local_id(0);
    group_sizze_8411 = get_local_size(0);
    wave_sizze_8410 = LOCKSTEP_WIDTH;
    group_tid_8409 = get_group_id(0);
    
    int32_t phys_tid_7767 = global_tid_8407;
    int32_t gtid_7766 = group_tid_8409 * segmap_group_sizze_7771 +
            local_tid_8408;
    
    if (slt32(gtid_7766, sizze_7549)) {
        float x_7778 = ((__global float *) x_mem_8284)[gtid_7766];
        float res_7779 = d_7550 - x_7778;
        
        ((__global float *) mem_8287)[gtid_7766] = res_7779;
    }
}
__kernel void segmap_7781(int32_t sizze_7555, __global
                          unsigned char *x_mem_8284, __global
                          unsigned char *y_mem_8285, __global
                          unsigned char *mem_8288)
{
    const int32_t segmap_group_sizze_7785 = multiply2zisegmap_group_sizze_7784;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_8414;
    int32_t local_tid_8415;
    int32_t group_sizze_8418;
    int32_t wave_sizze_8417;
    int32_t group_tid_8416;
    
    global_tid_8414 = get_global_id(0);
    local_tid_8415 = get_local_id(0);
    group_sizze_8418 = get_local_size(0);
    wave_sizze_8417 = LOCKSTEP_WIDTH;
    group_tid_8416 = get_group_id(0);
    
    int32_t phys_tid_7781 = global_tid_8414;
    int32_t gtid_7780 = group_tid_8416 * segmap_group_sizze_7785 +
            local_tid_8415;
    
    if (slt32(gtid_7780, sizze_7555)) {
        float x_7792 = ((__global float *) x_mem_8284)[gtid_7780];
        float x_7793 = ((__global float *) y_mem_8285)[gtid_7780];
        float res_7794 = x_7792 * x_7793;
        
        ((__global float *) mem_8288)[gtid_7780] = res_7794;
    }
}
__kernel void segmap_7796(int32_t sizze_7570, __global
                          unsigned char *x_mem_8284, __global
                          unsigned char *y_mem_8285, __global
                          unsigned char *mem_8288)
{
    const int32_t segmap_group_sizze_7800 = add2zisegmap_group_sizze_7799;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_8421;
    int32_t local_tid_8422;
    int32_t group_sizze_8425;
    int32_t wave_sizze_8424;
    int32_t group_tid_8423;
    
    global_tid_8421 = get_global_id(0);
    local_tid_8422 = get_local_id(0);
    group_sizze_8425 = get_local_size(0);
    wave_sizze_8424 = LOCKSTEP_WIDTH;
    group_tid_8423 = get_group_id(0);
    
    int32_t phys_tid_7796 = global_tid_8421;
    int32_t gtid_7795 = group_tid_8423 * segmap_group_sizze_7800 +
            local_tid_8422;
    
    if (slt32(gtid_7795, sizze_7570)) {
        float x_7807 = ((__global float *) x_mem_8284)[gtid_7795];
        float x_7808 = ((__global float *) y_mem_8285)[gtid_7795];
        float res_7809 = x_7807 + x_7808;
        
        ((__global float *) mem_8288)[gtid_7795] = res_7809;
    }
}
__kernel void segmap_7811(int32_t sizze_7585, __global
                          unsigned char *x_mem_8284, __global
                          unsigned char *y_mem_8285, __global
                          unsigned char *mem_8288)
{
    const int32_t segmap_group_sizze_7815 = substract2zisegmap_group_sizze_7814;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_8428;
    int32_t local_tid_8429;
    int32_t group_sizze_8432;
    int32_t wave_sizze_8431;
    int32_t group_tid_8430;
    
    global_tid_8428 = get_global_id(0);
    local_tid_8429 = get_local_id(0);
    group_sizze_8432 = get_local_size(0);
    wave_sizze_8431 = LOCKSTEP_WIDTH;
    group_tid_8430 = get_group_id(0);
    
    int32_t phys_tid_7811 = global_tid_8428;
    int32_t gtid_7810 = group_tid_8430 * segmap_group_sizze_7815 +
            local_tid_8429;
    
    if (slt32(gtid_7810, sizze_7585)) {
        float x_7822 = ((__global float *) x_mem_8284)[gtid_7810];
        float x_7823 = ((__global float *) y_mem_8285)[gtid_7810];
        float res_7824 = x_7822 - x_7823;
        
        ((__global float *) mem_8288)[gtid_7810] = res_7824;
    }
}
__kernel void segmap_7827(int32_t sizze_7600, int32_t sizze_7601, float d_7602,
                          __global unsigned char *x_mem_8284, __global
                          unsigned char *mem_8289)
{
    const int32_t segmap_group_sizze_7833 =
                  lmatsubstractzisegmap_group_sizze_7832;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_8436;
    int32_t local_tid_8437;
    int32_t group_sizze_8440;
    int32_t wave_sizze_8439;
    int32_t group_tid_8438;
    
    global_tid_8436 = get_global_id(0);
    local_tid_8437 = get_local_id(0);
    group_sizze_8440 = get_local_size(0);
    wave_sizze_8439 = LOCKSTEP_WIDTH;
    group_tid_8438 = get_group_id(0);
    
    int32_t phys_tid_7827 = global_tid_8436;
    int32_t gtid_7825 = squot32(group_tid_8438 * segmap_group_sizze_7833 +
                                local_tid_8437, sizze_7601);
    int32_t gtid_7826;
    
    gtid_7826 = group_tid_8438 * segmap_group_sizze_7833 + local_tid_8437 -
        squot32(group_tid_8438 * segmap_group_sizze_7833 + local_tid_8437,
                sizze_7601) * sizze_7601;
    if (slt32(gtid_7825, sizze_7600) && slt32(gtid_7826, sizze_7601)) {
        float x_7840 = ((__global float *) x_mem_8284)[gtid_7825 * sizze_7601 +
                                                       gtid_7826];
        float res_7841 = d_7602 - x_7840;
        
        ((__global float *) mem_8289)[gtid_7825 * sizze_7601 + gtid_7826] =
            res_7841;
    }
}
__kernel void segmap_7844(int32_t sizze_7609, int32_t sizze_7610, __global
                          unsigned char *x_mem_8284, __global
                          unsigned char *mem_8289)
{
    const int32_t segmap_group_sizze_7850 = sigmoidzisegmap_group_sizze_7849;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_8444;
    int32_t local_tid_8445;
    int32_t group_sizze_8448;
    int32_t wave_sizze_8447;
    int32_t group_tid_8446;
    
    global_tid_8444 = get_global_id(0);
    local_tid_8445 = get_local_id(0);
    group_sizze_8448 = get_local_size(0);
    wave_sizze_8447 = LOCKSTEP_WIDTH;
    group_tid_8446 = get_group_id(0);
    
    int32_t phys_tid_7844 = global_tid_8444;
    int32_t gtid_7842 = squot32(group_tid_8446 * segmap_group_sizze_7850 +
                                local_tid_8445, sizze_7610);
    int32_t gtid_7843;
    
    gtid_7843 = group_tid_8446 * segmap_group_sizze_7850 + local_tid_8445 -
        squot32(group_tid_8446 * segmap_group_sizze_7850 + local_tid_8445,
                sizze_7610) * sizze_7610;
    if (slt32(gtid_7842, sizze_7609) && slt32(gtid_7843, sizze_7610)) {
        float x_7857 = ((__global float *) x_mem_8284)[gtid_7842 * sizze_7610 +
                                                       gtid_7843];
        float res_7858 = 0.0F - x_7857;
        float res_7859 = fpow32(2.7182817F, res_7858);
        float res_7860 = 1.0F + res_7859;
        float res_7861 = 1.0F / res_7860;
        
        ((__global float *) mem_8289)[gtid_7842 * sizze_7610 + gtid_7843] =
            res_7861;
    }
}
__kernel void segmap_7864(int32_t sizze_7620, int32_t sizze_7621,
                          int32_t sizze_7622, __global
                          unsigned char *u_mem_8284, __global
                          unsigned char *b_mem_8285, __global
                          unsigned char *mem_8288)
{
    const int32_t segmap_group_sizze_7868 = lvecmulzisegmap_group_sizze_7867;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_8451;
    int32_t local_tid_8452;
    int32_t group_sizze_8455;
    int32_t wave_sizze_8454;
    int32_t group_tid_8453;
    
    global_tid_8451 = get_global_id(0);
    local_tid_8452 = get_local_id(0);
    group_sizze_8455 = get_local_size(0);
    wave_sizze_8454 = LOCKSTEP_WIDTH;
    group_tid_8453 = get_group_id(0);
    
    int32_t phys_tid_7864 = global_tid_8451;
    int32_t gtid_7863 = group_tid_8453 * segmap_group_sizze_7868 +
            local_tid_8452;
    
    if (slt32(gtid_7863, sizze_7622)) {
        int32_t binop_x_8262 = sizze_7620 * gtid_7863;
        float res_7876;
        float redout_8256 = 0.0F;
        
        for (int32_t i_8257 = 0; i_8257 < sizze_7620; i_8257++) {
            float x_7880 = ((__global float *) u_mem_8284)[i_8257];
            int32_t binop_x_8263 = i_8257 + binop_x_8262;
            int32_t new_index_8264 = squot32(binop_x_8263, sizze_7621);
            int32_t binop_y_8270 = sizze_7621 * new_index_8264;
            int32_t new_index_8271 = binop_x_8263 - binop_y_8270;
            float x_7881 = ((__global float *) b_mem_8285)[new_index_8271 *
                                                           sizze_7622 +
                                                           new_index_8264];
            float res_7882 = x_7880 * x_7881;
            float res_7879 = res_7882 + redout_8256;
            float redout_tmp_8456 = res_7879;
            
            redout_8256 = redout_tmp_8456;
        }
        res_7876 = redout_8256;
        ((__global float *) mem_8288)[gtid_7863] = res_7876;
    }
}
__kernel void segmap_7886(int32_t sizze_7642, int32_t sizze_7643,
                          int32_t sizze_7645, __global
                          unsigned char *x_mem_8284, __global
                          unsigned char *y_mem_8285, __global
                          unsigned char *mem_8290)
{
    const int32_t segmap_group_sizze_7892 =
                  matsubstractzisegmap_group_sizze_7891;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_8460;
    int32_t local_tid_8461;
    int32_t group_sizze_8464;
    int32_t wave_sizze_8463;
    int32_t group_tid_8462;
    
    global_tid_8460 = get_global_id(0);
    local_tid_8461 = get_local_id(0);
    group_sizze_8464 = get_local_size(0);
    wave_sizze_8463 = LOCKSTEP_WIDTH;
    group_tid_8462 = get_group_id(0);
    
    int32_t phys_tid_7886 = global_tid_8460;
    int32_t gtid_7884 = squot32(group_tid_8462 * segmap_group_sizze_7892 +
                                local_tid_8461, sizze_7643);
    int32_t gtid_7885;
    
    gtid_7885 = group_tid_8462 * segmap_group_sizze_7892 + local_tid_8461 -
        squot32(group_tid_8462 * segmap_group_sizze_7892 + local_tid_8461,
                sizze_7643) * sizze_7643;
    if (slt32(gtid_7884, sizze_7642) && slt32(gtid_7885, sizze_7643)) {
        float x_7899 = ((__global float *) x_mem_8284)[gtid_7884 * sizze_7643 +
                                                       gtid_7885];
        int32_t binop_x_7953 = sizze_7643 * gtid_7884;
        int32_t binop_x_7954 = gtid_7885 + binop_x_7953;
        int32_t new_index_7955 = squot32(binop_x_7954, sizze_7645);
        int32_t binop_y_7961 = sizze_7645 * new_index_7955;
        int32_t new_index_7962 = binop_x_7954 - binop_y_7961;
        float x_7900 = ((__global float *) y_mem_8285)[new_index_7955 *
                                                       sizze_7645 +
                                                       new_index_7962];
        float res_7901 = x_7899 - x_7900;
        
        ((__global float *) mem_8290)[gtid_7884 * sizze_7643 + gtid_7885] =
            res_7901;
    }
}
__kernel void segmap_intragroup_7976(__local volatile
                                     int64_t *mem_8294_backing_aligned_0,
                                     __local volatile
                                     int64_t *mem_8299_backing_aligned_1,
                                     __local volatile
                                     int64_t *mem_8309_backing_aligned_2,
                                     __local volatile
                                     int64_t *mem_8314_backing_aligned_3,
                                     int32_t sizze_7671, int32_t sizze_7672,
                                     int32_t sizze_7674,
                                     int32_t num_groups_y_7974,
                                     int32_t num_whole_tiles_7977,
                                     int32_t residual_input_8110,
                                     unsigned char cond_8111, __global
                                     unsigned char *a_mem_8284, __global
                                     unsigned char *b_mem_8285, __global
                                     unsigned char *mem_8324)
{
    const int32_t tile_sizze_7967 = matmulzitile_sizze_7966;
    const int32_t group_sizze_7968 = matmulzitile_sizze_7966 *
                  matmulzitile_sizze_7966;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_8294_backing_0 = (__local volatile
                                                          char *) mem_8294_backing_aligned_0;
    __local volatile char *restrict mem_8299_backing_1 = (__local volatile
                                                          char *) mem_8299_backing_aligned_1;
    __local volatile char *restrict mem_8309_backing_2 = (__local volatile
                                                          char *) mem_8309_backing_aligned_2;
    __local volatile char *restrict mem_8314_backing_3 = (__local volatile
                                                          char *) mem_8314_backing_aligned_3;
    int32_t global_tid_8468;
    int32_t local_tid_8469;
    int32_t group_sizze_8472;
    int32_t wave_sizze_8471;
    int32_t group_tid_8470;
    
    global_tid_8468 = get_global_id(0);
    local_tid_8469 = get_local_id(0);
    group_sizze_8472 = get_local_size(0);
    wave_sizze_8471 = LOCKSTEP_WIDTH;
    group_tid_8470 = get_group_id(0);
    
    int32_t gid_flat_7976 = group_tid_8470;
    int32_t gid_x_7964 = squot32(group_tid_8470, num_groups_y_7974);
    int32_t gid_y_7965;
    
    gid_y_7965 = group_tid_8470 - squot32(group_tid_8470, num_groups_y_7974) *
        num_groups_y_7974;
    
    float mem_8289;
    int32_t ltid_x_7993 = squot32(local_tid_8469, tile_sizze_7967);
    int32_t ltid_y_7994;
    
    ltid_y_7994 = local_tid_8469 - squot32(local_tid_8469, tile_sizze_7967) *
        tile_sizze_7967;
    
    int32_t ltid_flat_7995;
    
    ltid_flat_7995 = local_tid_8469;
    if (slt32(ltid_x_7993, tile_sizze_7967) && slt32(ltid_y_7994,
                                                     tile_sizze_7967)) {
        mem_8289 = 0.0F;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t binop_x_8072 = gid_x_7964 * tile_sizze_7967;
    int32_t binop_x_8074 = gid_y_7965 * tile_sizze_7967;
    __local char *mem_8294;
    
    mem_8294 = (__local char *) mem_8294_backing_0;
    
    __local char *mem_8299;
    
    mem_8299 = (__local char *) mem_8299_backing_1;
    
    float mem_8303;
    
    for (int32_t tile_id_8004 = 0; tile_id_8004 < num_whole_tiles_7977;
         tile_id_8004++) {
        int32_t binop_x_8068 = tile_sizze_7967 * tile_id_8004;
        int32_t ltid_x_8005 = squot32(local_tid_8469, tile_sizze_7967);
        int32_t ltid_y_8006;
        
        ltid_y_8006 = local_tid_8469 - squot32(local_tid_8469,
                                               tile_sizze_7967) *
            tile_sizze_7967;
        
        int32_t ltid_flat_8007;
        
        ltid_flat_8007 = local_tid_8469;
        if (slt32(ltid_x_8005, tile_sizze_7967) && slt32(ltid_y_8006,
                                                         tile_sizze_7967)) {
            int32_t i_8069 = ltid_x_8005 + binop_x_8068;
            int32_t j_8071 = ltid_y_8006 + binop_x_8068;
            int32_t gtid_8073 = ltid_x_8005 + binop_x_8072;
            int32_t gtid_8075 = ltid_y_8006 + binop_x_8074;
            float tile_elem_8079 = ((__global float *) a_mem_8284)[gtid_8073 *
                                                                   sizze_7672 +
                                                                   j_8071];
            float tile_elem_8080 = ((__global float *) b_mem_8285)[i_8069 *
                                                                   sizze_7674 +
                                                                   gtid_8075];
            
            ((__local float *) mem_8294)[ltid_x_8005 * tile_sizze_7967 +
                                         ltid_y_8006] = tile_elem_8079;
            ((__local float *) mem_8299)[ltid_x_8005 * tile_sizze_7967 +
                                         ltid_y_8006] = tile_elem_8080;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_8031 = squot32(local_tid_8469, tile_sizze_7967);
        int32_t ltid_y_8032;
        
        ltid_y_8032 = local_tid_8469 - squot32(local_tid_8469,
                                               tile_sizze_7967) *
            tile_sizze_7967;
        
        int32_t ltid_flat_8033;
        
        ltid_flat_8033 = local_tid_8469;
        if (slt32(ltid_x_8031, tile_sizze_7967) && slt32(ltid_y_8032,
                                                         tile_sizze_7967)) {
            int32_t gtid_8083 = ltid_x_8031 + binop_x_8072;
            int32_t gtid_8085 = ltid_y_8032 + binop_x_8074;
            float acc_8089 = mem_8289;
            bool binop_x_8092 = slt32(gtid_8083, sizze_7671);
            bool binop_y_8093 = slt32(gtid_8085, sizze_7674);
            bool cond_8094 = binop_x_8092 && binop_y_8093;
            float acc_8095;
            
            if (cond_8094) {
                float x_8096;
                float redout_8256 = acc_8089;
                
                for (int32_t i_8257 = 0; i_8257 < tile_sizze_7967; i_8257++) {
                    float x_8100 = ((__local float *) mem_8294)[ltid_x_8031 *
                                                                tile_sizze_7967 +
                                                                i_8257];
                    float x_8101 = ((__local float *) mem_8299)[i_8257 *
                                                                tile_sizze_7967 +
                                                                ltid_y_8032];
                    float res_8102 = x_8100 * x_8101;
                    float res_8099 = res_8102 + redout_8256;
                    float redout_tmp_8474 = res_8099;
                    
                    redout_8256 = redout_tmp_8474;
                }
                x_8096 = redout_8256;
                acc_8095 = x_8096;
            } else {
                acc_8095 = acc_8089;
            }
            mem_8303 = acc_8095;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_8475 = 0; i_8475 < squot32(tile_sizze_7967 *
                                                  tile_sizze_7967 -
                                                  local_tid_8469 +
                                                  group_sizze_7968 - 1,
                                                  group_sizze_7968); i_8475++) {
            mem_8289 = mem_8303;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    __local char *mem_8309;
    
    mem_8309 = (__local char *) mem_8309_backing_2;
    
    __local char *mem_8314;
    
    mem_8314 = (__local char *) mem_8314_backing_3;
    
    float mem_8318;
    float mem_8336;
    
    if (cond_8111) {
        for (int32_t i_8476 = 0; i_8476 < squot32(tile_sizze_7967 *
                                                  tile_sizze_7967 -
                                                  local_tid_8469 +
                                                  group_sizze_7968 - 1,
                                                  group_sizze_7968); i_8476++) {
            mem_8336 = mem_8289;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
        int32_t binop_x_8197 = tile_sizze_7967 * num_whole_tiles_7977;
        int32_t ltid_x_8112 = squot32(local_tid_8469, tile_sizze_7967);
        int32_t ltid_y_8113;
        
        ltid_y_8113 = local_tid_8469 - squot32(local_tid_8469,
                                               tile_sizze_7967) *
            tile_sizze_7967;
        
        int32_t ltid_flat_8114;
        
        ltid_flat_8114 = local_tid_8469;
        if (slt32(ltid_x_8112, tile_sizze_7967) && slt32(ltid_y_8113,
                                                         tile_sizze_7967)) {
            int32_t i_8198 = ltid_x_8112 + binop_x_8197;
            int32_t j_8200 = ltid_y_8113 + binop_x_8197;
            int32_t gtid_8202 = binop_x_8072 + ltid_x_8112;
            int32_t gtid_8204 = binop_x_8074 + ltid_y_8113;
            bool binop_x_8208 = slt32(j_8200, sizze_7672);
            bool binop_y_8209 = slt32(gtid_8202, sizze_7671);
            bool cond_8210 = binop_x_8208 && binop_y_8209;
            float pre_8211;
            
            if (cond_8210) {
                float x_8212 = ((__global float *) a_mem_8284)[gtid_8202 *
                                                               sizze_7672 +
                                                               j_8200];
                
                pre_8211 = x_8212;
            } else {
                pre_8211 = 0.0F;
            }
            
            bool binop_x_8214 = slt32(i_8198, sizze_7672);
            bool binop_y_8215 = slt32(gtid_8204, sizze_7674);
            bool cond_8216 = binop_x_8214 && binop_y_8215;
            float pre_8217;
            
            if (cond_8216) {
                float x_8218 = ((__global float *) b_mem_8285)[i_8198 *
                                                               sizze_7674 +
                                                               gtid_8204];
                
                pre_8217 = x_8218;
            } else {
                pre_8217 = 0.0F;
            }
            ((__local float *) mem_8309)[ltid_x_8112 * tile_sizze_7967 +
                                         ltid_y_8113] = pre_8211;
            ((__local float *) mem_8314)[ltid_x_8112 * tile_sizze_7967 +
                                         ltid_y_8113] = pre_8217;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_8160 = squot32(local_tid_8469, tile_sizze_7967);
        int32_t ltid_y_8161;
        
        ltid_y_8161 = local_tid_8469 - squot32(local_tid_8469,
                                               tile_sizze_7967) *
            tile_sizze_7967;
        
        int32_t ltid_flat_8162;
        
        ltid_flat_8162 = local_tid_8469;
        if (slt32(ltid_x_8160, tile_sizze_7967) && slt32(ltid_y_8161,
                                                         tile_sizze_7967)) {
            int32_t gtid_8224 = binop_x_8072 + ltid_x_8160;
            int32_t gtid_8226 = binop_x_8074 + ltid_y_8161;
            float acc_8230 = mem_8289;
            bool binop_x_8233 = slt32(gtid_8224, sizze_7671);
            bool binop_y_8234 = slt32(gtid_8226, sizze_7674);
            bool cond_8235 = binop_x_8233 && binop_y_8234;
            float acc_8236;
            
            if (cond_8235) {
                float x_8237;
                float redout_8258 = acc_8230;
                
                for (int32_t i_8259 = 0; i_8259 < residual_input_8110;
                     i_8259++) {
                    float x_8241 = ((__local float *) mem_8309)[ltid_x_8160 *
                                                                tile_sizze_7967 +
                                                                i_8259];
                    float x_8242 = ((__local float *) mem_8314)[i_8259 *
                                                                tile_sizze_7967 +
                                                                ltid_y_8161];
                    float res_8243 = x_8241 * x_8242;
                    float res_8240 = res_8243 + redout_8258;
                    float redout_tmp_8477 = res_8240;
                    
                    redout_8258 = redout_tmp_8477;
                }
                x_8237 = redout_8258;
                acc_8236 = x_8237;
            } else {
                acc_8236 = acc_8230;
            }
            mem_8318 = acc_8236;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_8478 = 0; i_8478 < squot32(tile_sizze_7967 *
                                                  tile_sizze_7967 -
                                                  local_tid_8469 +
                                                  group_sizze_7968 - 1,
                                                  group_sizze_7968); i_8478++) {
            mem_8336 = mem_8318;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    int32_t thread_out_index_8479 = gid_x_7964 * tile_sizze_7967 +
            squot32(local_tid_8469, tile_sizze_7967);
    int32_t thread_out_index_8480;
    
    thread_out_index_8480 = gid_y_7965 * tile_sizze_7967 + (local_tid_8469 -
                                                            squot32(local_tid_8469,
                                                                    tile_sizze_7967) *
                                                            tile_sizze_7967);
    if (slt32(thread_out_index_8479, sizze_7671) && slt32(thread_out_index_8480,
                                                          sizze_7674)) {
        ((__global float *) mem_8324)[thread_out_index_8479 * sizze_7674 +
                                      thread_out_index_8480] = mem_8336;
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
  entry_points = {"matmul": (["[][]f32", "[][]f32"], ["[][]f32"]),
                  "matsubstract": (["[][]f32", "[][]f32"], ["[][]f32"]),
                  "lvecmul": (["[]f32", "[][]f32"], ["[]f32"]),
                  "sigmoid": (["[][]f32"], ["[][]f32"]),
                  "lmatsubstract": (["f32", "[][]f32"], ["[][]f32"]),
                  "substract2": (["[]f32", "[]f32"], ["[]f32"]),
                  "add2": (["[]f32", "[]f32"], ["[]f32"]),
                  "multiply2": (["[]f32", "[]f32"], ["[]f32"]),
                  "substract": (["f32", "[]f32"], ["[]f32"]), "add": (["[]f32",
                                                                       "f32"],
                                                                      ["[]f32"]),
                  "multiply": (["[]f32", "f32"], ["[]f32"]), "divide": (["f32",
                                                                         "[]f32"],
                                                                        ["[]f32"]),
                  "negation": (["[]f32"], ["[]f32"]), "exp": (["[]f32"],
                                                              ["[]f32"]),
                  "transp": (["[][]f32"], ["[][]f32"])}
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
                                       all_sizes={"add.segmap_group_size_7756": {"class": "group_size", "value": None},
                                        "add2.segmap_group_size_7799": {"class": "group_size", "value": None},
                                        "divide.segmap_group_size_7728": {"class": "group_size", "value": None},
                                        "exp.segmap_group_size_7700": {"class": "group_size", "value": None},
                                        "lmatsubstract.segmap_group_size_7832": {"class": "group_size", "value": None},
                                        "lvecmul.segmap_group_size_7867": {"class": "group_size", "value": None},
                                        "matmul.tile_size_7966": {"class": "tile_size", "value": None},
                                        "matsubstract.segmap_group_size_7891": {"class": "group_size", "value": None},
                                        "multiply.segmap_group_size_7742": {"class": "group_size", "value": None},
                                        "multiply2.segmap_group_size_7784": {"class": "group_size", "value": None},
                                        "negation.segmap_group_size_7714": {"class": "group_size", "value": None},
                                        "sigmoid.segmap_group_size_7849": {"class": "group_size", "value": None},
                                        "substract.segmap_group_size_7770": {"class": "group_size", "value": None},
                                        "substract2.segmap_group_size_7814": {"class": "group_size", "value": None}})
    self.map_transpose_f32_var = program.map_transpose_f32
    self.map_transpose_f32_low_height_var = program.map_transpose_f32_low_height
    self.map_transpose_f32_low_width_var = program.map_transpose_f32_low_width
    self.map_transpose_f32_small_var = program.map_transpose_f32_small
    self.segmap_7697_var = program.segmap_7697
    self.segmap_7711_var = program.segmap_7711
    self.segmap_7725_var = program.segmap_7725
    self.segmap_7739_var = program.segmap_7739
    self.segmap_7753_var = program.segmap_7753
    self.segmap_7767_var = program.segmap_7767
    self.segmap_7781_var = program.segmap_7781
    self.segmap_7796_var = program.segmap_7796
    self.segmap_7811_var = program.segmap_7811
    self.segmap_7827_var = program.segmap_7827
    self.segmap_7844_var = program.segmap_7844
    self.segmap_7864_var = program.segmap_7864
    self.segmap_7886_var = program.segmap_7886
    self.segmap_intragroup_7976_var = program.segmap_intragroup_7976
  def futhark_matmul(self, a_mem_8284, b_mem_8285, sizze_7671, sizze_7672,
                     sizze_7673, sizze_7674):
    dim_zzero_7678 = (np.int32(0) == sizze_7673)
    dim_zzero_7679 = (np.int32(0) == sizze_7672)
    both_empty_7680 = (dim_zzero_7678 and dim_zzero_7679)
    dim_match_7681 = (sizze_7672 == sizze_7673)
    empty_or_match_7682 = (both_empty_7680 or dim_match_7681)
    empty_or_match_cert_7683 = True
    assert empty_or_match_7682, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:10:19-30\n   #1  GPU.fut:14:21-38\n   #2  GPU.fut:14:3-15:17\n   #3  GPU.fut:23:18-33\n   #4  GPU.fut:23:3-36\n   #5  GPU.fut:22:1-23:36\n" % ("function arguments of wrong shape",))
    tile_sizze_7967 = self.sizes["matmul.tile_size_7966"]
    group_sizze_7968 = (tile_sizze_7967 * tile_sizze_7967)
    y_7969 = (tile_sizze_7967 - np.int32(1))
    x_7970 = (sizze_7671 + y_7969)
    num_groups_x_7971 = squot32(x_7970, tile_sizze_7967)
    x_7973 = (sizze_7674 + y_7969)
    num_groups_y_7974 = squot32(x_7973, tile_sizze_7967)
    num_groups_top_7975 = (num_groups_x_7971 * num_groups_y_7974)
    num_whole_tiles_7977 = squot32(sizze_7672, tile_sizze_7967)
    residual_input_8110 = srem32(sizze_7672, tile_sizze_7967)
    cond_8111 = (residual_input_8110 == np.int32(0))
    binop_x_8321 = sext_i32_i64(sizze_7671)
    binop_y_8322 = sext_i32_i64(sizze_7674)
    binop_x_8323 = (binop_x_8321 * binop_y_8322)
    bytes_8320 = (np.int64(4) * binop_x_8323)
    mem_8324 = opencl_alloc(self, bytes_8320, "mem_8324")
    binop_x_8288 = sext_i32_i64(group_sizze_7968)
    bytes_8286 = (np.int64(4) * binop_x_8288)
    binop_x_8291 = sext_i32_i64(tile_sizze_7967)
    binop_x_8293 = (binop_x_8291 * binop_x_8291)
    bytes_8290 = (np.int64(4) * binop_x_8293)
    if ((1 * (np.long(num_groups_top_7975) * np.long(group_sizze_7968))) != 0):
      self.segmap_intragroup_7976_var.set_args(cl.LocalMemory(np.long(bytes_8290)),
                                               cl.LocalMemory(np.long(bytes_8290)),
                                               cl.LocalMemory(np.long(bytes_8290)),
                                               cl.LocalMemory(np.long(bytes_8290)),
                                               np.int32(sizze_7671),
                                               np.int32(sizze_7672),
                                               np.int32(sizze_7674),
                                               np.int32(num_groups_y_7974),
                                               np.int32(num_whole_tiles_7977),
                                               np.int32(residual_input_8110),
                                               np.byte(cond_8111), a_mem_8284,
                                               b_mem_8285, mem_8324)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_intragroup_7976_var,
                                 ((np.long(num_groups_top_7975) * np.long(group_sizze_7968)),),
                                 (np.long(group_sizze_7968),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8466 = sizze_7671
    out_arrsizze_8467 = sizze_7674
    out_mem_8465 = mem_8324
    return (out_mem_8465, out_arrsizze_8466, out_arrsizze_8467)
  def futhark_matsubstract(self, x_mem_8284, y_mem_8285, sizze_7642, sizze_7643,
                           sizze_7644, sizze_7645):
    dim_zzero_7648 = (np.int32(0) == sizze_7644)
    dim_zzero_7649 = (np.int32(0) == sizze_7645)
    old_empty_7650 = (dim_zzero_7648 or dim_zzero_7649)
    dim_zzero_7651 = (np.int32(0) == sizze_7642)
    new_empty_7652 = (dim_zzero_7649 or dim_zzero_7651)
    both_empty_7653 = (old_empty_7650 and new_empty_7652)
    dim_match_7654 = (sizze_7642 == sizze_7644)
    empty_or_match_7655 = (both_empty_7653 or dim_match_7654)
    empty_or_match_cert_7656 = True
    assert empty_or_match_7655, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:104:3-21\n   #1  GPU.fut:103:1-104:21\n" % ("function arguments of wrong shape",))
    dim_zzero_7658 = (np.int32(0) == sizze_7643)
    both_empty_7659 = (dim_zzero_7649 and dim_zzero_7658)
    dim_match_7660 = (sizze_7643 == sizze_7645)
    empty_or_match_7661 = (both_empty_7659 or dim_match_7660)
    empty_or_match_cert_7662 = True
    assert empty_or_match_7661, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:96:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPU.fut:104:3-21\n   #4  GPU.fut:103:1-104:21\n" % ("function arguments of wrong shape",))
    sizze_7887 = sext_i32_i64(sizze_7642)
    sizze_7888 = sext_i32_i64(sizze_7643)
    nest_sizze_7890 = (sizze_7887 * sizze_7888)
    segmap_group_sizze_7892 = self.sizes["matsubstract.segmap_group_size_7891"]
    segmap_group_sizze_7893 = sext_i32_i64(segmap_group_sizze_7892)
    y_7894 = (segmap_group_sizze_7893 - np.int64(1))
    x_7895 = (nest_sizze_7890 + y_7894)
    segmap_usable_groups_64_7897 = squot64(x_7895, segmap_group_sizze_7893)
    segmap_usable_groups_7898 = sext_i64_i32(segmap_usable_groups_64_7897)
    bytes_8286 = (np.int64(4) * nest_sizze_7890)
    mem_8290 = opencl_alloc(self, bytes_8286, "mem_8290")
    if ((1 * (np.long(segmap_usable_groups_7898) * np.long(segmap_group_sizze_7892))) != 0):
      self.segmap_7886_var.set_args(np.int32(sizze_7642), np.int32(sizze_7643),
                                    np.int32(sizze_7645), x_mem_8284,
                                    y_mem_8285, mem_8290)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_7886_var,
                                 ((np.long(segmap_usable_groups_7898) * np.long(segmap_group_sizze_7892)),),
                                 (np.long(segmap_group_sizze_7892),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8458 = sizze_7642
    out_arrsizze_8459 = sizze_7643
    out_mem_8457 = mem_8290
    return (out_mem_8457, out_arrsizze_8458, out_arrsizze_8459)
  def futhark_lvecmul(self, u_mem_8284, b_mem_8285, sizze_7620, sizze_7621,
                      sizze_7622):
    dim_zzero_7626 = (np.int32(0) == sizze_7621)
    dim_zzero_7627 = (np.int32(0) == sizze_7620)
    both_empty_7628 = (dim_zzero_7626 and dim_zzero_7627)
    dim_match_7629 = (sizze_7620 == sizze_7621)
    empty_or_match_7630 = (both_empty_7628 or dim_match_7629)
    empty_or_match_cert_7631 = True
    assert empty_or_match_7630, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:10:19-30\n   #1  GPU.fut:14:21-38\n   #2  GPU.fut:14:3-15:17\n   #3  GPU.fut:13:1-15:17\n" % ("function arguments of wrong shape",))
    sizze_7865 = sext_i32_i64(sizze_7622)
    segmap_group_sizze_7868 = self.sizes["lvecmul.segmap_group_size_7867"]
    segmap_group_sizze_7869 = sext_i32_i64(segmap_group_sizze_7868)
    y_7870 = (segmap_group_sizze_7869 - np.int64(1))
    x_7871 = (sizze_7865 + y_7870)
    segmap_usable_groups_64_7873 = squot64(x_7871, segmap_group_sizze_7869)
    segmap_usable_groups_7874 = sext_i64_i32(segmap_usable_groups_64_7873)
    bytes_8286 = (np.int64(4) * sizze_7865)
    mem_8288 = opencl_alloc(self, bytes_8286, "mem_8288")
    if ((1 * (np.long(segmap_usable_groups_7874) * np.long(segmap_group_sizze_7868))) != 0):
      self.segmap_7864_var.set_args(np.int32(sizze_7620), np.int32(sizze_7621),
                                    np.int32(sizze_7622), u_mem_8284,
                                    b_mem_8285, mem_8288)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_7864_var,
                                 ((np.long(segmap_usable_groups_7874) * np.long(segmap_group_sizze_7868)),),
                                 (np.long(segmap_group_sizze_7868),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8450 = sizze_7622
    out_mem_8449 = mem_8288
    return (out_mem_8449, out_arrsizze_8450)
  def futhark_sigmoid(self, x_mem_8284, sizze_7609, sizze_7610):
    sizze_7845 = sext_i32_i64(sizze_7609)
    sizze_7846 = sext_i32_i64(sizze_7610)
    nest_sizze_7848 = (sizze_7845 * sizze_7846)
    segmap_group_sizze_7850 = self.sizes["sigmoid.segmap_group_size_7849"]
    segmap_group_sizze_7851 = sext_i32_i64(segmap_group_sizze_7850)
    y_7852 = (segmap_group_sizze_7851 - np.int64(1))
    x_7853 = (nest_sizze_7848 + y_7852)
    segmap_usable_groups_64_7855 = squot64(x_7853, segmap_group_sizze_7851)
    segmap_usable_groups_7856 = sext_i64_i32(segmap_usable_groups_64_7855)
    bytes_8285 = (np.int64(4) * nest_sizze_7848)
    mem_8289 = opencl_alloc(self, bytes_8285, "mem_8289")
    if ((1 * (np.long(segmap_usable_groups_7856) * np.long(segmap_group_sizze_7850))) != 0):
      self.segmap_7844_var.set_args(np.int32(sizze_7609), np.int32(sizze_7610),
                                    x_mem_8284, mem_8289)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_7844_var,
                                 ((np.long(segmap_usable_groups_7856) * np.long(segmap_group_sizze_7850)),),
                                 (np.long(segmap_group_sizze_7850),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8442 = sizze_7609
    out_arrsizze_8443 = sizze_7610
    out_mem_8441 = mem_8289
    return (out_mem_8441, out_arrsizze_8442, out_arrsizze_8443)
  def futhark_lmatsubstract(self, x_mem_8284, sizze_7600, sizze_7601, d_7602):
    sizze_7828 = sext_i32_i64(sizze_7600)
    sizze_7829 = sext_i32_i64(sizze_7601)
    nest_sizze_7831 = (sizze_7828 * sizze_7829)
    segmap_group_sizze_7833 = self.sizes["lmatsubstract.segmap_group_size_7832"]
    segmap_group_sizze_7834 = sext_i32_i64(segmap_group_sizze_7833)
    y_7835 = (segmap_group_sizze_7834 - np.int64(1))
    x_7836 = (nest_sizze_7831 + y_7835)
    segmap_usable_groups_64_7838 = squot64(x_7836, segmap_group_sizze_7834)
    segmap_usable_groups_7839 = sext_i64_i32(segmap_usable_groups_64_7838)
    bytes_8285 = (np.int64(4) * nest_sizze_7831)
    mem_8289 = opencl_alloc(self, bytes_8285, "mem_8289")
    if ((1 * (np.long(segmap_usable_groups_7839) * np.long(segmap_group_sizze_7833))) != 0):
      self.segmap_7827_var.set_args(np.int32(sizze_7600), np.int32(sizze_7601),
                                    np.float32(d_7602), x_mem_8284, mem_8289)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_7827_var,
                                 ((np.long(segmap_usable_groups_7839) * np.long(segmap_group_sizze_7833)),),
                                 (np.long(segmap_group_sizze_7833),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8434 = sizze_7600
    out_arrsizze_8435 = sizze_7601
    out_mem_8433 = mem_8289
    return (out_mem_8433, out_arrsizze_8434, out_arrsizze_8435)
  def futhark_substract2(self, x_mem_8284, y_mem_8285, sizze_7585, sizze_7586):
    dim_zzero_7589 = (np.int32(0) == sizze_7586)
    dim_zzero_7590 = (np.int32(0) == sizze_7585)
    both_empty_7591 = (dim_zzero_7589 and dim_zzero_7590)
    dim_match_7592 = (sizze_7585 == sizze_7586)
    empty_or_match_7593 = (both_empty_7591 or dim_match_7592)
    empty_or_match_cert_7594 = True
    assert empty_or_match_7593, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:96:3-14\n   #1  GPU.fut:95:1-96:14\n" % ("function arguments of wrong shape",))
    sizze_7812 = sext_i32_i64(sizze_7585)
    segmap_group_sizze_7815 = self.sizes["substract2.segmap_group_size_7814"]
    segmap_group_sizze_7816 = sext_i32_i64(segmap_group_sizze_7815)
    y_7817 = (segmap_group_sizze_7816 - np.int64(1))
    x_7818 = (sizze_7812 + y_7817)
    segmap_usable_groups_64_7820 = squot64(x_7818, segmap_group_sizze_7816)
    segmap_usable_groups_7821 = sext_i64_i32(segmap_usable_groups_64_7820)
    bytes_8286 = (np.int64(4) * sizze_7812)
    mem_8288 = opencl_alloc(self, bytes_8286, "mem_8288")
    if ((1 * (np.long(segmap_usable_groups_7821) * np.long(segmap_group_sizze_7815))) != 0):
      self.segmap_7811_var.set_args(np.int32(sizze_7585), x_mem_8284,
                                    y_mem_8285, mem_8288)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_7811_var,
                                 ((np.long(segmap_usable_groups_7821) * np.long(segmap_group_sizze_7815)),),
                                 (np.long(segmap_group_sizze_7815),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8427 = sizze_7585
    out_mem_8426 = mem_8288
    return (out_mem_8426, out_arrsizze_8427)
  def futhark_add2(self, x_mem_8284, y_mem_8285, sizze_7570, sizze_7571):
    dim_zzero_7574 = (np.int32(0) == sizze_7571)
    dim_zzero_7575 = (np.int32(0) == sizze_7570)
    both_empty_7576 = (dim_zzero_7574 and dim_zzero_7575)
    dim_match_7577 = (sizze_7570 == sizze_7571)
    empty_or_match_7578 = (both_empty_7576 or dim_match_7577)
    empty_or_match_cert_7579 = True
    assert empty_or_match_7578, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:80:3-14\n   #1  GPU.fut:79:1-80:14\n" % ("function arguments of wrong shape",))
    sizze_7797 = sext_i32_i64(sizze_7570)
    segmap_group_sizze_7800 = self.sizes["add2.segmap_group_size_7799"]
    segmap_group_sizze_7801 = sext_i32_i64(segmap_group_sizze_7800)
    y_7802 = (segmap_group_sizze_7801 - np.int64(1))
    x_7803 = (sizze_7797 + y_7802)
    segmap_usable_groups_64_7805 = squot64(x_7803, segmap_group_sizze_7801)
    segmap_usable_groups_7806 = sext_i64_i32(segmap_usable_groups_64_7805)
    bytes_8286 = (np.int64(4) * sizze_7797)
    mem_8288 = opencl_alloc(self, bytes_8286, "mem_8288")
    if ((1 * (np.long(segmap_usable_groups_7806) * np.long(segmap_group_sizze_7800))) != 0):
      self.segmap_7796_var.set_args(np.int32(sizze_7570), x_mem_8284,
                                    y_mem_8285, mem_8288)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_7796_var,
                                 ((np.long(segmap_usable_groups_7806) * np.long(segmap_group_sizze_7800)),),
                                 (np.long(segmap_group_sizze_7800),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8420 = sizze_7570
    out_mem_8419 = mem_8288
    return (out_mem_8419, out_arrsizze_8420)
  def futhark_multiply2(self, x_mem_8284, y_mem_8285, sizze_7555, sizze_7556):
    dim_zzero_7559 = (np.int32(0) == sizze_7556)
    dim_zzero_7560 = (np.int32(0) == sizze_7555)
    both_empty_7561 = (dim_zzero_7559 and dim_zzero_7560)
    dim_match_7562 = (sizze_7555 == sizze_7556)
    empty_or_match_7563 = (both_empty_7561 or dim_match_7562)
    empty_or_match_cert_7564 = True
    assert empty_or_match_7563, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:64:3-14\n   #1  GPU.fut:63:1-64:14\n" % ("function arguments of wrong shape",))
    sizze_7782 = sext_i32_i64(sizze_7555)
    segmap_group_sizze_7785 = self.sizes["multiply2.segmap_group_size_7784"]
    segmap_group_sizze_7786 = sext_i32_i64(segmap_group_sizze_7785)
    y_7787 = (segmap_group_sizze_7786 - np.int64(1))
    x_7788 = (sizze_7782 + y_7787)
    segmap_usable_groups_64_7790 = squot64(x_7788, segmap_group_sizze_7786)
    segmap_usable_groups_7791 = sext_i64_i32(segmap_usable_groups_64_7790)
    bytes_8286 = (np.int64(4) * sizze_7782)
    mem_8288 = opencl_alloc(self, bytes_8286, "mem_8288")
    if ((1 * (np.long(segmap_usable_groups_7791) * np.long(segmap_group_sizze_7785))) != 0):
      self.segmap_7781_var.set_args(np.int32(sizze_7555), x_mem_8284,
                                    y_mem_8285, mem_8288)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_7781_var,
                                 ((np.long(segmap_usable_groups_7791) * np.long(segmap_group_sizze_7785)),),
                                 (np.long(segmap_group_sizze_7785),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8413 = sizze_7555
    out_mem_8412 = mem_8288
    return (out_mem_8412, out_arrsizze_8413)
  def futhark_substract(self, x_mem_8284, sizze_7549, d_7550):
    sizze_7768 = sext_i32_i64(sizze_7549)
    segmap_group_sizze_7771 = self.sizes["substract.segmap_group_size_7770"]
    segmap_group_sizze_7772 = sext_i32_i64(segmap_group_sizze_7771)
    y_7773 = (segmap_group_sizze_7772 - np.int64(1))
    x_7774 = (sizze_7768 + y_7773)
    segmap_usable_groups_64_7776 = squot64(x_7774, segmap_group_sizze_7772)
    segmap_usable_groups_7777 = sext_i64_i32(segmap_usable_groups_64_7776)
    bytes_8285 = (np.int64(4) * sizze_7768)
    mem_8287 = opencl_alloc(self, bytes_8285, "mem_8287")
    if ((1 * (np.long(segmap_usable_groups_7777) * np.long(segmap_group_sizze_7771))) != 0):
      self.segmap_7767_var.set_args(np.int32(sizze_7549), np.float32(d_7550),
                                    x_mem_8284, mem_8287)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_7767_var,
                                 ((np.long(segmap_usable_groups_7777) * np.long(segmap_group_sizze_7771)),),
                                 (np.long(segmap_group_sizze_7771),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8406 = sizze_7549
    out_mem_8405 = mem_8287
    return (out_mem_8405, out_arrsizze_8406)
  def futhark_add(self, x_mem_8284, sizze_7543, s_7545):
    sizze_7754 = sext_i32_i64(sizze_7543)
    segmap_group_sizze_7757 = self.sizes["add.segmap_group_size_7756"]
    segmap_group_sizze_7758 = sext_i32_i64(segmap_group_sizze_7757)
    y_7759 = (segmap_group_sizze_7758 - np.int64(1))
    x_7760 = (sizze_7754 + y_7759)
    segmap_usable_groups_64_7762 = squot64(x_7760, segmap_group_sizze_7758)
    segmap_usable_groups_7763 = sext_i64_i32(segmap_usable_groups_64_7762)
    bytes_8285 = (np.int64(4) * sizze_7754)
    mem_8287 = opencl_alloc(self, bytes_8285, "mem_8287")
    if ((1 * (np.long(segmap_usable_groups_7763) * np.long(segmap_group_sizze_7757))) != 0):
      self.segmap_7753_var.set_args(np.int32(sizze_7543), np.float32(s_7545),
                                    x_mem_8284, mem_8287)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_7753_var,
                                 ((np.long(segmap_usable_groups_7763) * np.long(segmap_group_sizze_7757)),),
                                 (np.long(segmap_group_sizze_7757),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8399 = sizze_7543
    out_mem_8398 = mem_8287
    return (out_mem_8398, out_arrsizze_8399)
  def futhark_multiply(self, x_mem_8284, sizze_7537, m_7539):
    sizze_7740 = sext_i32_i64(sizze_7537)
    segmap_group_sizze_7743 = self.sizes["multiply.segmap_group_size_7742"]
    segmap_group_sizze_7744 = sext_i32_i64(segmap_group_sizze_7743)
    y_7745 = (segmap_group_sizze_7744 - np.int64(1))
    x_7746 = (sizze_7740 + y_7745)
    segmap_usable_groups_64_7748 = squot64(x_7746, segmap_group_sizze_7744)
    segmap_usable_groups_7749 = sext_i64_i32(segmap_usable_groups_64_7748)
    bytes_8285 = (np.int64(4) * sizze_7740)
    mem_8287 = opencl_alloc(self, bytes_8285, "mem_8287")
    if ((1 * (np.long(segmap_usable_groups_7749) * np.long(segmap_group_sizze_7743))) != 0):
      self.segmap_7739_var.set_args(np.int32(sizze_7537), np.float32(m_7539),
                                    x_mem_8284, mem_8287)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_7739_var,
                                 ((np.long(segmap_usable_groups_7749) * np.long(segmap_group_sizze_7743)),),
                                 (np.long(segmap_group_sizze_7743),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8392 = sizze_7537
    out_mem_8391 = mem_8287
    return (out_mem_8391, out_arrsizze_8392)
  def futhark_divide(self, x_mem_8284, sizze_7531, div_7532):
    sizze_7726 = sext_i32_i64(sizze_7531)
    segmap_group_sizze_7729 = self.sizes["divide.segmap_group_size_7728"]
    segmap_group_sizze_7730 = sext_i32_i64(segmap_group_sizze_7729)
    y_7731 = (segmap_group_sizze_7730 - np.int64(1))
    x_7732 = (sizze_7726 + y_7731)
    segmap_usable_groups_64_7734 = squot64(x_7732, segmap_group_sizze_7730)
    segmap_usable_groups_7735 = sext_i64_i32(segmap_usable_groups_64_7734)
    bytes_8285 = (np.int64(4) * sizze_7726)
    mem_8287 = opencl_alloc(self, bytes_8285, "mem_8287")
    if ((1 * (np.long(segmap_usable_groups_7735) * np.long(segmap_group_sizze_7729))) != 0):
      self.segmap_7725_var.set_args(np.int32(sizze_7531), np.float32(div_7532),
                                    x_mem_8284, mem_8287)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_7725_var,
                                 ((np.long(segmap_usable_groups_7735) * np.long(segmap_group_sizze_7729)),),
                                 (np.long(segmap_group_sizze_7729),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8385 = sizze_7531
    out_mem_8384 = mem_8287
    return (out_mem_8384, out_arrsizze_8385)
  def futhark_negation(self, x_mem_8284, sizze_7526):
    sizze_7712 = sext_i32_i64(sizze_7526)
    segmap_group_sizze_7715 = self.sizes["negation.segmap_group_size_7714"]
    segmap_group_sizze_7716 = sext_i32_i64(segmap_group_sizze_7715)
    y_7717 = (segmap_group_sizze_7716 - np.int64(1))
    x_7718 = (sizze_7712 + y_7717)
    segmap_usable_groups_64_7720 = squot64(x_7718, segmap_group_sizze_7716)
    segmap_usable_groups_7721 = sext_i64_i32(segmap_usable_groups_64_7720)
    bytes_8285 = (np.int64(4) * sizze_7712)
    mem_8287 = opencl_alloc(self, bytes_8285, "mem_8287")
    if ((1 * (np.long(segmap_usable_groups_7721) * np.long(segmap_group_sizze_7715))) != 0):
      self.segmap_7711_var.set_args(np.int32(sizze_7526), x_mem_8284, mem_8287)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_7711_var,
                                 ((np.long(segmap_usable_groups_7721) * np.long(segmap_group_sizze_7715)),),
                                 (np.long(segmap_group_sizze_7715),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8378 = sizze_7526
    out_mem_8377 = mem_8287
    return (out_mem_8377, out_arrsizze_8378)
  def futhark_exp(self, x_mem_8284, sizze_7521):
    sizze_7698 = sext_i32_i64(sizze_7521)
    segmap_group_sizze_7701 = self.sizes["exp.segmap_group_size_7700"]
    segmap_group_sizze_7702 = sext_i32_i64(segmap_group_sizze_7701)
    y_7703 = (segmap_group_sizze_7702 - np.int64(1))
    x_7704 = (sizze_7698 + y_7703)
    segmap_usable_groups_64_7706 = squot64(x_7704, segmap_group_sizze_7702)
    segmap_usable_groups_7707 = sext_i64_i32(segmap_usable_groups_64_7706)
    bytes_8285 = (np.int64(4) * sizze_7698)
    mem_8287 = opencl_alloc(self, bytes_8285, "mem_8287")
    if ((1 * (np.long(segmap_usable_groups_7707) * np.long(segmap_group_sizze_7701))) != 0):
      self.segmap_7697_var.set_args(np.int32(sizze_7521), x_mem_8284, mem_8287)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_7697_var,
                                 ((np.long(segmap_usable_groups_7707) * np.long(segmap_group_sizze_7701)),),
                                 (np.long(segmap_group_sizze_7701),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_8371 = sizze_7521
    out_mem_8370 = mem_8287
    return (out_mem_8370, out_arrsizze_8371)
  def futhark_transp(self, x_mem_8284, sizze_7517, sizze_7518):
    binop_x_8286 = sext_i32_i64(sizze_7518)
    binop_y_8287 = sext_i32_i64(sizze_7517)
    binop_x_8288 = (binop_x_8286 * binop_y_8287)
    bytes_8285 = (np.int64(4) * binop_x_8288)
    mem_8289 = opencl_alloc(self, bytes_8285, "mem_8289")
    self.futhark__map_transpose_f32(mem_8289, np.int32(0), x_mem_8284,
                                    np.int32(0), np.int32(1), sizze_7518,
                                    sizze_7517, (sizze_7518 * sizze_7517),
                                    (sizze_7518 * sizze_7517))
    out_arrsizze_8368 = sizze_7518
    out_arrsizze_8369 = sizze_7517
    out_mem_8367 = mem_8289
    return (out_mem_8367, out_arrsizze_8368, out_arrsizze_8369)
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
  def matmul(self, a_mem_8284_ext, b_mem_8285_ext):
    try:
      assert ((type(a_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (a_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7671 = np.int32(a_mem_8284_ext.shape[0])
      sizze_7672 = np.int32(a_mem_8284_ext.shape[1])
      if (type(a_mem_8284_ext) == cl.array.Array):
        a_mem_8284 = a_mem_8284_ext.data
      else:
        a_mem_8284 = opencl_alloc(self, np.int64(a_mem_8284_ext.nbytes),
                                  "a_mem_8284")
        if (np.int64(a_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, a_mem_8284,
                          normaliseArray(a_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(a_mem_8284_ext),
                                                                                                                            a_mem_8284_ext))
    try:
      assert ((type(b_mem_8285_ext) in [np.ndarray,
                                        cl.array.Array]) and (b_mem_8285_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7673 = np.int32(b_mem_8285_ext.shape[0])
      sizze_7674 = np.int32(b_mem_8285_ext.shape[1])
      if (type(b_mem_8285_ext) == cl.array.Array):
        b_mem_8285 = b_mem_8285_ext.data
      else:
        b_mem_8285 = opencl_alloc(self, np.int64(b_mem_8285_ext.nbytes),
                                  "b_mem_8285")
        if (np.int64(b_mem_8285_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, b_mem_8285,
                          normaliseArray(b_mem_8285_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(b_mem_8285_ext),
                                                                                                                            b_mem_8285_ext))
    (out_mem_8465, out_arrsizze_8466,
     out_arrsizze_8467) = self.futhark_matmul(a_mem_8284, b_mem_8285,
                                              sizze_7671, sizze_7672,
                                              sizze_7673, sizze_7674)
    return cl.array.Array(self.queue, (out_arrsizze_8466, out_arrsizze_8467),
                          ct.c_float, data=out_mem_8465)
  def matsubstract(self, x_mem_8284_ext, y_mem_8285_ext):
    try:
      assert ((type(x_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (x_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7642 = np.int32(x_mem_8284_ext.shape[0])
      sizze_7643 = np.int32(x_mem_8284_ext.shape[1])
      if (type(x_mem_8284_ext) == cl.array.Array):
        x_mem_8284 = x_mem_8284_ext.data
      else:
        x_mem_8284 = opencl_alloc(self, np.int64(x_mem_8284_ext.nbytes),
                                  "x_mem_8284")
        if (np.int64(x_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_8284,
                          normaliseArray(x_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_8284_ext),
                                                                                                                            x_mem_8284_ext))
    try:
      assert ((type(y_mem_8285_ext) in [np.ndarray,
                                        cl.array.Array]) and (y_mem_8285_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7644 = np.int32(y_mem_8285_ext.shape[0])
      sizze_7645 = np.int32(y_mem_8285_ext.shape[1])
      if (type(y_mem_8285_ext) == cl.array.Array):
        y_mem_8285 = y_mem_8285_ext.data
      else:
        y_mem_8285 = opencl_alloc(self, np.int64(y_mem_8285_ext.nbytes),
                                  "y_mem_8285")
        if (np.int64(y_mem_8285_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_8285,
                          normaliseArray(y_mem_8285_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(y_mem_8285_ext),
                                                                                                                            y_mem_8285_ext))
    (out_mem_8457, out_arrsizze_8458,
     out_arrsizze_8459) = self.futhark_matsubstract(x_mem_8284, y_mem_8285,
                                                    sizze_7642, sizze_7643,
                                                    sizze_7644, sizze_7645)
    return cl.array.Array(self.queue, (out_arrsizze_8458, out_arrsizze_8459),
                          ct.c_float, data=out_mem_8457)
  def lvecmul(self, u_mem_8284_ext, b_mem_8285_ext):
    try:
      assert ((type(u_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (u_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7620 = np.int32(u_mem_8284_ext.shape[0])
      if (type(u_mem_8284_ext) == cl.array.Array):
        u_mem_8284 = u_mem_8284_ext.data
      else:
        u_mem_8284 = opencl_alloc(self, np.int64(u_mem_8284_ext.nbytes),
                                  "u_mem_8284")
        if (np.int64(u_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, u_mem_8284,
                          normaliseArray(u_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(u_mem_8284_ext),
                                                                                                                            u_mem_8284_ext))
    try:
      assert ((type(b_mem_8285_ext) in [np.ndarray,
                                        cl.array.Array]) and (b_mem_8285_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7621 = np.int32(b_mem_8285_ext.shape[0])
      sizze_7622 = np.int32(b_mem_8285_ext.shape[1])
      if (type(b_mem_8285_ext) == cl.array.Array):
        b_mem_8285 = b_mem_8285_ext.data
      else:
        b_mem_8285 = opencl_alloc(self, np.int64(b_mem_8285_ext.nbytes),
                                  "b_mem_8285")
        if (np.int64(b_mem_8285_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, b_mem_8285,
                          normaliseArray(b_mem_8285_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(b_mem_8285_ext),
                                                                                                                            b_mem_8285_ext))
    (out_mem_8449, out_arrsizze_8450) = self.futhark_lvecmul(u_mem_8284,
                                                             b_mem_8285,
                                                             sizze_7620,
                                                             sizze_7621,
                                                             sizze_7622)
    return cl.array.Array(self.queue, (out_arrsizze_8450,), ct.c_float,
                          data=out_mem_8449)
  def sigmoid(self, x_mem_8284_ext):
    try:
      assert ((type(x_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (x_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7609 = np.int32(x_mem_8284_ext.shape[0])
      sizze_7610 = np.int32(x_mem_8284_ext.shape[1])
      if (type(x_mem_8284_ext) == cl.array.Array):
        x_mem_8284 = x_mem_8284_ext.data
      else:
        x_mem_8284 = opencl_alloc(self, np.int64(x_mem_8284_ext.nbytes),
                                  "x_mem_8284")
        if (np.int64(x_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_8284,
                          normaliseArray(x_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_8284_ext),
                                                                                                                            x_mem_8284_ext))
    (out_mem_8441, out_arrsizze_8442,
     out_arrsizze_8443) = self.futhark_sigmoid(x_mem_8284, sizze_7609,
                                               sizze_7610)
    return cl.array.Array(self.queue, (out_arrsizze_8442, out_arrsizze_8443),
                          ct.c_float, data=out_mem_8441)
  def lmatsubstract(self, d_7602_ext, x_mem_8284_ext):
    try:
      d_7602 = np.float32(ct.c_float(d_7602_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(d_7602_ext),
                                                                                                                            d_7602_ext))
    try:
      assert ((type(x_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (x_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7600 = np.int32(x_mem_8284_ext.shape[0])
      sizze_7601 = np.int32(x_mem_8284_ext.shape[1])
      if (type(x_mem_8284_ext) == cl.array.Array):
        x_mem_8284 = x_mem_8284_ext.data
      else:
        x_mem_8284 = opencl_alloc(self, np.int64(x_mem_8284_ext.nbytes),
                                  "x_mem_8284")
        if (np.int64(x_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_8284,
                          normaliseArray(x_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_8284_ext),
                                                                                                                            x_mem_8284_ext))
    (out_mem_8433, out_arrsizze_8434,
     out_arrsizze_8435) = self.futhark_lmatsubstract(x_mem_8284, sizze_7600,
                                                     sizze_7601, d_7602)
    return cl.array.Array(self.queue, (out_arrsizze_8434, out_arrsizze_8435),
                          ct.c_float, data=out_mem_8433)
  def substract2(self, x_mem_8284_ext, y_mem_8285_ext):
    try:
      assert ((type(x_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (x_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7585 = np.int32(x_mem_8284_ext.shape[0])
      if (type(x_mem_8284_ext) == cl.array.Array):
        x_mem_8284 = x_mem_8284_ext.data
      else:
        x_mem_8284 = opencl_alloc(self, np.int64(x_mem_8284_ext.nbytes),
                                  "x_mem_8284")
        if (np.int64(x_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_8284,
                          normaliseArray(x_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_8284_ext),
                                                                                                                            x_mem_8284_ext))
    try:
      assert ((type(y_mem_8285_ext) in [np.ndarray,
                                        cl.array.Array]) and (y_mem_8285_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7586 = np.int32(y_mem_8285_ext.shape[0])
      if (type(y_mem_8285_ext) == cl.array.Array):
        y_mem_8285 = y_mem_8285_ext.data
      else:
        y_mem_8285 = opencl_alloc(self, np.int64(y_mem_8285_ext.nbytes),
                                  "y_mem_8285")
        if (np.int64(y_mem_8285_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_8285,
                          normaliseArray(y_mem_8285_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(y_mem_8285_ext),
                                                                                                                            y_mem_8285_ext))
    (out_mem_8426, out_arrsizze_8427) = self.futhark_substract2(x_mem_8284,
                                                                y_mem_8285,
                                                                sizze_7585,
                                                                sizze_7586)
    return cl.array.Array(self.queue, (out_arrsizze_8427,), ct.c_float,
                          data=out_mem_8426)
  def add2(self, x_mem_8284_ext, y_mem_8285_ext):
    try:
      assert ((type(x_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (x_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7570 = np.int32(x_mem_8284_ext.shape[0])
      if (type(x_mem_8284_ext) == cl.array.Array):
        x_mem_8284 = x_mem_8284_ext.data
      else:
        x_mem_8284 = opencl_alloc(self, np.int64(x_mem_8284_ext.nbytes),
                                  "x_mem_8284")
        if (np.int64(x_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_8284,
                          normaliseArray(x_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_8284_ext),
                                                                                                                            x_mem_8284_ext))
    try:
      assert ((type(y_mem_8285_ext) in [np.ndarray,
                                        cl.array.Array]) and (y_mem_8285_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7571 = np.int32(y_mem_8285_ext.shape[0])
      if (type(y_mem_8285_ext) == cl.array.Array):
        y_mem_8285 = y_mem_8285_ext.data
      else:
        y_mem_8285 = opencl_alloc(self, np.int64(y_mem_8285_ext.nbytes),
                                  "y_mem_8285")
        if (np.int64(y_mem_8285_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_8285,
                          normaliseArray(y_mem_8285_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(y_mem_8285_ext),
                                                                                                                            y_mem_8285_ext))
    (out_mem_8419, out_arrsizze_8420) = self.futhark_add2(x_mem_8284,
                                                          y_mem_8285,
                                                          sizze_7570,
                                                          sizze_7571)
    return cl.array.Array(self.queue, (out_arrsizze_8420,), ct.c_float,
                          data=out_mem_8419)
  def multiply2(self, x_mem_8284_ext, y_mem_8285_ext):
    try:
      assert ((type(x_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (x_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7555 = np.int32(x_mem_8284_ext.shape[0])
      if (type(x_mem_8284_ext) == cl.array.Array):
        x_mem_8284 = x_mem_8284_ext.data
      else:
        x_mem_8284 = opencl_alloc(self, np.int64(x_mem_8284_ext.nbytes),
                                  "x_mem_8284")
        if (np.int64(x_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_8284,
                          normaliseArray(x_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_8284_ext),
                                                                                                                            x_mem_8284_ext))
    try:
      assert ((type(y_mem_8285_ext) in [np.ndarray,
                                        cl.array.Array]) and (y_mem_8285_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7556 = np.int32(y_mem_8285_ext.shape[0])
      if (type(y_mem_8285_ext) == cl.array.Array):
        y_mem_8285 = y_mem_8285_ext.data
      else:
        y_mem_8285 = opencl_alloc(self, np.int64(y_mem_8285_ext.nbytes),
                                  "y_mem_8285")
        if (np.int64(y_mem_8285_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_8285,
                          normaliseArray(y_mem_8285_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(y_mem_8285_ext),
                                                                                                                            y_mem_8285_ext))
    (out_mem_8412, out_arrsizze_8413) = self.futhark_multiply2(x_mem_8284,
                                                               y_mem_8285,
                                                               sizze_7555,
                                                               sizze_7556)
    return cl.array.Array(self.queue, (out_arrsizze_8413,), ct.c_float,
                          data=out_mem_8412)
  def substract(self, d_7550_ext, x_mem_8284_ext):
    try:
      d_7550 = np.float32(ct.c_float(d_7550_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(d_7550_ext),
                                                                                                                            d_7550_ext))
    try:
      assert ((type(x_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (x_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7549 = np.int32(x_mem_8284_ext.shape[0])
      if (type(x_mem_8284_ext) == cl.array.Array):
        x_mem_8284 = x_mem_8284_ext.data
      else:
        x_mem_8284 = opencl_alloc(self, np.int64(x_mem_8284_ext.nbytes),
                                  "x_mem_8284")
        if (np.int64(x_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_8284,
                          normaliseArray(x_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_8284_ext),
                                                                                                                            x_mem_8284_ext))
    (out_mem_8405, out_arrsizze_8406) = self.futhark_substract(x_mem_8284,
                                                               sizze_7549,
                                                               d_7550)
    return cl.array.Array(self.queue, (out_arrsizze_8406,), ct.c_float,
                          data=out_mem_8405)
  def add(self, x_mem_8284_ext, s_7545_ext):
    try:
      assert ((type(x_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (x_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7543 = np.int32(x_mem_8284_ext.shape[0])
      if (type(x_mem_8284_ext) == cl.array.Array):
        x_mem_8284 = x_mem_8284_ext.data
      else:
        x_mem_8284 = opencl_alloc(self, np.int64(x_mem_8284_ext.nbytes),
                                  "x_mem_8284")
        if (np.int64(x_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_8284,
                          normaliseArray(x_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_8284_ext),
                                                                                                                            x_mem_8284_ext))
    try:
      s_7545 = np.float32(ct.c_float(s_7545_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(s_7545_ext),
                                                                                                                            s_7545_ext))
    (out_mem_8398, out_arrsizze_8399) = self.futhark_add(x_mem_8284, sizze_7543,
                                                         s_7545)
    return cl.array.Array(self.queue, (out_arrsizze_8399,), ct.c_float,
                          data=out_mem_8398)
  def multiply(self, x_mem_8284_ext, m_7539_ext):
    try:
      assert ((type(x_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (x_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7537 = np.int32(x_mem_8284_ext.shape[0])
      if (type(x_mem_8284_ext) == cl.array.Array):
        x_mem_8284 = x_mem_8284_ext.data
      else:
        x_mem_8284 = opencl_alloc(self, np.int64(x_mem_8284_ext.nbytes),
                                  "x_mem_8284")
        if (np.int64(x_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_8284,
                          normaliseArray(x_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_8284_ext),
                                                                                                                            x_mem_8284_ext))
    try:
      m_7539 = np.float32(ct.c_float(m_7539_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(m_7539_ext),
                                                                                                                            m_7539_ext))
    (out_mem_8391, out_arrsizze_8392) = self.futhark_multiply(x_mem_8284,
                                                              sizze_7537,
                                                              m_7539)
    return cl.array.Array(self.queue, (out_arrsizze_8392,), ct.c_float,
                          data=out_mem_8391)
  def divide(self, div_7532_ext, x_mem_8284_ext):
    try:
      div_7532 = np.float32(ct.c_float(div_7532_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(div_7532_ext),
                                                                                                                            div_7532_ext))
    try:
      assert ((type(x_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (x_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7531 = np.int32(x_mem_8284_ext.shape[0])
      if (type(x_mem_8284_ext) == cl.array.Array):
        x_mem_8284 = x_mem_8284_ext.data
      else:
        x_mem_8284 = opencl_alloc(self, np.int64(x_mem_8284_ext.nbytes),
                                  "x_mem_8284")
        if (np.int64(x_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_8284,
                          normaliseArray(x_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_8284_ext),
                                                                                                                            x_mem_8284_ext))
    (out_mem_8384, out_arrsizze_8385) = self.futhark_divide(x_mem_8284,
                                                            sizze_7531,
                                                            div_7532)
    return cl.array.Array(self.queue, (out_arrsizze_8385,), ct.c_float,
                          data=out_mem_8384)
  def negation(self, x_mem_8284_ext):
    try:
      assert ((type(x_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (x_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7526 = np.int32(x_mem_8284_ext.shape[0])
      if (type(x_mem_8284_ext) == cl.array.Array):
        x_mem_8284 = x_mem_8284_ext.data
      else:
        x_mem_8284 = opencl_alloc(self, np.int64(x_mem_8284_ext.nbytes),
                                  "x_mem_8284")
        if (np.int64(x_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_8284,
                          normaliseArray(x_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_8284_ext),
                                                                                                                            x_mem_8284_ext))
    (out_mem_8377, out_arrsizze_8378) = self.futhark_negation(x_mem_8284,
                                                              sizze_7526)
    return cl.array.Array(self.queue, (out_arrsizze_8378,), ct.c_float,
                          data=out_mem_8377)
  def exp(self, x_mem_8284_ext):
    try:
      assert ((type(x_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (x_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7521 = np.int32(x_mem_8284_ext.shape[0])
      if (type(x_mem_8284_ext) == cl.array.Array):
        x_mem_8284 = x_mem_8284_ext.data
      else:
        x_mem_8284 = opencl_alloc(self, np.int64(x_mem_8284_ext.nbytes),
                                  "x_mem_8284")
        if (np.int64(x_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_8284,
                          normaliseArray(x_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_8284_ext),
                                                                                                                            x_mem_8284_ext))
    (out_mem_8370, out_arrsizze_8371) = self.futhark_exp(x_mem_8284, sizze_7521)
    return cl.array.Array(self.queue, (out_arrsizze_8371,), ct.c_float,
                          data=out_mem_8370)
  def transp(self, x_mem_8284_ext):
    try:
      assert ((type(x_mem_8284_ext) in [np.ndarray,
                                        cl.array.Array]) and (x_mem_8284_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_7517 = np.int32(x_mem_8284_ext.shape[0])
      sizze_7518 = np.int32(x_mem_8284_ext.shape[1])
      if (type(x_mem_8284_ext) == cl.array.Array):
        x_mem_8284 = x_mem_8284_ext.data
      else:
        x_mem_8284 = opencl_alloc(self, np.int64(x_mem_8284_ext.nbytes),
                                  "x_mem_8284")
        if (np.int64(x_mem_8284_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_8284,
                          normaliseArray(x_mem_8284_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_8284_ext),
                                                                                                                            x_mem_8284_ext))
    (out_mem_8367, out_arrsizze_8368,
     out_arrsizze_8369) = self.futhark_transp(x_mem_8284, sizze_7517,
                                              sizze_7518)
    return cl.array.Array(self.queue, (out_arrsizze_8368, out_arrsizze_8369),
                          ct.c_float, data=out_mem_8367)