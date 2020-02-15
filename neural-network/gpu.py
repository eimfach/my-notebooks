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
__kernel void segmap_10010(int32_t sizze_9732, float m_9733, __global
                           unsigned char *x_mem_10627, __global
                           unsigned char *mem_10630)
{
    const int32_t segmap_group_sizze_10014 = multiplyzisegmap_group_sizze_10013;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10744;
    int32_t local_tid_10745;
    int32_t group_sizze_10748;
    int32_t wave_sizze_10747;
    int32_t group_tid_10746;
    
    global_tid_10744 = get_global_id(0);
    local_tid_10745 = get_local_id(0);
    group_sizze_10748 = get_local_size(0);
    wave_sizze_10747 = LOCKSTEP_WIDTH;
    group_tid_10746 = get_group_id(0);
    
    int32_t phys_tid_10010 = global_tid_10744;
    int32_t gtid_10009 = group_tid_10746 * segmap_group_sizze_10014 +
            local_tid_10745;
    
    if (slt32(gtid_10009, sizze_9732)) {
        float x_10021 = ((__global float *) x_mem_10627)[gtid_10009];
        float res_10022 = m_9733 * x_10021;
        
        ((__global float *) mem_10630)[gtid_10009] = res_10022;
    }
}
__kernel void segmap_10024(int32_t sizze_9738, float s_9739, __global
                           unsigned char *x_mem_10627, __global
                           unsigned char *mem_10630)
{
    const int32_t segmap_group_sizze_10028 = addzisegmap_group_sizze_10027;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10751;
    int32_t local_tid_10752;
    int32_t group_sizze_10755;
    int32_t wave_sizze_10754;
    int32_t group_tid_10753;
    
    global_tid_10751 = get_global_id(0);
    local_tid_10752 = get_local_id(0);
    group_sizze_10755 = get_local_size(0);
    wave_sizze_10754 = LOCKSTEP_WIDTH;
    group_tid_10753 = get_group_id(0);
    
    int32_t phys_tid_10024 = global_tid_10751;
    int32_t gtid_10023 = group_tid_10753 * segmap_group_sizze_10028 +
            local_tid_10752;
    
    if (slt32(gtid_10023, sizze_9738)) {
        float x_10035 = ((__global float *) x_mem_10627)[gtid_10023];
        float res_10036 = s_9739 + x_10035;
        
        ((__global float *) mem_10630)[gtid_10023] = res_10036;
    }
}
__kernel void segmap_10038(int32_t sizze_9744, float d_9745, __global
                           unsigned char *x_mem_10627, __global
                           unsigned char *mem_10630)
{
    const int32_t segmap_group_sizze_10042 =
                  substractzisegmap_group_sizze_10041;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10758;
    int32_t local_tid_10759;
    int32_t group_sizze_10762;
    int32_t wave_sizze_10761;
    int32_t group_tid_10760;
    
    global_tid_10758 = get_global_id(0);
    local_tid_10759 = get_local_id(0);
    group_sizze_10762 = get_local_size(0);
    wave_sizze_10761 = LOCKSTEP_WIDTH;
    group_tid_10760 = get_group_id(0);
    
    int32_t phys_tid_10038 = global_tid_10758;
    int32_t gtid_10037 = group_tid_10760 * segmap_group_sizze_10042 +
            local_tid_10759;
    
    if (slt32(gtid_10037, sizze_9744)) {
        float x_10049 = ((__global float *) x_mem_10627)[gtid_10037];
        float res_10050 = d_9745 - x_10049;
        
        ((__global float *) mem_10630)[gtid_10037] = res_10050;
    }
}
__kernel void segmap_10052(int32_t sizze_9750, __global
                           unsigned char *x_mem_10627, __global
                           unsigned char *y_mem_10628, __global
                           unsigned char *mem_10631)
{
    const int32_t segmap_group_sizze_10056 =
                  multiply2zisegmap_group_sizze_10055;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10765;
    int32_t local_tid_10766;
    int32_t group_sizze_10769;
    int32_t wave_sizze_10768;
    int32_t group_tid_10767;
    
    global_tid_10765 = get_global_id(0);
    local_tid_10766 = get_local_id(0);
    group_sizze_10769 = get_local_size(0);
    wave_sizze_10768 = LOCKSTEP_WIDTH;
    group_tid_10767 = get_group_id(0);
    
    int32_t phys_tid_10052 = global_tid_10765;
    int32_t gtid_10051 = group_tid_10767 * segmap_group_sizze_10056 +
            local_tid_10766;
    
    if (slt32(gtid_10051, sizze_9750)) {
        float x_10063 = ((__global float *) x_mem_10627)[gtid_10051];
        float x_10064 = ((__global float *) y_mem_10628)[gtid_10051];
        float res_10065 = x_10063 * x_10064;
        
        ((__global float *) mem_10631)[gtid_10051] = res_10065;
    }
}
__kernel void segmap_10068(int32_t sizze_9765, int32_t sizze_9766, float p_9767,
                           __global unsigned char *x_mem_10627, __global
                           unsigned char *mem_10632)
{
    const int32_t segmap_group_sizze_10074 =
                  lmatmultiplyzisegmap_group_sizze_10073;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10773;
    int32_t local_tid_10774;
    int32_t group_sizze_10777;
    int32_t wave_sizze_10776;
    int32_t group_tid_10775;
    
    global_tid_10773 = get_global_id(0);
    local_tid_10774 = get_local_id(0);
    group_sizze_10777 = get_local_size(0);
    wave_sizze_10776 = LOCKSTEP_WIDTH;
    group_tid_10775 = get_group_id(0);
    
    int32_t phys_tid_10068 = global_tid_10773;
    int32_t gtid_10066 = squot32(group_tid_10775 * segmap_group_sizze_10074 +
                                 local_tid_10774, sizze_9766);
    int32_t gtid_10067;
    
    gtid_10067 = group_tid_10775 * segmap_group_sizze_10074 + local_tid_10774 -
        squot32(group_tid_10775 * segmap_group_sizze_10074 + local_tid_10774,
                sizze_9766) * sizze_9766;
    if (slt32(gtid_10066, sizze_9765) && slt32(gtid_10067, sizze_9766)) {
        float x_10081 = ((__global float *) x_mem_10627)[gtid_10066 *
                                                         sizze_9766 +
                                                         gtid_10067];
        float res_10082 = p_9767 * x_10081;
        
        ((__global float *) mem_10632)[gtid_10066 * sizze_9766 + gtid_10067] =
            res_10082;
    }
}
__kernel void segmap_10084(int32_t sizze_9774, __global
                           unsigned char *x_mem_10627, __global
                           unsigned char *y_mem_10628, __global
                           unsigned char *mem_10631)
{
    const int32_t segmap_group_sizze_10088 = add2zisegmap_group_sizze_10087;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10780;
    int32_t local_tid_10781;
    int32_t group_sizze_10784;
    int32_t wave_sizze_10783;
    int32_t group_tid_10782;
    
    global_tid_10780 = get_global_id(0);
    local_tid_10781 = get_local_id(0);
    group_sizze_10784 = get_local_size(0);
    wave_sizze_10783 = LOCKSTEP_WIDTH;
    group_tid_10782 = get_group_id(0);
    
    int32_t phys_tid_10084 = global_tid_10780;
    int32_t gtid_10083 = group_tid_10782 * segmap_group_sizze_10088 +
            local_tid_10781;
    
    if (slt32(gtid_10083, sizze_9774)) {
        float x_10095 = ((__global float *) x_mem_10627)[gtid_10083];
        float x_10096 = ((__global float *) y_mem_10628)[gtid_10083];
        float res_10097 = x_10095 + x_10096;
        
        ((__global float *) mem_10631)[gtid_10083] = res_10097;
    }
}
__kernel void segmap_10099(int32_t sizze_9789, __global
                           unsigned char *x_mem_10627, __global
                           unsigned char *y_mem_10628, __global
                           unsigned char *mem_10631)
{
    const int32_t segmap_group_sizze_10103 =
                  substract2zisegmap_group_sizze_10102;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10787;
    int32_t local_tid_10788;
    int32_t group_sizze_10791;
    int32_t wave_sizze_10790;
    int32_t group_tid_10789;
    
    global_tid_10787 = get_global_id(0);
    local_tid_10788 = get_local_id(0);
    group_sizze_10791 = get_local_size(0);
    wave_sizze_10790 = LOCKSTEP_WIDTH;
    group_tid_10789 = get_group_id(0);
    
    int32_t phys_tid_10099 = global_tid_10787;
    int32_t gtid_10098 = group_tid_10789 * segmap_group_sizze_10103 +
            local_tid_10788;
    
    if (slt32(gtid_10098, sizze_9789)) {
        float x_10110 = ((__global float *) x_mem_10627)[gtid_10098];
        float x_10111 = ((__global float *) y_mem_10628)[gtid_10098];
        float res_10112 = x_10110 - x_10111;
        
        ((__global float *) mem_10631)[gtid_10098] = res_10112;
    }
}
__kernel void segmap_10115(int32_t sizze_9804, int32_t sizze_9805, float s_9806,
                           __global unsigned char *x_mem_10627, __global
                           unsigned char *mem_10632)
{
    const int32_t segmap_group_sizze_10121 = lmataddzisegmap_group_sizze_10120;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10795;
    int32_t local_tid_10796;
    int32_t group_sizze_10799;
    int32_t wave_sizze_10798;
    int32_t group_tid_10797;
    
    global_tid_10795 = get_global_id(0);
    local_tid_10796 = get_local_id(0);
    group_sizze_10799 = get_local_size(0);
    wave_sizze_10798 = LOCKSTEP_WIDTH;
    group_tid_10797 = get_group_id(0);
    
    int32_t phys_tid_10115 = global_tid_10795;
    int32_t gtid_10113 = squot32(group_tid_10797 * segmap_group_sizze_10121 +
                                 local_tid_10796, sizze_9805);
    int32_t gtid_10114;
    
    gtid_10114 = group_tid_10797 * segmap_group_sizze_10121 + local_tid_10796 -
        squot32(group_tid_10797 * segmap_group_sizze_10121 + local_tid_10796,
                sizze_9805) * sizze_9805;
    if (slt32(gtid_10113, sizze_9804) && slt32(gtid_10114, sizze_9805)) {
        float x_10128 = ((__global float *) x_mem_10627)[gtid_10113 *
                                                         sizze_9805 +
                                                         gtid_10114];
        float res_10129 = s_9806 + x_10128;
        
        ((__global float *) mem_10632)[gtid_10113 * sizze_9805 + gtid_10114] =
            res_10129;
    }
}
__kernel void segmap_10132(int32_t sizze_9813, int32_t sizze_9814, float d_9815,
                           __global unsigned char *x_mem_10627, __global
                           unsigned char *mem_10632)
{
    const int32_t segmap_group_sizze_10138 =
                  lmatsubstractzisegmap_group_sizze_10137;
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
    
    int32_t phys_tid_10132 = global_tid_10803;
    int32_t gtid_10130 = squot32(group_tid_10805 * segmap_group_sizze_10138 +
                                 local_tid_10804, sizze_9814);
    int32_t gtid_10131;
    
    gtid_10131 = group_tid_10805 * segmap_group_sizze_10138 + local_tid_10804 -
        squot32(group_tid_10805 * segmap_group_sizze_10138 + local_tid_10804,
                sizze_9814) * sizze_9814;
    if (slt32(gtid_10130, sizze_9813) && slt32(gtid_10131, sizze_9814)) {
        float x_10145 = ((__global float *) x_mem_10627)[gtid_10130 *
                                                         sizze_9814 +
                                                         gtid_10131];
        float res_10146 = d_9815 - x_10145;
        
        ((__global float *) mem_10632)[gtid_10130 * sizze_9814 + gtid_10131] =
            res_10146;
    }
}
__kernel void segmap_10149(int32_t sizze_9822, int32_t sizze_9823, __global
                           unsigned char *x_mem_10627, __global
                           unsigned char *mem_10632)
{
    const int32_t segmap_group_sizze_10155 = sigmoidzisegmap_group_sizze_10154;
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
    
    int32_t phys_tid_10149 = global_tid_10811;
    int32_t gtid_10147 = squot32(group_tid_10813 * segmap_group_sizze_10155 +
                                 local_tid_10812, sizze_9823);
    int32_t gtid_10148;
    
    gtid_10148 = group_tid_10813 * segmap_group_sizze_10155 + local_tid_10812 -
        squot32(group_tid_10813 * segmap_group_sizze_10155 + local_tid_10812,
                sizze_9823) * sizze_9823;
    if (slt32(gtid_10147, sizze_9822) && slt32(gtid_10148, sizze_9823)) {
        float x_10162 = ((__global float *) x_mem_10627)[gtid_10147 *
                                                         sizze_9823 +
                                                         gtid_10148];
        float res_10163 = 0.0F - x_10162;
        float res_10164 = fpow32(2.7182817F, res_10163);
        float res_10165 = 1.0F + res_10164;
        float res_10166 = 1.0F / res_10165;
        
        ((__global float *) mem_10632)[gtid_10147 * sizze_9823 + gtid_10148] =
            res_10166;
    }
}
__kernel void segmap_10169(int32_t sizze_9833, int32_t sizze_9834,
                           int32_t sizze_9835, __global
                           unsigned char *u_mem_10627, __global
                           unsigned char *b_mem_10628, __global
                           unsigned char *mem_10631)
{
    const int32_t segmap_group_sizze_10173 = lvecmulzisegmap_group_sizze_10172;
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
    
    int32_t phys_tid_10169 = global_tid_10818;
    int32_t gtid_10168 = group_tid_10820 * segmap_group_sizze_10173 +
            local_tid_10819;
    
    if (slt32(gtid_10168, sizze_9835)) {
        int32_t binop_x_10605 = sizze_9833 * gtid_10168;
        float res_10181;
        float redout_10599 = 0.0F;
        
        for (int32_t i_10600 = 0; i_10600 < sizze_9833; i_10600++) {
            float x_10185 = ((__global float *) u_mem_10627)[i_10600];
            int32_t binop_x_10606 = i_10600 + binop_x_10605;
            int32_t new_index_10607 = squot32(binop_x_10606, sizze_9834);
            int32_t binop_y_10613 = sizze_9834 * new_index_10607;
            int32_t new_index_10614 = binop_x_10606 - binop_y_10613;
            float x_10186 = ((__global float *) b_mem_10628)[new_index_10614 *
                                                             sizze_9835 +
                                                             new_index_10607];
            float res_10187 = x_10185 * x_10186;
            float res_10184 = res_10187 + redout_10599;
            float redout_tmp_10823 = res_10184;
            
            redout_10599 = redout_tmp_10823;
        }
        res_10181 = redout_10599;
        ((__global float *) mem_10631)[gtid_10168] = res_10181;
    }
}
__kernel void segmap_10191(int32_t sizze_9855, int32_t sizze_9856,
                           int32_t sizze_9858, __global
                           unsigned char *x_mem_10627, __global
                           unsigned char *y_mem_10628, __global
                           unsigned char *mem_10633)
{
    const int32_t segmap_group_sizze_10197 =
                  matmultiplyzisegmap_group_sizze_10196;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10827;
    int32_t local_tid_10828;
    int32_t group_sizze_10831;
    int32_t wave_sizze_10830;
    int32_t group_tid_10829;
    
    global_tid_10827 = get_global_id(0);
    local_tid_10828 = get_local_id(0);
    group_sizze_10831 = get_local_size(0);
    wave_sizze_10830 = LOCKSTEP_WIDTH;
    group_tid_10829 = get_group_id(0);
    
    int32_t phys_tid_10191 = global_tid_10827;
    int32_t gtid_10189 = squot32(group_tid_10829 * segmap_group_sizze_10197 +
                                 local_tid_10828, sizze_9856);
    int32_t gtid_10190;
    
    gtid_10190 = group_tid_10829 * segmap_group_sizze_10197 + local_tid_10828 -
        squot32(group_tid_10829 * segmap_group_sizze_10197 + local_tid_10828,
                sizze_9856) * sizze_9856;
    if (slt32(gtid_10189, sizze_9855) && slt32(gtid_10190, sizze_9856)) {
        float x_10204 = ((__global float *) x_mem_10627)[gtid_10189 *
                                                         sizze_9856 +
                                                         gtid_10190];
        int32_t binop_x_10296 = sizze_9856 * gtid_10189;
        int32_t binop_x_10297 = gtid_10190 + binop_x_10296;
        int32_t new_index_10298 = squot32(binop_x_10297, sizze_9858);
        int32_t binop_y_10304 = sizze_9858 * new_index_10298;
        int32_t new_index_10305 = binop_x_10297 - binop_y_10304;
        float x_10205 = ((__global float *) y_mem_10628)[new_index_10298 *
                                                         sizze_9858 +
                                                         new_index_10305];
        float res_10206 = x_10204 * x_10205;
        
        ((__global float *) mem_10633)[gtid_10189 * sizze_9856 + gtid_10190] =
            res_10206;
    }
}
__kernel void segmap_10210(int32_t sizze_9884, int32_t sizze_9885,
                           int32_t sizze_9887, __global
                           unsigned char *x_mem_10627, __global
                           unsigned char *y_mem_10628, __global
                           unsigned char *mem_10633)
{
    const int32_t segmap_group_sizze_10216 = mataddzisegmap_group_sizze_10215;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10835;
    int32_t local_tid_10836;
    int32_t group_sizze_10839;
    int32_t wave_sizze_10838;
    int32_t group_tid_10837;
    
    global_tid_10835 = get_global_id(0);
    local_tid_10836 = get_local_id(0);
    group_sizze_10839 = get_local_size(0);
    wave_sizze_10838 = LOCKSTEP_WIDTH;
    group_tid_10837 = get_group_id(0);
    
    int32_t phys_tid_10210 = global_tid_10835;
    int32_t gtid_10208 = squot32(group_tid_10837 * segmap_group_sizze_10216 +
                                 local_tid_10836, sizze_9885);
    int32_t gtid_10209;
    
    gtid_10209 = group_tid_10837 * segmap_group_sizze_10216 + local_tid_10836 -
        squot32(group_tid_10837 * segmap_group_sizze_10216 + local_tid_10836,
                sizze_9885) * sizze_9885;
    if (slt32(gtid_10208, sizze_9884) && slt32(gtid_10209, sizze_9885)) {
        float x_10223 = ((__global float *) x_mem_10627)[gtid_10208 *
                                                         sizze_9885 +
                                                         gtid_10209];
        int32_t binop_x_10296 = sizze_9885 * gtid_10208;
        int32_t binop_x_10297 = gtid_10209 + binop_x_10296;
        int32_t new_index_10298 = squot32(binop_x_10297, sizze_9887);
        int32_t binop_y_10304 = sizze_9887 * new_index_10298;
        int32_t new_index_10305 = binop_x_10297 - binop_y_10304;
        float x_10224 = ((__global float *) y_mem_10628)[new_index_10298 *
                                                         sizze_9887 +
                                                         new_index_10305];
        float res_10225 = x_10223 + x_10224;
        
        ((__global float *) mem_10633)[gtid_10208 * sizze_9885 + gtid_10209] =
            res_10225;
    }
}
__kernel void segmap_10229(int32_t sizze_9913, int32_t sizze_9914,
                           int32_t sizze_9916, __global
                           unsigned char *x_mem_10627, __global
                           unsigned char *y_mem_10628, __global
                           unsigned char *mem_10633)
{
    const int32_t segmap_group_sizze_10235 =
                  matsubstractzisegmap_group_sizze_10234;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10843;
    int32_t local_tid_10844;
    int32_t group_sizze_10847;
    int32_t wave_sizze_10846;
    int32_t group_tid_10845;
    
    global_tid_10843 = get_global_id(0);
    local_tid_10844 = get_local_id(0);
    group_sizze_10847 = get_local_size(0);
    wave_sizze_10846 = LOCKSTEP_WIDTH;
    group_tid_10845 = get_group_id(0);
    
    int32_t phys_tid_10229 = global_tid_10843;
    int32_t gtid_10227 = squot32(group_tid_10845 * segmap_group_sizze_10235 +
                                 local_tid_10844, sizze_9914);
    int32_t gtid_10228;
    
    gtid_10228 = group_tid_10845 * segmap_group_sizze_10235 + local_tid_10844 -
        squot32(group_tid_10845 * segmap_group_sizze_10235 + local_tid_10844,
                sizze_9914) * sizze_9914;
    if (slt32(gtid_10227, sizze_9913) && slt32(gtid_10228, sizze_9914)) {
        float x_10242 = ((__global float *) x_mem_10627)[gtid_10227 *
                                                         sizze_9914 +
                                                         gtid_10228];
        int32_t binop_x_10296 = sizze_9914 * gtid_10227;
        int32_t binop_x_10297 = gtid_10228 + binop_x_10296;
        int32_t new_index_10298 = squot32(binop_x_10297, sizze_9916);
        int32_t binop_y_10304 = sizze_9916 * new_index_10298;
        int32_t new_index_10305 = binop_x_10297 - binop_y_10304;
        float x_10243 = ((__global float *) y_mem_10628)[new_index_10298 *
                                                         sizze_9916 +
                                                         new_index_10305];
        float res_10244 = x_10242 - x_10243;
        
        ((__global float *) mem_10633)[gtid_10227 * sizze_9914 + gtid_10228] =
            res_10244;
    }
}
__kernel void segmap_9968(int32_t sizze_9716, __global
                          unsigned char *x_mem_10627, __global
                          unsigned char *mem_10630)
{
    const int32_t segmap_group_sizze_9972 = expzisegmap_group_sizze_9971;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10723;
    int32_t local_tid_10724;
    int32_t group_sizze_10727;
    int32_t wave_sizze_10726;
    int32_t group_tid_10725;
    
    global_tid_10723 = get_global_id(0);
    local_tid_10724 = get_local_id(0);
    group_sizze_10727 = get_local_size(0);
    wave_sizze_10726 = LOCKSTEP_WIDTH;
    group_tid_10725 = get_group_id(0);
    
    int32_t phys_tid_9968 = global_tid_10723;
    int32_t gtid_9967 = group_tid_10725 * segmap_group_sizze_9972 +
            local_tid_10724;
    
    if (slt32(gtid_9967, sizze_9716)) {
        float x_9979 = ((__global float *) x_mem_10627)[gtid_9967];
        float res_9980 = fpow32(2.7182817F, x_9979);
        
        ((__global float *) mem_10630)[gtid_9967] = res_9980;
    }
}
__kernel void segmap_9982(int32_t sizze_9721, __global
                          unsigned char *x_mem_10627, __global
                          unsigned char *mem_10630)
{
    const int32_t segmap_group_sizze_9986 = negationzisegmap_group_sizze_9985;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10730;
    int32_t local_tid_10731;
    int32_t group_sizze_10734;
    int32_t wave_sizze_10733;
    int32_t group_tid_10732;
    
    global_tid_10730 = get_global_id(0);
    local_tid_10731 = get_local_id(0);
    group_sizze_10734 = get_local_size(0);
    wave_sizze_10733 = LOCKSTEP_WIDTH;
    group_tid_10732 = get_group_id(0);
    
    int32_t phys_tid_9982 = global_tid_10730;
    int32_t gtid_9981 = group_tid_10732 * segmap_group_sizze_9986 +
            local_tid_10731;
    
    if (slt32(gtid_9981, sizze_9721)) {
        float x_9993 = ((__global float *) x_mem_10627)[gtid_9981];
        float res_9994 = 0.0F - x_9993;
        
        ((__global float *) mem_10630)[gtid_9981] = res_9994;
    }
}
__kernel void segmap_9996(int32_t sizze_9726, float d_9727, __global
                          unsigned char *x_mem_10627, __global
                          unsigned char *mem_10630)
{
    const int32_t segmap_group_sizze_10000 = dividezisegmap_group_sizze_9999;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t global_tid_10737;
    int32_t local_tid_10738;
    int32_t group_sizze_10741;
    int32_t wave_sizze_10740;
    int32_t group_tid_10739;
    
    global_tid_10737 = get_global_id(0);
    local_tid_10738 = get_local_id(0);
    group_sizze_10741 = get_local_size(0);
    wave_sizze_10740 = LOCKSTEP_WIDTH;
    group_tid_10739 = get_group_id(0);
    
    int32_t phys_tid_9996 = global_tid_10737;
    int32_t gtid_9995 = group_tid_10739 * segmap_group_sizze_10000 +
            local_tid_10738;
    
    if (slt32(gtid_9995, sizze_9726)) {
        float x_10007 = ((__global float *) x_mem_10627)[gtid_9995];
        float res_10008 = d_9727 / x_10007;
        
        ((__global float *) mem_10630)[gtid_9995] = res_10008;
    }
}
__kernel void segmap_intragroup_10319(__local volatile
                                      int64_t *mem_10637_backing_aligned_0,
                                      __local volatile
                                      int64_t *mem_10642_backing_aligned_1,
                                      __local volatile
                                      int64_t *mem_10652_backing_aligned_2,
                                      __local volatile
                                      int64_t *mem_10657_backing_aligned_3,
                                      int32_t sizze_9942, int32_t sizze_9943,
                                      int32_t sizze_9945,
                                      int32_t num_groups_y_10317,
                                      int32_t num_whole_tiles_10320,
                                      int32_t residual_input_10453,
                                      unsigned char cond_10454, __global
                                      unsigned char *a_mem_10627, __global
                                      unsigned char *b_mem_10628, __global
                                      unsigned char *mem_10667)
{
    const int32_t tile_sizze_10310 = dotzitile_sizze_10309;
    const int32_t group_sizze_10311 = dotzitile_sizze_10309 *
                  dotzitile_sizze_10309;
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_10637_backing_0 = (__local volatile
                                                           char *) mem_10637_backing_aligned_0;
    __local volatile char *restrict mem_10642_backing_1 = (__local volatile
                                                           char *) mem_10642_backing_aligned_1;
    __local volatile char *restrict mem_10652_backing_2 = (__local volatile
                                                           char *) mem_10652_backing_aligned_2;
    __local volatile char *restrict mem_10657_backing_3 = (__local volatile
                                                           char *) mem_10657_backing_aligned_3;
    int32_t global_tid_10851;
    int32_t local_tid_10852;
    int32_t group_sizze_10855;
    int32_t wave_sizze_10854;
    int32_t group_tid_10853;
    
    global_tid_10851 = get_global_id(0);
    local_tid_10852 = get_local_id(0);
    group_sizze_10855 = get_local_size(0);
    wave_sizze_10854 = LOCKSTEP_WIDTH;
    group_tid_10853 = get_group_id(0);
    
    int32_t gid_flat_10319 = group_tid_10853;
    int32_t gid_x_10307 = squot32(group_tid_10853, num_groups_y_10317);
    int32_t gid_y_10308;
    
    gid_y_10308 = group_tid_10853 - squot32(group_tid_10853,
                                            num_groups_y_10317) *
        num_groups_y_10317;
    
    float mem_10632;
    int32_t ltid_x_10336 = squot32(local_tid_10852, tile_sizze_10310);
    int32_t ltid_y_10337;
    
    ltid_y_10337 = local_tid_10852 - squot32(local_tid_10852,
                                             tile_sizze_10310) *
        tile_sizze_10310;
    
    int32_t ltid_flat_10338;
    
    ltid_flat_10338 = local_tid_10852;
    if (slt32(ltid_x_10336, tile_sizze_10310) && slt32(ltid_y_10337,
                                                       tile_sizze_10310)) {
        mem_10632 = 0.0F;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t binop_x_10415 = gid_x_10307 * tile_sizze_10310;
    int32_t binop_x_10417 = gid_y_10308 * tile_sizze_10310;
    __local char *mem_10637;
    
    mem_10637 = (__local char *) mem_10637_backing_0;
    
    __local char *mem_10642;
    
    mem_10642 = (__local char *) mem_10642_backing_1;
    
    float mem_10646;
    
    for (int32_t tile_id_10347 = 0; tile_id_10347 < num_whole_tiles_10320;
         tile_id_10347++) {
        int32_t binop_x_10411 = tile_sizze_10310 * tile_id_10347;
        int32_t ltid_x_10348 = squot32(local_tid_10852, tile_sizze_10310);
        int32_t ltid_y_10349;
        
        ltid_y_10349 = local_tid_10852 - squot32(local_tid_10852,
                                                 tile_sizze_10310) *
            tile_sizze_10310;
        
        int32_t ltid_flat_10350;
        
        ltid_flat_10350 = local_tid_10852;
        if (slt32(ltid_x_10348, tile_sizze_10310) && slt32(ltid_y_10349,
                                                           tile_sizze_10310)) {
            int32_t i_10412 = ltid_x_10348 + binop_x_10411;
            int32_t j_10414 = ltid_y_10349 + binop_x_10411;
            int32_t gtid_10416 = ltid_x_10348 + binop_x_10415;
            int32_t gtid_10418 = ltid_y_10349 + binop_x_10417;
            float tile_elem_10422 = ((__global
                                      float *) a_mem_10627)[gtid_10416 *
                                                            sizze_9943 +
                                                            j_10414];
            float tile_elem_10423 = ((__global float *) b_mem_10628)[i_10412 *
                                                                     sizze_9945 +
                                                                     gtid_10418];
            
            ((__local float *) mem_10637)[ltid_x_10348 * tile_sizze_10310 +
                                          ltid_y_10349] = tile_elem_10422;
            ((__local float *) mem_10642)[ltid_x_10348 * tile_sizze_10310 +
                                          ltid_y_10349] = tile_elem_10423;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_10374 = squot32(local_tid_10852, tile_sizze_10310);
        int32_t ltid_y_10375;
        
        ltid_y_10375 = local_tid_10852 - squot32(local_tid_10852,
                                                 tile_sizze_10310) *
            tile_sizze_10310;
        
        int32_t ltid_flat_10376;
        
        ltid_flat_10376 = local_tid_10852;
        if (slt32(ltid_x_10374, tile_sizze_10310) && slt32(ltid_y_10375,
                                                           tile_sizze_10310)) {
            int32_t gtid_10426 = ltid_x_10374 + binop_x_10415;
            int32_t gtid_10428 = ltid_y_10375 + binop_x_10417;
            float acc_10432 = mem_10632;
            bool binop_x_10435 = slt32(gtid_10426, sizze_9942);
            bool binop_y_10436 = slt32(gtid_10428, sizze_9945);
            bool cond_10437 = binop_x_10435 && binop_y_10436;
            float acc_10438;
            
            if (cond_10437) {
                float x_10439;
                float redout_10599 = acc_10432;
                
                for (int32_t i_10600 = 0; i_10600 < tile_sizze_10310;
                     i_10600++) {
                    float x_10443 = ((__local float *) mem_10637)[ltid_x_10374 *
                                                                  tile_sizze_10310 +
                                                                  i_10600];
                    float x_10444 = ((__local float *) mem_10642)[i_10600 *
                                                                  tile_sizze_10310 +
                                                                  ltid_y_10375];
                    float res_10445 = x_10443 * x_10444;
                    float res_10442 = res_10445 + redout_10599;
                    float redout_tmp_10857 = res_10442;
                    
                    redout_10599 = redout_tmp_10857;
                }
                x_10439 = redout_10599;
                acc_10438 = x_10439;
            } else {
                acc_10438 = acc_10432;
            }
            mem_10646 = acc_10438;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_10858 = 0; i_10858 < squot32(tile_sizze_10310 *
                                                    tile_sizze_10310 -
                                                    local_tid_10852 +
                                                    group_sizze_10311 - 1,
                                                    group_sizze_10311);
             i_10858++) {
            mem_10632 = mem_10646;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    __local char *mem_10652;
    
    mem_10652 = (__local char *) mem_10652_backing_2;
    
    __local char *mem_10657;
    
    mem_10657 = (__local char *) mem_10657_backing_3;
    
    float mem_10661;
    float mem_10679;
    
    if (cond_10454) {
        for (int32_t i_10859 = 0; i_10859 < squot32(tile_sizze_10310 *
                                                    tile_sizze_10310 -
                                                    local_tid_10852 +
                                                    group_sizze_10311 - 1,
                                                    group_sizze_10311);
             i_10859++) {
            mem_10679 = mem_10632;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
        int32_t binop_x_10540 = tile_sizze_10310 * num_whole_tiles_10320;
        int32_t ltid_x_10455 = squot32(local_tid_10852, tile_sizze_10310);
        int32_t ltid_y_10456;
        
        ltid_y_10456 = local_tid_10852 - squot32(local_tid_10852,
                                                 tile_sizze_10310) *
            tile_sizze_10310;
        
        int32_t ltid_flat_10457;
        
        ltid_flat_10457 = local_tid_10852;
        if (slt32(ltid_x_10455, tile_sizze_10310) && slt32(ltid_y_10456,
                                                           tile_sizze_10310)) {
            int32_t i_10541 = ltid_x_10455 + binop_x_10540;
            int32_t j_10543 = ltid_y_10456 + binop_x_10540;
            int32_t gtid_10545 = binop_x_10415 + ltid_x_10455;
            int32_t gtid_10547 = binop_x_10417 + ltid_y_10456;
            bool binop_x_10551 = slt32(j_10543, sizze_9943);
            bool binop_y_10552 = slt32(gtid_10545, sizze_9942);
            bool cond_10553 = binop_x_10551 && binop_y_10552;
            float pre_10554;
            
            if (cond_10553) {
                float x_10555 = ((__global float *) a_mem_10627)[gtid_10545 *
                                                                 sizze_9943 +
                                                                 j_10543];
                
                pre_10554 = x_10555;
            } else {
                pre_10554 = 0.0F;
            }
            
            bool binop_x_10557 = slt32(i_10541, sizze_9943);
            bool binop_y_10558 = slt32(gtid_10547, sizze_9945);
            bool cond_10559 = binop_x_10557 && binop_y_10558;
            float pre_10560;
            
            if (cond_10559) {
                float x_10561 = ((__global float *) b_mem_10628)[i_10541 *
                                                                 sizze_9945 +
                                                                 gtid_10547];
                
                pre_10560 = x_10561;
            } else {
                pre_10560 = 0.0F;
            }
            ((__local float *) mem_10652)[ltid_x_10455 * tile_sizze_10310 +
                                          ltid_y_10456] = pre_10554;
            ((__local float *) mem_10657)[ltid_x_10455 * tile_sizze_10310 +
                                          ltid_y_10456] = pre_10560;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_10503 = squot32(local_tid_10852, tile_sizze_10310);
        int32_t ltid_y_10504;
        
        ltid_y_10504 = local_tid_10852 - squot32(local_tid_10852,
                                                 tile_sizze_10310) *
            tile_sizze_10310;
        
        int32_t ltid_flat_10505;
        
        ltid_flat_10505 = local_tid_10852;
        if (slt32(ltid_x_10503, tile_sizze_10310) && slt32(ltid_y_10504,
                                                           tile_sizze_10310)) {
            int32_t gtid_10567 = binop_x_10415 + ltid_x_10503;
            int32_t gtid_10569 = binop_x_10417 + ltid_y_10504;
            float acc_10573 = mem_10632;
            bool binop_x_10576 = slt32(gtid_10567, sizze_9942);
            bool binop_y_10577 = slt32(gtid_10569, sizze_9945);
            bool cond_10578 = binop_x_10576 && binop_y_10577;
            float acc_10579;
            
            if (cond_10578) {
                float x_10580;
                float redout_10601 = acc_10573;
                
                for (int32_t i_10602 = 0; i_10602 < residual_input_10453;
                     i_10602++) {
                    float x_10584 = ((__local float *) mem_10652)[ltid_x_10503 *
                                                                  tile_sizze_10310 +
                                                                  i_10602];
                    float x_10585 = ((__local float *) mem_10657)[i_10602 *
                                                                  tile_sizze_10310 +
                                                                  ltid_y_10504];
                    float res_10586 = x_10584 * x_10585;
                    float res_10583 = res_10586 + redout_10601;
                    float redout_tmp_10860 = res_10583;
                    
                    redout_10601 = redout_tmp_10860;
                }
                x_10580 = redout_10601;
                acc_10579 = x_10580;
            } else {
                acc_10579 = acc_10573;
            }
            mem_10661 = acc_10579;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_10861 = 0; i_10861 < squot32(tile_sizze_10310 *
                                                    tile_sizze_10310 -
                                                    local_tid_10852 +
                                                    group_sizze_10311 - 1,
                                                    group_sizze_10311);
             i_10861++) {
            mem_10679 = mem_10661;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    int32_t thread_out_index_10862 = gid_x_10307 * tile_sizze_10310 +
            squot32(local_tid_10852, tile_sizze_10310);
    int32_t thread_out_index_10863;
    
    thread_out_index_10863 = gid_y_10308 * tile_sizze_10310 + (local_tid_10852 -
                                                               squot32(local_tid_10852,
                                                                       tile_sizze_10310) *
                                                               tile_sizze_10310);
    if (slt32(thread_out_index_10862, sizze_9942) &&
        slt32(thread_out_index_10863, sizze_9945)) {
        ((__global float *) mem_10667)[thread_out_index_10862 * sizze_9945 +
                                       thread_out_index_10863] = mem_10679;
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
                                       all_sizes={"add.segmap_group_size_10027": {"class": "group_size", "value": None},
                                        "add2.segmap_group_size_10087": {"class": "group_size", "value": None},
                                        "divide.segmap_group_size_9999": {"class": "group_size", "value": None},
                                        "dot.tile_size_10309": {"class": "tile_size", "value": None},
                                        "exp.segmap_group_size_9971": {"class": "group_size", "value": None},
                                        "lmatadd.segmap_group_size_10120": {"class": "group_size", "value": None},
                                        "lmatmultiply.segmap_group_size_10073": {"class": "group_size", "value": None},
                                        "lmatsubstract.segmap_group_size_10137": {"class": "group_size",
                                                                                  "value": None},
                                        "lvecmul.segmap_group_size_10172": {"class": "group_size", "value": None},
                                        "matadd.segmap_group_size_10215": {"class": "group_size", "value": None},
                                        "matmultiply.segmap_group_size_10196": {"class": "group_size", "value": None},
                                        "matsubstract.segmap_group_size_10234": {"class": "group_size", "value": None},
                                        "multiply.segmap_group_size_10013": {"class": "group_size", "value": None},
                                        "multiply2.segmap_group_size_10055": {"class": "group_size", "value": None},
                                        "negation.segmap_group_size_9985": {"class": "group_size", "value": None},
                                        "sigmoid.segmap_group_size_10154": {"class": "group_size", "value": None},
                                        "substract.segmap_group_size_10041": {"class": "group_size", "value": None},
                                        "substract2.segmap_group_size_10102": {"class": "group_size", "value": None}})
    self.map_transpose_f32_var = program.map_transpose_f32
    self.map_transpose_f32_low_height_var = program.map_transpose_f32_low_height
    self.map_transpose_f32_low_width_var = program.map_transpose_f32_low_width
    self.map_transpose_f32_small_var = program.map_transpose_f32_small
    self.segmap_10010_var = program.segmap_10010
    self.segmap_10024_var = program.segmap_10024
    self.segmap_10038_var = program.segmap_10038
    self.segmap_10052_var = program.segmap_10052
    self.segmap_10068_var = program.segmap_10068
    self.segmap_10084_var = program.segmap_10084
    self.segmap_10099_var = program.segmap_10099
    self.segmap_10115_var = program.segmap_10115
    self.segmap_10132_var = program.segmap_10132
    self.segmap_10149_var = program.segmap_10149
    self.segmap_10169_var = program.segmap_10169
    self.segmap_10191_var = program.segmap_10191
    self.segmap_10210_var = program.segmap_10210
    self.segmap_10229_var = program.segmap_10229
    self.segmap_9968_var = program.segmap_9968
    self.segmap_9982_var = program.segmap_9982
    self.segmap_9996_var = program.segmap_9996
    self.segmap_intragroup_10319_var = program.segmap_intragroup_10319
  def futhark_dot(self, a_mem_10627, b_mem_10628, sizze_9942, sizze_9943,
                  sizze_9944, sizze_9945):
    dim_zzero_9949 = (np.int32(0) == sizze_9944)
    dim_zzero_9950 = (np.int32(0) == sizze_9943)
    both_empty_9951 = (dim_zzero_9949 and dim_zzero_9950)
    dim_match_9952 = (sizze_9943 == sizze_9944)
    empty_or_match_9953 = (both_empty_9951 or dim_match_9952)
    empty_or_match_cert_9954 = True
    assert empty_or_match_9953, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:10:19-30\n   #1  GPU.fut:15:21-38\n   #2  GPU.fut:15:3-16:17\n   #3  GPU.fut:25:18-33\n   #4  GPU.fut:25:3-36\n   #5  GPU.fut:24:1-25:36\n" % ("function arguments of wrong shape",))
    tile_sizze_10310 = self.sizes["dot.tile_size_10309"]
    group_sizze_10311 = (tile_sizze_10310 * tile_sizze_10310)
    y_10312 = (tile_sizze_10310 - np.int32(1))
    x_10313 = (sizze_9942 + y_10312)
    num_groups_x_10314 = squot32(x_10313, tile_sizze_10310)
    x_10316 = (sizze_9945 + y_10312)
    num_groups_y_10317 = squot32(x_10316, tile_sizze_10310)
    num_groups_top_10318 = (num_groups_x_10314 * num_groups_y_10317)
    num_whole_tiles_10320 = squot32(sizze_9943, tile_sizze_10310)
    residual_input_10453 = srem32(sizze_9943, tile_sizze_10310)
    cond_10454 = (residual_input_10453 == np.int32(0))
    binop_x_10664 = sext_i32_i64(sizze_9942)
    binop_y_10665 = sext_i32_i64(sizze_9945)
    binop_x_10666 = (binop_x_10664 * binop_y_10665)
    bytes_10663 = (np.int64(4) * binop_x_10666)
    mem_10667 = opencl_alloc(self, bytes_10663, "mem_10667")
    binop_x_10631 = sext_i32_i64(group_sizze_10311)
    bytes_10629 = (np.int64(4) * binop_x_10631)
    binop_x_10634 = sext_i32_i64(tile_sizze_10310)
    binop_x_10636 = (binop_x_10634 * binop_x_10634)
    bytes_10633 = (np.int64(4) * binop_x_10636)
    if ((1 * (np.long(num_groups_top_10318) * np.long(group_sizze_10311))) != 0):
      self.segmap_intragroup_10319_var.set_args(cl.LocalMemory(np.long(bytes_10633)),
                                                cl.LocalMemory(np.long(bytes_10633)),
                                                cl.LocalMemory(np.long(bytes_10633)),
                                                cl.LocalMemory(np.long(bytes_10633)),
                                                np.int32(sizze_9942),
                                                np.int32(sizze_9943),
                                                np.int32(sizze_9945),
                                                np.int32(num_groups_y_10317),
                                                np.int32(num_whole_tiles_10320),
                                                np.int32(residual_input_10453),
                                                np.byte(cond_10454),
                                                a_mem_10627, b_mem_10628,
                                                mem_10667)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_intragroup_10319_var,
                                 ((np.long(num_groups_top_10318) * np.long(group_sizze_10311)),),
                                 (np.long(group_sizze_10311),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10849 = sizze_9942
    out_arrsizze_10850 = sizze_9945
    out_mem_10848 = mem_10667
    return (out_mem_10848, out_arrsizze_10849, out_arrsizze_10850)
  def futhark_matsubstract(self, x_mem_10627, y_mem_10628, sizze_9913,
                           sizze_9914, sizze_9915, sizze_9916):
    dim_zzero_9919 = (np.int32(0) == sizze_9915)
    dim_zzero_9920 = (np.int32(0) == sizze_9916)
    old_empty_9921 = (dim_zzero_9919 or dim_zzero_9920)
    dim_zzero_9922 = (np.int32(0) == sizze_9913)
    new_empty_9923 = (dim_zzero_9920 or dim_zzero_9922)
    both_empty_9924 = (old_empty_9921 and new_empty_9923)
    dim_match_9925 = (sizze_9913 == sizze_9915)
    empty_or_match_9926 = (both_empty_9924 or dim_match_9925)
    empty_or_match_cert_9927 = True
    assert empty_or_match_9926, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:150:3-21\n   #1  GPU.fut:149:1-150:21\n" % ("function arguments of wrong shape",))
    dim_zzero_9929 = (np.int32(0) == sizze_9914)
    both_empty_9930 = (dim_zzero_9920 and dim_zzero_9929)
    dim_match_9931 = (sizze_9914 == sizze_9916)
    empty_or_match_9932 = (both_empty_9930 or dim_match_9931)
    empty_or_match_cert_9933 = True
    assert empty_or_match_9932, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:141:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPU.fut:150:3-21\n   #4  GPU.fut:149:1-150:21\n" % ("function arguments of wrong shape",))
    sizze_10230 = sext_i32_i64(sizze_9913)
    sizze_10231 = sext_i32_i64(sizze_9914)
    nest_sizze_10233 = (sizze_10230 * sizze_10231)
    segmap_group_sizze_10235 = self.sizes["matsubstract.segmap_group_size_10234"]
    segmap_group_sizze_10236 = sext_i32_i64(segmap_group_sizze_10235)
    y_10237 = (segmap_group_sizze_10236 - np.int64(1))
    x_10238 = (nest_sizze_10233 + y_10237)
    segmap_usable_groups_64_10240 = squot64(x_10238, segmap_group_sizze_10236)
    segmap_usable_groups_10241 = sext_i64_i32(segmap_usable_groups_64_10240)
    bytes_10629 = (np.int64(4) * nest_sizze_10233)
    mem_10633 = opencl_alloc(self, bytes_10629, "mem_10633")
    if ((1 * (np.long(segmap_usable_groups_10241) * np.long(segmap_group_sizze_10235))) != 0):
      self.segmap_10229_var.set_args(np.int32(sizze_9913), np.int32(sizze_9914),
                                     np.int32(sizze_9916), x_mem_10627,
                                     y_mem_10628, mem_10633)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10229_var,
                                 ((np.long(segmap_usable_groups_10241) * np.long(segmap_group_sizze_10235)),),
                                 (np.long(segmap_group_sizze_10235),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10841 = sizze_9913
    out_arrsizze_10842 = sizze_9914
    out_mem_10840 = mem_10633
    return (out_mem_10840, out_arrsizze_10841, out_arrsizze_10842)
  def futhark_matadd(self, x_mem_10627, y_mem_10628, sizze_9884, sizze_9885,
                     sizze_9886, sizze_9887):
    dim_zzero_9890 = (np.int32(0) == sizze_9886)
    dim_zzero_9891 = (np.int32(0) == sizze_9887)
    old_empty_9892 = (dim_zzero_9890 or dim_zzero_9891)
    dim_zzero_9893 = (np.int32(0) == sizze_9884)
    new_empty_9894 = (dim_zzero_9891 or dim_zzero_9893)
    both_empty_9895 = (old_empty_9892 and new_empty_9894)
    dim_match_9896 = (sizze_9884 == sizze_9886)
    empty_or_match_9897 = (both_empty_9895 or dim_match_9896)
    empty_or_match_cert_9898 = True
    assert empty_or_match_9897, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:114:3-15\n   #1  GPU.fut:113:1-114:15\n" % ("function arguments of wrong shape",))
    dim_zzero_9900 = (np.int32(0) == sizze_9885)
    both_empty_9901 = (dim_zzero_9891 and dim_zzero_9900)
    dim_match_9902 = (sizze_9885 == sizze_9887)
    empty_or_match_9903 = (both_empty_9901 or dim_match_9902)
    empty_or_match_cert_9904 = True
    assert empty_or_match_9903, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:105:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPU.fut:114:3-15\n   #4  GPU.fut:113:1-114:15\n" % ("function arguments of wrong shape",))
    sizze_10211 = sext_i32_i64(sizze_9884)
    sizze_10212 = sext_i32_i64(sizze_9885)
    nest_sizze_10214 = (sizze_10211 * sizze_10212)
    segmap_group_sizze_10216 = self.sizes["matadd.segmap_group_size_10215"]
    segmap_group_sizze_10217 = sext_i32_i64(segmap_group_sizze_10216)
    y_10218 = (segmap_group_sizze_10217 - np.int64(1))
    x_10219 = (nest_sizze_10214 + y_10218)
    segmap_usable_groups_64_10221 = squot64(x_10219, segmap_group_sizze_10217)
    segmap_usable_groups_10222 = sext_i64_i32(segmap_usable_groups_64_10221)
    bytes_10629 = (np.int64(4) * nest_sizze_10214)
    mem_10633 = opencl_alloc(self, bytes_10629, "mem_10633")
    if ((1 * (np.long(segmap_usable_groups_10222) * np.long(segmap_group_sizze_10216))) != 0):
      self.segmap_10210_var.set_args(np.int32(sizze_9884), np.int32(sizze_9885),
                                     np.int32(sizze_9887), x_mem_10627,
                                     y_mem_10628, mem_10633)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10210_var,
                                 ((np.long(segmap_usable_groups_10222) * np.long(segmap_group_sizze_10216)),),
                                 (np.long(segmap_group_sizze_10216),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10833 = sizze_9884
    out_arrsizze_10834 = sizze_9885
    out_mem_10832 = mem_10633
    return (out_mem_10832, out_arrsizze_10833, out_arrsizze_10834)
  def futhark_matmultiply(self, x_mem_10627, y_mem_10628, sizze_9855,
                          sizze_9856, sizze_9857, sizze_9858):
    dim_zzero_9861 = (np.int32(0) == sizze_9857)
    dim_zzero_9862 = (np.int32(0) == sizze_9858)
    old_empty_9863 = (dim_zzero_9861 or dim_zzero_9862)
    dim_zzero_9864 = (np.int32(0) == sizze_9855)
    new_empty_9865 = (dim_zzero_9862 or dim_zzero_9864)
    both_empty_9866 = (old_empty_9863 and new_empty_9865)
    dim_match_9867 = (sizze_9855 == sizze_9857)
    empty_or_match_9868 = (both_empty_9866 or dim_match_9867)
    empty_or_match_cert_9869 = True
    assert empty_or_match_9868, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:80:3-20\n   #1  GPU.fut:79:1-80:20\n" % ("function arguments of wrong shape",))
    dim_zzero_9871 = (np.int32(0) == sizze_9856)
    both_empty_9872 = (dim_zzero_9862 and dim_zzero_9871)
    dim_match_9873 = (sizze_9856 == sizze_9858)
    empty_or_match_9874 = (both_empty_9872 or dim_match_9873)
    empty_or_match_cert_9875 = True
    assert empty_or_match_9874, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:71:3-14\n   #1  /futlib/soacs.fut:51:19-23\n   #2  /futlib/soacs.fut:51:3-37\n   #3  GPU.fut:80:3-20\n   #4  GPU.fut:79:1-80:20\n" % ("function arguments of wrong shape",))
    sizze_10192 = sext_i32_i64(sizze_9855)
    sizze_10193 = sext_i32_i64(sizze_9856)
    nest_sizze_10195 = (sizze_10192 * sizze_10193)
    segmap_group_sizze_10197 = self.sizes["matmultiply.segmap_group_size_10196"]
    segmap_group_sizze_10198 = sext_i32_i64(segmap_group_sizze_10197)
    y_10199 = (segmap_group_sizze_10198 - np.int64(1))
    x_10200 = (nest_sizze_10195 + y_10199)
    segmap_usable_groups_64_10202 = squot64(x_10200, segmap_group_sizze_10198)
    segmap_usable_groups_10203 = sext_i64_i32(segmap_usable_groups_64_10202)
    bytes_10629 = (np.int64(4) * nest_sizze_10195)
    mem_10633 = opencl_alloc(self, bytes_10629, "mem_10633")
    if ((1 * (np.long(segmap_usable_groups_10203) * np.long(segmap_group_sizze_10197))) != 0):
      self.segmap_10191_var.set_args(np.int32(sizze_9855), np.int32(sizze_9856),
                                     np.int32(sizze_9858), x_mem_10627,
                                     y_mem_10628, mem_10633)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10191_var,
                                 ((np.long(segmap_usable_groups_10203) * np.long(segmap_group_sizze_10197)),),
                                 (np.long(segmap_group_sizze_10197),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10825 = sizze_9855
    out_arrsizze_10826 = sizze_9856
    out_mem_10824 = mem_10633
    return (out_mem_10824, out_arrsizze_10825, out_arrsizze_10826)
  def futhark_lvecmul(self, u_mem_10627, b_mem_10628, sizze_9833, sizze_9834,
                      sizze_9835):
    dim_zzero_9839 = (np.int32(0) == sizze_9834)
    dim_zzero_9840 = (np.int32(0) == sizze_9833)
    both_empty_9841 = (dim_zzero_9839 and dim_zzero_9840)
    dim_match_9842 = (sizze_9833 == sizze_9834)
    empty_or_match_9843 = (both_empty_9841 or dim_match_9842)
    empty_or_match_cert_9844 = True
    assert empty_or_match_9843, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:10:19-30\n   #1  GPU.fut:15:21-38\n   #2  GPU.fut:15:3-16:17\n   #3  GPU.fut:14:1-16:17\n" % ("function arguments of wrong shape",))
    sizze_10170 = sext_i32_i64(sizze_9835)
    segmap_group_sizze_10173 = self.sizes["lvecmul.segmap_group_size_10172"]
    segmap_group_sizze_10174 = sext_i32_i64(segmap_group_sizze_10173)
    y_10175 = (segmap_group_sizze_10174 - np.int64(1))
    x_10176 = (sizze_10170 + y_10175)
    segmap_usable_groups_64_10178 = squot64(x_10176, segmap_group_sizze_10174)
    segmap_usable_groups_10179 = sext_i64_i32(segmap_usable_groups_64_10178)
    bytes_10629 = (np.int64(4) * sizze_10170)
    mem_10631 = opencl_alloc(self, bytes_10629, "mem_10631")
    if ((1 * (np.long(segmap_usable_groups_10179) * np.long(segmap_group_sizze_10173))) != 0):
      self.segmap_10169_var.set_args(np.int32(sizze_9833), np.int32(sizze_9834),
                                     np.int32(sizze_9835), u_mem_10627,
                                     b_mem_10628, mem_10631)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10169_var,
                                 ((np.long(segmap_usable_groups_10179) * np.long(segmap_group_sizze_10173)),),
                                 (np.long(segmap_group_sizze_10173),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10817 = sizze_9835
    out_mem_10816 = mem_10631
    return (out_mem_10816, out_arrsizze_10817)
  def futhark_sigmoid(self, x_mem_10627, sizze_9822, sizze_9823):
    sizze_10150 = sext_i32_i64(sizze_9822)
    sizze_10151 = sext_i32_i64(sizze_9823)
    nest_sizze_10153 = (sizze_10150 * sizze_10151)
    segmap_group_sizze_10155 = self.sizes["sigmoid.segmap_group_size_10154"]
    segmap_group_sizze_10156 = sext_i32_i64(segmap_group_sizze_10155)
    y_10157 = (segmap_group_sizze_10156 - np.int64(1))
    x_10158 = (nest_sizze_10153 + y_10157)
    segmap_usable_groups_64_10160 = squot64(x_10158, segmap_group_sizze_10156)
    segmap_usable_groups_10161 = sext_i64_i32(segmap_usable_groups_64_10160)
    bytes_10628 = (np.int64(4) * nest_sizze_10153)
    mem_10632 = opencl_alloc(self, bytes_10628, "mem_10632")
    if ((1 * (np.long(segmap_usable_groups_10161) * np.long(segmap_group_sizze_10155))) != 0):
      self.segmap_10149_var.set_args(np.int32(sizze_9822), np.int32(sizze_9823),
                                     x_mem_10627, mem_10632)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10149_var,
                                 ((np.long(segmap_usable_groups_10161) * np.long(segmap_group_sizze_10155)),),
                                 (np.long(segmap_group_sizze_10155),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10809 = sizze_9822
    out_arrsizze_10810 = sizze_9823
    out_mem_10808 = mem_10632
    return (out_mem_10808, out_arrsizze_10809, out_arrsizze_10810)
  def futhark_lmatsubstract(self, x_mem_10627, sizze_9813, sizze_9814, d_9815):
    sizze_10133 = sext_i32_i64(sizze_9813)
    sizze_10134 = sext_i32_i64(sizze_9814)
    nest_sizze_10136 = (sizze_10133 * sizze_10134)
    segmap_group_sizze_10138 = self.sizes["lmatsubstract.segmap_group_size_10137"]
    segmap_group_sizze_10139 = sext_i32_i64(segmap_group_sizze_10138)
    y_10140 = (segmap_group_sizze_10139 - np.int64(1))
    x_10141 = (nest_sizze_10136 + y_10140)
    segmap_usable_groups_64_10143 = squot64(x_10141, segmap_group_sizze_10139)
    segmap_usable_groups_10144 = sext_i64_i32(segmap_usable_groups_64_10143)
    bytes_10628 = (np.int64(4) * nest_sizze_10136)
    mem_10632 = opencl_alloc(self, bytes_10628, "mem_10632")
    if ((1 * (np.long(segmap_usable_groups_10144) * np.long(segmap_group_sizze_10138))) != 0):
      self.segmap_10132_var.set_args(np.int32(sizze_9813), np.int32(sizze_9814),
                                     np.float32(d_9815), x_mem_10627, mem_10632)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10132_var,
                                 ((np.long(segmap_usable_groups_10144) * np.long(segmap_group_sizze_10138)),),
                                 (np.long(segmap_group_sizze_10138),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10801 = sizze_9813
    out_arrsizze_10802 = sizze_9814
    out_mem_10800 = mem_10632
    return (out_mem_10800, out_arrsizze_10801, out_arrsizze_10802)
  def futhark_lmatadd(self, x_mem_10627, sizze_9804, sizze_9805, s_9806):
    sizze_10116 = sext_i32_i64(sizze_9804)
    sizze_10117 = sext_i32_i64(sizze_9805)
    nest_sizze_10119 = (sizze_10116 * sizze_10117)
    segmap_group_sizze_10121 = self.sizes["lmatadd.segmap_group_size_10120"]
    segmap_group_sizze_10122 = sext_i32_i64(segmap_group_sizze_10121)
    y_10123 = (segmap_group_sizze_10122 - np.int64(1))
    x_10124 = (nest_sizze_10119 + y_10123)
    segmap_usable_groups_64_10126 = squot64(x_10124, segmap_group_sizze_10122)
    segmap_usable_groups_10127 = sext_i64_i32(segmap_usable_groups_64_10126)
    bytes_10628 = (np.int64(4) * nest_sizze_10119)
    mem_10632 = opencl_alloc(self, bytes_10628, "mem_10632")
    if ((1 * (np.long(segmap_usable_groups_10127) * np.long(segmap_group_sizze_10121))) != 0):
      self.segmap_10115_var.set_args(np.int32(sizze_9804), np.int32(sizze_9805),
                                     np.float32(s_9806), x_mem_10627, mem_10632)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10115_var,
                                 ((np.long(segmap_usable_groups_10127) * np.long(segmap_group_sizze_10121)),),
                                 (np.long(segmap_group_sizze_10121),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10793 = sizze_9804
    out_arrsizze_10794 = sizze_9805
    out_mem_10792 = mem_10632
    return (out_mem_10792, out_arrsizze_10793, out_arrsizze_10794)
  def futhark_substract2(self, x_mem_10627, y_mem_10628, sizze_9789,
                         sizze_9790):
    dim_zzero_9793 = (np.int32(0) == sizze_9790)
    dim_zzero_9794 = (np.int32(0) == sizze_9789)
    both_empty_9795 = (dim_zzero_9793 and dim_zzero_9794)
    dim_match_9796 = (sizze_9789 == sizze_9790)
    empty_or_match_9797 = (both_empty_9795 or dim_match_9796)
    empty_or_match_cert_9798 = True
    assert empty_or_match_9797, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:141:3-14\n   #1  GPU.fut:140:1-141:14\n" % ("function arguments of wrong shape",))
    sizze_10100 = sext_i32_i64(sizze_9789)
    segmap_group_sizze_10103 = self.sizes["substract2.segmap_group_size_10102"]
    segmap_group_sizze_10104 = sext_i32_i64(segmap_group_sizze_10103)
    y_10105 = (segmap_group_sizze_10104 - np.int64(1))
    x_10106 = (sizze_10100 + y_10105)
    segmap_usable_groups_64_10108 = squot64(x_10106, segmap_group_sizze_10104)
    segmap_usable_groups_10109 = sext_i64_i32(segmap_usable_groups_64_10108)
    bytes_10629 = (np.int64(4) * sizze_10100)
    mem_10631 = opencl_alloc(self, bytes_10629, "mem_10631")
    if ((1 * (np.long(segmap_usable_groups_10109) * np.long(segmap_group_sizze_10103))) != 0):
      self.segmap_10099_var.set_args(np.int32(sizze_9789), x_mem_10627,
                                     y_mem_10628, mem_10631)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10099_var,
                                 ((np.long(segmap_usable_groups_10109) * np.long(segmap_group_sizze_10103)),),
                                 (np.long(segmap_group_sizze_10103),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10786 = sizze_9789
    out_mem_10785 = mem_10631
    return (out_mem_10785, out_arrsizze_10786)
  def futhark_add2(self, x_mem_10627, y_mem_10628, sizze_9774, sizze_9775):
    dim_zzero_9778 = (np.int32(0) == sizze_9775)
    dim_zzero_9779 = (np.int32(0) == sizze_9774)
    both_empty_9780 = (dim_zzero_9778 and dim_zzero_9779)
    dim_match_9781 = (sizze_9774 == sizze_9775)
    empty_or_match_9782 = (both_empty_9780 or dim_match_9781)
    empty_or_match_cert_9783 = True
    assert empty_or_match_9782, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:105:3-14\n   #1  GPU.fut:104:1-105:14\n" % ("function arguments of wrong shape",))
    sizze_10085 = sext_i32_i64(sizze_9774)
    segmap_group_sizze_10088 = self.sizes["add2.segmap_group_size_10087"]
    segmap_group_sizze_10089 = sext_i32_i64(segmap_group_sizze_10088)
    y_10090 = (segmap_group_sizze_10089 - np.int64(1))
    x_10091 = (sizze_10085 + y_10090)
    segmap_usable_groups_64_10093 = squot64(x_10091, segmap_group_sizze_10089)
    segmap_usable_groups_10094 = sext_i64_i32(segmap_usable_groups_64_10093)
    bytes_10629 = (np.int64(4) * sizze_10085)
    mem_10631 = opencl_alloc(self, bytes_10629, "mem_10631")
    if ((1 * (np.long(segmap_usable_groups_10094) * np.long(segmap_group_sizze_10088))) != 0):
      self.segmap_10084_var.set_args(np.int32(sizze_9774), x_mem_10627,
                                     y_mem_10628, mem_10631)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10084_var,
                                 ((np.long(segmap_usable_groups_10094) * np.long(segmap_group_sizze_10088)),),
                                 (np.long(segmap_group_sizze_10088),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10779 = sizze_9774
    out_mem_10778 = mem_10631
    return (out_mem_10778, out_arrsizze_10779)
  def futhark_lmatmultiply(self, x_mem_10627, sizze_9765, sizze_9766, p_9767):
    sizze_10069 = sext_i32_i64(sizze_9765)
    sizze_10070 = sext_i32_i64(sizze_9766)
    nest_sizze_10072 = (sizze_10069 * sizze_10070)
    segmap_group_sizze_10074 = self.sizes["lmatmultiply.segmap_group_size_10073"]
    segmap_group_sizze_10075 = sext_i32_i64(segmap_group_sizze_10074)
    y_10076 = (segmap_group_sizze_10075 - np.int64(1))
    x_10077 = (nest_sizze_10072 + y_10076)
    segmap_usable_groups_64_10079 = squot64(x_10077, segmap_group_sizze_10075)
    segmap_usable_groups_10080 = sext_i64_i32(segmap_usable_groups_64_10079)
    bytes_10628 = (np.int64(4) * nest_sizze_10072)
    mem_10632 = opencl_alloc(self, bytes_10628, "mem_10632")
    if ((1 * (np.long(segmap_usable_groups_10080) * np.long(segmap_group_sizze_10074))) != 0):
      self.segmap_10068_var.set_args(np.int32(sizze_9765), np.int32(sizze_9766),
                                     np.float32(p_9767), x_mem_10627, mem_10632)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10068_var,
                                 ((np.long(segmap_usable_groups_10080) * np.long(segmap_group_sizze_10074)),),
                                 (np.long(segmap_group_sizze_10074),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10771 = sizze_9765
    out_arrsizze_10772 = sizze_9766
    out_mem_10770 = mem_10632
    return (out_mem_10770, out_arrsizze_10771, out_arrsizze_10772)
  def futhark_multiply2(self, x_mem_10627, y_mem_10628, sizze_9750, sizze_9751):
    dim_zzero_9754 = (np.int32(0) == sizze_9751)
    dim_zzero_9755 = (np.int32(0) == sizze_9750)
    both_empty_9756 = (dim_zzero_9754 and dim_zzero_9755)
    dim_match_9757 = (sizze_9750 == sizze_9751)
    empty_or_match_9758 = (both_empty_9756 or dim_match_9757)
    empty_or_match_cert_9759 = True
    assert empty_or_match_9758, ("Error: %s\n\nBacktrace:\n-> #0  GPU.fut:71:3-14\n   #1  GPU.fut:70:1-71:14\n" % ("function arguments of wrong shape",))
    sizze_10053 = sext_i32_i64(sizze_9750)
    segmap_group_sizze_10056 = self.sizes["multiply2.segmap_group_size_10055"]
    segmap_group_sizze_10057 = sext_i32_i64(segmap_group_sizze_10056)
    y_10058 = (segmap_group_sizze_10057 - np.int64(1))
    x_10059 = (sizze_10053 + y_10058)
    segmap_usable_groups_64_10061 = squot64(x_10059, segmap_group_sizze_10057)
    segmap_usable_groups_10062 = sext_i64_i32(segmap_usable_groups_64_10061)
    bytes_10629 = (np.int64(4) * sizze_10053)
    mem_10631 = opencl_alloc(self, bytes_10629, "mem_10631")
    if ((1 * (np.long(segmap_usable_groups_10062) * np.long(segmap_group_sizze_10056))) != 0):
      self.segmap_10052_var.set_args(np.int32(sizze_9750), x_mem_10627,
                                     y_mem_10628, mem_10631)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10052_var,
                                 ((np.long(segmap_usable_groups_10062) * np.long(segmap_group_sizze_10056)),),
                                 (np.long(segmap_group_sizze_10056),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10764 = sizze_9750
    out_mem_10763 = mem_10631
    return (out_mem_10763, out_arrsizze_10764)
  def futhark_substract(self, x_mem_10627, sizze_9744, d_9745):
    sizze_10039 = sext_i32_i64(sizze_9744)
    segmap_group_sizze_10042 = self.sizes["substract.segmap_group_size_10041"]
    segmap_group_sizze_10043 = sext_i32_i64(segmap_group_sizze_10042)
    y_10044 = (segmap_group_sizze_10043 - np.int64(1))
    x_10045 = (sizze_10039 + y_10044)
    segmap_usable_groups_64_10047 = squot64(x_10045, segmap_group_sizze_10043)
    segmap_usable_groups_10048 = sext_i64_i32(segmap_usable_groups_64_10047)
    bytes_10628 = (np.int64(4) * sizze_10039)
    mem_10630 = opencl_alloc(self, bytes_10628, "mem_10630")
    if ((1 * (np.long(segmap_usable_groups_10048) * np.long(segmap_group_sizze_10042))) != 0):
      self.segmap_10038_var.set_args(np.int32(sizze_9744), np.float32(d_9745),
                                     x_mem_10627, mem_10630)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10038_var,
                                 ((np.long(segmap_usable_groups_10048) * np.long(segmap_group_sizze_10042)),),
                                 (np.long(segmap_group_sizze_10042),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10757 = sizze_9744
    out_mem_10756 = mem_10630
    return (out_mem_10756, out_arrsizze_10757)
  def futhark_add(self, x_mem_10627, sizze_9738, s_9739):
    sizze_10025 = sext_i32_i64(sizze_9738)
    segmap_group_sizze_10028 = self.sizes["add.segmap_group_size_10027"]
    segmap_group_sizze_10029 = sext_i32_i64(segmap_group_sizze_10028)
    y_10030 = (segmap_group_sizze_10029 - np.int64(1))
    x_10031 = (sizze_10025 + y_10030)
    segmap_usable_groups_64_10033 = squot64(x_10031, segmap_group_sizze_10029)
    segmap_usable_groups_10034 = sext_i64_i32(segmap_usable_groups_64_10033)
    bytes_10628 = (np.int64(4) * sizze_10025)
    mem_10630 = opencl_alloc(self, bytes_10628, "mem_10630")
    if ((1 * (np.long(segmap_usable_groups_10034) * np.long(segmap_group_sizze_10028))) != 0):
      self.segmap_10024_var.set_args(np.int32(sizze_9738), np.float32(s_9739),
                                     x_mem_10627, mem_10630)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10024_var,
                                 ((np.long(segmap_usable_groups_10034) * np.long(segmap_group_sizze_10028)),),
                                 (np.long(segmap_group_sizze_10028),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10750 = sizze_9738
    out_mem_10749 = mem_10630
    return (out_mem_10749, out_arrsizze_10750)
  def futhark_multiply(self, x_mem_10627, sizze_9732, m_9733):
    sizze_10011 = sext_i32_i64(sizze_9732)
    segmap_group_sizze_10014 = self.sizes["multiply.segmap_group_size_10013"]
    segmap_group_sizze_10015 = sext_i32_i64(segmap_group_sizze_10014)
    y_10016 = (segmap_group_sizze_10015 - np.int64(1))
    x_10017 = (sizze_10011 + y_10016)
    segmap_usable_groups_64_10019 = squot64(x_10017, segmap_group_sizze_10015)
    segmap_usable_groups_10020 = sext_i64_i32(segmap_usable_groups_64_10019)
    bytes_10628 = (np.int64(4) * sizze_10011)
    mem_10630 = opencl_alloc(self, bytes_10628, "mem_10630")
    if ((1 * (np.long(segmap_usable_groups_10020) * np.long(segmap_group_sizze_10014))) != 0):
      self.segmap_10010_var.set_args(np.int32(sizze_9732), np.float32(m_9733),
                                     x_mem_10627, mem_10630)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_10010_var,
                                 ((np.long(segmap_usable_groups_10020) * np.long(segmap_group_sizze_10014)),),
                                 (np.long(segmap_group_sizze_10014),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10743 = sizze_9732
    out_mem_10742 = mem_10630
    return (out_mem_10742, out_arrsizze_10743)
  def futhark_divide(self, x_mem_10627, sizze_9726, d_9727):
    sizze_9997 = sext_i32_i64(sizze_9726)
    segmap_group_sizze_10000 = self.sizes["divide.segmap_group_size_9999"]
    segmap_group_sizze_10001 = sext_i32_i64(segmap_group_sizze_10000)
    y_10002 = (segmap_group_sizze_10001 - np.int64(1))
    x_10003 = (sizze_9997 + y_10002)
    segmap_usable_groups_64_10005 = squot64(x_10003, segmap_group_sizze_10001)
    segmap_usable_groups_10006 = sext_i64_i32(segmap_usable_groups_64_10005)
    bytes_10628 = (np.int64(4) * sizze_9997)
    mem_10630 = opencl_alloc(self, bytes_10628, "mem_10630")
    if ((1 * (np.long(segmap_usable_groups_10006) * np.long(segmap_group_sizze_10000))) != 0):
      self.segmap_9996_var.set_args(np.int32(sizze_9726), np.float32(d_9727),
                                    x_mem_10627, mem_10630)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_9996_var,
                                 ((np.long(segmap_usable_groups_10006) * np.long(segmap_group_sizze_10000)),),
                                 (np.long(segmap_group_sizze_10000),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10736 = sizze_9726
    out_mem_10735 = mem_10630
    return (out_mem_10735, out_arrsizze_10736)
  def futhark_negation(self, x_mem_10627, sizze_9721):
    sizze_9983 = sext_i32_i64(sizze_9721)
    segmap_group_sizze_9986 = self.sizes["negation.segmap_group_size_9985"]
    segmap_group_sizze_9987 = sext_i32_i64(segmap_group_sizze_9986)
    y_9988 = (segmap_group_sizze_9987 - np.int64(1))
    x_9989 = (sizze_9983 + y_9988)
    segmap_usable_groups_64_9991 = squot64(x_9989, segmap_group_sizze_9987)
    segmap_usable_groups_9992 = sext_i64_i32(segmap_usable_groups_64_9991)
    bytes_10628 = (np.int64(4) * sizze_9983)
    mem_10630 = opencl_alloc(self, bytes_10628, "mem_10630")
    if ((1 * (np.long(segmap_usable_groups_9992) * np.long(segmap_group_sizze_9986))) != 0):
      self.segmap_9982_var.set_args(np.int32(sizze_9721), x_mem_10627,
                                    mem_10630)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_9982_var,
                                 ((np.long(segmap_usable_groups_9992) * np.long(segmap_group_sizze_9986)),),
                                 (np.long(segmap_group_sizze_9986),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10729 = sizze_9721
    out_mem_10728 = mem_10630
    return (out_mem_10728, out_arrsizze_10729)
  def futhark_exp(self, x_mem_10627, sizze_9716):
    sizze_9969 = sext_i32_i64(sizze_9716)
    segmap_group_sizze_9972 = self.sizes["exp.segmap_group_size_9971"]
    segmap_group_sizze_9973 = sext_i32_i64(segmap_group_sizze_9972)
    y_9974 = (segmap_group_sizze_9973 - np.int64(1))
    x_9975 = (sizze_9969 + y_9974)
    segmap_usable_groups_64_9977 = squot64(x_9975, segmap_group_sizze_9973)
    segmap_usable_groups_9978 = sext_i64_i32(segmap_usable_groups_64_9977)
    bytes_10628 = (np.int64(4) * sizze_9969)
    mem_10630 = opencl_alloc(self, bytes_10628, "mem_10630")
    if ((1 * (np.long(segmap_usable_groups_9978) * np.long(segmap_group_sizze_9972))) != 0):
      self.segmap_9968_var.set_args(np.int32(sizze_9716), x_mem_10627,
                                    mem_10630)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_9968_var,
                                 ((np.long(segmap_usable_groups_9978) * np.long(segmap_group_sizze_9972)),),
                                 (np.long(segmap_group_sizze_9972),))
      if synchronous:
        self.queue.finish()
    out_arrsizze_10722 = sizze_9716
    out_mem_10721 = mem_10630
    return (out_mem_10721, out_arrsizze_10722)
  def futhark_transp(self, x_mem_10627, sizze_9712, sizze_9713):
    binop_x_10629 = sext_i32_i64(sizze_9713)
    binop_y_10630 = sext_i32_i64(sizze_9712)
    binop_x_10631 = (binop_x_10629 * binop_y_10630)
    bytes_10628 = (np.int64(4) * binop_x_10631)
    mem_10632 = opencl_alloc(self, bytes_10628, "mem_10632")
    self.futhark__map_transpose_f32(mem_10632, np.int32(0), x_mem_10627,
                                    np.int32(0), np.int32(1), sizze_9713,
                                    sizze_9712, (sizze_9713 * sizze_9712),
                                    (sizze_9713 * sizze_9712))
    out_arrsizze_10719 = sizze_9713
    out_arrsizze_10720 = sizze_9712
    out_mem_10718 = mem_10632
    return (out_mem_10718, out_arrsizze_10719, out_arrsizze_10720)
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
  def dot(self, a_mem_10627_ext, b_mem_10628_ext):
    try:
      assert ((type(a_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (a_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9942 = np.int32(a_mem_10627_ext.shape[0])
      sizze_9943 = np.int32(a_mem_10627_ext.shape[1])
      if (type(a_mem_10627_ext) == cl.array.Array):
        a_mem_10627 = a_mem_10627_ext.data
      else:
        a_mem_10627 = opencl_alloc(self, np.int64(a_mem_10627_ext.nbytes),
                                   "a_mem_10627")
        if (np.int64(a_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, a_mem_10627,
                          normaliseArray(a_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(a_mem_10627_ext),
                                                                                                                            a_mem_10627_ext))
    try:
      assert ((type(b_mem_10628_ext) in [np.ndarray,
                                         cl.array.Array]) and (b_mem_10628_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9944 = np.int32(b_mem_10628_ext.shape[0])
      sizze_9945 = np.int32(b_mem_10628_ext.shape[1])
      if (type(b_mem_10628_ext) == cl.array.Array):
        b_mem_10628 = b_mem_10628_ext.data
      else:
        b_mem_10628 = opencl_alloc(self, np.int64(b_mem_10628_ext.nbytes),
                                   "b_mem_10628")
        if (np.int64(b_mem_10628_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, b_mem_10628,
                          normaliseArray(b_mem_10628_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(b_mem_10628_ext),
                                                                                                                            b_mem_10628_ext))
    (out_mem_10848, out_arrsizze_10849,
     out_arrsizze_10850) = self.futhark_dot(a_mem_10627, b_mem_10628,
                                            sizze_9942, sizze_9943, sizze_9944,
                                            sizze_9945)
    return cl.array.Array(self.queue, (out_arrsizze_10849, out_arrsizze_10850),
                          ct.c_float, data=out_mem_10848)
  def matsubstract(self, x_mem_10627_ext, y_mem_10628_ext):
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9913 = np.int32(x_mem_10627_ext.shape[0])
      sizze_9914 = np.int32(x_mem_10627_ext.shape[1])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    try:
      assert ((type(y_mem_10628_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_10628_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9915 = np.int32(y_mem_10628_ext.shape[0])
      sizze_9916 = np.int32(y_mem_10628_ext.shape[1])
      if (type(y_mem_10628_ext) == cl.array.Array):
        y_mem_10628 = y_mem_10628_ext.data
      else:
        y_mem_10628 = opencl_alloc(self, np.int64(y_mem_10628_ext.nbytes),
                                   "y_mem_10628")
        if (np.int64(y_mem_10628_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_10628,
                          normaliseArray(y_mem_10628_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(y_mem_10628_ext),
                                                                                                                            y_mem_10628_ext))
    (out_mem_10840, out_arrsizze_10841,
     out_arrsizze_10842) = self.futhark_matsubstract(x_mem_10627, y_mem_10628,
                                                     sizze_9913, sizze_9914,
                                                     sizze_9915, sizze_9916)
    return cl.array.Array(self.queue, (out_arrsizze_10841, out_arrsizze_10842),
                          ct.c_float, data=out_mem_10840)
  def matadd(self, x_mem_10627_ext, y_mem_10628_ext):
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9884 = np.int32(x_mem_10627_ext.shape[0])
      sizze_9885 = np.int32(x_mem_10627_ext.shape[1])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    try:
      assert ((type(y_mem_10628_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_10628_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9886 = np.int32(y_mem_10628_ext.shape[0])
      sizze_9887 = np.int32(y_mem_10628_ext.shape[1])
      if (type(y_mem_10628_ext) == cl.array.Array):
        y_mem_10628 = y_mem_10628_ext.data
      else:
        y_mem_10628 = opencl_alloc(self, np.int64(y_mem_10628_ext.nbytes),
                                   "y_mem_10628")
        if (np.int64(y_mem_10628_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_10628,
                          normaliseArray(y_mem_10628_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(y_mem_10628_ext),
                                                                                                                            y_mem_10628_ext))
    (out_mem_10832, out_arrsizze_10833,
     out_arrsizze_10834) = self.futhark_matadd(x_mem_10627, y_mem_10628,
                                               sizze_9884, sizze_9885,
                                               sizze_9886, sizze_9887)
    return cl.array.Array(self.queue, (out_arrsizze_10833, out_arrsizze_10834),
                          ct.c_float, data=out_mem_10832)
  def matmultiply(self, x_mem_10627_ext, y_mem_10628_ext):
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9855 = np.int32(x_mem_10627_ext.shape[0])
      sizze_9856 = np.int32(x_mem_10627_ext.shape[1])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    try:
      assert ((type(y_mem_10628_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_10628_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9857 = np.int32(y_mem_10628_ext.shape[0])
      sizze_9858 = np.int32(y_mem_10628_ext.shape[1])
      if (type(y_mem_10628_ext) == cl.array.Array):
        y_mem_10628 = y_mem_10628_ext.data
      else:
        y_mem_10628 = opencl_alloc(self, np.int64(y_mem_10628_ext.nbytes),
                                   "y_mem_10628")
        if (np.int64(y_mem_10628_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_10628,
                          normaliseArray(y_mem_10628_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(y_mem_10628_ext),
                                                                                                                            y_mem_10628_ext))
    (out_mem_10824, out_arrsizze_10825,
     out_arrsizze_10826) = self.futhark_matmultiply(x_mem_10627, y_mem_10628,
                                                    sizze_9855, sizze_9856,
                                                    sizze_9857, sizze_9858)
    return cl.array.Array(self.queue, (out_arrsizze_10825, out_arrsizze_10826),
                          ct.c_float, data=out_mem_10824)
  def lvecmul(self, u_mem_10627_ext, b_mem_10628_ext):
    try:
      assert ((type(u_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (u_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9833 = np.int32(u_mem_10627_ext.shape[0])
      if (type(u_mem_10627_ext) == cl.array.Array):
        u_mem_10627 = u_mem_10627_ext.data
      else:
        u_mem_10627 = opencl_alloc(self, np.int64(u_mem_10627_ext.nbytes),
                                   "u_mem_10627")
        if (np.int64(u_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, u_mem_10627,
                          normaliseArray(u_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(u_mem_10627_ext),
                                                                                                                            u_mem_10627_ext))
    try:
      assert ((type(b_mem_10628_ext) in [np.ndarray,
                                         cl.array.Array]) and (b_mem_10628_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9834 = np.int32(b_mem_10628_ext.shape[0])
      sizze_9835 = np.int32(b_mem_10628_ext.shape[1])
      if (type(b_mem_10628_ext) == cl.array.Array):
        b_mem_10628 = b_mem_10628_ext.data
      else:
        b_mem_10628 = opencl_alloc(self, np.int64(b_mem_10628_ext.nbytes),
                                   "b_mem_10628")
        if (np.int64(b_mem_10628_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, b_mem_10628,
                          normaliseArray(b_mem_10628_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(b_mem_10628_ext),
                                                                                                                            b_mem_10628_ext))
    (out_mem_10816, out_arrsizze_10817) = self.futhark_lvecmul(u_mem_10627,
                                                               b_mem_10628,
                                                               sizze_9833,
                                                               sizze_9834,
                                                               sizze_9835)
    return cl.array.Array(self.queue, (out_arrsizze_10817,), ct.c_float,
                          data=out_mem_10816)
  def sigmoid(self, x_mem_10627_ext):
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9822 = np.int32(x_mem_10627_ext.shape[0])
      sizze_9823 = np.int32(x_mem_10627_ext.shape[1])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    (out_mem_10808, out_arrsizze_10809,
     out_arrsizze_10810) = self.futhark_sigmoid(x_mem_10627, sizze_9822,
                                                sizze_9823)
    return cl.array.Array(self.queue, (out_arrsizze_10809, out_arrsizze_10810),
                          ct.c_float, data=out_mem_10808)
  def lmatsubstract(self, d_9815_ext, x_mem_10627_ext):
    try:
      d_9815 = np.float32(ct.c_float(d_9815_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(d_9815_ext),
                                                                                                                            d_9815_ext))
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9813 = np.int32(x_mem_10627_ext.shape[0])
      sizze_9814 = np.int32(x_mem_10627_ext.shape[1])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    (out_mem_10800, out_arrsizze_10801,
     out_arrsizze_10802) = self.futhark_lmatsubstract(x_mem_10627, sizze_9813,
                                                      sizze_9814, d_9815)
    return cl.array.Array(self.queue, (out_arrsizze_10801, out_arrsizze_10802),
                          ct.c_float, data=out_mem_10800)
  def lmatadd(self, s_9806_ext, x_mem_10627_ext):
    try:
      s_9806 = np.float32(ct.c_float(s_9806_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(s_9806_ext),
                                                                                                                            s_9806_ext))
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9804 = np.int32(x_mem_10627_ext.shape[0])
      sizze_9805 = np.int32(x_mem_10627_ext.shape[1])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    (out_mem_10792, out_arrsizze_10793,
     out_arrsizze_10794) = self.futhark_lmatadd(x_mem_10627, sizze_9804,
                                                sizze_9805, s_9806)
    return cl.array.Array(self.queue, (out_arrsizze_10793, out_arrsizze_10794),
                          ct.c_float, data=out_mem_10792)
  def substract2(self, x_mem_10627_ext, y_mem_10628_ext):
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9789 = np.int32(x_mem_10627_ext.shape[0])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    try:
      assert ((type(y_mem_10628_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_10628_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9790 = np.int32(y_mem_10628_ext.shape[0])
      if (type(y_mem_10628_ext) == cl.array.Array):
        y_mem_10628 = y_mem_10628_ext.data
      else:
        y_mem_10628 = opencl_alloc(self, np.int64(y_mem_10628_ext.nbytes),
                                   "y_mem_10628")
        if (np.int64(y_mem_10628_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_10628,
                          normaliseArray(y_mem_10628_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(y_mem_10628_ext),
                                                                                                                            y_mem_10628_ext))
    (out_mem_10785, out_arrsizze_10786) = self.futhark_substract2(x_mem_10627,
                                                                  y_mem_10628,
                                                                  sizze_9789,
                                                                  sizze_9790)
    return cl.array.Array(self.queue, (out_arrsizze_10786,), ct.c_float,
                          data=out_mem_10785)
  def add2(self, x_mem_10627_ext, y_mem_10628_ext):
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9774 = np.int32(x_mem_10627_ext.shape[0])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    try:
      assert ((type(y_mem_10628_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_10628_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9775 = np.int32(y_mem_10628_ext.shape[0])
      if (type(y_mem_10628_ext) == cl.array.Array):
        y_mem_10628 = y_mem_10628_ext.data
      else:
        y_mem_10628 = opencl_alloc(self, np.int64(y_mem_10628_ext.nbytes),
                                   "y_mem_10628")
        if (np.int64(y_mem_10628_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_10628,
                          normaliseArray(y_mem_10628_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(y_mem_10628_ext),
                                                                                                                            y_mem_10628_ext))
    (out_mem_10778, out_arrsizze_10779) = self.futhark_add2(x_mem_10627,
                                                            y_mem_10628,
                                                            sizze_9774,
                                                            sizze_9775)
    return cl.array.Array(self.queue, (out_arrsizze_10779,), ct.c_float,
                          data=out_mem_10778)
  def lmatmultiply(self, p_9767_ext, x_mem_10627_ext):
    try:
      p_9767 = np.float32(ct.c_float(p_9767_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(p_9767_ext),
                                                                                                                            p_9767_ext))
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9765 = np.int32(x_mem_10627_ext.shape[0])
      sizze_9766 = np.int32(x_mem_10627_ext.shape[1])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    (out_mem_10770, out_arrsizze_10771,
     out_arrsizze_10772) = self.futhark_lmatmultiply(x_mem_10627, sizze_9765,
                                                     sizze_9766, p_9767)
    return cl.array.Array(self.queue, (out_arrsizze_10771, out_arrsizze_10772),
                          ct.c_float, data=out_mem_10770)
  def multiply2(self, x_mem_10627_ext, y_mem_10628_ext):
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9750 = np.int32(x_mem_10627_ext.shape[0])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    try:
      assert ((type(y_mem_10628_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_10628_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9751 = np.int32(y_mem_10628_ext.shape[0])
      if (type(y_mem_10628_ext) == cl.array.Array):
        y_mem_10628 = y_mem_10628_ext.data
      else:
        y_mem_10628 = opencl_alloc(self, np.int64(y_mem_10628_ext.nbytes),
                                   "y_mem_10628")
        if (np.int64(y_mem_10628_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_10628,
                          normaliseArray(y_mem_10628_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(y_mem_10628_ext),
                                                                                                                            y_mem_10628_ext))
    (out_mem_10763, out_arrsizze_10764) = self.futhark_multiply2(x_mem_10627,
                                                                 y_mem_10628,
                                                                 sizze_9750,
                                                                 sizze_9751)
    return cl.array.Array(self.queue, (out_arrsizze_10764,), ct.c_float,
                          data=out_mem_10763)
  def substract(self, d_9745_ext, x_mem_10627_ext):
    try:
      d_9745 = np.float32(ct.c_float(d_9745_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(d_9745_ext),
                                                                                                                            d_9745_ext))
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9744 = np.int32(x_mem_10627_ext.shape[0])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    (out_mem_10756, out_arrsizze_10757) = self.futhark_substract(x_mem_10627,
                                                                 sizze_9744,
                                                                 d_9745)
    return cl.array.Array(self.queue, (out_arrsizze_10757,), ct.c_float,
                          data=out_mem_10756)
  def add(self, s_9739_ext, x_mem_10627_ext):
    try:
      s_9739 = np.float32(ct.c_float(s_9739_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(s_9739_ext),
                                                                                                                            s_9739_ext))
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9738 = np.int32(x_mem_10627_ext.shape[0])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    (out_mem_10749, out_arrsizze_10750) = self.futhark_add(x_mem_10627,
                                                           sizze_9738, s_9739)
    return cl.array.Array(self.queue, (out_arrsizze_10750,), ct.c_float,
                          data=out_mem_10749)
  def multiply(self, m_9733_ext, x_mem_10627_ext):
    try:
      m_9733 = np.float32(ct.c_float(m_9733_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(m_9733_ext),
                                                                                                                            m_9733_ext))
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9732 = np.int32(x_mem_10627_ext.shape[0])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    (out_mem_10742, out_arrsizze_10743) = self.futhark_multiply(x_mem_10627,
                                                                sizze_9732,
                                                                m_9733)
    return cl.array.Array(self.queue, (out_arrsizze_10743,), ct.c_float,
                          data=out_mem_10742)
  def divide(self, d_9727_ext, x_mem_10627_ext):
    try:
      d_9727 = np.float32(ct.c_float(d_9727_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(d_9727_ext),
                                                                                                                            d_9727_ext))
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9726 = np.int32(x_mem_10627_ext.shape[0])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    (out_mem_10735, out_arrsizze_10736) = self.futhark_divide(x_mem_10627,
                                                              sizze_9726,
                                                              d_9727)
    return cl.array.Array(self.queue, (out_arrsizze_10736,), ct.c_float,
                          data=out_mem_10735)
  def negation(self, x_mem_10627_ext):
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9721 = np.int32(x_mem_10627_ext.shape[0])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    (out_mem_10728, out_arrsizze_10729) = self.futhark_negation(x_mem_10627,
                                                                sizze_9721)
    return cl.array.Array(self.queue, (out_arrsizze_10729,), ct.c_float,
                          data=out_mem_10728)
  def exp(self, x_mem_10627_ext):
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9716 = np.int32(x_mem_10627_ext.shape[0])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    (out_mem_10721, out_arrsizze_10722) = self.futhark_exp(x_mem_10627,
                                                           sizze_9716)
    return cl.array.Array(self.queue, (out_arrsizze_10722,), ct.c_float,
                          data=out_mem_10721)
  def transp(self, x_mem_10627_ext):
    try:
      assert ((type(x_mem_10627_ext) in [np.ndarray,
                                         cl.array.Array]) and (x_mem_10627_ext.dtype == np.float32)), "Parameter has unexpected type"
      sizze_9712 = np.int32(x_mem_10627_ext.shape[0])
      sizze_9713 = np.int32(x_mem_10627_ext.shape[1])
      if (type(x_mem_10627_ext) == cl.array.Array):
        x_mem_10627 = x_mem_10627_ext.data
      else:
        x_mem_10627 = opencl_alloc(self, np.int64(x_mem_10627_ext.nbytes),
                                   "x_mem_10627")
        if (np.int64(x_mem_10627_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, x_mem_10627,
                          normaliseArray(x_mem_10627_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(x_mem_10627_ext),
                                                                                                                            x_mem_10627_ext))
    (out_mem_10718, out_arrsizze_10719,
     out_arrsizze_10720) = self.futhark_transp(x_mem_10627, sizze_9712,
                                               sizze_9713)
    return cl.array.Array(self.queue, (out_arrsizze_10719, out_arrsizze_10720),
                          ct.c_float, data=out_mem_10718)