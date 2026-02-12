// since we are using volk to load vulkan functions
// 	we need to include it here to properly build vma functionality
#include "volk.h"
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#undef VMA_IMPLEMENTATION
