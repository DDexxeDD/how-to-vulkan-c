#include <stdio.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ktx.h>
#include <math.h>

#include "volk.h"
#include <ktxvulkan.h>  // need to include _after_ volk
#include "cvector.h"
#include "vk_mem_alloc.h"
#include "struct.h"
#include "fast_obj.h"

typedef struct vertex_t
{
	vec3 position;
	vec3 normal;
	vec3 uv;
} vertex_t;

typedef struct shader_data_t
{
	mat4s projection;
	mat4s view;
	mat4s model[3];
	vec4s light_position;  // default {0.0f, -10.0f, 10.0f, 0.0f}
	uint32_t selected;  // default 1
} shader_data_t;

typedef struct shader_data_buffer_t
{
	VmaAllocation allocation;  // default VK_NULL_HANDLE
	VkBuffer buffer;  // default VK_NULL_HANDLE
	VkDeviceAddress device_address;
	void* mapped;  // default NULL
} shader_data_buffer_t;

typedef struct texture_t
{
	VmaAllocation allocation;  // default VK_NULL_HANDLE
	VkImage image;  // default VK_NULL_HANDLE
	VkImageView view;  // default VK_NULL_HANDLE
	VkSampler sampler;  // default VK_NULL_HANDLE
} texture_t;

/*
 * this is basically double buffering (can be more than 2, but every added frame adds latency)
 * in vulkan this is often referred to as "frames in flight"
 * resources which are shared by the cpu and gpu (duplicated on gpu and cpu)
 * 	will need MAX_FRAMES_IN_FLIGHT dimensions
 */
#define MAX_FRAMES_IN_FLIGHT 2

SDL_Window* window = NULL;

VkInstance instance = {VK_NULL_HANDLE};
VkDevice device = {VK_NULL_HANDLE};
VkQueue queue = {VK_NULL_HANDLE};
VkSurfaceKHR surface = {VK_NULL_HANDLE};
VkSwapchainKHR swapchain = {VK_NULL_HANDLE};
VkImage depth_image;
VkImageView depth_image_view;
VkBuffer vertex_buffer = {VK_NULL_HANDLE};
VkCommandPool command_pool = {VK_NULL_HANDLE};
VkDescriptorSetLayout descriptor_set_layout_texture = {VK_NULL_HANDLE};
VkDescriptorPool descriptor_pool = {VK_NULL_HANDLE};
VkDescriptorSet descriptor_set_texture = {VK_NULL_HANDLE};
VkPipelineLayout pipeline_layout = {VK_NULL_HANDLE};
VkPipeline pipeline = {VK_NULL_HANDLE};

cvector (VkImage) swapchain_images = NULL;
cvector (VkImageView) swapchain_image_views = NULL;

VmaAllocator allocator = {VK_NULL_HANDLE};
VmaAllocation depth_image_allocation;
VmaAllocation vertex_buffer_allocation = {VK_NULL_HANDLE};

cvector (texture_t) textures = NULL;  // init to size 3 later

// resources duplicated on cpu and gpu
shader_data_buffer_t shader_data_buffers[MAX_FRAMES_IN_FLIGHT];
VkCommandBuffer command_buffers[MAX_FRAMES_IN_FLIGHT];
VkFence fences[MAX_FRAMES_IN_FLIGHT];
cvector (VkSemaphore) render_semaphores = NULL;
VkSemaphore present_semaphores[MAX_FRAMES_IN_FLIGHT];

uint32_t image_index = 0;
uint32_t frame_index = 0;
bool update_swapchain = false;

shader_data_t shader_data;
vec3s camera_position = {{0.0f, 0.0f, -6.0f}};
vec3s object_rotations[3] = {GLMS_VEC3_ZERO_INIT, GLMS_VEC3_ZERO_INIT, GLMS_VEC3_ZERO_INIT};
ivec2s window_size;

// initialize and resize a vector to 'capacity'
// 	'value' is the value to want elements added by resize to have
#define cvector_init_and_size(vec, capacity, value, elem_destructor_fn) \
	do {                                                                 \
		if (!(vec)) {                                                     \
			cvector_init ((vec), (capacity), (elem_destructor_fn));        \
			cvector_resize ((vec), (capacity), (value));                   \
		}                                                                 \
	} while (0)

int setup_sdl (SDL_Window** window)
{
	if (!SDL_Init (SDL_INIT_VIDEO))
	{
		printf ("failed to initialize sdl video: %s\n", SDL_GetError ());
		return 1;
	}

	if (!SDL_Vulkan_LoadLibrary (NULL))
	{
		return 1;
	}

	return 0;
}

// this is a simple check to see if an error code was returned
// 	exit the program if an error code was returned
// in real life we would need more robust error handling
// 	the big switch case in here is basically a reminder of that
static inline int check_vulkan (int result, char* filename, int line)
{
	// this is all of the unique VkResult codes that are defined on my machine
	switch (result)
	{
		case VK_SUCCESS:
		case VK_NOT_READY:
		case VK_TIMEOUT:
		case VK_EVENT_SET:
		case VK_EVENT_RESET:
		case VK_INCOMPLETE:
		case VK_PIPELINE_COMPILE_REQUIRED:
		case VK_SUBOPTIMAL_KHR:
		case VK_THREAD_IDLE_KHR:
		case VK_THREAD_DONE_KHR:
		case VK_OPERATION_DEFERRED_KHR:
		case VK_OPERATION_NOT_DEFERRED_KHR:
			// success codes
			break;
	
		case VK_ERROR_OUT_OF_HOST_MEMORY:
		case VK_ERROR_OUT_OF_DEVICE_MEMORY:
		case VK_ERROR_INITIALIZATION_FAILED:
		case VK_ERROR_DEVICE_LOST:
		case VK_ERROR_MEMORY_MAP_FAILED:
		case VK_ERROR_LAYER_NOT_PRESENT:
		case VK_ERROR_EXTENSION_NOT_PRESENT:
		case VK_ERROR_FEATURE_NOT_PRESENT:
		case VK_ERROR_INCOMPATIBLE_DRIVER:
		case VK_ERROR_TOO_MANY_OBJECTS:
		case VK_ERROR_FORMAT_NOT_SUPPORTED:
		case VK_ERROR_FRAGMENTED_POOL:
		case VK_ERROR_UNKNOWN:
		case VK_ERROR_OUT_OF_POOL_MEMORY:
		case VK_ERROR_INVALID_EXTERNAL_HANDLE:
		case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS:
		case VK_ERROR_FRAGMENTATION:
		case VK_ERROR_SURFACE_LOST_KHR:
		case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
		case VK_ERROR_OUT_OF_DATE_KHR:
		case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
		case VK_ERROR_INVALID_SHADER_NV:
		case VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR:
		case VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR:
		case VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR:
		case VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR:
		case VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR:
		case VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR:
		case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT:
		case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
		case VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR:
		case VK_ERROR_COMPRESSION_EXHAUSTED_EXT:
		case VK_ERROR_VALIDATION_FAILED_EXT:
		case VK_ERROR_NOT_PERMITTED_EXT:
		case VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT:
			printf ("vulkan call returned an error\n");
			printf ("  %s:%d\n", filename, line);
			exit (result);
			break;
		default:
			break;
	}

	return result;
}
#define check_vk(x) check_vulkan((x), __FILE__, __LINE__)

static inline void check_swapchain (VkResult result )
{
	if (result < VK_SUCCESS)
	{
		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			update_swapchain = true;
			return;
		}
	}
}

static inline void check_int (int result, int expected, char* filename, int line)
{
	if (result != expected)
	{
		printf ("error at %s:%d\n", filename, line);
		exit (result);
	}
}
#define check(result, expected) check_int((result), (expected), __FILE__, __LINE__)

// this is whatever glm is doing to convert a vec3 euler angle to a quaternion
versors euler_to_quat (vec3s angle)
{
	versors out;
	vec3s half = {{0.5f, 0.5f, 0.5f}};

	vec3s cosine = glms_vec3_mul (angle, half);
	cosine.x = cosf (cosine.x);
	cosine.y = cosf (cosine.y);
	cosine.z = cosf (cosine.z);

	vec3s sine = glms_vec3_mul (angle, half);
	sine.x = sinf (sine.x);
	sine.y = sinf (sine.y);
	sine.z = sinf (sine.z);

	out.w = cosine.x * cosine.y * cosine.z + sine.x * sine.y * sine.z;
	out.x = sine.x * cosine.y * cosine.z - cosine.x * sine.y * sine.z;
	out.y = cosine.x * sine.y * cosine.z + sine.x * cosine.y * sine.z;
	out.z = cosine.x * cosine.y * sine.z - sine.x * sine.y * cosine.z;

	return out;
}

int main (int argc, char* argv[])
{
	check (setup_sdl (&window), 0);
	check (volkInitialize (), VK_SUCCESS);

	/*
	 * vulkan instance
	 *
	 * the vulkan instance connects the application to vulkan
	 * 	the instance is the basis for everything we will do with vulkan
	 */
	VkApplicationInfo app_info = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "how to vulkan",
		.apiVersion = VK_API_VERSION_1_3
	};

	uint32_t instance_extensions_count = 0;
	char const* const* instance_extensions = SDL_Vulkan_GetInstanceExtensions (&instance_extensions_count);
	VkInstanceCreateInfo instance_ci = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app_info,
		.enabledExtensionCount = instance_extensions_count,
		.ppEnabledExtensionNames = instance_extensions,
	};

	check_vk (vkCreateInstance (&instance_ci, NULL, &instance));
	volkLoadInstance (instance);

	/*
	 * vulkan device
	 *
	 * the vulkan device is the actual device which will be doing(:D) vulkan
	 * this is generally a gpu, but could also be a software implementation
	 *
	 * it is possible to have multiple vulkan devices in a system
	 * 	multiple gpus
	 * 	integrated and discrete gpu
	 * 	...
	 */
	uint32_t device_count = 0;
	// this is a common pattern for lists
	// 	call a function to get the number of elements in the list
	// 	then call the function again to actually fill the list
	check_vk (vkEnumeratePhysicalDevices (instance, &device_count, NULL));
	cvector (VkPhysicalDevice) devices = NULL;
	cvector_init_and_size (devices, device_count, VK_NULL_HANDLE, NULL);
	check_vk (vkEnumeratePhysicalDevices (instance, &device_count, devices));

	// most systems will only have 1 device
	// 	if a user wants to specify a device they can do so as a command line argument
	uint32_t device_index = 0;
	if (argc > 1)
	{
		device_index = atoi (argv[1]);
		assert (device_index < device_count);
	}

	VkPhysicalDeviceProperties2 device_properties = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
	vkGetPhysicalDeviceProperties2 (devices[device_index], &device_properties);
	printf ("selected device: %s\n", device_properties.properties.deviceName);

	/*
	 * queues
	 *
	 * work is submitted to the device via queues
	 */
	uint32_t queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties (devices[device_index], &queue_family_count, NULL);
	cvector (VkQueueFamilyProperties) queue_families = NULL;
	cvector_init_and_size (queue_families, queue_family_count, (VkQueueFamilyProperties) {0}, NULL);
	vkGetPhysicalDeviceQueueFamilyProperties (devices[device_index], &queue_family_count, queue_families);
	uint32_t queue_family = 0;

	for (size_t iter = 0; iter < cvector_size (queue_families); iter++)
	{
		if (queue_families[iter].queueFlags & VK_QUEUE_GRAPHICS_BIT)
		{
			queue_family = iter;
			break;
		}
	}

	check (SDL_Vulkan_GetPresentationSupport (instance, devices[device_index], queue_family), true);

	/*
	 * logical device
	 *
	 * the logical device is the actual device's (gpu's) implementation
	 * 	which the application will interact with
	 */
	const float qfpriorities = 1.0f;
	VkDeviceQueueCreateInfo queue_ci = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = queue_family,
		.queueCount = 1,
		.pQueuePriorities = &qfpriorities
	};

	// setting core features we want to use
	VkPhysicalDeviceVulkan12Features enabled_vk12_features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.descriptorIndexing = true,
		.shaderSampledImageArrayNonUniformIndexing = true,
		.descriptorBindingVariableDescriptorCount = true,
		.runtimeDescriptorArray = true,
		.bufferDeviceAddress = true
	};
	VkPhysicalDeviceVulkan13Features enabled_vk13_features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.pNext = &enabled_vk12_features,
		.synchronization2 = true,
		.dynamicRendering = true,
	};
	// core features from 1.2 and 1.3 are directly enabled above
	// 	extensions may not be present
	// 		will need to be checked for existing
	// 		will need fallback path(s) if they do not exist
	// vulkan wants device extensions to be a const char* array
	const char* const device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
	const VkPhysicalDeviceFeatures enabled_vk10_features = {
		.samplerAnisotropy = VK_TRUE,
	};

	VkDeviceCreateInfo device_ci = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = &enabled_vk13_features,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &queue_ci,
		.enabledExtensionCount = 1,
		.ppEnabledExtensionNames = device_extensions,
		.pEnabledFeatures = &enabled_vk10_features
	};
	check_vk (vkCreateDevice (devices[device_index], &device_ci, NULL, &device));
	// we need a queue to submit graphics commands to
	// 	which requires the device we just created
	vkGetDeviceQueue (device, queue_family, 0, &queue);

	/*
	 * vulkan memory allocator
	 */
	VmaVulkanFunctions vk_functions = {
		.vkGetInstanceProcAddr = vkGetInstanceProcAddr,
		.vkGetDeviceProcAddr = vkGetDeviceProcAddr,
		.vkCreateImage = vkCreateImage
	};
	VmaAllocatorCreateInfo allocator_ci = {
		.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
		.physicalDevice = devices[device_index],
		.device = device,
		.pVulkanFunctions = &vk_functions,
		.instance = instance
	};
	check_vk (vmaCreateAllocator (&allocator_ci, &allocator));

	/*
	 * window and surface
	 */
	window = SDL_CreateWindow ("how to vulkan", 1280, 720, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
	assert (window);

	check (SDL_Vulkan_CreateSurface (window, instance, NULL, &surface), true);
	check (SDL_GetWindowSize (window, &window_size.x, &window_size.y), true);

	VkSurfaceCapabilitiesKHR surface_capabilities;
	check_vk (vkGetPhysicalDeviceSurfaceCapabilitiesKHR (devices[device_index], surface, &surface_capabilities));

	/*
	 * swapchain
	 *
	 * basically a series of images which are enqueued
	 * 	to the presentation engine of the operating system
	 */
	const VkFormat image_format = {VK_FORMAT_B8G8R8A8_SRGB};
	VkSwapchainCreateInfoKHR swapchain_ci = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.surface = surface,
		.minImageCount = surface_capabilities.minImageCount,
		.imageFormat = image_format,
		.imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR,
		.imageExtent = {surface_capabilities.currentExtent.width, surface_capabilities.currentExtent.height},
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
		.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode = VK_PRESENT_MODE_FIFO_KHR
	};
	check_vk (vkCreateSwapchainKHR (device, &swapchain_ci, NULL, &swapchain));

	uint32_t image_count = 0;
	check_vk (vkGetSwapchainImagesKHR (device, swapchain, &image_count, NULL));
	cvector_init_and_size (swapchain_images, image_count, VK_NULL_HANDLE, NULL);
	check_vk (vkGetSwapchainImagesKHR (device, swapchain, &image_count, swapchain_images));

	cvector_init_and_size (swapchain_image_views, image_count, VK_NULL_HANDLE, NULL);
	for (int iter = 0; iter < image_count; iter++)
	{
		VkImageViewCreateInfo view_ci = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = swapchain_images[iter],
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = image_format,
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1}
		};
		check_vk (vkCreateImageView (device, &view_ci, NULL, &swapchain_image_views[iter]));
	}

	/*
	 * depth attachment
	 */
	VkFormat depth_format_list[] = {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
	VkFormat depth_format = VK_FORMAT_UNDEFINED;
	for (int iter = 0; iter < 2; iter++)
	{
		VkFormat format = depth_format_list[iter];
		VkFormatProperties2 format_properties = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
		vkGetPhysicalDeviceFormatProperties2 (devices[device_index], format, &format_properties);

		if (format_properties.formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			depth_format = format;
			break;
		}
	}
	assert (depth_format != VK_FORMAT_UNDEFINED);

	VkImageCreateInfo depth_image_ci = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = depth_format,
		.extent = (VkExtent3D) {.width = window_size.x, .height = window_size.y, .depth = 1},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
	};

	VmaAllocationCreateInfo alloc_ci = {
		.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO
	};
	check_vk (vmaCreateImage (allocator, &depth_image_ci, &alloc_ci, &depth_image, &depth_image_allocation, NULL));

	VkImageViewCreateInfo depth_view_ci = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = depth_image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = depth_format,
		.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .levelCount = 1, .layerCount = 1}
	};
	check_vk (vkCreateImageView (device, &depth_view_ci, NULL, &depth_image_view));

	/*
	 * mesh data
	 */
	// i didnt want to handle triangulating the suzanne model myself
	// 	so we load the already triangulated model
	fastObjMesh* mesh = fast_obj_read ("resources/suzanne_triangulated.obj");
	if (!mesh)
	{
		printf ("failed to load mesh\n");
		return 1;
	}
	const VkDeviceSize index_count = mesh->index_count;
	cvector (vertex_t) vertices = NULL;
	cvector_init (vertices, 128, NULL);
	cvector (uint16_t) indices = NULL;
	cvector_init (indices, 128, NULL);

	for (unsigned int iter = 0; iter < mesh->index_count; iter++)
	{
		fastObjIndex* mesh_vertex = &mesh->indices[iter];
		vertex_t new_vertex = {0};

		glm_vec3_copy (&mesh->positions[mesh_vertex->p * 3], new_vertex.position);
		glm_vec3_copy (&mesh->normals[mesh_vertex->n * 3], new_vertex.normal);
		glm_vec2_copy (&mesh->texcoords[mesh_vertex->t * 2], new_vertex.uv);

		// flip y-axis values to accomodate vulkan's coordinate system
		new_vertex.position[1] = -new_vertex.position[1];
		new_vertex.normal[1] = -new_vertex.normal[1];
		new_vertex.uv[1] = 1.0f - new_vertex.uv[1];

		cvector_push_back (vertices, new_vertex);
		cvector_push_back (indices, cvector_size (indices));
	}

	VkDeviceSize vertex_buffer_size = sizeof (vertex_t) * cvector_size (vertices);
	VkDeviceSize index_buffer_size = sizeof (uint16_t) * cvector_size (indices);
	VkBufferCreateInfo buffer_ci = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = vertex_buffer_size + index_buffer_size,
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
	};

	VmaAllocationCreateInfo buffer_alloc_ci = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO
	};
	check_vk (vmaCreateBuffer (allocator, &buffer_ci, &buffer_alloc_ci, &vertex_buffer, &vertex_buffer_allocation, NULL));

	void* buffer_pointer = NULL;
	check_vk (vmaMapMemory (allocator, vertex_buffer_allocation, &buffer_pointer));
	memcpy (buffer_pointer, vertices, vertex_buffer_size);
	memcpy (((char*) buffer_pointer) + vertex_buffer_size, indices, index_buffer_size);
	vmaUnmapMemory (allocator, vertex_buffer_allocation);

	/*
	 * shader data buffers
	 */
	for (int iter = 0; iter < MAX_FRAMES_IN_FLIGHT; iter++)
	{
		// create and allocate a uniform buffer (for each frame in flight)
		VkBufferCreateInfo uniform_buffer_ci = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof (shader_data_t),
			.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		};
		VmaAllocationCreateInfo uniform_buffer_alloc_ci = {
			.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO
		};
		check_vk (vmaCreateBuffer (allocator, &uniform_buffer_ci, &uniform_buffer_alloc_ci, &shader_data_buffers[iter].buffer, &shader_data_buffers[iter].allocation, NULL));
		check_vk (vmaMapMemory (allocator, shader_data_buffers[iter].allocation, &shader_data_buffers[iter].mapped));

		// to be able to access the buffer in our shader
		// 	we get its address and store it for later use
		VkBufferDeviceAddressInfo uniform_buffer_bda_info = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = shader_data_buffers[iter].buffer
		};
		shader_data_buffers[iter].device_address = vkGetBufferDeviceAddress (device, &uniform_buffer_bda_info);
	}

	/*
	 * synchronization objects
	 *
	 * fences
	 * 	used to signal to the cpu that the gpu has completed work
	 * 	make sure a shared gpu/cpu resource can be modified on the cpu
	 * 	have to be created and stored
	 *
	 * semaphores
	 * 	used to control access to resources on the gpu
	 * 	use them to ensure proper ordering for things like presentation
	 * 	have to be created and stored
	 *
	 * pipeline barriers
	 * 	used to control resource access within a gpu queue
	 * 	we use them for layout transitions of images
	 * 	issued as commands (not created and stored)
	 */
	VkSemaphoreCreateInfo semaphore_ci = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
	};
	VkFenceCreateInfo fence_ci = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT
	};

	for (int iter = 0; iter < MAX_FRAMES_IN_FLIGHT; iter++)
	{
		check_vk (vkCreateFence (device, &fence_ci, NULL, &fences[iter]));
		check_vk (vkCreateSemaphore (device, &semaphore_ci, NULL, &present_semaphores[iter]));
	}

	// the number of semaphores needs to match the number of swapchain images
	// 	this is relevant later to command buffer submission
	cvector_init_and_size (render_semaphores, cvector_size (swapchain_images), VK_NULL_HANDLE, NULL);
	for (int iter = 0; iter < cvector_size (render_semaphores); iter++)
	{
		check_vk (vkCreateSemaphore (device, &semaphore_ci, NULL, &render_semaphores[iter]));
	}

	/*
	 * command buffers
	 */
	VkCommandPoolCreateInfo command_pool_ci = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = queue_family
	};
	check_vk (vkCreateCommandPool (device, &command_pool_ci, NULL, &command_pool));

	VkCommandBufferAllocateInfo command_buffer_alloc_ci = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = command_pool,
		.commandBufferCount = MAX_FRAMES_IN_FLIGHT
	};
	check_vk (vkAllocateCommandBuffers (device, &command_buffer_alloc_ci, command_buffers));

	/*
	 * loading textures
	 */
	cvector (VkDescriptorImageInfo) texture_descriptors = NULL;
	cvector_init (texture_descriptors, 1, NULL);
	cvector_init (textures, 3, NULL);
	// initialize the textures
	for (int iter = 0; iter < 3; iter++)
	{
		texture_t texture = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
		cvector_push_back (textures, texture);
	}

	// going to re-use this inside of this for loop
	// 	declare it here so we arent re-allocating it every loop
	cvector (VkBufferImageCopy) copy_regions = NULL;
	cvector_init (copy_regions, 1, NULL);
	for (int iter = 0; iter < cvector_size (textures); iter++)
	{
		ktxTexture* ktx_texture = NULL;
		char filename[64] = {0};
		sprintf (filename, "resources/suzanne%i.ktx", iter);

		ktxTexture_CreateFromNamedFile (filename, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktx_texture);
		VkImageCreateInfo texture_image_ci = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = ktxTexture_GetVkFormat (ktx_texture),
			.extent = {.width = ktx_texture->baseWidth, .height = ktx_texture->baseHeight, .depth = 1},
			.mipLevels = ktx_texture->numLevels,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
		};
		VmaAllocationCreateInfo texture_image_alloc_ci = {
			.usage = VMA_MEMORY_USAGE_AUTO
		};
		check_vk (vmaCreateImage (allocator, &texture_image_ci, &texture_image_alloc_ci, &textures[iter].image, &textures[iter].allocation, NULL));

		VkImageViewCreateInfo texture_view_ci = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = textures[iter].image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = texture_image_ci.format,
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = ktx_texture->numLevels, .layerCount = 1}
		};
		check_vk (vkCreateImageView (device, &texture_view_ci, NULL, &textures[iter].view));

		// storing the image using VK_IMAGE_TILING_OPTIMAL
		// 	means using an implementation dependent arrangement
		// 	which is opaque to us
		// we have to create an intermediate buffer to copy the image data to
		// 	then issue a command to the gpu to copy the buffer to the image
		// 		converting the buffer to the internal arrangement during the copy
		VkBuffer image_source_buffer;
		VmaAllocation image_source_allocation;
		VkBufferCreateInfo image_source_buffer_ci = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = (uint32_t) ktx_texture->dataSize,
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		};
		VmaAllocationCreateInfo image_source_alloc_ci = {
			.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO
		};
		check_vk (vmaCreateBuffer (allocator, &image_source_buffer_ci, &image_source_alloc_ci, &image_source_buffer, &image_source_allocation, NULL));

		void* image_source_buffer_pointer = NULL;
		check_vk (vmaMapMemory (allocator, image_source_allocation, &image_source_buffer_pointer));
		memcpy (image_source_buffer_pointer, ktx_texture->pData, ktx_texture->dataSize);

		// need a fence to let the cpu know the gpu is done
		VkFenceCreateInfo fence_one_time_ci = {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
		};
		VkFence fence_one_time;
		check_vk (vkCreateFence (device, &fence_one_time_ci, NULL, &fence_one_time));

		VkCommandBuffer command_buffer_one_time;
		VkCommandBufferAllocateInfo command_buffer_one_time_ai = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = command_pool,
			.commandBufferCount = 1
		};
		check_vk (vkAllocateCommandBuffers (device, &command_buffer_one_time_ai, &command_buffer_one_time));

		// start recording the commands required to get image data to its destination
		VkCommandBufferBeginInfo command_buffer_one_time_bi = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
		};
		check_vk (vkBeginCommandBuffer (command_buffer_one_time, &command_buffer_one_time_bi));

		VkImageMemoryBarrier2 barrier_texture_image = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_NONE,
			.srcAccessMask = VK_ACCESS_2_NONE,
			.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.image = textures[iter].image,
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = ktx_texture->numLevels, .layerCount = 1}
		};
		VkDependencyInfo barrier_texture_info = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &barrier_texture_image
		};
		vkCmdPipelineBarrier2 (command_buffer_one_time, &barrier_texture_info);

		cvector_clear (copy_regions);
		for (int jter = 0; jter < ktx_texture->numLevels; jter++)
		{
			ktx_size_t mip_offset = 0;
			/*KTX_error_code code =*/ ktxTexture_GetImageOffset (ktx_texture, jter, 0, 0, &mip_offset);
			VkBufferImageCopy new_copy_region = {
				.bufferOffset = mip_offset,
				.imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = (uint32_t) jter, .layerCount = 1},
				.imageExtent = {.width = ktx_texture->baseWidth >> jter, .height = ktx_texture->baseHeight >> jter, .depth = 1}
			};
			cvector_push_back (copy_regions, new_copy_region);
		}
		vkCmdCopyBufferToImage (command_buffer_one_time, image_source_buffer, textures[iter].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, cvector_size (copy_regions), copy_regions);

		VkImageMemoryBarrier2 barrier_texture_read = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
			.image = textures[iter].image,
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = ktx_texture->numLevels, .layerCount = 1}
		};
		barrier_texture_info.pImageMemoryBarriers = &barrier_texture_read;
		vkCmdPipelineBarrier2 (command_buffer_one_time, &barrier_texture_info);
		check_vk (vkEndCommandBuffer (command_buffer_one_time));

		VkSubmitInfo one_time_si = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &command_buffer_one_time
		};
		check_vk (vkQueueSubmit (queue, 1, &one_time_si, fence_one_time));
		check_vk (vkWaitForFences (device, 1, &fence_one_time, VK_TRUE, UINT64_MAX));

		vkDestroyFence (device, fence_one_time, NULL);
		vmaUnmapMemory (allocator, image_source_allocation);
		vmaDestroyBuffer (allocator, image_source_buffer, image_source_allocation);

		// sampler
		VkSamplerCreateInfo sampler_ci = {
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.anisotropyEnable = VK_TRUE,
			.maxAnisotropy = 8.0f, // 8 is a widely supported value for max anisotropy
			.maxLod = (float) ktx_texture->numLevels
		};
		check_vk (vkCreateSampler (device, &sampler_ci, NULL, &textures[iter].sampler));

		ktxTexture_Destroy (ktx_texture);
		VkDescriptorImageInfo descriptor = {
			.sampler = textures[iter].sampler,
			.imageView = textures[iter].view,
			.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL
		};
		cvector_push_back (texture_descriptors, descriptor);
	}
	cvector_free (copy_regions);

	// descriptor (indexing)
	VkDescriptorBindingFlags descriptor_variable_flag = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
	VkDescriptorSetLayoutBindingFlagsCreateInfo descriptor_binding_flags = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
		.bindingCount = 1,
		.pBindingFlags = &descriptor_variable_flag
	};
	VkDescriptorSetLayoutBinding descriptor_layout_binding_texture = {
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = cvector_size (textures),
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
	};
	VkDescriptorSetLayoutCreateInfo descriptor_layout_texture_ci = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = &descriptor_binding_flags,
		.bindingCount = 1,
		.pBindings = &descriptor_layout_binding_texture
	};
	check_vk (vkCreateDescriptorSetLayout (device, &descriptor_layout_texture_ci, NULL, &descriptor_set_layout_texture));

	VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = cvector_size (textures)
	};
	VkDescriptorPoolCreateInfo descriptor_pool_ci = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size
	};
	check_vk (vkCreateDescriptorPool (device, &descriptor_pool_ci, NULL, &descriptor_pool));

	uint32_t variable_descriptor_count = cvector_size (textures);
	VkDescriptorSetVariableDescriptorCountAllocateInfo variable_descriptor_count_ai = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT,
		.descriptorSetCount = 1,
		.pDescriptorCounts = &variable_descriptor_count
	};
	VkDescriptorSetAllocateInfo texture_descriptor_set_alloc = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext = &variable_descriptor_count_ai,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &descriptor_set_layout_texture
	};
	check_vk (vkAllocateDescriptorSets (device, &texture_descriptor_set_alloc, &descriptor_set_texture));

	// initialize the descriptor set with the above texture descriptors
	VkWriteDescriptorSet write_descriptor_set = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptor_set_texture,
		.dstBinding = 0,
		.descriptorCount = cvector_size (texture_descriptors),
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = texture_descriptors
	};
	vkUpdateDescriptorSets (device, 1, &write_descriptor_set, 0, NULL);

	FILE* spirv_file = fopen ("source/shader.spv", "r");
	if (!spirv_file)
	{
		printf ("failed to load spir-v file\n");
		exit (1);
	}
	fseek (spirv_file, 0, SEEK_END);
	size_t length = ftell (spirv_file);
	fseek (spirv_file, 0, SEEK_SET);
	char* spirv_buffer = malloc (length);
	fread (spirv_buffer, 1, length, spirv_file);
	fclose (spirv_file);

	VkShaderModuleCreateInfo shader_module_ci = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = length,
		.pCode = (uint32_t*) spirv_buffer
	};
	VkShaderModule shader_module;
	check_vk (vkCreateShaderModule (device, &shader_module_ci, NULL, &shader_module));
	free (spirv_buffer);

	/*
	 * pipeline
	 */
	VkPushConstantRange push_constant_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.size = sizeof (VkDeviceAddress)
	};
	VkPipelineLayoutCreateInfo pipeline_layout_ci = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout_texture,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range
	};
	check_vk (vkCreatePipelineLayout (device, &pipeline_layout_ci, NULL, &pipeline_layout));

	VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof (vertex_t),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
	};
	// hard coding this to 3 attributes
	VkVertexInputAttributeDescription vertex_attributes[3] = {
		{.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
		{.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof (vertex_t, normal)},
		{.location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof (vertex_t, uv)}
	};

	VkPipelineVertexInputStateCreateInfo vertex_input_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 3,  // hard coded array of 3
		.pVertexAttributeDescriptions = vertex_attributes
	};
	VkPipelineInputAssemblyStateCreateInfo input_assembly_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
	};

	// hard coding this to 2 stages
	VkPipelineShaderStageCreateInfo shader_stages[2] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = shader_module,
			.pName = "main"
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = shader_module,
			.pName = "main"
		}
	};

	VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1
	};
	VkDynamicState dynamic_states[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamic_state_ci = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states
	};

	VkPipelineDepthStencilStateCreateInfo depth_stencil_state_ci = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL
	};

	// we dont use the following states
	// 	but they must be specified, and need some sane default values
	VkPipelineColorBlendAttachmentState blend_attachment = {.colorWriteMask = 0xf};
	VkPipelineColorBlendStateCreateInfo color_blend_state_ci = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment
	};
	VkPipelineRasterizationStateCreateInfo rasterization_state_ci = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.lineWidth = 1.0f
	};
	VkPipelineMultisampleStateCreateInfo multisample_state_ci = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
	};
	// end unused states
	
	// this tells the pipeline we want to use dynamic rendering instead of render pass objects
	// this functionality was added later in vulkan
	// 	there is no dedicated member for it in pipeline create info
	// 	it will be passed to pNext
	VkPipelineRenderingCreateInfo rendering_ci = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = 1,
		.pColorAttachmentFormats = &image_format,
		.depthAttachmentFormat = depth_format
	};
	VkGraphicsPipelineCreateInfo pipeline_ci = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &rendering_ci,
		.stageCount = 2,
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input_state,
		.pInputAssemblyState = &input_assembly_state,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization_state_ci,
		.pMultisampleState = &multisample_state_ci,
		.pDepthStencilState = &depth_stencil_state_ci,
		.pColorBlendState = &color_blend_state_ci,
		.pDynamicState = &dynamic_state_ci,
		.layout = pipeline_layout
	};
	check_vk (vkCreateGraphicsPipelines (device, VK_NULL_HANDLE, 1, &pipeline_ci, NULL, &pipeline));

	/*
	 * render loop
	 */
	uint64_t last_time = SDL_GetTicks ();
	bool quit = false;

	while (!quit)
	{
		// sync gpu and cpu
		check_vk (vkWaitForFences (device, 1, &fences[frame_index], true, UINT64_MAX));
		check_vk (vkResetFences (device, 1, &fences[frame_index]));
		check_swapchain (vkAcquireNextImageKHR (device, swapchain, UINT64_MAX, present_semaphores[frame_index], VK_NULL_HANDLE, &image_index));

		// update shader data
		shader_data.projection = glms_perspective (glm_rad (45.0f), (float) window_size.x / (float) window_size.y, 0.1f, 32.0f);
		shader_data.view = glms_translate (GLMS_MAT4_IDENTITY, camera_position);
		for (int iter = 0; iter < 3; iter++)
		{
			vec3s instance_position = {{(float) (iter - 1) * 3.0f, 0.0f, 0.0f}};
			shader_data.model[iter] = glms_mat4_mul (glms_translate (GLMS_MAT4_IDENTITY, instance_position), glms_quat_mat4 (euler_to_quat (object_rotations[iter])));
		}
		memcpy (shader_data_buffers[frame_index].mapped, &shader_data, sizeof (shader_data));

		/*
		 * build command buffer
		 */
		// put command buffer into initial state, by resetting it
		VkCommandBuffer command_buffer = command_buffers[frame_index];
		check_vk (vkResetCommandBuffer (command_buffer, 0));

		VkCommandBufferBeginInfo command_buffer_bi = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
		};
		check_vk (vkBeginCommandBuffer (command_buffer, &command_buffer_bi));

		VkImageMemoryBarrier2 output_barriers[2] = {
			{
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = 0,
				.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
				.image = swapchain_images[image_index],
				.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1}
			},
			{
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
				.image = depth_image,
				.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, .levelCount = 1, .layerCount = 1}
			}
		};
		VkDependencyInfo barrier_dependency_info = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.imageMemoryBarrierCount = 2,
			.pImageMemoryBarriers = output_barriers
		};
		vkCmdPipelineBarrier2 (command_buffer, &barrier_dependency_info);

		VkRenderingAttachmentInfo color_attachment_info = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = swapchain_image_views[image_index],
			.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = {.color = {{0.0f, 0.2f, 0.2f, 1.0f}}}
		};
		VkRenderingAttachmentInfo depth_attachment_info = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = depth_image_view,
			.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.clearValue = {.depthStencil = {1.0f, 0}}
		};
		VkRenderingInfo rendering_info = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
			.renderArea = {.extent = {.width = window_size.x, .height = window_size.y}},
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &color_attachment_info,
			.pDepthAttachment = &depth_attachment_info
		};
		// put command buffer into writable state
		vkCmdBeginRendering (command_buffer, &rendering_info);

		VkViewport viewport = {
			.width = window_size.x,
			.height = window_size.y,
			.minDepth = 0.0f,
			.maxDepth = 1.0f
		};
		vkCmdSetViewport (command_buffer, 0, 1, &viewport);

		VkRect2D scissor = {.extent = {.width = window_size.x, .height = window_size.y}};
		vkCmdSetScissor (command_buffer, 0, 1, &scissor);

		vkCmdBindPipeline (command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		vkCmdBindDescriptorSets (command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set_texture, 0, NULL);
		VkDeviceSize vertex_offset = 0;
		vkCmdBindVertexBuffers (command_buffer, 0, 1, &vertex_buffer, &vertex_offset);
		vkCmdBindIndexBuffer (command_buffer, vertex_buffer, vertex_buffer_size, VK_INDEX_TYPE_UINT16);
		vkCmdPushConstants (command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof (VkDeviceAddress), &shader_data_buffers[frame_index].device_address);

		vkCmdDrawIndexed (command_buffer, index_count, 3, 0, 0, 0);

		vkCmdEndRendering (command_buffer);

		VkImageMemoryBarrier2 barrier_present = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstAccessMask = 0,
			.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			.image = swapchain_images[image_index],
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1}
		};
		VkDependencyInfo barrier_present_dependency_info = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &barrier_present
		};
		vkCmdPipelineBarrier2 (command_buffer, &barrier_present_dependency_info);

		vkEndCommandBuffer (command_buffer);

		VkPipelineStageFlags wait_stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo submit_info = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &present_semaphores[frame_index],
			.pWaitDstStageMask = &wait_stages,
			.commandBufferCount = 1,
			.pCommandBuffers = &command_buffer,
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &render_semaphores[image_index]
		};
		check_vk (vkQueueSubmit (queue, 1, &submit_info, fences[frame_index]));

		frame_index = (frame_index + 1) % MAX_FRAMES_IN_FLIGHT;

		VkPresentInfoKHR present_info = {
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &render_semaphores[image_index],
			.swapchainCount = 1,
			.pSwapchains = &swapchain,
			.pImageIndices = &image_index
		};
		// this will enqueue the image for presentation after waiting for the render semaphore
		check_swapchain (vkQueuePresentKHR (queue, &present_info));

		/*
		 * poll events
		 */
		float elapsed_time = (SDL_GetTicks () - last_time) / 1000.0f;
		last_time = SDL_GetTicks ();
		SDL_Event event;

		while (SDL_PollEvent (&event))
		{
			if (event.type == SDL_EVENT_QUIT)
			{
				quit = true;
				break;
			}
			if (event.type == SDL_EVENT_MOUSE_MOTION)
			{
				if (event.button.button == SDL_BUTTON_LEFT)
				{
					object_rotations[shader_data.selected].x -= (float) event.motion.yrel * elapsed_time;
					object_rotations[shader_data.selected].y += (float) event.motion.xrel * elapsed_time;
				}
			}
			if (event.type == SDL_EVENT_MOUSE_WHEEL)
			{
				camera_position.z += (float) event.wheel.y * elapsed_time * 10.0f;
			}
			if (event.type == SDL_EVENT_KEY_DOWN)
			{
				if (event.key.key == SDLK_ESCAPE)
				{
					quit = true;
					break;
				}
				if (event.key.key == SDLK_EQUALS || event.key.key == SDLK_KP_PLUS)
				{
					shader_data.selected = (shader_data.selected < 2) ? shader_data.selected + 1 : 0;
				}
				if (event.key.key == SDLK_MINUS || event.key.key == SDLK_KP_MINUS)
				{
					shader_data.selected = (shader_data.selected > 0) ? shader_data.selected - 1 : 2;
				}
			}
			if (event.type == SDL_EVENT_WINDOW_RESIZED)
			{
				update_swapchain = true;
			}
		}

		if (update_swapchain)
		{
			update_swapchain = false;

			vkDeviceWaitIdle (device);
			check_vk (vkGetPhysicalDeviceSurfaceCapabilitiesKHR (devices[device_index], surface, &surface_capabilities));

			swapchain_ci.oldSwapchain = swapchain;
			swapchain_ci.imageExtent = (VkExtent2D) {.width = (uint32_t) window_size.x, .height = (uint32_t) window_size.y};
			check_vk (vkCreateSwapchainKHR (device, &swapchain_ci, NULL, &swapchain));

			for (int iter = 0; iter < image_count; iter++)
			{
				vkDestroyImageView (device, swapchain_image_views[iter], NULL);
			}
			check_vk (vkGetSwapchainImagesKHR (device, swapchain, &image_count, NULL));
			cvector_resize (swapchain_images, image_count, VK_NULL_HANDLE);
			check_vk (vkGetSwapchainImagesKHR (device, swapchain, &image_count, swapchain_images));

			cvector_resize (swapchain_image_views, image_count, VK_NULL_HANDLE);
			for (int iter = 0; iter < image_count; iter++)
			{
				VkImageViewCreateInfo view_ci = {
					.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
					.image = swapchain_images[iter],
					.viewType = VK_IMAGE_VIEW_TYPE_2D,
					.format = image_format,
					.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1}
				};
				check_vk (vkCreateImageView (device, &view_ci, NULL, &swapchain_image_views[iter]));
			}

			vkDestroySwapchainKHR (device, swapchain_ci.oldSwapchain, NULL);
			vmaDestroyImage (allocator, depth_image, depth_image_allocation);
			vkDestroyImageView (device, depth_image_view, NULL);
			depth_image_ci.extent = (VkExtent3D) {.width = window_size.x, .height = window_size.y, .depth = 1};
			VmaAllocationCreateInfo alloc_ci = {
				.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				.usage = VMA_MEMORY_USAGE_AUTO
			};
			check_vk (vmaCreateImage (allocator, &depth_image_ci, &alloc_ci, &depth_image, &depth_image_allocation, NULL));
			VkImageViewCreateInfo view_ci = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = depth_image,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = depth_format,
				.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .levelCount = 1, .layerCount = 1}
			};
			check_vk (vkCreateImageView (device, &view_ci, NULL, &depth_image_view));
		}
	}

	/*
	 * tear down
	 */
	// make sure none of the gpu resources we are going to destroy are still in use
	check_vk (vkDeviceWaitIdle (device));

	/*
	 * ordering of commands only matters for the vma allocator, vulkan device, and vulkan instance
	 * the instance should be deleted last
	 * 	when using validation layers we will be notified of anything we did not delete properly
	 */

	for (int iter = 0; iter < MAX_FRAMES_IN_FLIGHT; iter++)
	{
		vkDestroyFence (device, fences[iter], NULL);
		vkDestroySemaphore (device, present_semaphores[iter], NULL);
		vmaUnmapMemory (allocator, shader_data_buffers[iter].allocation);
		vmaDestroyBuffer (allocator, shader_data_buffers[iter].buffer, shader_data_buffers[iter].allocation);
	}
	for (int iter = 0; iter < cvector_size (render_semaphores); iter++)
	{
		vkDestroySemaphore (device, render_semaphores[iter], NULL);
	}

	vmaDestroyImage (allocator, depth_image, depth_image_allocation);
	vkDestroyImageView (device, depth_image_view, NULL);

	for (int iter = 0; iter < cvector_size (swapchain_image_views); iter++)
	{
		vkDestroyImageView (device, swapchain_image_views[iter], NULL);
	}
	vmaDestroyBuffer (allocator, vertex_buffer, vertex_buffer_allocation);
	for (int iter = 0; iter < cvector_size (textures); iter++)
	{
		vkDestroyImageView (device, textures[iter].view, NULL);
		vkDestroySampler (device, textures[iter].sampler, NULL);
		vmaDestroyImage (allocator, textures[iter].image, textures[iter].allocation);
	}

	vkDestroyDescriptorSetLayout (device, descriptor_set_layout_texture, NULL);
	vkDestroyDescriptorPool (device, descriptor_pool, NULL);
	vkDestroyPipelineLayout (device, pipeline_layout, NULL);
	vkDestroyPipeline (device, pipeline, NULL);
	vkDestroySwapchainKHR (device, swapchain, NULL);
	vkDestroySurfaceKHR (instance, surface, NULL);
	vkDestroyCommandPool (device, command_pool, NULL);
	vkDestroyShaderModule (device, shader_module, NULL);
	vmaDestroyAllocator (allocator);

	cvector_free (swapchain_images);
	cvector_free (swapchain_image_views);
	cvector_free (textures);
	cvector_free (render_semaphores);
	cvector_free (devices);
	cvector_free (queue_families);
	cvector_free (vertices);
	cvector_free (indices);
	cvector_free (texture_descriptors);

	SDL_DestroyWindow (window);
	SDL_QuitSubSystem (SDL_INIT_VIDEO);
	SDL_Quit ();

	vkDestroyDevice (device, NULL);
	vkDestroyInstance (instance, NULL);

	return 0;
}
