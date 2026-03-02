project('vulkan', 'c', 'cpp')

cc = meson.get_compiler('c')
slang = find_program('slangc', required : true)

run_command(slang, 'source/shader.slang', '-matrix-layout-column-major', '-target', 'spirv', '-o', 'source/shader.spv', check : true)

vulkan_dependencies = [
  cc.find_library('m'),
  cc.find_library('ktx'),
  dependency('sdl3'),
  dependency('vulkan'),
]

include = [
  include_directories('.'),
  include_directories('external/volk'),
  include_directories('external/vma'),
  include_directories('external/cglm'),
  include_directories('external/cvector'),
  include_directories('external/fast_obj'),
]

external_source_files = []

subdir('external/volk')
subdir('external/vma')
subdir('external/fast_obj')

internal_source_files = [
  'source/main.c',
]

source_files = external_source_files + internal_source_files

add_global_arguments('-DVK_USE_PLATFORM_XLIB_KHR', language : 'c')

vulkan_binary = executable (
  'vulkan',
  source_files,
  include_directories : include,
  dependencies : [vulkan_dependencies],
)
