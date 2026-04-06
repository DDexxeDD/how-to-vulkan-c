# How to Vulkan C

A C implementation of the [How to Vulkan](https://www.howtovulkan.com) tutorial.

The tutorial is written in a way that is mostly compatible with C so there is very little difference between this code and the reference code.  The biggest difference is that Slang does not currently have a C API, so the slang shader has to be compiled to spir-v outside of the program and loaded in.

This implementation also includes `resources/suzanne_triangulated.obj` which is the same suzanne model that is included in the reference repo, but has already been triangulated.


## Building

You will need [Meson](https://mesonbuild.com/Getting-meson.html) to build.
You will also need [slang](https://shader-slang.org/) to compile the shader.

This was built and tested on linux, if you are on a platform besides linux you will need to change `VK_USE_PLATFORM_XLIB_KHR` in `meson.build` to the correct argument for your [platform](https://docs.vulkan.org/spec/latest/appendices/boilerplate.html#boilerplate-wsi-header).

## Running

The executable will be in whatever build folder you specified, but the paths to resources and the shader are relative to the top level directory (the directory this README is in). You will need to run the executable from the top level directory.

## License

The included licenses are copied from the reference implementation.
