#!/bin/zsh

slangc shader.slang -matrix-layout-column-major -target spirv -o shader.spv
