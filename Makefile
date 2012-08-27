##===- culeidoscope/Makefile -------------------------------*- Makefile -*-===##
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
TOOLNAME = culeidoscope
EXAMPLE_TOOL = 1
REQUIRES_RTTI := 1

LINK_COMPONENTS := core jit native

include $(LEVEL)/Makefile.common
