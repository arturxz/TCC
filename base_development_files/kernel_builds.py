import pyopencl as cl

def is_kernel_compiled(programs, method_name):
	for program in programs:
		if( method_name in program.kernel_names ):
			return program
	return None

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
programs = []

program = cl.Program(ctx, open("/home/artur/visiongl/src/CL/vglClCopy.cl", "r").read() )
programs.append( program.build() )

program = cl.Program(ctx, open("/home/artur/visiongl/src/CL/vglClConvolution.cl", "r").read() )
programs.append( program.build() )

print( is_kernel_compiled(programs, "vglClCopy") )
print( is_kernel_compiled(programs, "vglClCatatau") )
print( is_kernel_compiled(programs, "vglClConvolution") )