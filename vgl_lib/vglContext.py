def vglIsContextValid(x):
	return ( (x >= 1) and (x <= 15) )
	
def vglIsContextUnique(x):
	return ( (x is 0) or (x is 1) or (x is 2) or (x is 4), (x is 8) )
	
def vglIsInContext(img, x):
	return ( ( img.inContext & x ) or ( (img.inContext is 0) and (x is 0)  ) )

"""
	EQUIVALENT TO vglContext.vglAddContext() METHOD.
	MUST CALL AFTER COPY. CONTEXT MUST BE UNIQUE.
	RETURN 0 IN CASE OF ERROR AND RESULTING CONTEXT IF SUCCESS
"""
def vglAddContext(img, context):
	if( not vglIsContextUnique(context) ):
		print("vglAddContext: Error: context =", context, "is not unique or invalid")
		return 0
	
	img.inContext = img.inContext | context
	return img.inContext

"""
	CALL AFTER PROCESSING OR CREATING IMAGE.
	CONTEXT MUST BE UNIQUE
	AFTER CREATING, USE VGL_BLANK_CONTEXT.
	RETURN 0 IN A ERROR OCCURS AND RESULTING CONTEXT IF SUCCESS.
"""
def vglSetContext(img, context):
	if( not vglIsContextUnique(context) and (context is not 0) ):
		print("vglSetContext: Error: context =", context, "is not unique")
		return 0
	
	img.inContext = context
	return img.inContext

"""
	CALL BEFORE USING IMAGE.
	CONTEXT MUST BE UNIQUE.
	RETURN 0 IF ERROR OCCURS AND RESULTING CONTEXT IF SUCCESS
	
	Shaders (vglCopy, vglDilateSq3, vglCudaInvertOnPlace etc)
	CALL ->
	vglCheckContext
	CALL ->
	vglUpload, vglDownload, vglGlToCuda, vglCudaToGl
	CALL ->
	vglSetContext, vglAddContext
"""
def vglCheckContext(img, context):
	print('vglCheckContext')

"""
	CALL BEFORE WRITING TO IMAGE. CONTEXT MUST BE UNIQUE.

    GL AND RAM IMAGES ARE ALWAYS ALLOCATED. 
	CUDA IMAGES MUST BE ALLOCATED IF cudaPbo == -1.

	BY NOW, ALL CUDA-SPECIFIC CODE WILL BE IGNORED 
	ON THIS METHOD. FUTURE INCREMENTS CAN IMPROVE THAT.
"""
def vglCheckContextForOutput(img, context):
	return 0