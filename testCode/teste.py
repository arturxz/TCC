import numpy as np

def copy_into_byte_array(value, byte_array, offset):
	for i,b in enumerate( value.tobytes() ):
		byte_array[i+offset] = b

arr1 = np.zeros(32, np.uint8)
arr2 = np.ones(16, np.uint8)

print(arr1)
print(arr2)

copy_into_byte_array(arr2, arr1, 0)

print(arr1)
print(arr2)