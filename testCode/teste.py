def method(**kwargs):
	mystr="arg7"
	print( any(key.startswith(mystr) for key in kwargs) )

method(arg1="ola", arg2="tchau", arg3=123)

