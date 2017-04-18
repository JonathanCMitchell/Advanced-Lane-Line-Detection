
def add_recent_centers(num, lst, arrayToTakeCentersFrom, smooth):
	if bool(len(arrayToTakeCentersFrom)):
		# print('we are inside add_recent_centers arrayToTakeCentersFrom not none and is: ', arrayToTakeCentersFrom)
		# add centers from this array
		i = 0
		while num > len(lst):
			if len(arrayToTakeCentersFrom) > smooth:
				lst.append(arrayToTakeCentersFrom[-smooth:][i])	
			else:
				lst.append(arrayToTakeCentersFrom[i])
			i += 1
		return lst
	else:
		for i in reversed(range(len(lst))):
			if len(lst) >= num:
				return lst
			else:
				lst.append(lst[i])
				print('lst is: ', lst)
		return lst