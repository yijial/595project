import os

input_directory = "../json"

max_size = 0
max_file = ""
for filename in os.listdir(input_directory):
	if filename.endswith(".json"):
		input_file = input_directory+"/"+filename
		cur_size = os.path.getsize(input_file)
		if cur_size > max_size:
			max_file = filename

print(max_file)
