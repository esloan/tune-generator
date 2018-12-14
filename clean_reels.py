
# TODO: actual data goes here
path_to_file = "ReelsABC.txt"
text = open(path_to_file)

out_text = open('ReelsNoMetadata.txt', 'w')

metadata_chars = ['X', 'T', 'Z', 'M', 'L', 'K']

for line in text:
	if line[0] not in metadata_chars:
		out_text.write(line)

out_text.close()
text.close()