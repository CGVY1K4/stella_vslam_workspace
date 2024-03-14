import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('--frame', action='store_true')

args = parser.parse_args()

if not args.frame:
    file_name = 'keyframe_trajectory'
else:
    file_name = 'frame_trajectory'

input_file_path = "results/" + args.name + "/" + file_name + ".txt"
output_file_path = "results/" + args.name + "/" + file_name + "_extracted.txt"

with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line in input_file:
        # Split each line into values
        values = line.strip().split()

        if len(values) >= 4:
            # Extract values at index 1, 2, and 3
            extracted_values = " ".join(values[1:4])

            # Write extracted values to the output file
            output_file.write(extracted_values + "\n")

print(f"Extracted values saved to {output_file_path}")
