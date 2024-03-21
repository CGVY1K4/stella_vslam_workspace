import os
import argparse

# need to specify below variable
path_to_executable = '/media/eunix/disk1/lib/stella_vslam_examples/build'

parser = argparse.ArgumentParser()
parser.add_argument('name') #folder name 
parser.add_argument('--yaml', default="rist_stereo360.yaml") #default rist_stereo360.yaml
parser.add_argument('--in_msg', default=None)
parser.add_argument('--use_image' , action='store_true')

args = parser.parse_args()

name = args.name
yaml = args.yaml
in_msg = args.in_msg

# hello_world_ver1 -> hello_world, ver1 -> hello_world.mp4
video_file_name = name.rsplit("_", 1)[0]+".mp4"

os.system(f'export PATH="{path_to_executable}:$PATH"')

if not os.path.exists(f"results/{name}"):
    os.system(f"mkdir results/{name}")


if not args.use_image:
	if in_msg:
		os.system(f"run_video_slam -v configs/orb_vocab.fbow --eval-log-dir results/{name} -c configs/{yaml} -m dataset/{video_file_name} -i results/{in_msg}/{in_msg}.msg -o results/{name}/{name}.msg --no-sleep")
		# os.system(f"run_image_slam -v configs/orb_vocab.fbow --eval-log-dir results/{name} -c configs/{yaml} -d . -i 	results/{name}/{in_msg}.msg -o results/{name}/{name}.msg")
	else:
	    os.system(f"run_video_slam -v configs/orb_vocab.fbow --eval-log-dir results/{name} -c configs/{yaml} -m dataset/{video_file_name} -o results/{name}/{name}.msg --no-sleep")

else:
	if in_msg:
		os.system(f"run_image_slam -v configs/orb_vocab.fbow --eval-log-dir results/{name} -c configs/{yaml} -d dataset/{name} -i results/{in_msg}/{in_msg}.msg --disable-mapping --no-sleep")
	else:
	#image_slam
		os.system(f"run_image_slam -v configs/orb_vocab.fbow --eval-log-dir results/{name} -c configs/{yaml} -i results/{in_msg}/{in_msg}.msg -d dataset/{name} --no-sleep")

