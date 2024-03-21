import numpy as np
import cv2
import sys
from transforms3d import quaternions
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('scale')

args = parser.parse_args()
name = args.name
scale = int(args.scale)

dir_path = "results/" + name


def transform_3d_to_2d(point_3d, scale_factor, grid_min, grid_max):
    #((grid_max_x - grid_min_x) - 1) / (grid_max_x - grid_min_x)
    norm_x = ((grid_max[0]-grid_min[0])-1) / (grid_max[0]-grid_min[0])
    norm_z = ((grid_max[1]-grid_min[1])-1) / (grid_max[1]-grid_min[1])
    x = int((np.floor(point_3d[0] * scale_factor) - grid_min[0]) * norm_x)
    z = int((np.floor(point_3d[2] * scale_factor) - grid_min[1]) * norm_z)
    return (x, z)


#
def get_line_bresenham(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


# seq_name = 'tum'
# inverse of cell size
scale_factor = scale
resize_factor =  1
filter_ground_points = 0
load_counters = 0               # 데이터가 이미 로드되었는지를 확인하는 데 사용

point_cloud_fname = f'{dir_path}/map_database.txt'
keyframe_trajectory_fname = f'{dir_path}/keyframe_trajectory.txt'
visit_counter_fname = '{:s}/{:s}_filtered_{:d}_scale_{:d}_visit_counter.txt'.format(
    dir_path, name, filter_ground_points, scale_factor)
occupied_counter_fname = '{:s}/{:s}_filtered_{:d}_scale_{:d}_occupied_counter.txt'.format(
    dir_path, name, filter_ground_points, scale_factor)

print('seq_name: ', name)
print('scale_factor: ', scale_factor)
print('resize_factor: ', resize_factor)
print('filter_ground_points: ', filter_ground_points)

counters_loaded = False
if load_counters:
    try:
        print('Loading counters...')
        visit_counter = np.loadtxt(visit_counter_fname)
        occupied_counter = np.loadtxt(occupied_counter_fname)
        grid_res = visit_counter.shape
        print('grid_res: ', grid_res)
        counters_loaded = True
    except:
        print('One or more counter files: {:s}, {:s} could not be found'.format(
            occupied_counter_fname, visit_counter_fname))
        counters_loaded = False

if not counters_loaded:
    # keyframe_trajectory_data = np.loadtxt(keyframe_trajectory_fname, dtype=np.float32)
    # keyframe_timestamps = keyframe_trajectory_data[:, 0]
    # keyframe_locations = keyframe_trajectory_data[:, 1:4]
    # keyframe_quaternions = keyframe_trajectory_data[:, 5:9]

    # read keyframes
    keyframe_trajectory_data = open(keyframe_trajectory_fname, 'r').readlines()
    keyframe_timestamps = []
    keyframe_locations = []
    keyframe_quaternions = []
    for line in keyframe_trajectory_data:
        line_tokens = line.strip().split()
        timestamp = float(line_tokens[0])
        keyframe_x = float(line_tokens[1]) * scale_factor
        keyframe_y = float(line_tokens[2]) * scale_factor
        keyframe_z = float(line_tokens[3]) * scale_factor
        keyframe_timestamps.append(timestamp)
        keyframe_locations.append([keyframe_x, keyframe_y, keyframe_z])
        keyframe_quaternions.append([float(line_tokens[4]), float(line_tokens[5]),
                                     float(line_tokens[6]), float(line_tokens[7])])

    keyframe_locations_dict = dict(zip(keyframe_timestamps, keyframe_locations))  #타임스탬프를 키로, 위치와 쿼터니언을 값으로 하는 딕셔너리를 각각 만듭니다.
    keyframe_quaternions_dict = dict(zip(keyframe_timestamps, keyframe_quaternions))
    keyframe_locations = np.array(keyframe_locations)  ##위치와 타임스탬프 리스트를 각각 넘파이 배열로 변환합니다.
    keyframe_timestamps = np.array(keyframe_timestamps)
    n_keyframes = keyframe_locations.shape[0]  #num of data

    # read point cloud
    point_cloud_data = open(point_cloud_fname, 'r').readlines()
    point_locations = []
    point_timestamps = []
    n_lines = len(point_cloud_data)
    n_points = 0
    is_ground_point = []
    for line in point_cloud_data:
        #문자열 양쪽의 공백을 제거하고, split() 함수는 공백을 기준으로 문자열을 나눠 각각의 요소를 리스트에 저장합니다. 결과적으로 이 코드는 한 줄의 텍스트를 공백을 기준으로 나눠 리스트로 만드는 역할을 합니다.
        line_tokens = line.strip().split()
        point_x = float(line_tokens[0]) * scale_factor
        point_y = float(line_tokens[1]) * scale_factor
        point_z = float(line_tokens[2]) * scale_factor

        timestamps = []

        for i in range(3, len(line_tokens)):#세 번째 토큰부터 마지막 토큰까지 반복문을 실행
            timestamps.append(float(line_tokens[i]))
        if len(timestamps) == 0:
            raise StandardError('Point {:d} has no keyframes'.format(n_points))

        is_ground_point.append(False)

        #한개의 map_point에 대해서 수행
        if filter_ground_points:
            key_frame_id = 0
            keyframe_quaternion = None
            keyframe_location = None
            while key_frame_id < len(timestamps):
                try:#키 프레임의 쿼터니언과 위치를 타임스탬프에 따라 찾습니다.
                    keyframe_quaternion = np.array(keyframe_quaternions_dict[timestamps[key_frame_id]])
                    keyframe_location = keyframe_locations_dict[timestamps[key_frame_id]]
                    break
                except KeyError:
                    key_frame_id += 1
            if keyframe_quaternion is None:
                raise StandardError('No valid keyframes found for point {:d}'.format(n_points))
            # normalize quaternion
            keyframe_quaternion /= np.linalg.norm(keyframe_quaternion)  #쿼터니언을 정규화
            keyframe_rotation = quaternions.quat2mat(keyframe_quaternion)  #쿼터니언을 회전 행렬로 변환
            keyframe_translation = np.matrix(keyframe_location).transpose()  #위치를 전치 행렬로 변환
            transform_mat = np.zeros([4, 4], dtype=np.float64)  #4x4 변환 행렬을 0으로 초기화
            transform_mat[0:3, 0:3] = keyframe_rotation.transpose()  #변환 행렬의 좌상단 3x3 부분을 회전 행렬의 전치로 설정
            transform_mat[3, 0:3] = (np.matrix(-keyframe_rotation.transpose()) * keyframe_translation).ravel()  #변환 행렬의 마지막 행의 처음 3개 요소를 설정합니다.
            transform_mat[3, 3] = 1  #변환 행렬의 마지막 요소를 1로 설정합니다.
            point_location = np.matrix([point_x, point_y, point_z, 1]).transpose()  #점의 위치를 4x1 행렬로 만들고 전치합니다.
            transformed_point_location = np.matrix(transform_mat) * point_location  #변환 행렬과 점의 위치를 곱해 변환된 위치를 구합니다.
            # homogeneous to non homogeneous coordinates
            transformed_point_height = transformed_point_location[1] / transformed_point_location[3] #변환된 위치의 높이를 구합니다.
            if transformed_point_height < 0:  #변환된 위치의 높이가 0보다 작으면, 마지막으로 추가된 지상점을 참(True)으로 설정합니다.
                is_ground_point[-1] = True  

        point_locations.append([point_x, point_y, point_z])  #점의 위치를 리스트에 추가
        point_timestamps.append(timestamps)  #타임스탬프를 리스트에 추가합니다.
        n_points += 1  #점의 개수를 1 증가시킵니다.

    point_locations = np.array(point_locations)
    print('n_keyframes: ', n_keyframes)
    print('n_points: ', n_points)
    if filter_ground_points:
        n_ground_points = np.count_nonzero(np.array(is_ground_point))
        print('n_ground_points: ', n_ground_points)
    
    #키 프레임 위치에서 x와 z 좌표의 최소값과 최대값을 계산하고, 이들을 각각 내림(np.floor)과 올림(np.ceil) 처리
    kf_min_x = np.floor(np.min(keyframe_locations[:, 0]))
    kf_min_z = np.floor(np.min(keyframe_locations[:, 2]))
    kf_max_x = np.ceil(np.max(keyframe_locations[:, 0]))
    kf_max_z = np.ceil(np.max(keyframe_locations[:, 2]))
    
    #포인트 위치에서 x와 z 좌표의 최소값과 최대값을 계산하고, 이들을 각각 내림과 올림 처리
    pc_min_x = np.floor(np.min(point_locations[:, 0]))
    pc_min_z = np.floor(np.min(point_locations[:, 2]))
    pc_max_x = np.ceil(np.max(point_locations[:, 0]))
    pc_max_z = np.ceil(np.max(point_locations[:, 2]))

    #격자의 x와 z 좌표의 최소값과 최대값을 계산합니다. 
    #이는 키 프레임 위치와 포인트 위치에서 계산한 값들 중에서 최소값과 최대값을 선택하는 것입니다.

    global grid_min_x
    global grid_min_z
    global grid_max_x
    global grid_max_z

    grid_min_x = min(kf_min_x, pc_min_x)
    grid_min_z = min(kf_min_z, pc_min_z)
    grid_max_x = max(kf_max_x, pc_max_x)
    grid_max_z = max(kf_max_z, pc_max_z)    

    print('grid_max_x: ', grid_max_x)
    print('grid_min_x: ', grid_min_x)
    print('grid_max_z: ', grid_max_z)
    print('grid_min_z: ', grid_min_z)

    #격자의 해상도(resolution)를 계산합니다. 이는 격자의 x와 z 좌표의 범위를 나타내는 것입니다
    grid_res = [int(grid_max_x - grid_min_x), int(grid_max_z - grid_min_z)]
    print('grid_res: ', grid_res)

    #방문 카운터와 점유 카운터를 초기화합니다. 이들은 각 격자 셀이 방문되거나 점유된 횟수를 저장하는 데 사용됩니다.
    visit_counter = np.zeros(grid_res, dtype=np.int32)
    occupied_counter = np.zeros(grid_res, dtype=np.int32)

    print('grid extends from ({:f}, {:f}) to ({:f}, {:f})'.format(
        grid_min_x, grid_min_z, grid_max_x, grid_max_z))

    #각 격자 셀의 x와 z 방향의 크기를 계산합니다.
    grid_cell_size_x = (grid_max_x - grid_min_x) / float(grid_res[0])
    grid_cell_size_z = (grid_max_z - grid_min_z) / float(grid_res[1])

    # print('using cell size: {:f} x {:f}'.format(grid_cell_size_x, grid_cell_size_z)

    #정규화 계수를 계산합니다. 이는 x와 z 좌표를 격자 셀의 인덱스로 변환하는 데 사용됩니다....???
    
    # ((grid_max_x - grid_min_x) - 1) / (grid_max_x - grid_min_x)
    
    norm_factor_x = float(grid_res[0] - 1) / float(grid_max_x - grid_min_x)
    norm_factor_z = float(grid_res[1] - 1) / float(grid_max_z - grid_min_z)
    print('norm_factor_x: ', norm_factor_x)
    print('norm_factor_z: ', norm_factor_z)

    # print('len(is_ground_point):', len(is_ground_point)

    for point_id in range(n_points):
        point_location = point_locations[point_id]  #현재 점의 위치를 가져옵니다.
        for timestamp in point_timestamps[point_id]:  #현재 점에 해당하는 각 타임스탬프에 대해 반복문을 실행
            try:
                keyframe_location = keyframe_locations_dict[timestamp] #타임스탬프에 해당하는 키 프레임 위치를 딕셔너리에서 가져옵니다.
            except KeyError:
                #print('Timestamp: {:f} not found'.format(timestamp))
                continue
            keyframe_x = int(keyframe_location[0])
            keyframe_z = int(keyframe_location[2])
            point_x = int(point_location[0])
            point_z = int(point_location[2])
            ray_points = get_line_bresenham([keyframe_x, keyframe_z], [point_x, point_z])  #키 프레임 위치에서 점 위치까지의 레이를 계산합니다. 여기서는 Bresenham의 알고리즘을 사용하여 격자에 맞춰 레이를 계산
            n_ray_pts = len(ray_points)

            for ray_point_id in range(n_ray_pts - 1): # 마지막 레이 포인트를 제외하고 계산
                #레이 포인트의 x, z 좌표를 격자의 인덱스로 변환합니다.
                # ex) if the ray_point coordinate is (0,0), then the normalized coordinate will be (grid_min_x, grid_min_z).
                # This makes the left top corner of the image (0,0) ??????????? not sure 
                ray_point_x_norm = int(np.floor((ray_points[ray_point_id][0] - grid_min_x) * norm_factor_x))
                ray_point_z_norm = int(np.floor((ray_points[ray_point_id][1] - grid_min_z) * norm_factor_z))
                # start_x = ray_point_x_norm - resize_factor / 2
                # start_z = ray_point_z_norm - resize_factor / 2
                # end_x = ray_point_x_norm + resize_factor / 2
                # end_z = ray_point_z_norm + resize_factor / 2
                try:
                    # visit_counter[start_x:end_x, start_z:end_z] += 1

                    #해당 격자 셀을 방문했음을 카운트
                    visit_counter[ray_point_x_norm, ray_point_z_norm] += 1
                except IndexError:
                    print('Out of bound point: ({:d}, {:d}) -> ({:f}, {:f})'.format(
                        ray_points[ray_point_id][0], ray_points[ray_point_id][1],
                        ray_point_x_norm, ray_point_z_norm))
                    sys.exit(0)
            #마지막 레이 포인트의 x, z 좌표를 격자의 인덱스로 변환
            ray_point_x_norm = int(np.floor((ray_points[-1][0] - grid_min_x) * norm_factor_x))
            ray_point_z_norm = int(np.floor((ray_points[-1][1] - grid_min_z) * norm_factor_z))
            # start_x = ray_point_x_norm - resize_factor / 2
            # start_z = ray_point_z_norm - resize_factor / 2
            # end_x = ray_point_x_norm + resize_factor / 2
            # end_z = ray_point_z_norm + resize_factor / 2
            try: #해당 격자 셀이 지상점이면 방문 카운터를 증가시키고, 그렇지 않으면 점유 카운터를 증가
                if is_ground_point[point_id]:
                    visit_counter[ray_point_x_norm, ray_point_z_norm] += 1
                else:
                    # occupied_counter[start_x:end_x, start_z:end_z] += 1
                    occupied_counter[ray_point_x_norm, ray_point_z_norm] += 1
            except IndexError:
                print('Out of bound point: ({:d}, {:d}) -> ({:f}, {:f})'.format(
                    ray_points[-1][0], ray_points[-1][1], ray_point_x_norm, ray_point_z_norm))
                sys.exit(0)
        if (point_id + 1) % 1000 == 0:
            print('Done {:d} points of {:d}'.format(point_id + 1, n_points))

    print('Saving counters to {:s} and {:s}'.format(occupied_counter_fname, visit_counter_fname))
    #점유 카운터와 방문 카운터를 각각의 파일에 저장
    np.savetxt(occupied_counter_fname, occupied_counter, fmt='%d')
    np.savetxt(visit_counter_fname, visit_counter, fmt='%d')

# occupied_counter_zeros = occupied_counter == 0
# occupied_counter[occupied_counter_zeros] = 2 * visit_counter[occupied_counter_zeros]
# grid_map = visit_counter.astype(np.float32) / occupied_counter.astype(np.float32)

free_thresh = 0.53
occupied_thresh = 0.50

grid_map = np.zeros(grid_res, dtype=np.float32)                 # grid map stores the possibility of each cell being occupied
grid_map_thresh = np.zeros(list(grid_res) + [3], dtype=np.uint8)  # create 3 channel image matrix(?)


occupied_cells = []
for x in range(grid_res[0]):
    for z in range(grid_res[1]):
        if visit_counter[x, z] == 0 or occupied_counter[x, z] == 0:
            #만약 해당 격자 셀이 방문되지 않았거나 점유되지 않았다면, 그 확률을 0.5로 설정
            grid_map[x, z] = 0.5
        else:
            # 격자 셀이 점유된 확률을 계산하고, 그 값을 1에서 빼서 빈 공간인 확률로 변환
            grid_map[x, z] = 1 - occupied_counter[x, z] / visit_counter[x, z]

        if grid_map[x, z] >= free_thresh:
            # free cell
            grid_map_thresh[x, z] = [255, 255, 255] # white
        elif occupied_thresh <= grid_map[x, z] < free_thresh:
            # unknown cell
            grid_map_thresh[x, z] = [200, 200, 200] # white
        else:
            # occupied cell
            occupied_cells.append((x,z))


# inflate occupied cell
for cell in occupied_cells: 
    x,z = cell
    for dx in range(-3,4): 
        for dz in range(-3,4):
            grid_map_thresh[x+dx, z+dz] = [0, 0, 0] # black


# draw occupied cell
for cell in occupied_cells:
    x,z = cell
    grid_map_thresh[x, z] = [255, 0, 0] # blue 


if resize_factor != 1:
    grid_res_resized = (grid_res[0] * resize_factor, grid_res[1] * resize_factor)
    print('grid_res_resized: ', grid_res_resized)
    grid_map_resized = cv2.resize(grid_map_thresh, grid_res_resized)
else:
    grid_map_resized = grid_map_thresh


out_fname = '{:s}/grid_map_{:s}_filtered_{:d}_scale_{:d}_resize_{:d}_{:.2f}_{:.2f}'.format(
    dir_path, name, filter_ground_points, scale_factor, resize_factor, occupied_thresh, free_thresh)


# convert to color scale
# color_image = cv2.cvtColor(grid_map_thresh, cv2.COLOR_GRAY2BGR)
color_image = grid_map_thresh

# draw red dot on (0, 0) coordinate of the image
origin_point = transform_3d_to_2d((0,0,0), scale_factor, (grid_min_x, grid_min_z), (grid_max_x, grid_max_z))
color_image = cv2.circle(color_image, (origin_point[1], origin_point[0]), 5, (0, 0, 255), -1) 


grid_map_resized = color_image      # this line ignores previous grid_map_resized value


cv2.imwrite('{:s}.png'.format(out_fname), grid_map_resized)
print("image written!")
cv2.imshow(out_fname, grid_map_resized)
cv2.waitKey(0)



