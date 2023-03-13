import os
import re
import subprocess
from video_split import split_video


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':
    # all videos source path
    video_dir = "./Videos"

    # result save path
    scene_video_save_dir = "./VideoSplit/VideoScenes"
    scene_info_save_dir = "./VideoSplit/SceneInfo"
    scene_image_save_dir = "./VideoSplit/SceneImages"
    scene_sketch_save_dir = "./VideoSplit/SceneSketch"


    # Start processing - split videos
    print("Start: Split Videos")
    video_dict = {}
    video_index = 0
    command_list = [] # to sketch image使用
    for each_video in os.listdir(video_dir):
        video_dict[str(video_index)] = each_video
        video_name = re.search(r"(.*)\.", each_video).group(1)
        # path to save result'
        each_video_path = os.path.join(video_dir, each_video) # OK
        each_scene_info_path = os.path.join(scene_info_save_dir, str(video_index) + ".csv") # OK
        each_scene_video_path = create_dir(os.path.join(scene_video_save_dir, str(video_index))) # OK
        each_scene_image_path = create_dir(os.path.join(scene_image_save_dir, str(video_index)))
        each_scene_sketch_path = create_dir(os.path.join(scene_sketch_save_dir, str(video_index)))
        # split the video
        print("- Split video (path): {}".format(each_video_path))
        scene_list = split_video(each_video_path, each_scene_video_path,
            each_scene_info_path, each_scene_image_path)
        # to sketch image 使用
        command_to_sketch = "python ./AdaAttN/test.py \
            --content_path ./VideoSplit/SceneImages/{}/ \
            --style_path ./AdaAttN/datasets/styles/ \
            --name AdaAttN \
            --model adaattn \
            --dataset_mode unaligned \
            --load_size 512 \
            --crop_size 512 \
            --image_encoder_path ./AdaAttN/checkpoints/vgg_normalised.pth \
            --gpu_ids 0 \
            --skip_connection_3 \
            --my_image_dir ./VideoSplit/SceneSketch/{}/ \
            --num_test {}\
            --shallow_layer".format(video_index, video_index, len(scene_list))
            # to sketch command
        command_list.append(command_to_sketch)
        video_index += 1

    # processing - Image to Sketch
    print("Start: Scene to Sketch image")
    subprocess.run(*command_list, shell=False)

    # recognize attribute
