# coding:utf-8
import os
import scenedetect
from scenedetect import open_video
from scenedetect.stats_manager import StatsManager
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors.content_detector import ContentDetector

'''
http://scenedetect.com/projects/Manual/en/latest/api/scene_manager.html
'''
def split_video(video_path, video_split_path, scene_save_path, img_save_path, save_scene=True, save_img=True, num_images=1):
    '''
    Split the video by different scenes (optional save scene list information & save img for each scene)
    Params:
    - video_path: video that need to be splitted
    - video_split_path: path of directory to save video
    - scene_save_path(optional): path for saving the scene information file (.csv)
    - img_save_path(optional): path of a directory for saving images for each scene
    - save_scene(boolean): boolean, True or False
    - save_img(boolean): boolean, True or False
    - num_images(int): save how many images for each scene
    Return:
    - scene_list: list of each scene
    '''
    assert video_path != None, "Empty 'video_path' param!"

    # detect each video scene
    video = open_video(video_path) # open the video
    scene_manager = SceneManager(StatsManager())
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    # save resource
    if save_scene == True:
        with open(scene_save_path, "w", encoding="utf-8") as file:
            scenedetect.scene_manager.write_scene_list(file, scene_list)

    if save_img == True:
        ret = scenedetect.scene_manager.save_images(scene_list, video=video, output_dir=img_save_path, num_images=num_images,
          show_progress=True, image_name_template="$SCENE_NUMBER")
    # split and save video
    assert scenedetect.video_splitter.is_ffmpeg_available(), "Please install and check ffmpeg in your device"
    scenedetect.video_splitter.split_video_ffmpeg(video_path, scene_list=scene_list,
        output_file_template = os.path.join(video_split_path, "$SCENE_NUMBER.mp4"),show_progress=True)
    return scene_list


# if __name__ == '__main__':
#     split_video("../Videos/毕业季VLOG.mp4",
#         "./VideoSplit/VideoScenes/毕业季VLOG/",
#         "./VideoSplit/毕业季VLOG.csv",
#         "./VideoSplit/SceneImages/毕业季VLOG/")
