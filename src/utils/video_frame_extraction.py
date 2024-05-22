import cv2
from cv2 import CAP_PROP_FPS
import os
import numpy as np
import time

'''
Returns True if pixel is black, False otherwise
'''
def pixel_is_black(pixel):
  BLACK = np.array([0, 0, 0])
  return (pixel == BLACK).all()

'''
Crop out the larynx video area of the frame.
Searches for first 10 black pixels at the bottom and to the left of the frame.
'''
def crop_frame(frame):
    height, width, _ = frame.shape
    BLACK_PIXEL_BUFFER = 10

    # Start in top right corner and go downwards until # of BLACK_PIXEL_BUFFER is found
    black_pixel_count = 0
    bottom_right_y = None

    for y in range(height):
        pixel = frame[y, width-1]
        if pixel_is_black(pixel):
            black_pixel_count += 1
        if black_pixel_count >= BLACK_PIXEL_BUFFER:
            bottom_right_y = y - BLACK_PIXEL_BUFFER
            break
        
    # x
    black_pixel_count = 0
    bottom_right_x = None

    for x in reversed(range(width)):
      pixel = frame[bottom_right_y, x]
      if pixel_is_black(pixel):
          black_pixel_count += 1
      if black_pixel_count >= BLACK_PIXEL_BUFFER:
          bottom_right_x = x + BLACK_PIXEL_BUFFER
          break
      
    return frame[0:bottom_right_y, bottom_right_x:width]

'''
Get the crop points for the larynx area of the frame. Returns a tuple (bottom_right_x, bottom_right_y, width).
'''
def get_crop_points(frame):
    height, width, _ = frame.shape
    BLACK_PIXEL_BUFFER = 1

    # Start in top right corner and go downwards until # of BLACK_PIXEL_BUFFER is found
    black_pixel_count = 0
    bottom_right_y = None

    for y in range(height):
        pixel = frame[y, width-1]
        if pixel_is_black(pixel):
            black_pixel_count += 1
        if black_pixel_count >= BLACK_PIXEL_BUFFER:
            bottom_right_y = y - BLACK_PIXEL_BUFFER
            break
        
    RATIO = 1.25173
    bottom_right_x = bottom_right_y * RATIO
    bottom_right_x = round(bottom_right_x)
    bottom_right_x = width - bottom_right_x
      
    return bottom_right_x, bottom_right_y, width

'''
Crop and covert a single video to frames.

Arguments:
- folder_path: Path to folder containing the videos to process
- output_path: Path to folder where the frames will be saved
- frames_per_second: How many frames to be captured each second
- num_frames: Total amount of frames to capture before stopping
- crop: Crop the video to only include the larynx area
'''
def video_to_frames(video_path, output_path, frames_per_second=None, num_frames=None, crop=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(CAP_PROP_FPS)
    capture_frequency = 1
    if frames_per_second is not None:
      capture_frequency = int(round(fps * frames_per_second, 0))

    index = 0       
    has_calculated_crop_props = False
    while cap.isOpened() and (num_frames is None or index < (capture_frequency * num_frames)):
        Ret, Mat = cap.read()
        if Ret:
            if index % capture_frequency == 0:
                if not has_calculated_crop_props:
                  crop_x, crop_y, width = get_crop_points(Mat)
                  has_calculated_crop_props = True
                output = Mat[0:crop_y, crop_x:width] if crop else Mat
                cv2.imwrite(output_path + '/' + output_path.split('/')[-1] + '_' + str(index) + '.png', output)
            index += 1
        else:
            break
    cap.release()
    return

'''
Crop and covert multiple videos to frames.
Arguments:
- folder_path: Path to folder containing the videos to process
- output_path: Path to folder where the frames will be saved
- frames_per_second: How many frames to be stored for every second of video
- num_frames: Total amount of frames to capture before stopping
- file_extension: File extension of the videos to process
- crop: Crop the video to only include the larynx area
'''
def multiple_videos_to_frames(folder_path, output_path, frames_per_second=None, num_frames=None, file_extension='mp4', crop=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    videos = list(filter(lambda x: x.endswith(file_extension), os.listdir(folder_path)))
    n_videos = len(videos)
    print(f'Number of videos: {n_videos}')
    counter = 1
    
    for video in videos:
        print(f"\nProcessing video ({counter}/{n_videos}) '{video}'...")
        start = time.perf_counter()
        video_id = video.split('_')[0]
        video_to_frames(folder_path + '/' + video, output_path + '/' + video_id, frames_per_second, num_frames, crop)
        end = time.perf_counter()
        diff = end - start
        minutes, seconds = int(diff // 60), int(diff % 60)
        print(f'Processing fininshed ({str(minutes).zfill(2)}:{str(seconds).zfill(2)})')
        counter += 1

# Example usage
src_path = '/example/path/to/source_videos'
dst_path = '/example/path/to/destination_frames'
multiple_videos_to_frames(f'{src_path}', f'{dst_path}', frames_per_second=2, file_extension='mpeg', num_frames=None, crop=True)
