import cv2
import os
import math

def createFrame(cam, dest_dir):
    try:

        # creating a folder named data
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0
    list_frames = []

    while (True):
        # reading from frame
        ret, frame = cam.read()
        if ret:
            list_frames.append(frame)
        else:
            break
    num_of_frames = 64
    N = len(list_frames)
    interval = (N-1)//(num_of_frames-1)
    if interval == 0:
        os.rmdir(dest_dir)
        return
    for i in range(0, N, interval):
        frame = list_frames[i]
        name = './{}/frame'.format(dest_dir) + str(i//interval) + '.jpg'
        print('Creating...' + name)
        # writing the extracted images
        cv2.imwrite(name, frame)
        if i//interval==num_of_frames-1:
            break

# Release all space and windows once done
if __name__ == '__main__':
    src = "RWF-2000/train/Fight/"
    dest_root = "RWF-2000-C/train/Fight/"
    dest_sub_folder = "Video_"
    src_videos = os.listdir(src)

    for i, src_video in enumerate(src_videos):
        src_path = src + str(src_video)
        cam = cv2.VideoCapture(src_path)
        dest_path = dest_root + dest_sub_folder + str(i) + "/"
        createFrame(cam, dest_path)
        cam.release()
        cv2.destroyAllWindows()