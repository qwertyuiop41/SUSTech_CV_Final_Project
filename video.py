import cv2


def split_video_to_frames(video_path, output_folder):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 确定帧计数器的初始值
    frame_count = 0

    # 逐帧读取视频，并保存为图像
    while True:
        # 读取一帧
        ret, frame = video.read()

        # 检查是否成功读取帧
        if not ret:
            break

        # 构建输出图像文件名
        output_file = f'{output_folder}/frame_{frame_count}.jpg'

        # 保存帧为图像文件
        cv2.imwrite(output_file, frame)

        # 更新帧计数器
        frame_count += 1

    # 关闭视频文件
    video.release()


# 示例用法
video_path = 'path/to/video.mp4'  # 替换为视频文件的路径
output_folder = 'path/to/output/folder'  # 替换为保存图像的文件夹路径

# 调用函数进行视频分割
split_video_to_frames(video_path, output_folder)
