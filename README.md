# 进度
已完成使用opencv库调用摄像头提取人脸，使用dlib库完成人脸数据特征向量的提取，使用opencv库对人脸数据进行实时标定，检测者可通过眨眼、张嘴、点头、摇头四种方式进行验活，通过验活后将人脸与已有的人脸信息进行检测比较进行人脸认证。

项目功能已基本实现。


# 使用方法
1. 执行【save_features.py】进行照片录入：右键run。在控制台输入根据提示输入照片对应的姓名，按回车；然后注意弹出的camera视频小框，摆好姿势，按【P】字母键进行截图，图片会自动保存在【data】文件夹的【user_pic】目录下。
3. 执行【camera_get_face.py】进行人脸识别：右键run。注意弹出的camera视频小框。有眨眼、张嘴、点头、摇头四种验活机制，当四种检测机制有三种及以上成功时，即可验活成功。按【q】字母键可退出验活。


# 人脸识别阈值可自行调整
我们的阈值在这里设为了400，可根据需要自行修改。在【camera_get_face.py】的以下代码处修改：
`if simi_value < 400:`
