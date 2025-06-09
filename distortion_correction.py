import cv2
import numpy as np
import torch
from PIL import Image
# import exifread

class Insta360InspProcessor:
    def __init__(self, device='cuda'):
        # 硬件加速配置
        self.use_gpu = torch.cuda.is_available() and device == 'cuda'
        self.device = torch.device(device if self.use_gpu else 'cpu')
        
        # Insta360 X4 默认相机参数（需要根据实际校准结果调整）
        # self.camera_params = {
        #     'fx': 1520.23,  # 焦距x
        #     'fy': 1520.86,  # 焦距y
        #     'cx': 1472,   # 光心x
        #     'cy': 1472,    # 光心y
        #     'k1': -0.1085,  # 径向畸变系数1
        #     'k2': 0.1034,   # 径向畸变系数2
        #     'p1': 0.0,      # 切向畸变系数1
        #     'p2': 0.0       # 切向畸变系数2
        # }
        self.camera_params = {
            'fx': 254.87597174247458,
            'fy': 254.06164868269013,
            'cx': 476.66470100570314,
            'cy': 482.477893158099,
            'k1': 0.08,
            'k2': -0.02,
            'p1': 0.0,
            'p2': 0.0,
        }
        # 初始化映射缓存
        self.undistortion_maps = {}
        
    def load_insp_image(self, file_path):
        """加载并解析Insta360 .insp图像文件"""
        with open(file_path, 'rb') as f:
            # 解析EXIF元数据
            # tags = exifread.process_file(f)
            
            # 从EXIF获取原始图像数据
            f.seek(0)
            img = Image.open(f)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            print(img.shape)
            # exit()

            # 自动检测拼接方式（水平或垂直）
            if img.shape[1] / img.shape[0] > 2:  # 水平拼接
                self.split_mode = 'horizontal'
            else:  # 垂直拼接
                self.split_mode = 'vertical'

            self.split_mode = 'horizontal'
            print(self.split_mode)
            return img

    def create_undistort_maps(self, height, width):
        """创建去畸变映射"""
        K = np.array([
            [self.camera_params['fx'], 0, self.camera_params['cx']],
            [0, self.camera_params['fy'], self.camera_params['cy']],
            [0, 0, 1]
        ])
        D = np.array([
            self.camera_params['k1'],
            self.camera_params['k2'],
            self.camera_params['p1'],
            self.camera_params['p2']
        ])
        
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (width, height), cv2.CV_32FC1)
        
        if self.use_gpu:
            map1_tensor = torch.from_numpy(map1).to(self.device)
            map2_tensor = torch.from_numpy(map2).to(self.device)
            map1_normalized = 2.0 * map1_tensor / (width - 1) - 1.0
            map2_normalized = 2.0 * map2_tensor / (height - 1) - 1.0
            return torch.stack([map1_normalized, map2_normalized], dim=-1).unsqueeze(0)
        return map1, map2

    def undistort(self, image):
        """执行去畸变校正"""
        h, w = image.shape[:2]
        
        if self.use_gpu:
            grid = self.undistortion_maps.get((h, w)) or self.create_undistort_maps(h, w)
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
            undistorted = torch.nn.functional.grid_sample(
                img_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            return undistorted.squeeze().permute(1, 2, 0).byte().cpu().numpy()
        else:
            map1, map2 = self.undistortion_maps.get((h, w)) or self.create_undistort_maps(h, w)
            return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

    def process_image(self, img):
        """处理完整流程"""
        # 分割双鱼眼图像

        img = cv2.resize(img, (1920, 960))

        if self.split_mode == 'horizontal':
            mid = img.shape[1] // 2
            # print(img.shape,mid)
            front = img[:, :mid]
            back = img[:, mid:]
        else:
            mid = img.shape[0] // 2
            front = img[:mid, :]
            back = img[mid:, :]
        
        # 去畸变处理
        front_undistorted = self.undistort(front)
        back_undistorted = self.undistort(back)

        front_distorted = front
        back_distorted = back
        return front_undistorted, back_undistorted, front_distorted, back_distorted

if __name__ == "__main__":
    processor = Insta360InspProcessor()
    
    # 加载图像文件
    input_path = "IMG_20201026_154628_00_069.insp"
    try:
        dual_image = processor.load_insp_image(input_path)
    except Exception as e:
        print(f"无法读取文件: {str(e)}")
        exit()
    
    # 处理图像
    front, back, front_distorted, back_distorted = processor.process_image(dual_image)
    
    # 保存结果
    cv2.imwrite("front_undistorted.jpg", front)
    cv2.imwrite("back_undistorted.jpg", back)

    cv2.imwrite("front_distorted.jpg", front_distorted)
    cv2.imwrite("back_distorted.jpg", back_distorted)
    # 显示结果
    # cv2.imshow('Front Camera', front)
    # cv2.imshow('Back Camera', back)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
