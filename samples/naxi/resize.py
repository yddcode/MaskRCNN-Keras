import os
import cv2
 
def resize_img(DATADIR, data_k, img_size):
    w = img_size[0]
    h = img_size[1]
    
    '''设置目标像素大小，此处设为300'''
    path = os.path.join(DATADIR, data_k)
    #返回path路径下所有文件的名字，以及文件夹的名字，
    img_list = os.listdir(path)
 
    for i in img_list:
        if i.endswith('.jpg'):
            # 调用cv2.imread读入图片，读入格式为IMREAD_COLOR
            img_array = cv2.imread((path + '/' + i), cv2.IMREAD_COLOR)
            # 调用cv2.resize函数resize图片
            # INTER_NEAREST         最近邻插值
            # INTER_LINEAR     双线性插值（默认设置） 将根据源图像附近的4 个(2X2范围)邻近像素的线性加权计算得出，权重由这4个像素到精确目标点的距离决定。
            # INTER_AREA       使用像素区域关系进行重采样。用新的像素点覆盖原来的像素点，然后求取覆盖区域的平均值，这种插值算法称为区域插值
            # INTER_CUBIC       4x4像素邻域的双三次插值 首先对源图像附近的4x4个邻近像素进行三次样条拟合，然后将目标像素对应的三次样条值作为目标图像对应像素点的值
            # INTER_LANCZOS4     8x8像素邻域的Lanczos插值
            (x, y)= img_array.shape[:2]
            aio = w / x
            y_ture = aio * x     #按图片比例缩放
            y_ture = int(y_ture)
            # out = im.resize((x_s, y_s), Image.ANTIALIAS)
            new_array = cv2.resize(img_array, (w, y_ture), interpolation=cv2.INTER_CUBIC)
            img_name = str(i)
            '''生成图片存储的目标路径'''
            save_path = path + '_new/'
            if os.path.exists(save_path):
                print(i)
                '''调用cv.2的imwrite函数保存图片'''
                save_img=save_path+img_name
                cv2.imwrite(save_img, new_array)
            else:
                os.mkdir(save_path)
                save_img = save_path + img_name
                cv2.imwrite(save_img, new_array)
 
 
if __name__ == '__main__':
    #设置图片路径
    DATADIR = "C:/Users/37455/Pictures/"
    data_k = 'naxi_jie'
    # data_k = 'naxi_doc'
    #需要修改的新的尺寸
    img_size = [512, 512]
    resize_img(DATADIR, data_k, img_size)