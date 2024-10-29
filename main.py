from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor,QCursor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, Qt
from ui.main_window import Ui_MainWindow
from ui.pop.pop_box import MessageBox
from ui.ui_function import *
from ui.toast.toast import DialogOver
from ui.dialog.rtsp_win import Window
from ui.dialog.id_win import id_Window
from utils.main_utils import check_url, check_path
from utils.AtestCamera import Camera
from classes.yolo import YoloPredictor
from classes.main_config import MainConfig
from classes.car_chart import WorkerThread
from PIL import Image
import numpy as np
import supervision as sv
import subprocess
import sys
import cv2
import os
import datetime
from ultralytics import YOLO
from realtime_predict import screen_predict
class MainWindow(QMainWindow, Ui_MainWindow):
    # 主窗口向yolo实例发送执行信号
    main2yolo_begin_sgl = Signal()

    def __init__(self, parent=None):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # 背景设置为半透明 & 无边框窗口
        self.setAttribute(Qt.WA_TranslucentBackground) 
        self.setWindowFlags(Qt.FramelessWindowHint)

        # UI动作（动效）
        UIFuncitons.uiDefinitions(self)

        # 变量设置
        self.smoke_threshold = 0 # 烟火阈值
        self.web_flag = True # 可以开启
        self.server_process = None # 服务器进程
        self.image_id = 0 # 图片 id
        self.txt_id = 0 # 标签 id

        # 设置阴影
        # UIFuncitons.shadow_style(self, self.Class_QF, QColor(162,129,247)) # 第一个类别计数窗口
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))

        # 设置 为 -- （好看）
        # self.Class_num.setText('--') # 第一个类别计数窗口的文本设置
        self.Target_num.setText('--')
        self.fps_label.setText('--')

        # 计时器：每2秒监视模型文件更改一次
        self.Qtimer_ModelBox = QTimer(self)
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)


        self.yolo_init()  # 实例化YOLO
        self.model_bind() # 预测参数-数值变动绑定 主页面
        # 主要功能绑定
        self.main_function_bind()
        self.load_config() # 配置加载
        self.model_load()  # 模型加载

        # 屏幕实时检测画图线程
        self.show_status('欢迎使用烟雾火焰智能监测系统!')

    # 实例化YOLO
    def yolo_init(self):
        # Yolo-v8 thread
        self.yolo_predict = YoloPredictor()                           #实例化yolo检测
        self.select_model = self.model_box.currentText()
        self.yolo_thread = QThread()

        # 显示预测视频（左，右）
        self.yolo_predict.yolo2main_trail_img.connect(lambda x: self.show_image(x, self.pre_video))
        self.yolo_predict.yolo2main_box_img.connect(lambda x: self.show_image(x, self.res_video))

        # 输出信息、FPS、类数、总数
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))
        # self.yolo_predict.yolo2main_class_num.connect(lambda x:self.Class_num.setText(str(x)))
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))

        # 移动线程里面去（通过main2yolo_begin_sgl信号控制-先要开启yolo_thread线程，才能启动yolo_predict的run方法）
        self.yolo_predict.moveToThread(self.yolo_thread)
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)

        # 显示总车流量
        self.yolo_predict.yolo2main_target_num.connect(lambda x:self.Target_setText(x))

    # 模型加载
    def model_load(self):
        # 创建模型文件夹
        check_path(self.config.models_path)
        self.model_box.clear()
        self.pt_list = os.listdir(f'./{self.config.models_path}')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt') or file.endswith('.engine')]
        self.pt_list.sort(key=lambda x: os.path.getsize(f'./{self.config.models_path}/' + x))   #按文件大小排序
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)

    # 主页面各功能绑定
    def main_function_bind(self):
        self.src_file_button.clicked.connect(self.open_src_file) # 打开文件夹
        self.src_cam_button.clicked.connect(self.camera_select) # 摄像头
        self.src_rtsp_button.clicked.connect(self.screen_predict) # RTPS改为屏幕实时检测
        self.src_graph_button.clicked.connect(self.open_det_file) # 流量图
        self.src_lock_button.clicked.connect(self.open_det_txt_file) # 单目追踪

        self.run_button.clicked.connect(self.run_or_continue) # 开始
        self.stop_button.clicked.connect(self.stop) # 终止

        self.save_res_button.toggled.connect(self.is_save_res) # 是否 保存 视频
        self.save_txt_button.toggled.connect(self.is_save_txt) # 是否 保存 标签
        self.show_labels_checkbox.toggled.connect(self.is_show_labels)  # 是否 显示标签

        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))

    # 模型参数绑定
    def model_bind(self):
        self.model_box.currentTextChanged.connect(self.change_model)
        self.iou_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'iou_spinbox'))    # iou box
        self.iou_slider.valueChanged.connect(lambda x:self.change_val(x, 'iou_slider'))      # iou scroll bar

        self.conf_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x:self.change_val(x, 'conf_slider'))    # conf scroll bar

        self.speed_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'speed_spinbox'))# speed box
        self.speed_slider.valueChanged.connect(lambda x:self.change_val(x, 'speed_slider'))  # speed scroll bar

    # JSON配置文件初始化
    def load_config(self):
        self.config = MainConfig("D:/postgraduate0/YHJC/Fire_smoke_monitoring_system-master/config/config.json")

        self.save_res_button.setChecked(self.config.save_res)
        self.save_txt_button.setChecked(self.config.save_txt)

        self.iou_slider.setValue(self.config.iou * 100)
        self.conf_slider.setValue(self.config.conf * 100)
        self.speed_slider.setValue(self.config.rate)

        self.yolo_predict.save_txt = self.config.save_txt # 保存标签
        self.yolo_predict.save_res = self.config.save_res # 保存结果
        self.yolo_predict.save_txt_path = self.config.save_txt_path
        self.yolo_predict.save_res_path = self.config.save_res_path
        self.yolo_predict.new_model_name = f"./{self.config.models_path}/%s" % self.select_model

        self.yolo_predict.show_trace = self.config.show_trace  # 轨迹
        # self.show_trace_checkbox.setChecked(self.config.show_trace) # 轨迹
        self.yolo_predict.show_labels = self.config.show_labels  # 标签
        self.show_labels_checkbox.setChecked(self.config.show_labels) # 标签

        self.open_fold = self.config.open_fold
        self.car_id = self.config.car_id

        self.run_button.setChecked(False)

    # 如果超出了烟火阈值，那么就变红！！
    def Target_setText(self, num):
        num = str(num)
        self.Target_num.setText(num)

        if (int(num) >= int(self.smoke_threshold)):
            self.char_label.setStyleSheet("color: red;")
            self.char_label.setText(
                f"发现烟火！发现烟火！")
        else:
            self.char_label.setStyleSheet("color: green;")
            self.char_label.setText(f"暂无风险")


    #主窗口显示轨迹图像和检测图像 （缩放在这里）
    @staticmethod
    def show_image(img_src, label):
        try:
            # 检查图像的通道数，确定图像是否为彩色图像
            if len(img_src.shape) == 3:
                ih, iw, _ = img_src.shape
            if len(img_src.shape) == 2:
                ih, iw = img_src.shape

            # 根据标签窗口的大小调整图像的大小
            w = label.geometry().width()
            h = label.geometry().height()

            # 根据图像宽高比例进行缩放
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            # 将OpenCV图像从BGR格式转换为RGB格式，并创建QImage对象
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)

            # 在标签窗口中显示图像
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    #控制开始|暂停
    def run_or_continue(self):
        # 检测是否有模型
        if self.yolo_predict.new_model_name == '' or self.yolo_predict.new_model_name == None:
            DialogOver(parent=self, text="请检测模型", title="运行失败", flags="danger")
            self.run_button.setChecked(False)
            return
        # 检测输入源
        if self.yolo_predict.source == '' or self.yolo_predict.source == None:
            self.show_status('请在检测前选择输入源...')
            self.run_button.setChecked(False)
            DialogOver(parent=self, text="请检测输入源", title="运行失败", flags="danger")
            return

        self.yolo_predict.stop_dtc = False # 线程开始

        # 开始
        if self.run_button.isChecked():
            # 图片预测
            file_extension = self.yolo_predict.source[-3:].lower()
            if file_extension == 'png' or file_extension == 'jpg':
                self.img_predict()
                return

            # 视频预测
            DialogOver(parent=self, text="开始检测...", title="运行成功", flags="success")
            self.run_button.setChecked(True)

            # 不可再改变设置（config动态调整 关闭）
            self.save_txt_button.setEnabled(False)
            self.save_res_button.setEnabled(False)
            self.conf_slider.setEnabled(False)
            self.iou_slider.setEnabled(False)
            self.speed_slider.setEnabled(False)

            self.show_status('检测中...')
            if '0' in self.yolo_predict.source or 'rtsp' in self.yolo_predict.source:
                self.progress_bar.setFormat('实时视频流检测中...')
            if 'avi' in self.yolo_predict.source or 'mp4' in self.yolo_predict.source:
                self.progress_bar.setFormat("当前检测进度:%p%")
            self.yolo_predict.continue_dtc = True
            # 开始检测
            if not self.yolo_thread.isRunning():
                self.yolo_thread.start()
                self.main2yolo_begin_sgl.emit()
        # 暂停
        else:
            # self.draw_thread.pause()  # 折线图暂停
            self.yolo_predict.continue_dtc = False
            self.show_status("暂停...")
            DialogOver(parent=self, text="已暂停检测", title="运行暂停", flags="warning")
            self.run_button.setChecked(False)

    # 显示状态 （底部栏
    def show_status(self, msg):
        self.status_bar.setText(msg)  # 显示输出
        if msg == '检测完成':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            self.progress_bar.setValue(0)
            # 终止yolo线程
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()

        elif msg == '检测终止':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            self.progress_bar.setValue(0)
            # 终止yolo线程
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()

            self.pre_video.clear()          
            self.res_video.clear()
            self.Target_num.setText('--')
            self.fps_label.setText('--')

    # 打开文件
    def open_src_file(self):
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', self.open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.yolo_predict.source = name
            self.show_status('加载文件：{}'.format(os.path.basename(name)))
            self.open_fold = os.path.dirname(name)
            # 终止事件
            self.stop()
            DialogOver(parent=self, text=f"文件路径: {name}", title="加载成功", flags="success")

    # camera选择
    def camera_select(self):
        #try:
            # 关闭YOLO线程
            self.stop()
            #获取本地摄像头数量
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.src_cam_button.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 20px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 212, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(16,155,226,50);
                                            }
                                            ''')
            
            for cam in cams:
                exec("action_%s = QAction('%s 号摄像头')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)
            pos = QCursor.pos()
            action = popMenu.exec(pos)

            # 设置摄像头来源
            if action:
                str_temp = ''
                selected_stream_source = str_temp.join(filter(str.isdigit, action.text())) #获取摄像头号，去除非数字字符
                self.yolo_predict.source = selected_stream_source
                self.show_status(f'摄像头设备:{action.text()}')
                DialogOver(parent=self, text=f"当前摄像头为: {action.text()}", title="摄像头选择成功", flags="success")

    def screen_predict(self):
        # 是否默认窗口置顶
        top_most = True
        # 加载模型
        model = self.yolo_predict.load_yolo_model()
        screen_predict.run(model, top_most)
        # 存储屏幕检测图片
        if self.yolo_predict.save_res:
            check_path(self.config.save_res_path) # 检查保存路径
            for i in screen_predict.image_det:
                # 存在同名文件，自增 self.image_id 直至文件不存在
                while os.path.exists(f"{self.config.save_res_path}/发现烟火{self.image_id}.jpg"):
                    self.image_id += 1
                Image.fromarray(i).save(f"./{self.config.save_res_path}/发现烟火{self.image_id}.jpg")

        # 存储labels里的信息
        if self.yolo_predict.save_txt:
            check_path(self.config.save_txt_path) # 检查保存路径
            # 存在同名文件，自增 self.txt_id 直至文件不存在
            while os.path.exists(f"{self.config.save_txt_path}/result_{self.txt_id}.jpg"):
                self.txt_id += 1

            with open(f'{self.config.save_txt_path}/result_{self.txt_id}.txt', 'a') as f:
                f.write(screen_predict.text_output)

        return

    # 打开检测结果文件
    def open_det_file(self):

        # 定义文件夹路径
        folder_path = r"D:\postgraduate0\YHJC\Fire_smoke_monitoring_system-master\pre_result"

        # 使用操作系统默认的文件管理器打开文件夹
        os.startfile(folder_path)

    def open_det_txt_file(self):

        # 定义文件夹路径
        folder_path_txt = r"D:\postgraduate0\YHJC\Fire_smoke_monitoring_system-master\pre_labels"

        # 使用操作系统默认的文件管理器打开文件夹
        os.startfile(folder_path_txt)

    #保存提示（MP4）
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('提示: 监测结果不会被保存')
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status('提示: 监测结果将会被保存')
            self.yolo_predict.save_res = True
    
    #保存提示（txt）
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('提示: 标签信息不会被保存')
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status('提示: 标签信息将会被保存')
            self.yolo_predict.save_txt = True

    # 是否显示 标签
    def is_show_labels(self):
        if self.show_labels_checkbox.checkState() == Qt.CheckState.Unchecked:
            self.yolo_predict.show_labels = False
            self.show_status('提示: 不再显示标签')
        elif self.show_labels_checkbox.checkState() == Qt.CheckState.Checked:
            self.yolo_predict.show_labels = True
            self.show_status('提示: 显示标签')

    #终止事件（按下终止按钮 or 输入源更换的时候）
    def stop(self):
        try:
            # 摄像头释放
            self.yolo_predict.release_capture()  # 这里是为了终止使用摄像头检测函数的线程，改了yolo源码
            # 结束线程
            self.yolo_thread.quit()

        except:
            pass
        self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)    #恢复按钮状态

        # 终止后才可以修改设置
        self.save_res_button.setEnabled(True)   #把保存按钮设置为可用
        self.save_txt_button.setEnabled(True)   #把保存按钮设置为可用
        self.iou_slider.setEnabled(True)        #把滑块设置为可用
        self.conf_slider.setEnabled(True)       #把滑块设置为可用
        self.speed_slider.setEnabled(True)      #把滑块设置为可用
        self.pre_video.clear()           #清空视频显示
        self.res_video.clear()           #清空视频显示
        self.progress_bar.setValue(0)   #进度条清零
        # self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    #检测参数设置
    def change_val(self, x, flag):
        # 交互比
        if flag == 'iou_spinbox':
            self.iou_slider.setValue(int(x*100))    
        elif flag == 'iou_slider':
            self.iou_spinbox.setValue(x/100) 
            self.show_status('IOU Threshold: %s' % str(x/100))
            self.yolo_predict.iou_thres = x/100
        # 置信度
        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x*100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x/100)
            self.show_status('Conf Threshold: %s' % str(x/100))
            self.yolo_predict.conf_thres = x/100
        # 延时
        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.show_status('Delay: %s ms' % str(x))
            self.yolo_predict.speed_thres = x  # ms

    #模型更换
    def change_model(self,x):
        self.select_model = self.model_box.currentText()
        self.yolo_predict.new_model_name = f"./{self.config.models_path}/%s" % self.select_model
        self.show_status('更改模型：%s' % self.select_model)
        self.Model_name.setText(self.select_model)

    #循环监测文件夹的文件变化
    def ModelBoxRefre(self):
        pt_list = os.listdir(f'./{self.config.models_path}')
        pt_list = [file for file in pt_list if file.endswith('.pt') or file.endswith('.engine')]
        pt_list.sort(key=lambda x: os.path.getsize(f'./{self.config.models_path}/' + x))
        #必须排序后再比较，否则列表会一直刷新
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    #获取鼠标位置（用于按住标题栏拖动窗口）
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    #拖动窗口大小时优化调整
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    # 退出时退出线程，保存设置
    def closeEvent(self, event):
        try:
            self.stop()

            # config.json
            self.config.save_res = self.yolo_predict.save_res
            self.config.save_txt = self.yolo_predict.save_txt

            self.config.show_labels = self.yolo_predict.show_labels
            self.config.show_trace = self.yolo_predict.show_trace

            self.config.iou = self.yolo_predict.iou_thres
            self.config.conf = self.yolo_predict.conf_thres
            self.config.rate = self.yolo_predict.speed_thres

            self.config.car_threshold = self.smoke_threshold   # 警报 阈值
            self.config.car_id = self.car_id
            self.config.open_fold = self.open_fold

            self.config.save_config()  # 保存设置

            # 退出弹窗 1000ms
            MessageBox(
                self.close_button, title='Note', text='退出中，请等待...', time=1000, auto=True).exec()

            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(0)

    # 预测图片
    def img_predict(self):

        if check_url(self.yolo_predict.source):
            DialogOver(parent=self, text="目标路径含有中文！", title="程序取消", flags="danger")
            return

        self.run_button.setChecked(False)  # 按钮
        # 读取照片
        image = cv2.imread(self.yolo_predict.source)
        org_img = image.copy()
        # 加载模型
        model = self.yolo_predict.load_yolo_model()
        # 获取数据源
        iter_model = iter(model.track(source=image, show=False))
        result = next(iter_model)  # 这里是检测的核心，
        # 如果没有目标
        if result.boxes.id is None:
            DialogOver(parent=self, text="该图片中没有要检测的目标哟！", title="运行完成", flags="warning")
            self.show_image(image, self.pre_video)
            self.show_image(image, self.res_video)
            self.yolo_predict.source = ''
            return

        # 如果有目标
        detections = sv.Detections.from_ultralytics(result)
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        # 画标签
        labels_write, img_box = self.yolo_predict.creat_labels(detections, image, model)

        # 显示信息 —— 类别数 & 总数
        self.Target_num.setText(str(len(detections.tracker_id)))
        # 显示图片
        self.show_image(org_img, self.pre_video)  # left
        self.show_image(img_box, self.res_video)  # right
        self.yolo_predict.source = ''
        DialogOver(parent=self, text="图片检测完成", title="运行成功", flags="success")

        # 保存图片
        if self.yolo_predict.save_res:
            check_path(self.config.save_res_path) # 检查保存路径
            # 存在同名文件，自增 self.image_id 直至文件不存在
            while os.path.exists(f"{self.config.save_res_path}/烟火图片检测结果{self.image_id}.jpg"):
                self.image_id += 1
            # 将 BGR 格式的 frame 转换为 RGB 格式
            rgb_frame = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)
            # 把 rgb_frame 转换为 numpy格式 就行了
            numpy_frame = np.array(rgb_frame)
            Image.fromarray(numpy_frame).save(f"./{self.config.save_res_path}/烟火图片检测结果{self.image_id}.jpg")

        # 存储labels里的信息
        if self.yolo_predict.save_txt:
            check_path(self.config.save_txt_path) # 检查保存路径
            # 存在同名文件，自增 self.txt_id 直至文件不存在
            while os.path.exists(f"{self.config.save_txt_path}/result_{self.txt_id}.jpg"):
                self.txt_id += 1

            with open(f'{self.config.save_txt_path}/result_{self.txt_id}.txt', 'a') as f:
                f.write(
                        f'检测时间: {datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}' +
                        f' 烟雾火焰目标总数: {len(detections.tracker_id)}')
                f.write('\n')
        return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())  
