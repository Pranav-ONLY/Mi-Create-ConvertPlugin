#shendkarpranav0@proton.me
import os
import subprocess
import logging
import traceback
from utils.project import FprjProject, GMFProject
from PyQt6.QtWidgets import QPushButton, QLabel, QProgressBar, QComboBox, QFileDialog, QCheckBox, QHBoxLayout, QRadioButton
from PyQt6.QtCore import Qt
from plugin_api import PluginAPI

from PILS import Image, ImageDraw
import numpy as np
import cv2
import onnxruntime as ort


plugin_dir = os.path.dirname(os.path.realpath(__file__))
resource_dir = os.path.join(plugin_dir, "resources")


class Plugin:
    def __init__(self):
        self.api = PluginAPI()
        self.main_window = self.api.get_main_window()
        self.settings = self.api.main_window.pluginLoader.settings
        self.session = None
        self.convert_models = {# model name: ((screen resolution width, height), (preview image size width, height), device type, corner radius)
                        "Mi Color2/S1/S2": ((466, 466), (246,246), 3, 233),
                        "Mi Watch S1 pro": ((480, 480), (280, 280), 4, 240),
                        "Redmi Watch 3": ((390, 450), (234, 270), 7, 86),
                        "Redmi Watch 3 Active": ((240, 280), (156, 182), 12, 55),
                        "Redmi Watch 5 Active":((320, 385), (320, 385), 3651, 82),
                        "Redmi Watch 5 Lite":((410, 502), (410, 502), 3652, 116),
                        "Redmi band pro": ((194, 368), (110, 208), 8, 28),
                        "Mi band 8": ((192, 490), (122, 310), 9, 280),
                        "Mi band 9": ((192, 490), (122, 310), 366, 280),
                        "Mi band 10": ((212, 520), (134, 328), 466, 104),
                        "Mi band 9 pro": ((336, 480), (230, 328), 367, 48),
                        "Mi band 8 pro": ((336, 480), (230, 328), 11, 48),
                        "Mi band 7 pro": ((280, 456), (220, 358), 6, 48),
                        "Mi watch S3": ((466, 466), (326, 326), 362, 233),
                        "Mi watch S4": ((464, 464), (326, 326), 462, 233),
                        "Redmi Watch 4": ((390, 450), (234, 270), 365, 90),
                        "Redmi Watch 5": ((432, 514), (432, 514), 465, 103)
                        }
        

    def load_up_model(self):
        if self.session is None:
            onnx_model_path = os.path.join(resource_dir, f'models/realesrgan_dynamic.onnx')
            self.session = ort.InferenceSession(onnx_model_path)  

    def to_8bit(self, img_path):
        if not os.path.exists(img_path):
            return
        pngquant_path = os.path.join(resource_dir, 'pngquant/pngquant.exe')
        img_path = os.path.normpath(img_path)
        pngquant_path = os.path.normpath(pngquant_path)
        temp_output_path = img_path + '.temp'
        print([pngquant_path, "--nofs", "--strip", "--ext=.png", "--force", img_path])
        if self.qualityComboBox.currentText() == 'max':
            subprocess.run([pngquant_path, "--verbose", "--nofs", "--strip", "-o", temp_output_path, "--force", img_path], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            eightBitQuality = f'--quality={self.qualityComboBox.currentText()}'  # 80-90, 70-80, 60-70, 50-60
            print([pngquant_path, "--verbose", "--nofs", "--strip", "-o", temp_output_path, "--force", eightBitQuality, img_path])
            subprocess.run([pngquant_path, "--verbose", "--nofs", "--strip", "-o", temp_output_path, "--force", eightBitQuality, img_path], check=True, creationflags=subprocess.CREATE_NO_WINDOW)  
        os.replace(temp_output_path, img_path)

    def compressImage(self, imagePath):
        if not os.path.exists(imagePath):
            return
        oxipng_path = os.path.join(resource_dir, 'oxipng/oxipng.exe')
        imagePath = os.path.normpath(imagePath)
        oxipng_path = os.path.normpath(oxipng_path)
        print([oxipng_path, "--alpha", "--strip", "all", imagePath])
        subprocess.run([oxipng_path, "--alpha", "--strip", "all", imagePath], check=True, creationflags=subprocess.CREATE_NO_WINDOW)

    def inpaint_transparent_edges(self, img_bgra):
        bgr = img_bgra[:, :, :3]
        alpha = img_bgra[:, :, 3]
        # Create mask of transparent areas
        mask = (alpha == 0).astype(np.uint8) * 255
        # Inpaint only the RGB channels
        inpainted = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
        # Merge back with original alpha
        result = cv2.merge((inpainted, alpha))
        return result


    def resizeImageU(self, imagePath, newWidth, newHeight):   #realesrgan
        fullPath = os.path.join(self.projectImagepath, imagePath)
        if not os.path.exists(fullPath):
            return
        img = cv2.imread(fullPath, cv2.IMREAD_UNCHANGED)
        if img is None:
            try:
                pil_image = Image.open(fullPath).convert('RGBA')
                img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
            except Exception as e:
                print(f"Error loading {imagePath} with Pillow: {str(e)}")
                return
        # has_alpha = img.shape[2] == 4
        has_alpha = img.shape[2] == 4 if len(img.shape) == 3 else False
        if has_alpha:
            img = self.inpaint_transparent_edges(img)
        if has_alpha:
            bgr = img[:, :, :3]  # Extract BGR channels
            alpha = img[:, :, 3]
        else:
            bgr = img
            alpha = None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        input_tensor = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0) 

        outputs = self.session.run(None, {'input': input_tensor})

        output_tensor = outputs[0][0]  # Remove batch dimension
        output_tensor = np.transpose(output_tensor, (1, 2, 0))  # CHW to HWC
        output_tensor = np.clip(output_tensor * 255.0, 0, 255).astype(np.uint8)

        output = cv2.resize(output_tensor, (newWidth, newHeight), interpolation=cv2.INTER_AREA)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        if has_alpha:
            alpha_resized = cv2.resize(alpha, (newWidth, newHeight), interpolation=cv2.INTER_LANCZOS4)
            output_bgra = cv2.merge((output_bgr, alpha_resized))
            cv2.imwrite(fullPath, output_bgra)
        else:
            cv2.imwrite(fullPath, output_bgr)
        if self.is_8bit:
            self.to_8bit(fullPath)
        if self.compressPNG:
            self.compressImage(fullPath)

    def resizeImageD(self, imagePath, newWidth, newHeight):   #open-cv
        fullPath = os.path.join(self.projectImagepath, imagePath)
        if os.path.exists(fullPath):
            img = cv2.imread(fullPath, cv2.IMREAD_UNCHANGED)
            if img is None:
                try:
                    pil_image = Image.open(fullPath).convert('RGBA')
                    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
                except Exception as e:
                    print(f"Error loading {imagePath} with Pillow: {str(e)}")
                    return
            final = cv2.resize(img, (newWidth, newHeight), interpolation=cv2.INTER_AREA)
            final = cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA)
            Image.fromarray(final).save(fullPath)
            if self.is_8bit:
                self.to_8bit(fullPath)
            if self.compressPNG:
                self.compressImage(fullPath)

    def alternate_bg(self, imagePath, newWidth, newHeight): #generate alternate background image 
        fullPath = os.path.join(self.projectImagepath, imagePath)
        if not os.path.exists(fullPath):
            return
        img = cv2.imread(fullPath, cv2.IMREAD_UNCHANGED)
        if img is None:
            try:
                pil_image = Image.open(fullPath).convert('RGBA')
                img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
            except Exception as e:
                print(f"Error loading {imagePath} with Pillow: {str(e)}")
                return        
        bgr = self.inpaint_transparent_edges(img)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        input_tensor = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0) 

        outputs = self.session.run(None, {'input': input_tensor})

        output_tensor = outputs[0][0]  # Remove batch dimension
        output_tensor = np.transpose(output_tensor, (1, 2, 0))  # CHW to HWC
        output_tensor = np.clip(output_tensor * 255.0, 0, 255).astype(np.uint8)

        output = cv2.resize(output_tensor, (newWidth, newHeight), interpolation=cv2.INTER_AREA)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        alternate_bg_name = f"{fullPath.split(".")[0]}_x2.png"
        cv2.imwrite(alternate_bg_name, output_bgr)
        if self.is_8bit:
            self.to_8bit(alternate_bg_name)
        if self.compressPNG:
            self.compressImage(alternate_bg_name)
        return alternate_bg_name

    def cutBGcorner(self, imagePath):
        fullPath = os.path.join(self.projectImagepath, imagePath)
        if os.path.exists(fullPath):
            with Image.open(fullPath) as img:
                large_mask = Image.new("L", ((img.width * 4), (img.height * 4)), 0)
                draw = ImageDraw.Draw(large_mask)
                draw.rounded_rectangle((0, 0, (img.width * 4), (img.height * 4)), radius=self.Trad * 4, fill=255)
                mask = large_mask.resize(img.size, Image.LANCZOS)
                rounded_img = Image.new("RGBA", img.size)
                rounded_img.paste(img, mask=mask)
                rounded_img.save(fullPath)
            if self.is_8bit:
                self.to_8bit(fullPath)
            if self.compressPNG:
                self.compressImage(fullPath)
    
    # def reusedImage(self, imagePath, newWidth, newHeight):
    #     fullPath = os.path.join(self.projectImagepath, imagePath)
    #     if os.path.exists(fullPath):
    #         with Image.open(fullPath) as img:
    #             return img.size == (newWidth, newHeight)
    #     return False  

    def projectDataConversion(self, project):
        self.projectImagepath = project['imageFolder']
        self.from_model = self.fromModelComboBox.currentText()
        self.to_model = self.toModelComboBox.currentText()
        if self.from_model == self.to_model:
            print("Invalid Conversion, Please select different models to convert between.")
            return
        print(f"Converting from {self.from_model} to {self.to_model}")

        # getting screen resolution
        self.from_res_x, self.from_res_y = self.convert_models[self.from_model][0]
        self.to_res_x, self.to_res_y = self.convert_models[self.to_model][0]
        
        # screen resolution for preview image
        self.px, self.py = self.convert_models[self.to_model][1]

        # device type
        self.dtF = self.convert_models[self.from_model][2]
        self.dtT = self.convert_models[self.to_model][2]

        # for checking if deviceTo reuires 8bit images
        self.is_8bit = True if str(self.dtT) in ['12', '3651', '3652'] or self.eightBitCheckBox.isChecked() else False
        self.compressPNG = True if self.compressCheckBox.isChecked() else False

        # get corner radius
        self.Trad = self.convert_models[self.to_model][3]

        # dividing screen res to get size difference
        self.x_factor = float(self.to_res_x) / float(self.from_res_x)
        self.y_factor = float(self.to_res_y) / float(self.from_res_y)
        self.mean = float((self.x_factor + self.y_factor)/2)
        self.scale_factor = max(self.x_factor, self.y_factor)
        print(self.x_factor, self.y_factor, self.dtF, self.dtT)
        self.image_resized_map = {}  # to store resized images to avoid resizing same image again and again

        # Update DeviceType project['data']['FaceProject']['@DeviceType']
        if str(project['data']['FaceProject']['@DeviceType']) == str(self.dtF):
            project['data']['FaceProject']['@DeviceType'] = str(self.dtT)
        else:
            self.main_window.showDialog("error", _(f"Failed to Convert project: Please select {self.from_model} Project"))
            return
        if str(project['data']['FaceProject']['Screen']['@Bitmap']) != '':
            self.resizeImage(project['data']['FaceProject']['Screen']['@Bitmap'], int(self.px), int(self.py))
            self.cutBGcorner(project['data']['FaceProject']['Screen']['@Bitmap'])

        
        # Process each Widget
        widgets = project['widgets']
        if not isinstance(widgets, list):
            widgets = [widgets]  # Ensure widgets is a list

        self.progressBar.setMaximum(len(widgets))
        self.progressBar.setValue(0)
        # for RW3A
        if str(self.dtT) == "12":
            self.RW3A(widgets, project)
        
        for i, widget in enumerate(widgets):
            # Update X and Y coordinates
            widget['@X'] = str(round(float(widget['@X']) * self.x_factor))
            widget['@Y'] = str(round(float(widget['@Y']) * self.y_factor))

            widget['@Width'] = str(round(float(widget['@Width']) * self.x_factor))
            widget['@Height'] = str(round(float(widget['@Height']) * self.y_factor))
                        
             # for small progressbar
            if i == 0:
                if str(widget['@Shape']) == '30':
                    if int(widget['@Width']) == self.to_res_x and int(widget['@Height']) == self.to_res_y:
                        self.load_up_model()
                        alternate_BG_x2 = self.alternate_bg(widget['@Bitmap'], int(widget['@Width']), int(widget['@Height']))
                        self.cutBGcorner(alternate_BG_x2)
                        # widget['@Bitmap'] = alternate_BG_x2.split("\\")[-1]  # Update the bitmap path to the new image XGZXFY - og - Copy (8)\images\BG_x2.png
                        # continue
             
            # Update Width and Height
            # if '@Digits' in widget:
            #     # increase_width_by_spacing = (int(widget['@Digits'])-1) * int(widget['@Spacing'])
            #     firstImagePath = widget['@BitmapList'].split('|')[0]
            #     mainPath = os.path.join(self.projectImagepath, firstImagePath)
            #     if os.path.exists(mainPath):
            #         with Image.open(mainPath) as img:
            #             digit_width = img.width
            #             digit_height = img.height
            #             spacing = int(widget.get('@Spacing', 0))
            #             digits = int(widget['@Digits'])
            #             total_width = (digit_width * digits) + (spacing * (digits - 1))
            #             widget['@Width'] = str(round(total_width * self.x_factor))
            #             widget['@Height'] = str(round(digit_height * self.y_factor))
            # else:
            #     widget['@Width'] = str(round(float(widget['@Width']) * self.x_factor))
            #     widget['@Height'] = str(round(float(widget['@Height']) * self.y_factor))
            

            # Resize images
            if '@Bitmap' in widget:
                fullPath = os.path.join(self.projectImagepath, widget['@Bitmap'])
                if os.path.exists(fullPath):
                    with Image.open(fullPath) as img:
                        img_w, img_h = img.size
                        widget['@Width'] = str(round(img_w * self.x_factor))
                        widget['@Height'] = str(round(img_h * self.y_factor))
                # if not self.reusedImage(widget['@Bitmap'], int(widget['@Width']), int(widget['@Height'])):
                if not self.image_resized_map.get(widget['@Bitmap'], False):
                    self.resizeImage(widget['@Bitmap'], int(widget['@Width']), int(widget['@Height']))
                    self.image_resized_map[widget['@Bitmap']] = True  # Mark this image as resized
                    if int(widget['@Width']) == self.to_res_x and int(widget['@Height']) == self.to_res_y:
                        self.cutBGcorner(widget['@Bitmap'])

            elif '@BitmapList' in widget:
                for bitmap in widget['@BitmapList'].split('|'):
                    if bitmap == "":
                        break
                    bitmaplist_imagepath = bitmap.split(':')[1] if ':' in bitmap else bitmap
                    fullPath = os.path.join(self.projectImagepath, bitmaplist_imagepath)
                    if os.path.exists(fullPath):
                        with Image.open(fullPath) as img:
                            img_w, img_h = img.size
                            widget['@Width'] = str(round(img_w * self.x_factor))
                            widget['@Height'] = str(round(img_h * self.y_factor))
                    # if not self.reusedImage(bitmaplist_imagepath, int(widget['@Width']), int(widget['@Height'])):
                    if not self.image_resized_map.get(bitmaplist_imagepath, False):
                        # Resize the image
                        self.resizeImage(bitmaplist_imagepath, int(widget['@Width']), int(widget['@Height']))
                        self.image_resized_map[bitmaplist_imagepath] = True  # Mark this image as resized
            
            elif widget['@Shape'] == '42':
                widget['@Rotate_xc'] = round(int(widget['@Rotate_xc']) * self.mean)
                widget['@Rotate_yc'] = round(int(widget['@Rotate_yc']) * self.mean)
                widget['@Radius'] = round(int(widget['@Radius']) * self.mean)
                widget['@Line_Width'] = round(int(widget['@Line_Width']) * self.mean)

                if widget['@Background_ImageName'] != "":
                    img_path =  os.path.join(self.projectImagepath, widget['@Background_ImageName'])
                    if os.path.exists(img_path):
                        with Image.open(img_path) as imgCbg:
                            newWidthCbg = round(float(imgCbg.width) * self.mean)
                            newHeightCbg = round(float(imgCbg.height) * self.mean)
                        self.resizeImage(widget['@Background_ImageName'], newWidthCbg, newHeightCbg)

                if widget['@Foreground_ImageName'] != "":
                    img_path =  os.path.join(self.projectImagepath, widget['@Foreground_ImageName'])
                    if os.path.exists(img_path):
                        with Image.open(img_path) as imgC:
                            newWidthC = round(float(imgC.width) * self.mean)
                            newHeightC = round(float(imgC.height) * self.mean)
                        self.resizeImage(widget['@Foreground_ImageName'], newWidthC, newHeightC)
                
            elif '@HourHand_ImageName' in widget and str(widget['@HourHand_ImageName']) != '':
                img_path =  os.path.join(self.projectImagepath, widget['@HourHand_ImageName'])
                if os.path.exists(img_path):
                    with Image.open(img_path) as imgH:
                        newWidthH = round(float(imgH.width) * self.mean)
                        newHeightH = round(float(imgH.height) * self.mean)
                    self.resizeImage(widget['@HourHand_ImageName'], newWidthH, newHeightH)

                widget['@HourImage_rotate_xc'] = round((float(widget['@HourImage_rotate_xc'])*self.mean))
                widget['@HourImage_rotate_yc'] = round((float(widget['@HourImage_rotate_yc'])*self.mean))

            if '@MinuteHand_Image' in widget and str(widget['@MinuteHand_Image']) != '':
                img_path =  os.path.join(self.projectImagepath, widget['@MinuteHand_Image'])
                if os.path.exists(img_path):
                    with Image.open(img_path) as imgM:
                        newWidthM = round(float(imgM.width) * self.mean)
                        newHeightM = round(float(imgM.height) * self.mean)
                    self.resizeImage(widget['@MinuteHand_Image'], newWidthM, newHeightM)

                widget['@MinuteImage_rotate_xc'] = round((float(widget['@MinuteImage_rotate_xc'])*self.mean))
                widget['@MinuteImage_rotate_yc'] = round((float(widget['@MinuteImage_rotate_yc'])*self.mean))

            if '@SecondHand_Image' in widget and str(widget['@SecondHand_Image']) != '':
                img_path =  os.path.join(self.projectImagepath, widget['@SecondHand_Image'])
                if os.path.exists(img_path):
                    with Image.open(img_path) as imgS:
                        newWidthS = round(float(imgS.width) * self.mean)
                        newHeightS = round(float(imgS.height) * self.mean)
                    self.resizeImage(widget['@SecondHand_Image'], newWidthS, newHeightS)
                widget['@SecondImage_rotate_xc'] = round((float(widget['@SecondImage_rotate_xc'])*self.mean))
                widget['@SecondImage_rotate_yc'] = round((float(widget['@SecondImage_rotate_yc'])*self.mean))
            
            self.progressBar.setValue(i + 1)# Update progress bar

    def resizeImage(self, imagePath, newWidth, newHeight):
          # Use the larger scale to determine upscaling or downscaling
        if self.scale_factor > 1:
            self.load_up_model()
            self.resizeImageU(imagePath, newWidth, newHeight)  # Upscaling
        else:
            self.resizeImageD(imagePath, newWidth, newHeight) # Downscaling
    
    def RW3A(self, widgets, project):
        if str(widgets[0]['@Shape']) != '30':
            first_img = Image.new("RGB", (1, 1), (0, 0, 0))
            first_img_path = os.path.join(self.projectImagepath, "/first_bg.png")
            first_img.save(first_img_path)
            first_img_widget = {'@Shape': '30', '@Name': 'background', '@X': '0', '@Y': '0', '@Width': '1', '@Height': '1', '@Alpha': '255', '@Visible_Src': '0', '@Bitmap': 'first_bg.png'}
            widgets.insert(0, first_img_widget)
        elif str(widgets[0]['@Shape']) == '30':
            widgets[0]['@Name'] = 'background'
        data_src ={'811':('hour', '42'),
                '911':('hourLow', '39'),
                '1000911':('hourHigh', '38'),
                '1011':('minute', '43'),
                '1111':('minLow', '3C'),
                '1211':('minHigh', '3B'),
                '1811':('second', '44'),
                '1911':('secLow', '00'),
                '1001911':('secHigh', '00'),
                '1812':('day', '0F'),
                '1912':('dayLow', '00'),
                '1001912':('dayHigh', '00'),
                '2012':('week', '11'),
                '1012':('month', '0D'),
                '812':('year', '00'),
                '813':('isAM', '14'),
                '1013':('isPM', '15'),
                '3031':('weatherIcon', '00'),
                '841':('batt', '09'),
                '1841':('sleepStatus', '00'),
                '2041':('btStatus', '00'),
                '3041':('lockStatus', '00'),
                '822':('hrm', '31'),
                '1022':('intHrm', '00'),
                '821':('steps', '1C'),
                '1021':('stepsPercent', '18'),
                '823':('calories', '27'),
                '1023':('calPercent', '23'),
                '824':('stand', '00'),
                '826':('stress', '00'),
                '5031':('weatherSomething', '00'),
                '828':('sleepScrore', '00')}
        
        for i, wid in enumerate(widgets[1:]): # for watch 3 active color checking and naming
            if str(wid['@Shape']) == '31':
                first_Image_Path = wid['@BitmapList'].split('|')[0].split(':')[1]
                mainPath = os.path.join(self.projectImagepath, first_Image_Path)
                if os.path.exists(mainPath):
                    with Image.open(mainPath) as f_img:
                        if f_img.mode == "RGB":
                            r, g, b = f_img.resize((1, 1)).getpixel((0, 0))
                        elif f_img.mode == "RGBA":
                            r, g, b, _ = f_img.resize((1, 1)).getpixel((0, 0))
                        hex_rgb = f"{r:02x}{g:02x}{b:02x}"  
                if wid['@Index_Src'] == '0A11':
                    wid['@Index_Src'] = '1000911'

                if int(wid['@Index_Src']) == '841':
                    wid['@Name'] = f'04_imgList_{data_src[str(int(wid['@Index_Src']))][0]}_color[{hex_rgb}]'
                elif str(int(wid['@Index_Src'])) in data_src:
                    wid['@Name'] = f'{data_src[str(int(wid['@Index_Src']))][1]}_imgList_{data_src[str(int(wid['@Index_Src']))][0]}_color[{hex_rgb}]'
                else:
                    wid['@Name'] = f'0{i}_imglist_color[{hex_rgb}]'
            elif str(wid['@Shape']) == '32':
                first_Image_Path = wid['@BitmapList'].split('|')[0]
                mainPath = os.path.join(self.projectImagepath, first_Image_Path)
                if os.path.exists(mainPath):
                    with Image.open(mainPath) as f_img:
                        if f_img.mode == "RGB":
                            r, g, b = f_img.resize((1, 1)).getpixel((0, 0))
                        elif f_img.mode == "RGBA":
                            r, g, b, _ = f_img.resize((1, 1)).getpixel((0, 0))
                        hex_rgb = f"{r:02x}{g:02x}{b:02x}" 
                if wid['@Value_Src'] == '0A11':
                    wid['@Value_Src'] = '1000911'
                if str(int(wid['@Value_Src'])) in data_src:
                    wid['@Name'] = f'{data_src[str(int(wid['@Value_Src']))][1]}_num_{data_src[str(int(wid['@Value_Src']))][0]}_color[{hex_rgb}]'
                else:
                    wid['@Name'] = f'0{i}_num_color[{hex_rgb}]'
            elif str(wid['@Shape']) == '30':
                first_Image_Path = wid['@Bitmap']
                mainPath = os.path.join(self.projectImagepath, first_Image_Path)
                if os.path.exists(mainPath):
                    with Image.open(mainPath) as f_img:
                        if f_img.mode == "RGB":
                            r, g, b = f_img.resize((1, 1)).getpixel((0, 0))
                        elif f_img.mode == "RGBA":
                            r, g, b, a = f_img.resize((1, 1)).getpixel((0, 0))
                        hex_rgb = f"{r:02x}{g:02x}{b:02x}" 
                wid['@Name'] = f'0{i}_img_color[{hex_rgb}]'
            else:
                wid['@Name'] = f'0{i}_none'

    def conversion(self, event=None, projectLocation=None):
        print("Conversion Started")
        # Get where to open the project from
        if projectLocation == None:
            projectLocation = QFileDialog.getOpenFileName(self.main_window, _('Convert Project...'), "%userprofile%/", "Watchface Project (*.fprj wfDef.json)")

        if not isinstance(projectLocation, str):
            projectLocation = projectLocation[0].replace("\\", "/")

        if os.path.isfile(projectLocation):
            extension = os.path.splitext(projectLocation)[1]
            if extension == '.fprj':
                project = FprjProject()
            elif extension == ".json":
                self.showDialog("warning", "GMFProjects are experimental! Bugs may occur.")
                project = GMFProject()
            else:
                self.showDialog("error", "Invalid project!")
                return False
        else:
            # no file was selected
            logging.debug(f"openProject failed to open project {projectLocation}: isfile failed!")
            return False

        load = project.load(projectLocation)
        if project.themes['aod']['imageFolder'] != "":
            self.projectDataConversion(project.themes['aod'])
        self.projectDataConversion(project.themes['default'])
        if load[0]:
            try:
                self.main_window.createNewWorkspace(project)
                recentProjectList = self.settings.value("recentProjects")

                if recentProjectList == None:
                    recentProjectList = []

                path = os.path.normpath(projectLocation)

                if isinstance(project, FprjProject):
                    projectListing = [os.path.basename(path), path]
                elif isinstance(project, GMFProject):
                    projectListing = [project.getTitle(), path]

                if projectListing in recentProjectList:
                    recentProjectList.pop(recentProjectList.index(projectListing))

                recentProjectList.append(projectListing)

                self.settings.setValue("recentProjects", recentProjectList)
            except Exception as e:
                self.main_window.showDialog("error", _("Failed to open project: ") + str(e), traceback.format_exc())
                return False
        else:
            self.main_window.showDialog("error", _('Cannot open project: ') + load[1], load[2])
            return False
        self.progressBar.setValue(0)
        self.main_window.saveProjects("current")



    def projectCompression(self, project):
        if self.eightBitCheckBox.isChecked() or self.compressCheckBox.isChecked():
            print("Compression Started")
        else:
            print("No Compression Selected")
            self.main_window.showDialog("error", _("Failed to Compress project: Please select at least one compression option."))
            return

        self.projectImagepath = project['imageFolder']
        self.from_model = self.fromModelComboBox.currentText()
        self.to_model = self.toModelComboBox.currentText()
        self.is_8bit = True if self.eightBitCheckBox.isChecked() else False
        self.compressPNG = True if self.compressCheckBox.isChecked() else False
        self.image_compressed_map = {} # to store compressed images to avoid compressing same image again and again
        
        #compress preview image
        if str(project['data']['FaceProject']['Screen']['@Bitmap']) != '':
            preview_image = os.path.join(self.projectImagepath, str(project['data']['FaceProject']['Screen']['@Bitmap']))
            if self.is_8bit:
                self.to_8bit(preview_image)
            if self.compressPNG:
                self.compressImage(preview_image)

        # Process each Widget
        widgets = project['widgets']
        if not isinstance(widgets, list):
            widgets = [widgets]  # Ensure widgets is a list
        self.progressBar.setMaximum(len(widgets))
        self.progressBar.setValue(0)  # Reset progress bar

        for i, widget in enumerate(widgets):
            if '@Bitmap' in widget:
                fullPath = os.path.join(self.projectImagepath, widget['@Bitmap'])
                if os.path.exists(fullPath):
                    if not self.image_compressed_map.get(widget['@Bitmap'], False):
                        if self.is_8bit:
                            self.to_8bit(fullPath)
                        if self.compressPNG:
                            self.compressImage(fullPath)
                        self.image_compressed_map[widget['@Bitmap']] = True
            elif '@BitmapList' in widget:
                for bitmap in widget['@BitmapList'].split('|'):
                    if bitmap == "":
                        break
                    bitmaplist_imagepath = bitmap.split(':')[1] if ':' in bitmap else bitmap
                    fullPath = os.path.join(self.projectImagepath, bitmaplist_imagepath)
                    if os.path.exists(fullPath):
                        if not self.image_compressed_map.get(bitmaplist_imagepath, False):
                            if self.is_8bit:
                                self.to_8bit(fullPath)
                            if self.compressPNG:
                                self.compressImage(fullPath)
                            self.image_compressed_map[bitmaplist_imagepath] = True

            elif widget['@Shape'] == '42':
                if widget['@Background_ImageName'] != "":
                    img_path =  os.path.join(self.projectImagepath, widget['@Background_ImageName'])
                    if os.path.exists(img_path):
                        if not self.image_compressed_map.get(widget['@Background_ImageName'], False):
                            if self.is_8bit:
                                self.to_8bit(img_path)
                            if self.compressPNG:
                                self.compressImage(img_path)
                            self.image_compressed_map[widget['@Background_ImageName']] = True

                if widget['@Foreground_ImageName'] != "":
                    img_path =  os.path.join(self.projectImagepath, widget['@Foreground_ImageName'])
                    if os.path.exists(img_path):
                        if not self.image_compressed_map.get(widget['@Foreground_ImageName'], False):
                            if self.is_8bit:
                                self.to_8bit(img_path)
                            if self.compressPNG:
                                self.compressImage(img_path)
                            self.image_compressed_map[widget['@Foreground_ImageName']] = True

            elif '@HourHand_ImageName' in widget and str(widget['@HourHand_ImageName']) != '':
                img_path =  os.path.join(self.projectImagepath, widget['@HourHand_ImageName'])
                if os.path.exists(img_path):
                    if not self.image_compressed_map.get(widget['@HourHand_ImageName'], False):
                        if self.is_8bit:
                            self.to_8bit(img_path)
                        if self.compressPNG:
                            self.compressImage(img_path)
                        self.image_compressed_map[widget['@HourHand_ImageName']] = True
            if '@MinuteHand_Image' in widget and str(widget['@MinuteHand_Image']) != '':
                img_path =  os.path.join(self.projectImagepath, widget['@MinuteHand_Image'])
                if os.path.exists(img_path):
                    if not self.image_compressed_map.get(widget['@MinuteHand_Image'], False):
                        if self.is_8bit:
                            self.to_8bit(img_path)
                        if self.compressPNG:
                            self.compressImage(img_path)
                        self.image_compressed_map[widget['@MinuteHand_Image']] = True
            if '@SecondHand_Image' in widget and str(widget['@SecondHand_Image']) != '':
                img_path =  os.path.join(self.projectImagepath, widget['@SecondHand_Image'])
                if os.path.exists(img_path):
                    if not self.image_compressed_map.get(widget['@SecondHand_Image'], False):
                        if self.is_8bit:
                            self.to_8bit(img_path)
                        if self.compressPNG:
                            self.compressImage(img_path)
                        self.image_compressed_map[widget['@SecondHand_Image']] = True
            self.progressBar.setValue(i + 1)# Update progress bar

    def compression(self, event=None, projectLocation=None):
        print("Compression Started")
        # Get where to open the project from
        if projectLocation == None:
            projectLocation = QFileDialog.getOpenFileName(self.main_window, _('Compress Project...'), "%userprofile%/", "Watchface Project (*.fprj wfDef.json)")

        if not isinstance(projectLocation, str):
            projectLocation = projectLocation[0].replace("\\", "/")

        if os.path.isfile(projectLocation):
            extension = os.path.splitext(projectLocation)[1]
            if extension == '.fprj':
                project = FprjProject()
            elif extension == ".json":
                self.showDialog("warning", "GMFProjects are experimental! Bugs may occur.")
                project = GMFProject()
            else:
                self.showDialog("error", "Invalid project!")
                return False
        else:
            # no file was selected
            logging.debug(f"openProject failed to open project {projectLocation}: isfile failed!")
            return False

        load = project.load(projectLocation)
        if project.themes['aod']['imageFolder'] != "":
            self.projectCompression(project.themes['aod'])
        self.projectCompression(project.themes['default'])
        if load[0]:
            try:
                self.main_window.createNewWorkspace(project)
                recentProjectList = self.settings.value("recentProjects")

                if recentProjectList == None:
                    recentProjectList = []

                path = os.path.normpath(projectLocation)

                if isinstance(project, FprjProject):
                    projectListing = [os.path.basename(path), path]
                elif isinstance(project, GMFProject):
                    projectListing = [project.getTitle(), path]

                if projectListing in recentProjectList:
                    recentProjectList.pop(recentProjectList.index(projectListing))

                recentProjectList.append(projectListing)

                self.settings.setValue("recentProjects", recentProjectList)
            except Exception as e:
                self.main_window.showDialog("error", _("Failed to open project: ") + str(e), traceback.format_exc())
                return False
        else:
            self.main_window.showDialog("error", _('Cannot open project: ') + load[1], load[2])
            return False
        self.progressBar.setValue(0)
        self.main_window.saveProjects("current")


    def register(self):
        # Function is called upon plugin initialization
        # This function is only called when the plugin is not disabled
        print("register")
        # # For head Lable
        # self.headLabel = QLabel("Convert :", self.main_window.coreDialog)
        # self.headLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # self.main_window.coreDialog.welcomeSidebarLayout.addWidget(self.headLabel)

        #For creating separate section between convert and compress
        self.radio_btn_convert = QRadioButton("Convert", self.main_window.coreDialog)
        self.radio_btn_convert.setChecked(True)  # Set Convert as default
        self.radio_btn_convert.setToolTip("Convert between different watch models")
        self.main_window.coreDialog.welcomeSidebarLayout.addWidget(self.radio_btn_convert)
        self.radio_btn_compress = QRadioButton("Compress", self.main_window.coreDialog)
        self.radio_btn_compress.setToolTip("Compress images to reduce Project size")
        self.main_window.coreDialog.welcomeSidebarLayout.addWidget(self.radio_btn_compress)
        # Connect radio buttons to toggle between convert and compress
        self.radio_btn_convert.toggled.connect(self.toggleConvertCompress)
        self.radio_btn_compress.toggled.connect(self.toggleConvertCompress)

        # for fromModelComboBox selectList 
        self.fromModelComboBox = QComboBox(self.main_window.coreDialog)
        self.fromModelComboBox.addItems(self.convert_models.keys())
        self.fromModelComboBox.setToolTip("Select the model you want to convert FROM")
        self.main_window.coreDialog.welcomeSidebarLayout.addWidget(self.fromModelComboBox)
        # for middle arrow pointing down
        self.arrowLabel = QLabel("To ↓", self.main_window.coreDialog)
        self.arrowLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_window.coreDialog.welcomeSidebarLayout.addWidget(self.arrowLabel)
        # for toModelComboBox selectList
        self.toModelComboBox = QComboBox(self.main_window.coreDialog)
        self.toModelComboBox.addItems(self.convert_models.keys())
        self.toModelComboBox.setToolTip("Select the model you want to convert TO")
        self.main_window.coreDialog.welcomeSidebarLayout.addWidget(self.toModelComboBox)
        # for choosing different models in both Lists for conversion as indicator
        self.fromModelComboBox.setCurrentIndex(0)
        self.toModelComboBox.setCurrentIndex(1)

        # for 8-bit color quality list with horizontal alignment
        bitColorLayout = QHBoxLayout()
        
        # for 8-bit color checkbox
        self.eightBitCheckBox = QCheckBox("8-bit color", self.main_window.coreDialog)   
        self.eightBitCheckBox.setToolTip("Select to convert to images to 8-bit color depth which also help reduces file size")
        # self.main_window.coreDialog.welcomeSidebarLayout.addWidget(self.eightBitCheckBox)
        bitColorLayout.addWidget(self.eightBitCheckBox)
        # for quality selection combobox
        self.qualityComboBox = QComboBox(self.main_window.coreDialog)
        self.qualityComboBox.addItems(['max', '80-90', '70-80', '60-70', '50-60'])
        self.qualityComboBox.setToolTip("Select 8-bit color Image quality \n max= orignal quality \n lower the quality = smaller the size")
        self.qualityComboBox.setEnabled(False)  # Disabled by default
        bitColorLayout.addWidget(self.qualityComboBox)
        # for adding 8-bit color layout
        self.main_window.coreDialog.welcomeSidebarLayout.addLayout(bitColorLayout)
        # Connect checkbox state change to enable/disable quality selection
        self.eightBitCheckBox.stateChanged.connect(self.toggleQualitySelection)
        # for compress checkbox
        self.compressCheckBox = QCheckBox("Optimize PNG (lossless)", self.main_window.coreDialog)   
        self.compressCheckBox.setToolTip("Select to compress PNG images without losing quality")
        self.main_window.coreDialog.welcomeSidebarLayout.addWidget(self.compressCheckBox)
        # for convertButton to start the conversion
        self.convertButton = QPushButton("Convert", self.main_window.coreDialog)
        self.convertButton.setToolTip("Click to start the conversion")
        self.main_window.coreDialog.welcomeSidebarLayout.addWidget(self.convertButton)
        self.convertButton.clicked.connect(self.conversion)

        # for compressButton to start the compression
        self.compressButton = QPushButton("Compress", self.main_window.coreDialog)
        self.compressButton.setToolTip("Click to start the compression")
        self.main_window.coreDialog.welcomeSidebarLayout.addWidget(self.compressButton)
        self.compressButton.clicked.connect(self.compression)
        self.compressButton.hide()  # Hide the compress button for now, as compression is not implemented

        # Connect the convertButton and compressButton to their respective functions
        self.compressCheckBox.clicked.connect(self.updateButtonStatus)  # Update button state when checkbox is clicked
        self.eightBitCheckBox.clicked.connect(self.updateButtonStatus)  # Update button state when checkbox is clicked   

        # for little progress bar
        self.progressBar = QProgressBar(self.main_window.coreDialog)
        self.progressBar.setMaximum(100)
        self.main_window.coreDialog.welcomeSidebarLayout.addWidget(self.progressBar)
# to update the toModelComboBox options based on the fromModelComboBox so that user does not select same model for conversion   
        self.fromModelComboBox.currentIndexChanged.connect(self.updateToModelOptions)
        self.toModelComboBox.currentIndexChanged.connect(self.updateCheckboxState)

# to update the toModelComboBox options based on the fromModelComboBox so that user does not select same model for conversion
    def updateToModelOptions(self):
        selected_model = self.fromModelComboBox.currentText()
        self.toModelComboBox.clear()
        self.toModelComboBox.addItems([model for model in self.convert_models if model != selected_model])

    def toggleConvertCompress(self):
        if self.radio_btn_convert.isChecked():
            self.updateCheckboxState()
            self.convertButton.show()
            self.compressButton.hide()
            self.arrowLabel.show()
            self.fromModelComboBox.show()
            self.toModelComboBox.show()
            self.compressCheckBox.setChecked(False)
            self.eightBitCheckBox.setChecked(False)
        elif self.radio_btn_compress.isChecked():
            self.updateCheckboxState()
            self.convertButton.hide()
            self.compressButton.show()
            self.arrowLabel.hide()
            self.toModelComboBox.hide()
            self.fromModelComboBox.hide()
            self.compressCheckBox.setChecked(True)
            self.eightBitCheckBox.setChecked(True)
            self.compressButton.setDisabled(False)

    #     # Enable or disable the compress button based on the checkboxes
    def updateButtonStatus(self):
        if not (self.compressCheckBox.isChecked() or self.eightBitCheckBox.isChecked()):
            self.compressButton.setDisabled(True)
        else:
            self.compressButton.setDisabled(False)
        
    def updateCheckboxState(self):
        selected_model = self.toModelComboBox.currentText()
        if selected_model in ['Redmi Watch 3 Active', 'Redmi Watch 5 Active', 'Redmi Watch 5 Lite'] and self.radio_btn_convert.isChecked():
            self.eightBitCheckBox.setChecked(True)
            self.eightBitCheckBox.setEnabled(False)
        else:
            self.eightBitCheckBox.setEnabled(True)
    def toggleQualitySelection(self, state):
        # print(state)
        self.qualityComboBox.setEnabled(True if state == 2 else False)

    def unregister(self):
        # Function called upon disabling a plugin
        # This function only calls when the user disables the plugin
        print("unregister")
        # self.main_window.coreDialog.welcomeSidebarLayout.removeWidget(self.headLabel)
        self.main_window.coreDialog.welcomeSidebarLayout.removeWidget(self.radio_btn_convert)
        self.main_window.coreDialog.welcomeSidebarLayout.removeWidget(self.radio_btn_compress)
        self.main_window.coreDialog.welcomeSidebarLayout.removeWidget(self.fromModelComboBox)
        self.main_window.coreDialog.welcomeSidebarLayout.removeWidget(self.arrowLabel)
        self.main_window.coreDialog.welcomeSidebarLayout.removeWidget(self.toModelComboBox)
        self.main_window.coreDialog.welcomeSidebarLayout.removeWidget(self.eightBitCheckBox)
        self.main_window.coreDialog.welcomeSidebarLayout.removeWidget(self.qualityComboBox)
        self.main_window.coreDialog.welcomeSidebarLayout.removeWidget(self.compressCheckBox)
        self.main_window.coreDialog.welcomeSidebarLayout.removeWidget(self.convertButton)
        self.main_window.coreDialog.welcomeSidebarLayout.removeWidget(self.compressButton)
        self.main_window.coreDialog.welcomeSidebarLayout.removeWidget(self.progressBar)
        # self.headLabel.deleteLater()
        self.radio_btn_convert.deleteLater()
        self.radio_btn_compress.deleteLater()
        self.fromModelComboBox.deleteLater()
        self.arrowLabel.deleteLater()
        self.toModelComboBox.deleteLater()
        self.eightBitCheckBox.deleteLater()
        self.qualityComboBox.deleteLater()
        self.compressCheckBox.deleteLater()
        self.convertButton.deleteLater()
        self.compressButton.deleteLater()
        self.progressBar.deleteLater()