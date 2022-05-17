import lxml.etree as etree
import numpy as np
import cv2
TYPE_CONTOUR =  0   # 原始格式：[pt1_xy, pt2_xy, pt3_xy, ...]
TYPE_BOX =      1   # 原始格式：[pt_lt, pt_rt, pt_rb, pt_rl]
TYPE_ELLIPSE =  2   # 原始格式：[[x1, y1], [x2, y2]]
TYPE_ARROW =    3   # 原始格式：[hear_xy, tail_xy]
file = '/media/zhaobingchao/M008/nanfang_hospital/2021-9-28/2021-9-28/000003463/PA1700006_HE/1700006.xml'
svs_file = '/media/zhaobingchao/M008/nanfang_hospital/2021-9-28/2021-9-28/000003463/PA1700006_HE/1700006.svs'
def color_int_to_tuple(color_int):
    '''
    将RGB颜色元组转换为颜色整数
    :param color_int:
    :return:
    '''
    color_str = hex(color_int)[2:]
    assert len(color_str) <= 6, 'Found unknow color!'
    pad_count = 6 - len(color_str)
    color_str = ''.join(['0'] * pad_count) + color_str
    b, g, r = int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16)
    return r, g, b
    
class ImageScopeXmlReader:
    def __init__(self, file=None, keep_arrow_tail=False, use_box_y1x1y2x2=True):
        '''

        :param file:            读取文件路径
        :param keep_arrow_tail: 读取箭头标签时是否保留箭头的尾部
        :param use_box_y1x1y2x2:读取方盒标签时是否使用y1x1y2x2坐标，若设为False则使用[左上，右上，右下，左下]坐标
        '''
        self.keep_arrow_tail = keep_arrow_tail
        self.use_box_y1x1y2x2 = use_box_y1x1y2x2
        self.contour_color_regs = {}
        self.box_color_regs = {}
        self.arrow_color_regs = {}
        self.ellipse_color_regs = {}
        if file is not None:
            self.read(file)

    def read(self, file,color=65280):
        tree = etree.parse(file)
        for ann in tree.findall('./Annotation'):
            color_int = int(ann.attrib['LineColor'])
            color_tuple = color_int_to_tuple(color_int)
            if color>0 and color_int!=color:
                print("Color {} not equal to {} continue".format(color_int,color))
                continue
            for region in ann.findall('./Regions/Region'):
                reg_type = int(region.attrib['Type'])
                #print(reg_type)
                if reg_type == TYPE_ARROW:
                    # 读取箭头标签
                    self.arrow_color_regs.setdefault(color_tuple, [])
                    arrow_head_tail_points = []
                    for vertex in region.findall('./Vertices/Vertex'):
                        x = int(float(vertex.attrib['X']))
                        y = int(float(vertex.attrib['Y']))
                        arrow_head_tail_points.append((y, x))
                    arrow_points = np.asarray(arrow_head_tail_points, np.int)
                    if not self.keep_arrow_tail:
                        arrow_points = arrow_points[0]
                    self.arrow_color_regs[color_tuple].append(arrow_points)

                elif reg_type == TYPE_BOX:
                    # 读取盒状标签
                    self.box_color_regs.setdefault(color_tuple, [])
                    box_points = []
                    for vertex in region.findall('./Vertices/Vertex'):
                        x = int(float(vertex.attrib['X']))
                        y = int(float(vertex.attrib['Y']))
                        box_points.append((y, x))
                    box_points = np.asarray(box_points, np.int)
                    if self.use_box_y1x1y2x2:
                        y1, x1 = box_points[0]
                        y2, x2 = box_points[2]
                        box_points = np.array([y1, x1, y2, x2])
                    self.box_color_regs[color_tuple].append(box_points)

                elif reg_type == TYPE_CONTOUR:
                    # 读取轮廓标签
                   # print('get contour')
                    self.contour_color_regs.setdefault(color_tuple, [])
                    contours = []
                    #print(region.findall('./Vertices'))
                    for Coordinate in region.findall('./Vertices/Vertex'):
                        #print(Coordinate)
                        x = int(float(Coordinate.attrib['X']))
                        y = int(float(Coordinate.attrib['Y']))
                        contours.append((y, x))
                    contours = np.asarray(contours, np.int)
                    self.contour_color_regs[color_tuple].append(contours)

                elif reg_type == TYPE_ELLIPSE:
                    # 读取椭圆标签
                    self.ellipse_color_regs.setdefault(color_tuple, [])
                    ellipse = []
                    for vertex in region.findall('./Vertices/Vertex'):
                        x = int(float(vertex.attrib['X']))
                        y = int(float(vertex.attrib['Y']))
                        ellipse.append((y, x))
                    ellipse = np.asarray(ellipse, np.int)
                    self.ellipse_color_regs[color_tuple].append(ellipse)

                else:
                    print('Unknow type {}. Will be skip.'.format(reg_type))

    def get_contours(self):
        contours, colors = [], []
        for color in self.contour_color_regs:
            contours.extend(self.contour_color_regs[color])
            colors.extend([color]*len(self.contour_color_regs[color]))
        return self.contour_color_regs
        return contours, colors

    def get_boxes(self):
        boxes, colors = [], []
        for color in self.box_color_regs:
            boxes.extend(self.box_color_regs[color])
            colors.extend([color]*len(self.box_color_regs[color]))
        return boxes, colors

    def get_arrows(self):
        arrows, colors = [], []
        for color in self.arrow_color_regs:
            arrows.extend(self.arrow_color_regs[color])
            colors.extend([color]*len(self.arrow_color_regs[color]))
        return arrows, colors

    def get_ellipses(self):
        ellipses, colors = [], []
        for color in self.ellipse_color_regs:
            ellipses.extend(self.ellipse_color_regs[color])
            colors.extend([color]*len(self.ellipse_color_regs[color]))
        return ellipses, colors

def get_roi_from_color(ret_mask, rois):
    for i in rois:
        _temp = []
        for ii in i:
            _temp.append([ii[1],ii[0]])
        mask    = np.array(_temp,dtype=np.int)
        ret_mask = cv2.fillPoly(ret_mask, [mask], (255))



# a = ur.ImageScopeXmlReader()
# a.read(anno[0])
# b = a.get_contours()


# slide = openslide.OpenSlide('/media/zhaobingchao/My Book/glioma/data/raw_data/南方医/已切/000030908/PA1702155/1702155_001.svs')
# height1, width1     = slide.level_dimensions[0]
# mask  = np.zeros(( width1,height1), np.uint8)
# ur.get_roi_from_color(mask,b[(0,255,0)])
# mask = cv2.resize(mask, (mask.shape[1]//4, mask.shape[0]//4))