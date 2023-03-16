import sys
import time
import random
import os
import gc
import torch
import glob
import traceback
import numpy as np
last_color  =31
WARNING     = 2
INFO        = 1
FLOW        = 0 
TIPS        = 4
ERROR       = 3
HIGH_LIGHT = 5

COLORS = {
    'BLACK':30, "RED":31, "GREEN":32, "BLUE":34, "WHITE":37,\
        "YELLOW":33, "YANGHONG":35, "CYAN":36
}
COLOR_CONSTRUCT = "\033[0;{}m{}\033[0m"

LEVEL_COLOR = {
    WARNING : "YELLOW",
    INFO    : "WHITE",
    FLOW    : "BLUE",
    HIGH_LIGHT: "CYAN",
    TIPS    : "GREEN",
    ERROR   : "RED"
}

LEVEL_INFO = {
    WARNING : "WARNING",
    INFO    : "INFO",
    FLOW    : "FLOW",
    TIPS    : "TIPS",
    ERROR   : "ERROR",
    HIGH_LIGHT:'HIGH_LIGHT'
}

SMALL_PATCH = 20004

def find_file(path,depth_down,depth_up=0,suffix='.xml'):
    ret = []
    for i in range(depth_up,depth_down):
        _path = os.path.join(path,'*/'*i+'*'+suffix)
        ret.extend(glob.glob(_path))
    ret.sort()
    return ret
    
def free_memory(if_cuda = True):
    gc.collect()
    if if_cuda:
        torch.cuda.empty_cache()
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def rtime_print(str,end='\r'):
    print(f'\033[5;{random.randint(31, 37)}m{str}\033[0m', end=end,flush=True)


def com_str(str,rc=True,sep=' ',last=False):
	global last_color
	if rc:
		if last:
			last_color = last_color
		else:
			last_color = random.randint(31, 37)
		return f'\033[1;{last_color}m{str}{sep}\033[0m'
	else:
		return f'\033[1;36m{str}{sep}\033[0m'


def my_print(*args,rc=True,sep=' ',if_last=False,color = ''):
    if color != "":
        p = [a for a in args]
        p = ' '.join(p)
        p = "\033[0;{}m{}\033[0m".format(COLORS[color], p)
        print(p)
    else:
        for i in range(len(args)-1):
            if i==0:
                print(com_str(args[i],rc,'',last=if_last),end='')
                continue
            print(com_str(args[i],rc,sep,last=if_last),end='')
        print(com_str(args[len(args)-1],rc,sep,last=if_last))

def color_str(*args,color = COLORS['BLACK']):
    if len(args) == 1:
        return COLOR_CONSTRUCT.format(color, args[0])
    return [COLOR_CONSTRUCT.format(color,a) for a in args]


def info(*args, sep=' ', color=""):
    p = sep.join([str(a) for a in args])
    if color == "":
        p = COLOR_CONSTRUCT.\
            format(COLORS[LEVEL_COLOR[INFO]], p)
    else:
        p = COLOR_CONSTRUCT.\
            format(COLORS[color], p)
    print(p)

def tips(*args, sep=' '):
    p = sep.join([str(a) for a in args])
    file        = __file__.split("/")[-1]
    funcName    = sys._getframe().f_back.f_code.co_name  # 获取调用函数名
    lineNumber  = sys._getframe().f_back.f_lineno
    
    info_str, funcName, lineNumber, p = \
        color_str(LEVEL_INFO[TIPS],funcName, lineNumber, p, color = COLORS[LEVEL_COLOR[TIPS]])
    
    p   = "[{}][{}:{}] {}".\
                format(info_str, funcName, lineNumber,  p)
    print(p)

def flow(*args, sep=' '):
    p = sep.join([str(a) for a in args])
    file        = __file__.split("/")[-1]
    funcName    = sys._getframe().f_back.f_code.co_name  # 获取调用函数名
    lineNumber  = sys._getframe().f_back.f_lineno

    info_str, funcName, lineNumber, p = \
        color_str(LEVEL_INFO[FLOW],funcName, lineNumber, p, color = COLORS[LEVEL_COLOR[FLOW]])
    
    p   = "[{}][{}:{}] {}".\
                format(info_str, funcName, lineNumber,  p)
    print(p)

def high(*args, sep=' '):
    p = sep.join([str(a) for a in args])
    file        = __file__.split("/")[-1]
    funcName    = sys._getframe().f_back.f_code.co_name  # 获取调用函数名
    lineNumber  = sys._getframe().f_back.f_lineno

    info_str, file, funcName, lineNumber, p = \
        color_str(LEVEL_INFO[HIGH_LIGHT],file, funcName, lineNumber, p, color = COLORS[LEVEL_COLOR[HIGH_LIGHT]])
    
    p   = "[{}][{}][{}:{}] {}".\
                format(info_str, file, funcName, lineNumber,  p)
    print(p)

def err(*args, sep=' ', exit = True, show_traceback_line=5):
    p = sep.join([str(a) for a in args])
    file        = __file__.split("/")[-1]
    funcName    = sys._getframe().f_back.f_code.co_name  # 获取调用函数名
    lineNumber  = sys._getframe().f_back.f_lineno

    info_str, funcName, lineNumber, p = \
        color_str(LEVEL_INFO[ERROR], funcName, lineNumber, p, color = COLORS[LEVEL_COLOR[ERROR]])
    
    p   = "[{}][{}:{}] {}".\
                format(info_str, funcName, lineNumber,  p)
    print(p)
    fs = traceback.format_stack()
    for line in fs[0-show_traceback_line:len(fs)]:
        p   = color_str(line.strip(), color = COLORS[LEVEL_COLOR[ERROR]])
        print(p)

    if exit:
        sys.exit()

def warn(*args, sep=' '):
    p = sep.join([str(a) for a in args])
    file        = __file__.split("/")[-1]
    funcName    = sys._getframe().f_back.f_code.co_name  # 获取调用函数名
    lineNumber  = sys._getframe().f_back.f_lineno

    info_str, file, funcName, lineNumber, p = \
        color_str(LEVEL_INFO[WARNING],file, funcName, lineNumber, p, color = COLORS[LEVEL_COLOR[WARNING]])
    
    p   = "[{}][{}][{}:{}] {}".\
                format(info_str, file, funcName, lineNumber,  p)
    print(p)

class MyZip(object):
    def __init__(self, *args,batch = 1):
        self.value_len = len(args[0])
        for num, value in enumerate(args):
            if not hasattr(value, "__iter__") and not hasattr(value, "__getitem__"):
                err("The {} value has not attr {} and {}".format(num, "__iter__", "__getitem__"))
                sys.exit()
            elif len(value) != self.value_len:
                err("The zeros arg len ({}) donet equal to the {} arg len ({})".\
                    format(self.value_len, num,len(value)))
                sys.exit()
        self.value = args
        self.batch = batch

    def __len__(self):
        if self.value_len%self.batch > 0 : 
            return self.value_len//self.batch+1
        else: 
            return self.value_len//self.batch

    def __getitem__(self,index):
        if index >= len(self):
            raise StopIteration()
        start = index * self.batch
        end = start + self.batch if start + self.batch <= self.value_len else self.value_len
        ret = []
        for i in self.value:
            ret.append(i[start:end])

        return ret[:]


#判断file 或者floder是否存在
def just_ff(path:str,*,file=False,floder=True,create_floder=False, info=True):
    """
    Check the input path status. Exist or not.

    Args:
        path (str): _description_
        file (bool, optional): _description_. Defaults to False.
        floder (bool, optional): _description_. Defaults to True.
        create_floder (bool, optional): _description_. Defaults to False.
        info (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if file:
        return os.path.isfile(path)
    elif floder:
        if os.path.exists(path):
            return True
        else:
            if create_floder:
                try:
                    os.makedirs(path) 
                    if info:
                        tips(r"Path '{}' does not exists, but created ！！".format(path))
                    return True
                except ValueError:
                    if info:
                        err(r"Path '{}' does not exists, and the creation failed ！！".format(path))
                    pass
            else:
                if info:
                    tips(r"Path '{}' does not exists！！".format(path))
                return False
                

def just_dir_of_file(file_path:str, create_floder:bool=True):
    """_summary_
    Check the dir of the input file. If donot exist, creat it!
    Args:
        file_path (_type_): _description_
        create_floder (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    _dir = os.path.split(file_path)[0]
    return just_ff(_dir, create_floder = create_floder)

def ignore_warning():
    import warnings
    warnings.filterwarnings("ignore")

# Fixed random seed
def seed_enviroment(seed=5,if_torch=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if if_torch:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def show_config(*args, param = {}, bar_color = "CYAN"):
    show_info       = []
    max_len         =   0
    param_record    = {}

    for key, v in param.items():
        if len(key)> max_len: 
            max_len = len(key)
        param_record[key] = 0

    for value in args:
        if isinstance(value, str) or isinstance(value, int) or isinstance(float):
            for key, v in param.items():
                if value == v:
                    if param_record[key] <=0:
                        show_info.append([key, value])
                        param_record[key] +=1
                    break
    info("*"*70, color = bar_color)
    info("*"*70, color = bar_color)
    for _info in show_info:
        _str = "'{}' : '{}'".\
                format(_info[0].rjust(max_len), str(_info[1]))
        print(_str)
    info("*"*70, color = bar_color)
    info("*"*70, color = bar_color)

def convert_time_to_str(time):
    #时间数字转化成字符串，不够10的前面补个0
    if (time < 10):
        time = '0' + str(time)
    else:
        time=str(time)
    return time

def sec_to_data(y):

    h=int(y//3600 % 24)
    d = int(y // 86400)
    m =int((y % 3600) // 60)
    s = round(y % 60,2)
    
    d = '' if d<=0 else '{}D : '.format(convert_time_to_str(d))
    h = '' if (d=='') and (h<=0) else '{}H : '.format(convert_time_to_str(h))
    m = '' if (h=='') and (m<=0) else '{}M : '.format(convert_time_to_str(m))
    s = '{}S'.format(convert_time_to_str(s))
    
    #10d : 10h : 10m : 10s
    return ''.join([d, h, m, s])

class progress_predictor(object):
    def __init__(self, total_num:int, record:bool=False):
        self.start_time = time.time()
        self.remain_num = total_num
        self.handle_start_time = time.time()
        self.last_time = time.time()
        self.smooth = 1e-10
        self.total_num = total_num
        self.handle_num = 0
        self.pred_time = [0]
        self.record = record
        self.num_sec = 0.
        self.sec_num = 0.
        self.UNIT = {'S':1, 'M':60, 'H':3600, 'D':86400}

    def __call__(self, stride=1, record_time=True):
        if record_time:
            self.handle_num += stride
            self.remain_num -= stride
            self.last_time = time.time()
            cost_time = self.last_time - self.handle_start_time
            self.num_sec = cost_time/self.handle_num
            self.sec_num = self.handle_num/cost_time
            pred_time = self.remain_num*self.num_sec

            if self.record:
                self.pred_time.append(pred_time)
            else:
                self.pred_time[0] = pred_time
        else:
            self.remain_num -= stride
            if self.handle_num == 0:
                self.handle_start_time = time.time()

    def predict(self):
        return sec_to_data(self.pred_time[-1])
    
    def crosting_time(self):
        return sec_to_data(time.time()-self.start_time)

    def rate(self, unit='S'):
        assert isinstance(unit, str), 'unit must be:(s,m,h,d or S,M,H,D)'
        unit = unit.upper()
        return '{:.2f}'.format(self.sec_num*self.UNIT[unit])
        
    def remaining(self):
        return self.remain_num
        
    def has_handle(self):
        return self.total_num - self.remain_num

    def last_cost(self):
        return sec_to_data(time.time()-self.last_time)
        
def listdir(path:str):
    files_list = os.listdir(path)
    files_list.sort()
    return files_list

def clean_dir(floder_path:str):
    file_or_dir = listdir(floder_path)
    if len(file_or_dir) != 0:
        for fod in file_or_dir:
            remove_path = os.path.join(floder_path, fod)
            if just_ff(remove_path, file=True):
                os.unlink(remove_path)
            else:
                os.removedirs(remove_path)

def name_in_list(file_path:str, dis_list):
    n = get_name_from_path(file_path)
    if n in dis_list:
        return True
    return False

def get_name_from_path(file_path:str, ret_all:bool=False):
    """_summary_
    Return the file name from file path.
    The return name without the suffix.

    Args:
        file_path (_type_): str
    """
    dir, n = os.path.split(file_path)
    n, suffix = os.path.splitext(n)
    if ret_all:
        return dir, n, suffix
    return n

def listdir_com_p(path:str):
    """_summary_
    return complete path of the subfile
    Args:
        path (_type_): 
    """
    files_list = listdir(path)
    com_p = [os.path.join(path, f) for f in files_list]
    return com_p
