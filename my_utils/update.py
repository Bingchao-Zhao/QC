
import sys, os, time, atexit, string
import shutil
from signal import SIGTERM
import yaml
import glob
import argparse

BACKUP = r'/media/zhaobingchao/T003/utile_code/my_utils'
# BACKUP = r'/media/zhaobingchao/ZBC14T/backup/data/utile_code/my_utils'
DEST = os.path.abspath(os.path.dirname(__file__)) 
EXCLUDE = ['model/', '.pyc']
def just_ff(path,*,file=False,floder=True,create_floder = False, info = True):
    if file:
        return os.path.exists(path)
    elif floder:
        if os.path.exists(path):
            return True
        else:
            if create_floder:
                try:
                    os.makedirs(path) 
                    if info:
                        print(r"Path '{}' does not exists, but created ！！".format(path))
                    return True
                except ValueError:
                    if info:
                        print(r"Path '{}' does not exists, and the creation failed ！！".format(path))
                    pass
            else:
                if info:
                    print(r"Path '{}' does not exists！！".format(path))
                return False

def return_all_files(path, files=[]):
    _path = glob.glob(os.path.join(path,'*'))
    for p in  _path:
        _p = os.path.join(path,p)
        if os.path.isdir(_p):
            return_all_files(_p, files=files)
        else:
            if p[0] == '.': continue
            app_flage = True
            for i in EXCLUDE:
                if _p.find(i)>=0:
                    app_flage = False
                    break
            if app_flage:
                files.append(_p)
def update():
    update_list = []
    return_all_files(BACKUP, update_list)
    for f in update_list:
        # 为了使得到的目录名前面不带反斜杆
        file_ = f.split(BACKUP)[-1]
        while(file_[0]==os.sep):
            file_ = file_[1:len(file_)]

        d_p = os.path.join(DEST, file_)
        # 文件不存在则拷贝
        if not just_ff(d_p, file=True):
            just_ff(os.path.dirname(d_p),create_floder=True)
            # print(f, d_p)
            shutil.copy2(f, d_p)
        else:
            # 如果文件修改时间相同，则不操作
            if os.path.getmtime(d_p) == os.path.getmtime(f):
                pass
            else:
                print('Update: ',f, d_p)
                shutil.copy2(f, d_p)

def backup():
    backup_list = []
    return_all_files(DEST, backup_list)
    for f in backup_list:
        # 为了使得到的目录名前面不带反斜杆    
        file_ = f.split(DEST)[-1]
        while(file_[0]==os.sep):
            file_ = file_[1:len(file_)]

        d_p = os.path.join(BACKUP, file_)
        # 文件不存在则拷贝
        if not just_ff(d_p, file=True):
            just_ff(os.path.dirname(d_p),create_floder=True)
            print('Backup: ',f, d_p)
            shutil.copy2(f, d_p)
        else:
            # 如果文件修改时间相同，则不操作
            if os.path.getmtime(d_p) == os.path.getmtime(f):
                pass
            else:
                print('Backup: ',f, d_p)
                shutil.copy2(f, d_p)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # b is backup, u is update
    parser.add_argument('--mod',default='b',type=str)
    args = parser.parse_args()
    if args.mod =='b':
        backup()
    elif args.mod =='u':
        update()
    else:
        print("Error. model can only be 'b' and 'u'. b is backup, u is update")