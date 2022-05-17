import os
import json
import csv
import yaml

from .utils import *

def write_csv_row(file_name,context=[], model = 'a', encoding='utf-8', just=False):
    """write csv row

    Args:
        file_name (str): csv file name.
        context (list): [[context1], [context2]]
        model (str, optional): a: If exist, append context. Doesnot exist, create. 
                               w: IF exist overwrite. Doesnet exist, create. Defaults to 'a'.
        encoding (str, optional): Defaults to 'utf-8'.
        just (bool, optional): defuslts left just. Defaults to 'False'.
    """
    if just:
        longest = [0 for i in range(len(context[0]))]
        temp_context = []
        for conx in context:
            temp = []
            for j in range(len(conx)):
                temp.append(str(conx[j]))
                longest[j] = len(temp[j]) if len(temp[j])>longest[j] else longest[j]
            temp_context.append(temp)

        context = []
        for conx in temp_context:
            context.append([conx[j].ljust(longest[j]) for j in range(len(conx))])

    with open(file_name, model, encoding=encoding) as f:
        csv_writer = csv.writer(f)
        for con in context:
            csv_writer.writerow(con)


def write_json(file_name, context={}, model='a', encoding='utf-8'):
    """write json file

    Args:
        file_name (str): json file name.
        context (dict): dict
        model (str, optional): a: If exist, append context. Doesnot exist, create. 
                               w: IF exist overwrite. Doesnet exist, create. Defaults to 'a'.
        encoding (str, optional): Defaults to 'utf-8'.
    """
    with open(file_name, model, encoding=encoding) as f:
        json.dump(context,f)
        f.close()

def construct_file_context(context, mutil_line=False, split=' '):
    if mutil_line:
        write_context = '\n'.join([str(i) for i in context])+'\n'
    else:
        if isinstance(context, list):
            write_context = split.join([str(i) for i in context])+'\n'
        else:
            context = str(context)
            if len(context)>0:
                context = context + '\n'
            write_context = context
    return write_context

def write_file(file_name,context, model='a', encoding='utf-8'):
    """write file

    Args:
        file_name (str): csv file name.
        context (list): [[context1], [context2]]
        model (str, optional): a: If exist, append context. Doesnot exist, create. 
                               w: IF exist overwrite. Doesnet exist, create. Defaults to 'a'.
        encoding (str, optional): Defaults to 'utf-8'.
    """
    with open(file_name, model, encoding=encoding) as fo:
        fo.write(context)

def record_in_file(file_name,context, mutil_line=False, model='a', \
                    split=' ', encoding='utf-8'):
    """write record file

    Args:
        file_name (str): csv file name.
        context (list): [[context1], [context2]]
        model (str, optional): a: If exist, append context. Doesnot exist, create. 
                               w: IF exist overwrite. Doesnet exist, create. Defaults to 'a'.
        encoding (str, optional): Defaults to 'utf-8'.
    """
    write_context = construct_file_context(context, mutil_line=mutil_line, split=split)
    write_file(file_name,write_context, model=model, encoding=encoding)

def split_path(root_path, input_path):
    path_split = os.sep
    while(root_path[-1]==path_split):
        root_path = root_path[0:len(root_path)-1]
    ret_path = input_path[len(root_path):len(input_path)]
    if len(ret_path) == 0:
        return ''
    while(ret_path[0]==path_split):
        ret_path = ret_path[1:len(ret_path)]
    return ret_path

def csv_reader(file_name, encoding='utf-8'):
    ret = []
    with open(file_name, encoding=encoding) as f:
        reader = csv.reader(f)
        for i in reader:
            ret.append(i)
    return ret

def read_yaml(yaml_file, encoding='utf-8'):
    # 打开yaml文件
    with open(yaml_file, 'r', encoding="utf-8") as file:
        file_data = file.read()
    # 将字符串转化为字典或列表
    data = yaml.load(file_data,Loader=yaml.FullLoader)
    return data