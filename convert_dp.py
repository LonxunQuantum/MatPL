from src.user.convert_model import conver_to_dp_torch2_version
import os
import sys
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

color_list = ["#D8BFD8",  "#008080",  "#FF6347",  "#40E0D0",  "#EE82EE",  "#F5DEB3"]
mark_list = ["s", "^", "v", "^", "+", '*', ' ']

def do_convert():
    # 搜索模型列表
    print(sys.argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', help='specify input model file path', type=str, default=None)
    parser.add_argument('-d', '--data', help='specify pwdata', type=str, default=None)
    parser.add_argument('-s', '--save_dir', help='specify save_dir', type=str, default=None)
    parser.add_argument('-t', '--atom_type', help='specify atom type list', nargs='+', type=int, default=None)
    parser.add_argument('-f', '--format', help='specify pwdata', nargs='+', type=str, default=None)
    parser.add_argument('-j', '--sij', help='specify sij_max',  type=float, default=None)

    args = parser.parse_args(sys.argv[1:])
    data_dict = None
    if args.sij is not None:
        data_dict = {}
        data_dict["Sij_max"] = args.sij
    model_list = glob.glob(os.path.join(args.model_dir, "*/epoch_valid.dat"))
    model_list = sorted(model_list)
    for model_file in model_list:
        model_dir = os.path.dirname(model_file)
        model_file = os.path.join(model_dir, "checkpoint.pth.tar")
        if not os.path.exists(model_file):
            continue
        # 检查模型训练epoch，收集超过30个epoch的模型，顺带画出loss 图
        loss_picture = draw_loss(model_dir) #epoch less than 20
        if loss_picture is None:
            continue
        # 转换模型
        savename1 = "dp_torch2.ckpt"
        if not os.path.exists(os.path.join(model_dir, savename1)):
            data_dict = conver_to_dp_torch2_version(model_file, args.atom_type, args.data, args.format, savename1, model_dir, data_dict)
        
        model_file = os.path.join(model_dir, "best.pth.tar")
        savename2 = "dp_torch2_best.ckpt"
        if not os.path.exists(os.path.join(model_dir, savename2)):
            data_dict = conver_to_dp_torch2_version(model_file, args.atom_type, args.data, args.format, savename2, model_dir, data_dict)

        save_dir = os.path.join(args.save_dir, "models", os.path.basename(model_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        copy_file(loss_picture, os.path.join(save_dir, os.path.basename(loss_picture)))
        copy_file(os.path.join(model_dir, savename1), os.path.join(save_dir, os.path.basename(savename1)))
        copy_file(os.path.join(model_dir, savename2), os.path.join(save_dir, os.path.basename(savename2)))

        copy_file(os.path.join(model_dir, "epoch_valid.dat"), os.path.join(save_dir, "epoch_valid.dat"))
        copy_file(os.path.join(model_dir, "epoch_train.dat"), os.path.join(save_dir, "epoch_train.dat"))

        copy_file(os.path.join(model_dir, "checkpoint.pth.tar"), os.path.join(save_dir, "checkpoint.pth.tar"))
        copy_file(os.path.join(model_dir, "best.pth.tar"), os.path.join(save_dir, "best.pth.tar"))

        print("convert done! {}".format(model_dir))

def convert_test():
    model_dir = "/data/home/wuxingxing/datas/PWMLFF_library_data/Si/models"
    data = "/data/home/wuxingxing/datas/PWMLFF_library_data/Si/Si_39988images/PWdata/Si72"
    save_dir = "/data/home/wuxingxing/datas/PWMLFF_library_data/Si/best_model"
    davg_dir = "/data/home/wuxingxing/sy_wuchao_data/Si/train"
    atom_type = [14]
    format = None
    sij = None
    do_convert_delresnet(model_dir, data, save_dir, davg_dir, atom_type, format, sij)

def do_convert_delresnet(model_work_dir:str, data:str, save_dir:str, davg_dir:str, atom_type:list[int], format:str, sij:float):
    # 搜索模型列表
    # print(sys.argv[1:])
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model_dir', help='specify input model file path', type=str, default=None)
    # parser.add_argument('-d', '--data', help='specify pwdata', type=str, default=None)
    # parser.add_argument('-s', '--save_dir', help='specify save_dir', type=str, default=None)
    # parser.add_argument('-t', '--atom_type', help='specify atom type list', nargs='+', type=int, default=None)
    # parser.add_argument('-f', '--format', help='specify pwdata', nargs='+', type=str, default=None) # if the input is raw_file, need this
    # parser.add_argument('-j', '--sij', help='specify sij_max',  type=float, default=None)

    # args = parser.parse_args(sys.argv[1:])
    data_dict = None
    if sij is not None:
        data_dict = {}
        data_dict["Sij_max"] = sij
    model_list = glob.glob(os.path.join(model_work_dir, "*/epoch_valid.dat"))
    model_list = sorted(model_list)
    loss_info = []
    for model_file in model_list:
        model_dir = os.path.dirname(model_file)
        model_file = os.path.join(model_dir, "checkpoint.pth.tar")
        if not os.path.exists(model_file):
            continue
        # 检查模型训练epoch，收集超过30个epoch的模型，顺带画出loss 图
        loss_picture, min_loss = draw_loss(model_dir) #epoch less than 20

        # 转换模型
        savename1 = "dp_torch2.ckpt"
        if os.path.exists(os.path.join(model_dir, "dp_torch2.ckpt")):
            os.remove(os.path.join(model_dir, "dp_torch2.ckpt"))
        data_dict = conver_to_dp_torch2_version(model_file, atom_type, data, format, savename1, model_dir, davg_dir, data_dict)
        
        model_file = os.path.join(model_dir, "best.pth.tar")
        savename2 = "dp_torch2_best.ckpt"
        if not os.path.exists(os.path.join(model_dir, "dp_torch2_best.ckpt")):
            os.remove(os.path.join(model_dir, "dp_torch2_best.ckpt"))
        data_dict = conver_to_dp_torch2_version(model_file, atom_type, data, format, savename2, model_dir, davg_dir, data_dict)

        if loss_picture is None:
            continue
        loss_info.append([model_dir, min_loss])
    loss_info = sorted(loss_info, key=lambda x: float(x[1]))
    # save_dir = os.path.join(save_dir, "models", os.path.basename(model_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_model_dir = loss_info[0][0]
    copy_file(os.path.join(best_model_dir, "train_loss.png"), os.path.join(save_dir, os.path.basename(loss_picture)))
    copy_file(os.path.join(best_model_dir, savename1), os.path.join(save_dir, os.path.basename(savename1)))
    # copy_file(os.path.join(best_model_dir, savename2), os.path.join(save_dir, os.path.basename(savename2)))

    copy_file(os.path.join(best_model_dir, "epoch_valid.dat"), os.path.join(save_dir, "epoch_valid.dat"))
    copy_file(os.path.join(best_model_dir, "epoch_train.dat"), os.path.join(save_dir, "epoch_train.dat"))

    # copy_file(os.path.join(best_model_dir, "checkpoint.pth.tar"), os.path.join(save_dir, "checkpoint.pth.tar"))
    # copy_file(os.path.join(best_model_dir, "best.pth.tar"), os.path.join(save_dir, "best.pth.tar"))

    print("convert done! {}".format(model_work_dir))
    
def copy_file(source_file:str, target_file:str, follow_symlinks:bool=True):
    if not os.path.exists(os.path.dirname(target_file)):
        os.makedirs(os.path.dirname(target_file))
    shutil.copyfile(source_file, target_file, follow_symlinks=follow_symlinks)
    # 
def draw_loss(model_dir):
    train_file = os.path.join(model_dir, "epoch_train.dat")
    min_rmse_etot = 999999999.0
    try:
        epoch_train = np.loadtxt(train_file, skiprows=1)
        # np.genfromtxt(train_file, delimiter='\t', skip_header=1, usecols=(2, 4))
        rmse_etot = epoch_train[:, 2]
        rmse_force = epoch_train[:, 4]
        if len(rmse_force) < 20:
            return None
    except Exception:
        return None
    # read valid epoch
    try:
        epoch_valid = np.loadtxt(os.path.join(model_dir, "epoch_valid.dat"), skiprows=1)
        rmse_etot_valid = epoch_valid[2]
        if min_rmse_etot > np.min(rmse_etot_valid):
            min_rmse_etot = np.min(rmse_etot_valid)

        rmse_force_valid = epoch_valid[4]
        valid_loss_str = r"Energy RMSE {:.2f}  Force RMSE {:.2f}".format(rmse_etot_valid, rmse_force_valid)
        title="training loss\n({})".format(valid_loss_str)
    except Exception:
        title="training loss"
    save_file = os.path.join(model_dir, "train_loss.png")
    x_list = [list(range(1, len(rmse_force)+1)), list(range(1, len(rmse_force)+1))]
    y_list = [rmse_etot, rmse_force]
    legend_label = ["rmse_etot", "rmse_force"]
    xticks = list(range(1, len(rmse_force)+1, 10))
    xtick_loc = [_ -1 for _ in xticks]
    len_split = 5
    while True:
        if len(xtick_loc) <= 15:
            break
        xticks = list(range(1, len(rmse_force)+1, len_split))
        xtick_loc = [_ -1 for _ in xticks]
        len_split += 1

    draw_lines(x_list=x_list, y_list=y_list, legend_label=legend_label, \
                      x_label="epochs", y_label = r"Energy RMSE $\left(\mathrm{eV}\right)$  Force RMSE $\mathrm{(eV/\overset{o}{A})}$",
                       title=title, location = "upper right",\
                        picture_save_path = save_file, draw_config = None, \
                        xticks=xticks, xtick_loc=xtick_loc, withmark=True, withxlim=True, figsize=None)
    return save_file, min_rmse_etot

def draw_lines(x_list:list, y_list :list, legend_label:list, \
                      x_label, y_label, title, location, picture_save_path, draw_config = None, \
                        xticks:list=None, xtick_loc:list=None, withmark=True, withxlim=True, figsize=None):
    # force-kpu散点图
    fontsize = 70
    fontsize2 = 60
    font =  {'family' : 'Times New Roman',
        'weight' : 'normal',
        'fontsize' : fontsize,
        }
    if figsize is None:
        figsize = (40,20)
    plt.figure(figsize=figsize)
    plt.style.use('classic') # 画板主题风格
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] # 使用微软雅黑的字体
    for i in range(len(y_list)):
        if withmark:
            plt.plot(x_list[i], y_list[i], \
                color=color_list[i], marker=mark_list[i], markersize=8, \
                    label=legend_label[i], linewidth =5.0)
        else:
            plt.plot(x_list[i], y_list[i], \
                color=color_list[i], \
                    label=legend_label[i], linewidth =5.0)
                   
    if xticks is not None:
        plt.xticks(xtick_loc, xticks, fontsize=fontsize2)
    if withxlim is True:
        plt.xlim(left=0, right=max(x_list[0])+0.2)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)
    plt.xlabel(x_label, font)
    plt.yscale('log')
    plt.grid(linewidth =1.5) # 网格线
    # plt.xscale('log')
    plt.ylabel(y_label, font)
    plt.title(title, font)
    plt.legend(fontsize=fontsize, frameon=False, loc=location)
    plt.tight_layout()
    plt.savefig(picture_save_path)

def copy_files():
    work_dir = "/data/home/wuxingxing/datas/PWMLFF_library_data"
    save_dir = "/data/home/wuxingxing/datas/PWMLFF_library"
    model_dir_list = glob.glob(os.path.join(work_dir, "*/models"))#Al/models
    save_file_list = [
        "dp_torch2_best.ckpt",
        "epoch_train.dat",
        "epoch_valid.dat",
        "train_loss.png"
        ]

    for model_dir in model_dir_list:
        if os.path.exists(os.path.join(save_dir, os.path.basename(os.path.dirname(model_dir)), "models")):
                shutil.rmtree(os.path.join(save_dir, os.path.basename(os.path.dirname(model_dir)), "models"))
        model_list = glob.glob(os.path.join(model_dir, "*/dp_torch2_best.ckpt")) #Al/models/adam_bs1_t1/dp_torch2_best.ckpt
        for model in model_list:
            if '1024' in model or '512' in model or '256' in model or '128' in model or '64' in model:
                continue
            data_name = os.path.basename(os.path.dirname(model_dir)) #Al
            model_type = os.path.basename(os.path.dirname(model))
            _save_dir = os.path.join(save_dir, data_name, "models", model_type)
            # copy file
            if not os.path.exists(_save_dir):
                os.makedirs(_save_dir)
            for save_file in save_file_list:
                copy_file(os.path.join(os.path.dirname(model), save_file), os.path.join(_save_dir, save_file))
            print("copy file done {}".format(os.path.dirname(model)))

if __name__=="__main__":
    # do_convert()
    convert_test()
    # copy_files()