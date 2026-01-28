from pwdata import Config
import os, glob, shutil
import dpdata
import json

def xyz2dpdata(work_dir):
    os.chdir(work_dir)
    if not os.path.exists("dpdata"):
        os.mkdir("dpdata")
    source_list = glob.glob("*.xyz")

    for source in source_list:
        tmp = dpdata.MultiSystems().from_file(source,fmt="ase/structure")
        save_dir = source.split(".")[0]
        tmp.to_deepmd_npy("dpdata/dpdata_{}".format(save_dir))
        print("{} to dpdata success!".format(source))

def dpdata2pwdata(work_dir):
    os.chdir(work_dir)
    source_list = glob.glob("dpdata_*")
    if not os.path.exists("pwdata"):
        os.makedirs("pwdata")

    for source in source_list:
        save_dir = os.path.basename(source.split("_")[1])
        sub_sour_list = glob.glob(os.path.join(source, "*/type.raw"))
        for sub_sour in sub_sour_list:
            save_path = "pwdata/pwdata_{}/{}".format(save_dir, os.path.basename(os.path.dirname(sub_sour)))
            extract_pwdata(
                data_list = [os.path.dirname(sub_sour)],
                data_format = "deepmd/npy",
                datasets_path=save_path, 
                train_valid_ratio=1, 
                data_shuffle=False,
                merge_data=True
            )

            copy_source_dir = os.path.join(save_path, "train")
            tar_dir = os.path.join(save_path, "valid")
            if not os.path.exists(tar_dir):
                shutil.copytree(copy_source_dir, tar_dir)
            
        print(source, " to pwdata success !")


def extract_pwdata(data_list:list[str], 
                data_format:str="pwmat/movement", 
                datasets_path="PWdata", 
                train_valid_ratio:float=0.8, 
                data_shuffle:bool=True,
                merge_data:bool=False,
                interval:int=1
                ):
    # if data_format == DFT_STYLE.cp2k:
    #     raise Exception("not relized cp2k pwdata convert")

    data_name = None
    if merge_data:
        data_name = os.path.basename(datasets_path)
        if not os.path.isabs(datasets_path):
            # data_name = datasets_path
            datasets_path = os.path.dirname(os.path.join(os.getcwd(), datasets_path))
        else:
            datasets_path = os.path.dirname(datasets_path)
        image_data = None
        for data_path in data_list:
            if image_data is not None:
                tmp_config = Config(data_format, data_path)
                # if not isinstance(tmp_config, list):
                #     tmp_config = [tmp_config]
                image_data.append(tmp_config)
            else:
                image_data = Config(data_format, data_path)
                # if not isinstance(image_data, list):
                #     image_data = [image_data]
        image_data.to(
                    output_path=datasets_path,
                    save_format="pwmlff/npy",
                    data_name=data_name,
                    train_ratio = train_valid_ratio, 
                    train_data_path="train", 
                    valid_data_path="valid", 
                    random=data_shuffle,
                    seed = 2024, 
                    retain_raw = False
                    )
    else:
        for data_path in data_list:
            image_data = Config(data_format, data_path)
            image_data.to(
                output_path=datasets_path,
                save_format="pwmlff/npy",
                train_ratio = train_valid_ratio, 
                train_data_path="train", 
                valid_data_path="valid", 
                random=data_shuffle,
                seed = 2024, 
                retain_raw = False
                )

def print_dir(json_path:str):
    dirs = ["/data/home/wuxingxing/datas/PWMLFF_library_data/nep-data/16_metal/1-component/pwdata_1", "/data/home/wuxingxing/datas/PWMLFF_library_data/nep-data/16_metal/2-component/pwdata_2"]
    res = []
    for d in dirs:
        dir_list = os.listdir(d)
        for _dir in dir_list:
            if not os.path.isdir(os.path.join(d, _dir)): 
                continue
            sub_dir = os.listdir(os.path.join(d, _dir))
            # res.append(os.path.join(d, _dir,sub_dir[0]))
            for sub in sub_dir:
                if os.path.exists(os.path.join(d, _dir, sub, "valid/atom_type.npy")):
                    res.append(os.path.join(d, _dir, sub))
            
    # content = ""
    # for _ in res:
    #     line = "\"{}\",".format(_)
    #     print(line)
    #     content += "{}\n".format(line)
    # with open("/data/home/wuxingxing/datas/PWMLFF_library_data/nep-data/16_metal/train/tmp/dir_list", 'w') as wf:
    #     wf.writelines(content)

    train_dict = json.load(open(json_path))
    train_dict['datasets_path'] = res
    json.dump(train_dict, open(json_path, "w"), indent=4)

if __name__ == '__main__':
    # work_dir = "/data/home/wuxingxing/datas/PWMLFF_library_data/nep-data/16_metal/2-component"
    # xyz2dpdata(work_dir)
    # dpdata2pwdata(os.path.join(work_dir, "dpdata"))
    json_path = "/data/home/wuxingxing/datas/PWMLFF_library_data/nep-data/16_metal/train/train.json"
    print_dir(json_path)