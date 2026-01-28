import json
import os
from utils.json_operation import get_parameter, get_required_parameter
from utils.file_operation import is_alive_atomic_energy

class WorkFileStructure(object):
    '''
    description: 
    param {*} self
    param {*} json_dir: The trained models, features, and inference results are collected in this directory
    return {*}
    author: wuxingxing
    '''    
    def __init__(self, json_dir:str, reserve_work_dir:bool, reserve_feature:bool, model_type:str) -> None:
        self.model_type = model_type
        self.json_dir = json_dir
        self.reserve_work_dir = reserve_work_dir
        self.reserve_feature = reserve_feature
        self.movement_name = "MOVEMENT"
        # self.raw_path = []
        self.train_feature_path = []
        self.valid_feature_path = []
        self.test_feature_path = []
        # self.datasets_path = []
        self.model_load_path = ""
        
        self.train_data_path = []
        self.valid_data_path = []
        self.test_data_path  = []
        
        if self.model_type == "NN" or self.model_type == "LINEAR":
            self._set_NN_PWdata_dirs() 
    # def _set_training_path(self, train_raw_path:list, train_feature_path:list, train_dir: str):
    #     self.raw_path = train_raw_path
    #     self.train_feature_path = train_feature_path
    #     self.train_dir = os.path.join(self.json_dir, train_dir)


    # def _set_data_file_paths(self, trainSetDir:str, dRFeatureInputDir:str, dRFeatureOutputDir:str,\
    #                     trainDataPath:str, validDataPath:str):
    #     self.trainSetDir = trainSetDir
    #     self.dRFeatureInputDir = dRFeatureInputDir# it is not used 2024.04.03
    #     self.dRFeatureOutputDir = dRFeatureOutputDir# it is not used 2024.04.03
    #     self.trainDataPath = trainDataPath
    #     self.validDataPath = validDataPath

    def _set_p_matrix_paths(self, p_path, save_p_matrix:bool):
        self.save_p_matrix = save_p_matrix
        self.p_matrix_path = p_path

    def _set_model_paths(self,model_store_dir:str, model_name:str, best_model_path:str):
        self.model_store_dir = model_store_dir
        self.model_name = model_name
        self.best_model_path = best_model_path# it is not used 2024.04.03
        self.model_save_path = os.path.join(model_store_dir, self.model_name)

    def _set_model_load_path(self, model_load_path:str):
        self.model_load_path = model_load_path

    def set_inference_paths(self, json_input:dict, is_nep_txt:bool=False):
        test_dir_name = get_parameter("test_dir_name", json_input, "test_result")
        
        if json_input["model_type"].upper() in ["LINEAR", "NN"]:
            self.test_dir = os.path.join(self.nn_work, test_dir_name)
        else:
            self.test_dir = os.path.join(self.json_dir, test_dir_name)

        if not json_input["model_type"].upper() == "LINEAR":
            if is_nep_txt:
                self.model_load_path = None
            else:
                model_load_path = get_required_parameter("model_load_file", json_input)
                self.model_load_path = os.path.abspath(model_load_path)
                if os.path.exists(self.model_load_path) is False:
                    raise Exception("the model_load_path is not exist: {}, please speccified 'model_load_path' at json file".format(self.model_load_path))
        
        # if "trainDataPath" in json_input.keys():# for test, people could set the 'trainSetDir' to 'valid', so the valid data in train dir could be used for valid
        #     self.trainDataPath = json_input["trainDataPath"]

        '''alive_atomic_energy = is_alive_atomic_energy(datasets_path)
        self._set_alive_atomic_energy(alive_atomic_energy)'''
    

    '''
    description: set feature paths
        the feature path list could from user input.json, \
            the feature path list could auto set after generating feature
    param {*} self
    param {list} feature_path
    return {*}
    author: wuxingxing
    '''
    def set_train_feature_path(self, feature_path:list):
        self.train_feature_path.extend(feature_path)

    def set_valid_feature_path(self, feature_path:list):
        self.valid_feature_path.extend(feature_path)

    def set_test_feature_path(self, feature_path:list):
        self.test_feature_path.extend(feature_path)

    # delete in 2025
    def set_datasets_path(self, datasets_path:list):
        pass
        # self.datasets_path.extend(datasets_path)


    '''
    description: 
    set workdir structrues of dp/NN/linear model when doing initialization
    param {*} self
    param {dict} json_input
    return {*}
    author: wuxingxing
    '''
    def set_model_file_paths(self, json_input:dict):
        model_load_path = get_parameter("model_load_file", json_input, " ")
        if os.path.exists(model_load_path):
            model_load_path = os.path.abspath(model_load_path)
        self._set_model_load_path(model_load_path)
        model_name = ""
        best_model_path = ""
        # if self.recover_train is True and os.path.isfile(model_load_path):
        #     raise Exception("Error! The recover_train and model_load_path are simultaneously specified, please set recover_train to False or remove param model_load_path")
        if self.model_type == "NN":
            model_name = get_parameter("model_name", json_input, "nn_model.ckpt")
        elif self.model_type == "DP":
            model_name = get_parameter("model_name", json_input, "dp_model.ckpt")
        elif self.model_type == "NEP":
            model_name = get_parameter("model_name", json_input, "nep_model.ckpt")
        elif self.model_type == "CHEBY":
            model_name = get_parameter("model_name", json_input, "chey_model.ckpt")
        best_model_path = os.path.join(self.json_dir, "best_model.ckpt")
        forcefield_name = get_parameter("forcefield_name", json_input, "forcefield.ff")
        forcefield_dir = get_parameter("forcefield_dir", json_input, "forcefield")
        # p matix, resume p matrix when recover is not realized
        # p matrix should extract to checkpoint files or a single file.
        # current not realized
        save_p_matrix = get_parameter("save_p_matrix", json_input, False)
        if save_p_matrix is not False:
            Pmatrix_path = os.path.join(self.json_dir, "P.pkl")
            self._set_p_matrix_paths(Pmatrix_path, True)
        else:
            self._set_p_matrix_paths(None, False)

        # common dir 
        model_store_dir = get_parameter("model_store_dir", json_input, "model_record")
        if self.model_type == "NN":
            model_store_dir = os.path.join(self.nn_work, model_store_dir)
            self.forcefield_dir = os.path.join(self.nn_work, forcefield_dir)
            self.forcefield_name = forcefield_name
        else:
            self.forcefield_dir = os.path.join(self.json_dir, forcefield_dir)
            self.forcefield_name = forcefield_name
            model_store_dir = os.path.join(self.json_dir, model_store_dir)
        self._set_model_paths(model_store_dir = model_store_dir, \
                                    model_name = model_name, best_model_path=best_model_path)
        
        # self._set_PWdata_dirs(json_input)

    def set_train_valid_file(self, json_input:dict):
        # set trian movement file path
        self.format = get_parameter("format", json_input, "pwmat/movement").lower() # used in new file and raw_file
        if self.model_type.upper() in ["NN", "LINEAR"]:
            if self.format != "pwmat/movement":
                raise Exception("Error! For NN or Linear model, the input 'format' should be 'pwmat/movement'!")
        train_data = get_parameter("train_data", json_input, [])
        if not isinstance(train_data, list):
            train_data = [train_data]
        
        for _train_data in train_data:
            if os.path.exists(_train_data) is False:
                raise Exception("Error! train data: {} file not exist!".format(_train_data))
            else:
                self.train_data_path.append(os.path.abspath(_train_data))
        valid_data = get_parameter("valid_data", json_input, [])
        if not isinstance(valid_data, list):
            valid_data = [valid_data]
        for _valid_data in valid_data:
            if os.path.exists(_valid_data) is False:
                raise Exception("Error! valid data: {} file not exist!".format(_valid_data))
            else:
                self.valid_data_path.append(os.path.abspath(_valid_data))
        test_data = get_parameter("test_data", json_input, [])
        if not isinstance(test_data, list):
            test_data = [test_data]
        for _test_data in test_data:
            if os.path.exists(_test_data) is False:
                raise Exception("Error! test data: {} file not exist!".format(_test_data))
            else:
                self.test_data_path.append(os.path.abspath(_test_data))

        if self.format == "pwmat/movement": # for nn
            self.alive_atomic_energy = False
            if len(self.train_data_path) > 0:
                alive_atomic_energy = is_alive_atomic_energy(self.train_data_path)
                self.alive_atomic_energy = alive_atomic_energy

            if len(self.valid_data_path) > 0:
                alive_atomic_energy = is_alive_atomic_energy(self.valid_data_path)
                self.alive_atomic_energy = alive_atomic_energy
            
            if len(self.test_data_path) > 0:
                alive_atomic_energy = is_alive_atomic_energy(self.test_data_path)
                self.alive_atomic_energy = alive_atomic_energy

    def set_nn_file(self, json_input:dict):
        self.train_feature_path = []
        self.valid_feature_path = []
        self.test_feature_path  = []
        train_feature_path = get_parameter("train_feature_path", json_input, [])
        for feat_path in train_feature_path:
            if os.path.exists(feat_path) is False:
                raise Exception("Error! train_feature_path: {} file not exist!".format(feat_path))
        self.train_feature_path = [os.path.abspath(_) for _ in train_feature_path]

        valid_feature_path = get_parameter("valid_feature_path", json_input, [])
        for feat_path in valid_feature_path:
            if os.path.exists(feat_path) is False:
                raise Exception("Error! valid_feature_path: {} file not exist!".format(feat_path))
        self.valid_feature_path = [os.path.abspath(_) for _ in valid_feature_path]

        test_feature_path = get_parameter("test_feature_path", json_input, [])
        for feat_path in test_feature_path:
            if os.path.exists(feat_path) is False:
                raise Exception("Error! test_feature_path: {} file not exist!".format(feat_path))
        self.test_feature_path = [os.path.abspath(_) for _ in test_feature_path]
        
    def _set_NN_PWdata_dirs(self):
        # set Pwdata dir file structure, they are used in feature generation
        self.nn_work = os.path.join(os.getcwd(), "work_dir") # the work dir of nn training or test 
        self.trainSetDir = 'PWdata'
        self.dRFeatureInputDir = 'input'# it is not used 2024.04.03
        self.dRFeatureOutputDir = 'output'# it is not used 2024.04.03
        # self.trainDataPath = 'train'
        # self.validDataPath = 'valid'
        # self._set_data_file_paths(trainSetDir, dRFeatureInputDir, dRFeatureOutputDir, trainDataPath, validDataPath)

    def set_nep_native_file_paths(self):
        self.nep_model_file = "nep5.txt"
                                 
    def get_data_file_structure(self):
        file_dict = {}
        file_dict["trainSetDir"] = self.trainSetDir
        file_dict["dRFeatureInputDir"] = self.dRFeatureInputDir
        file_dict["dRFeatureOutputDir"] = self.dRFeatureOutputDir
        # file_dict["trainDataPath"] = self.trainDataPath
        # file_dict["validDataPath"] = self.validDataPath
        return file_dict

    def to_dict(self):
        dicts = {}
        dicts["format"] = self.format     
        if self.model_load_path is not None and os.path.exists(self.model_load_path):
            dicts["model_load_file"] = self.model_load_path
        if len(self.train_data_path) > 0:
            dicts["train_data"] = self.train_data_path
        if len(self.valid_data_path) > 0:
            dicts["valid_data"] = self.valid_data_path
        if len(self.test_data_path) > 0:
            dicts["test_data"] = self.test_data_path
        return dicts
