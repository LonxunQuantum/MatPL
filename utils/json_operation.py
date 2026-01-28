'''
description: 
    get value of param in json_input which is required parameters which need input by user
        if the parameter is not specified in json_input, raise error and print error log to user.
param {str} param
param {dict} json_input
param {str} info 
return {*}
author: wuxingxing
'''
def get_required_parameter(param:str, json_input:dict):
    if param not in json_input.keys():
        raise Exception("Input error! : The {} parameter is missing and must be specified in input json file!".format(param))
    return json_input[param]

'''
description: 
    get value of param in json_input,
        if the parameter is not specified in json_input, return the default parameter value 'default_value'
        convert the param of json_input to target format: out_format = 1 for int ; 2 for int array; 3 for float; 4 for float array
param {str} param
param {dict} json_input
param {*} default_value
return {*}
author: wuxingxing
'''
def get_parameter(param:str, json_input:dict, default_value, out_format=None):
    if param not in json_input.keys():
        return default_value
    else:
        if out_format is not None:
            if out_format == 1:
                return int(json_input[param])
            elif out_format == 2:
                return [int(_) for _ in json_input[param].split()]
            elif out_format == 3:
                return float(json_input[param])
            elif out_format == 4:
                return [float(_) for _ in json_input[param].split()]
        else:
            return json_input[param]
