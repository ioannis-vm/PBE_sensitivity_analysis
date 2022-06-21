def read_study_param(param_path:str) -> str:
    with open(param_path, 'r') as f:
        data = f.read()
    return data
