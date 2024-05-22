from pathlib import Path

path = (Path(__file__)).resolve()
path = '/'.join(str(path).split('/')[:-1])
path += '/pretained_model/'

config_dict = {
    'path_pretrain': path
}