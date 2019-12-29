import numpy as np

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def get_model_info(model):
    if model.early_stopping_rounds is False:
        raise Exception("You cannot get info about model if model trained without early_stoppings")
    params = model.params
    params_info = "\n     " + "\n     ".join(['{}: {}'.format(x[0], x[1]) for x in params.items()])
    print("Model trained with following parameters: {}".format(params_info))

    print(color.BOLD+"Best score is "+color.END + "{} on {} tree".format(model.best_score, model.best_ntree))

    return
