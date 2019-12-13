import numpy as np

class Callback():
    def __init__(self):
        pass

    def evaluate(self, model, eval_sets, eval_preds, ntree, score_len=0, n_estimators=0, binarise=False, print_stdout=True):
        print_string = ""
        gap = 0
        for eval in eval_sets:
            eval_set, eval_name = eval
            if binarise == False:
                score = model.eval_metric(eval_set[1], eval_preds[eval_name])
            else:
                score = model.eval_metric(eval_set[1], (eval_preds[eval_name]>.5).astype(int))

            score = np.round(score, 5)
            if ntree==0:
                score_len = len(str(score))

            first_gap = 1+len(str(n_estimators))-len(str(ntree))
            gap = 4-len(str(gap))
            if eval_name=="train":
                print_string += "[{}]".format(ntree) + " "*first_gap + "--- "
            print_string += "{}:".format(eval_name) + " "*gap + str(score)
            gap = 3+score_len-len(str(score))
            print_string += " "*gap
        if print_stdout:
            print(print_string)

        return score_len, score