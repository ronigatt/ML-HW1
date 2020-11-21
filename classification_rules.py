class ClassificationRules:

    @staticmethod
    def predict(x_1, x_2):
        if x_1 > 1 and x_2 > 3:
            my_class = 1
        elif x_2 < 1:
            my_class = 2
        else:
            my_class = 3
        return my_class