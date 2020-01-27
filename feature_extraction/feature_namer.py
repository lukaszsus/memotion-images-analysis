from deprecated import deprecated


@deprecated(reason="Idea of storing features names has been changed. Use FeatureSelector class instead.")
class FeatureNamer():
    def __init__(self):
        pass

    @staticmethod
    def get_features_names(methods):
        """
        Method created to retrieve features names returned by method pass as parameter.
        :param methods: method or list of method which returned features names we want to get
        :return: list of feature names
        """
        if type(methods) != list and type(methods) != tuple:
            methods = [methods]

        features_names = list()
        for m in methods:
            if m.__name__ == "mean_color_diffs":
                features_names.append("bilateral_filtering_{}".format(m.__name__))
            elif m.__name__ == "n_color_diff":
                features_names.append("bilateral_filtering_{}".format(m.__name__))
            elif m.__name__ == "norm_color_count":
                features_names.append("{}".format(m.__name__))
            elif m.__name__ == "grayscale_edges_factor":
                features_names.extend(FeatureNamer.grayscale_edges_factor_name())
            elif m.__name__ == "hsv_var":
                features_names.extend(FeatureNamer.hsv_var_names())
            elif m.__name__ == "saturation_distribution":
                features_names.extend(FeatureNamer.sat_dist_names())
            elif m.__name__ == "sat_value_distribution":
                features_names.extend(FeatureNamer.sat_val_dist_names())
        return features_names

    @staticmethod
    def grayscale_edges_factor_name():
        thresholds = [(25, 100),
                      (50, 100),
                      (50, 150),
                      (100, 150),
                      (100, 200),
                      (150, 200),
                      (150, 225)]
        prefix = "grayscale_edges_factor"
        names = list()
        for threshold in thresholds:
            names.append(prefix + "_" + str(threshold[0]) + "_" + str(threshold[1]))
        return names


    @staticmethod
    def hsv_var_names():
        prefix = "hsv_var"
        hsv = ["hue", "value", "saturation"]
        names = list()
        for component in hsv:
            for metric in FeatureNamer._get_stat_metrics_names():
                names.append(prefix + "_" + component + "_" + metric)
        return names

    @staticmethod
    def sat_dist_names():
        prefix = "sat_dist"
        n_bins = [15, 20, 25]
        names = list()
        for n_bin in n_bins:
            for i in range(n_bin):
                names.append(prefix + "_" + str(n_bin) + "_" + i)
        return names

    @staticmethod
    def sat_val_dist_names():
        prefix = "sat_val_dist"
        n = 20
        names = list()
        for i in range(n):
            for j in range(n):
                names.append(prefix + "_" + str(i) + "_" + str(j))
        return names

    @staticmethod
    def _get_stat_metrics_names():
        return ["mean", "p10", "p25", "p50", "p75", "p90"]
