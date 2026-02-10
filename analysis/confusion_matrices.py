"""
Confusion matrices from 16 experimental scenarios.

Source: M. Coban, MSc Thesis, KTO Karatay University, 2021
        Appendix 1: Confusion Matrices (pp. 122-137)

Format: [[TN, FP], [FN, TP]]

Table-Scenario Mapping:
    Table 22 = 5.1.1.1  Full Year, Current Day, Sequential, Default     (Thesis Table 31, p.122)
    Table 23 = 5.1.1.2  Full Year, Current Day, Sequential, Optimized   (Thesis Table 32, p.123)
    Table 24 = 5.1.2.1  Full Year, Current Day, Percentage, Default     (Thesis Table 33, p.124)
    Table 25 = 5.1.2.2  Full Year, Current Day, Percentage, Optimized   (Thesis Table 34, p.125)
    Table 26 = 5.2.1.1  2021 Q1, Current Day, Sequential, Default      (Thesis Table 35, p.126)
    Table 27 = 5.2.1.2  2021 Q1, Current Day, Sequential, Optimized    (Thesis Table 36, p.127)
    Table 28 = 5.2.2.1  2021 Q1, Current Day, Percentage, Default      (Thesis Table 37, p.128)
    Table 29 = 5.2.2.2  2021 Q1, Current Day, Percentage, Optimized    (Thesis Table 38, p.129)
    Table 30 = 5.3.1.1  Full Year, Next Day, Sequential, Default       (Thesis Table 39, p.130)
    Table 31 = 5.3.1.2  Full Year, Next Day, Sequential, Optimized     (Thesis Table 40, p.131)
    Table 32 = 5.3.2.1  Full Year, Next Day, Percentage, Default       (Thesis Table 41, p.132)
    Table 33 = 5.3.2.2  Full Year, Next Day, Percentage, Optimized     (Thesis Table 42, p.133)
    Table 34 = 5.4.1.1  2021 Q1, Next Day, Sequential, Default        (Thesis Table 43, p.134)
    Table 35 = 5.4.1.2  2021 Q1, Next Day, Sequential, Optimized      (Thesis Table 44, p.135)
    Table 36 = 5.4.2.1  2021 Q1, Next Day, Percentage, Default        (Thesis Table 45, p.136)
    Table 37 = 5.4.2.2  2021 Q1, Next Day, Percentage, Optimized      (Thesis Table 46, p.137)

Variants:
    "normal"            = base algorithm (default or optimized parameters)
    "feature_selection" = with feature selection applied (marked as * in article)
"""

results = {
    # 5.1 Full Year, Current Day

    # Table 22
    22: {
        "KNN": {
            "normal": [[142, 32], [142, 50]],
            "feature_selection": [[120, 54], [155, 37]]
        },
        "DTR": {
            "normal": [[125, 49], [38, 154]],
            "feature_selection": [[125, 49], [53, 139]]
        },
        "DTC": {
            "normal": [[125, 49], [38, 154]],
            "feature_selection": [[125, 49], [53, 139]]
        },
        "LR": {
            "normal": [[106, 68], [99, 93]],
            "feature_selection": [[125, 49], [122, 70]]
        },
        "SVM": {
            "normal": [[100, 74], [100, 92]],
            "feature_selection": [[87, 87], [104, 88]]
        },
        "RFC": {
            "normal": [[24, 150], [11, 181]],
            "feature_selection": [[99, 75], [44, 148]]
        },
        "NN": {
            "normal": [[150, 24], [155, 37]],
            "feature_selection": [[16, 158], [28, 164]]
        },
        "XGB": {
            "normal": [[121, 53], [38, 154]],
            "feature_selection": [[118, 56], [42, 150]]
        }
    },

    # Table 23
    23: {
        "KNN": {
            "normal": [[137, 37], [132, 60]],
            "feature_selection": [[115, 59], [129, 63]]
        },
        "DTR": {
            "normal": [[129, 45], [38, 154]],
            "feature_selection": [[123, 51], [61, 131]]
        },
        "DTC": {
            "normal": [[106, 68], [10, 182]],
            "feature_selection": [[114, 60], [29, 163]]
        },
        "LR": {
            "normal": [[103, 71], [92, 100]],
            "feature_selection": [[129, 45], [125, 67]]
        },
        "SVM": {
            "normal": [[94, 80], [99, 93]],
            "feature_selection": [[135, 39], [144, 48]]
        },
        "RFC": {
            "normal": [[0, 174], [0, 192]],
            "feature_selection": [[0, 174], [0, 192]]
        },
        "NN": {
            "normal": [[126, 48], [118, 74]],
            "feature_selection": [[136, 38], [138, 54]]
        },
        "XGB": {
            "normal": [[111, 63], [13, 179]],
            "feature_selection": [[118, 56], [28, 164]]
        }
    },

    # Table 24
    24: {
        "KNN": {
            "normal": [[144, 30], [139, 53]],
            "feature_selection": [[152, 22], [155, 37]]
        },
        "DTR": {
            "normal": [[118, 56], [38, 154]],
            "feature_selection": [[156, 18], [142, 50]]
        },
        "DTC": {
            "normal": [[118, 56], [38, 154]],
            "feature_selection": [[156, 18], [142, 50]]
        },
        "LR": {
            "normal": [[114, 60], [91, 101]],
            "feature_selection": [[125, 49], [116, 76]]
        },
        "SVM": {
            "normal": [[102, 72], [92, 100]],
            "feature_selection": [[1, 173], [3, 189]]
        },
        "RFC": {
            "normal": [[11, 163], [2, 190]],
            "feature_selection": [[93, 81], [37, 155]]
        },
        "NN": {
            "normal": [[151, 23], [149, 43]],
            "feature_selection": [[18, 156], [33, 159]]
        },
        "XGB": {
            "normal": [[119, 55], [24, 168]],
            "feature_selection": [[159, 15], [140, 52]]
        }
    },

    # Table 25
    25: {
        "KNN": {
            "normal": [[138, 36], [134, 58]],
            "feature_selection": [[149, 25], [158, 34]]
        },
        "DTR": {
            "normal": [[119, 55], [38, 154]],
            "feature_selection": [[121, 53], [39, 153]]
        },
        "DTC": {
            "normal": [[107, 67], [9, 183]],
            "feature_selection": [[105, 69], [19, 173]]
        },
        "LR": {
            "normal": [[117, 57], [97, 95]],
            "feature_selection": [[131, 43], [126, 66]]
        },
        "SVM": {
            "normal": [[87, 87], [79, 113]],
            "feature_selection": [[144, 30], [143, 49]]
        },
        "RFC": {
            "normal": [[0, 174], [0, 192]],
            "feature_selection": [[0, 174], [0, 192]]
        },
        "NN": {
            "normal": [[130, 44], [119, 73]],
            "feature_selection": [[133, 41], [130, 62]]
        },
        "XGB": {
            "normal": [[106, 68], [8, 184]],
            "feature_selection": [[137, 37], [74, 118]]
        }
    },

    # 5.2 2021 Q1, Current Day

    # Table 26
    26: {
        "KNN": {
            "normal": [[42, 9], [35, 4]],
            "feature_selection": [[31, 20], [31, 8]]
        },
        "DTR": {
            "normal": [[34, 17], [15, 24]],
            "feature_selection": [[35, 16], [17, 22]]
        },
        "DTC": {
            "normal": [[34, 17], [15, 24]],
            "feature_selection": [[35, 16], [18, 21]]
        },
        "LR": {
            "normal": [[37, 14], [30, 9]],
            "feature_selection": [[36, 15], [30, 9]]
        },
        "SVM": {
            "normal": [[35, 16], [19, 20]],
            "feature_selection": [[29, 22], [22, 17]]
        },
        "RFC": {
            "normal": [[21, 30], [9, 30]],
            "feature_selection": [[34, 17], [13, 26]]
        },
        "NN": {
            "normal": [[51, 0], [39, 0]],
            "feature_selection": [[0, 51], [1, 38]]
        },
        "XGB": {
            "normal": [[36, 15], [9, 30]],
            "feature_selection": [[34, 17], [12, 27]]
        }
    },

    # Table 27
    27: {
        "KNN": {
            "normal": [[44, 7], [31, 8]],
            "feature_selection": [[36, 15], [31, 8]]
        },
        "DTR": {
            "normal": [[37, 14], [16, 23]],
            "feature_selection": [[42, 9], [19, 20]]
        },
        "DTC": {
            "normal": [[32, 19], [8, 31]],
            "feature_selection": [[38, 13], [8, 31]]
        },
        "LR": {
            "normal": [[36, 15], [29, 10]],
            "feature_selection": [[37, 14], [30, 9]]
        },
        "SVM": {
            "normal": [[35, 16], [26, 13]],
            "feature_selection": [[41, 10], [27, 12]]
        },
        "RFC": {
            "normal": [[0, 51], [0, 39]],
            "feature_selection": [[0, 51], [0, 39]]
        },
        "NN": {
            "normal": [[32, 19], [32, 7]],
            "feature_selection": [[40, 11], [29, 10]]
        },
        "XGB": {
            "normal": [[33, 18], [5, 34]],
            "feature_selection": [[33, 18], [12, 27]]
        }
    },

    # Table 28
    28: {
        "KNN": {
            "normal": [[42, 9], [34, 5]],
            "feature_selection": [[44, 7], [36, 3]]
        },
        "DTR": {
            "normal": [[34, 17], [15, 24]],
            "feature_selection": [[35, 16], [12, 27]]
        },
        "DTC": {
            "normal": [[34, 17], [15, 24]],
            "feature_selection": [[35, 16], [12, 27]]
        },
        "LR": {
            "normal": [[35, 16], [31, 8]],
            "feature_selection": [[34, 17], [30, 9]]
        },
        "SVM": {
            "normal": [[32, 19], [27, 12]],
            "feature_selection": [[41, 10], [34, 5]]
        },
        "RFC": {
            "normal": [[18, 33], [5, 34]],
            "feature_selection": [[35, 16], [13, 26]]
        },
        "NN": {
            "normal": [[51, 0], [39, 0]],
            "feature_selection": [[1, 50], [1, 38]]
        },
        "XGB": {
            "normal": [[38, 13], [8, 31]],
            "feature_selection": [[40, 11], [8, 31]]
        }
    },

    # Table 29
    29: {
        "KNN": {
            "normal": [[40, 11], [30, 9]],
            "feature_selection": [[44, 7], [35, 4]]
        },
        "DTR": {
            "normal": [[34, 17], [14, 25]],
            "feature_selection": [[41, 10], [12, 27]]
        },
        "DTC": {
            "normal": [[26, 25], [0, 39]],
            "feature_selection": [[34, 17], [4, 35]]
        },
        "LR": {
            "normal": [[35, 16], [30, 9]],
            "feature_selection": [[34, 17], [30, 9]]
        },
        "SVM": {
            "normal": [[37, 14], [32, 7]],
            "feature_selection": [[50, 1], [34, 5]]
        },
        "RFC": {
            "normal": [[0, 51], [0, 39]],
            "feature_selection": [[0, 51], [0, 39]]
        },
        "NN": {
            "normal": [[33, 18], [31, 8]],
            "feature_selection": [[40, 11], [30, 9]]
        },
        "XGB": {
            "normal": [[36, 15], [5, 34]],
            "feature_selection": [[36, 15], [5, 34]]
        }
    },

    # 5.3 Full Year, Next Day

    # Table 30
    30: {
        "KNN": {
            "normal": [[131, 43], [76, 116]],
            "feature_selection": [[138, 36], [89, 103]]
        },
        "DTR": {
            "normal": [[123, 51], [48, 144]],
            "feature_selection": [[124, 50], [58, 134]]
        },
        "DTC": {
            "normal": [[123, 51], [47, 145]],
            "feature_selection": [[123, 51], [58, 134]]
        },
        "LR": {
            "normal": [[119, 55], [13, 179]],
            "feature_selection": [[92, 82], [73, 119]]
        },
        "SVM": {
            "normal": [[107, 67], [45, 147]],
            "feature_selection": [[104, 70], [101, 91]]
        },
        "RFC": {
            "normal": [[101, 73], [4, 188]],
            "feature_selection": [[111, 63], [6, 186]]
        },
        "NN": {
            "normal": [[174, 0], [192, 0]],
            "feature_selection": [[0, 174], [0, 192]]
        },
        "XGB": {
            "normal": [[128, 46], [29, 163]],
            "feature_selection": [[129, 45], [34, 158]]
        }
    },

    # Table 31
    31: {
        "KNN": {
            "normal": [[122, 52], [47, 145]],
            "feature_selection": [[125, 49], [78, 114]]
        },
        "DTR": {
            "normal": [[125, 49], [51, 141]],
            "feature_selection": [[126, 48], [52, 140]]
        },
        "DTC": {
            "normal": [[109, 65], [4, 188]],
            "feature_selection": [[120, 54], [21, 171]]
        },
        "LR": {
            "normal": [[118, 56], [14, 178]],
            "feature_selection": [[90, 84], [75, 117]]
        },
        "SVM": {
            "normal": [[111, 63], [33, 159]],
            "feature_selection": [[101, 73], [115, 77]]
        },
        "RFC": {
            "normal": [[0, 174], [0, 192]],
            "feature_selection": [[0, 174], [0, 192]]
        },
        "NN": {
            "normal": [[123, 51], [27, 165]],
            "feature_selection": [[58, 116], [39, 153]]
        },
        "XGB": {
            "normal": [[122, 52], [15, 177]],
            "feature_selection": [[126, 48], [22, 170]]
        }
    },

    # Table 32
    32: {
        "KNN": {
            "normal": [[122, 52], [84, 108]],
            "feature_selection": [[133, 41], [122, 70]]
        },
        "DTR": {
            "normal": [[129, 45], [53, 139]],
            "feature_selection": [[85, 89], [117, 75]]
        },
        "DTC": {
            "normal": [[129, 45], [53, 139]],
            "feature_selection": [[85, 89], [117, 75]]
        },
        "LR": {
            "normal": [[114, 60], [14, 178]],
            "feature_selection": [[79, 95], [75, 117]]
        },
        "SVM": {
            "normal": [[143, 31], [121, 71]],
            "feature_selection": [[174, 0], [192, 0]]
        },
        "RFC": {
            "normal": [[102, 72], [4, 188]],
            "feature_selection": [[111, 63], [5, 187]]
        },
        "NN": {
            "normal": [[174, 0], [192, 0]],
            "feature_selection": [[0, 174], [0, 192]]
        },
        "XGB": {
            "normal": [[127, 47], [17, 175]],
            "feature_selection": [[144, 30], [61, 131]]
        }
    },

    # Table 33
    33: {
        "KNN": {
            "normal": [[115, 59], [51, 141]],
            "feature_selection": [[127, 47], [107, 85]]
        },
        "DTR": {
            "normal": [[118, 56], [36, 156]],
            "feature_selection": [[118, 56], [37, 155]]
        },
        "DTC": {
            "normal": [[111, 63], [5, 187]],
            "feature_selection": [[128, 46], [46, 146]]
        },
        "LR": {
            "normal": [[117, 57], [20, 172]],
            "feature_selection": [[36, 138], [16, 176]]
        },
        "SVM": {
            "normal": [[131, 43], [72, 120]],
            "feature_selection": [[174, 0], [192, 0]]
        },
        "RFC": {
            "normal": [[0, 174], [0, 192]],
            "feature_selection": [[0, 174], [0, 192]]
        },
        "NN": {
            "normal": [[107, 67], [19, 173]],
            "feature_selection": [[174, 0], [192, 0]]
        },
        "XGB": {
            "normal": [[116, 58], [10, 182]],
            "feature_selection": [[132, 42], [67, 125]]
        }
    },

    # 5.4 2021 Q1, Next Day

    # Table 34
    34: {
        "KNN": {
            "normal": [[38, 13], [24, 15]],
            "feature_selection": [[42, 9], [29, 10]]
        },
        "DTR": {
            "normal": [[37, 14], [11, 28]],
            "feature_selection": [[38, 13], [12, 27]]
        },
        "DTC": {
            "normal": [[37, 14], [11, 28]],
            "feature_selection": [[37, 14], [12, 27]]
        },
        "LR": {
            "normal": [[32, 19], [5, 34]],
            "feature_selection": [[50, 1], [36, 3]]
        },
        "SVM": {
            "normal": [[40, 11], [20, 19]],
            "feature_selection": [[30, 21], [24, 15]]
        },
        "RFC": {
            "normal": [[27, 24], [0, 39]],
            "feature_selection": [[27, 24], [0, 39]]
        },
        "NN": {
            "normal": [[51, 0], [39, 0]],
            "feature_selection": [[0, 51], [0, 39]]
        },
        "XGB": {
            "normal": [[36, 15], [9, 30]],
            "feature_selection": [[33, 18], [16, 23]]
        }
    },

    # Table 35
    35: {
        "KNN": {
            "normal": [[34, 17], [16, 23]],
            "feature_selection": [[41, 10], [29, 10]]
        },
        "DTR": {
            "normal": [[36, 15], [11, 28]],
            "feature_selection": [[40, 11], [11, 28]]
        },
        "DTC": {
            "normal": [[27, 24], [0, 39]],
            "feature_selection": [[26, 25], [0, 39]]
        },
        "LR": {
            "normal": [[31, 20], [3, 36]],
            "feature_selection": [[51, 0], [39, 0]]
        },
        "SVM": {
            "normal": [[33, 18], [9, 30]],
            "feature_selection": [[31, 20], [20, 19]]
        },
        "RFC": {
            "normal": [[0, 51], [0, 39]],
            "feature_selection": [[0, 51], [0, 39]]
        },
        "NN": {
            "normal": [[43, 8], [18, 21]],
            "feature_selection": [[46, 5], [34, 5]]
        },
        "XGB": {
            "normal": [[35, 16], [7, 32]],
            "feature_selection": [[39, 12], [10, 29]]
        }
    },

    # Table 36
    36: {
        "KNN": {
            "normal": [[36, 15], [24, 15]],
            "feature_selection": [[47, 4], [32, 7]]
        },
        "DTR": {
            "normal": [[40, 11], [15, 24]],
            "feature_selection": [[41, 10], [26, 13]]
        },
        "DTC": {
            "normal": [[40, 11], [15, 24]],
            "feature_selection": [[41, 10], [26, 13]]
        },
        "LR": {
            "normal": [[32, 19], [1, 38]],
            "feature_selection": [[50, 1], [37, 2]]
        },
        "SVM": {
            "normal": [[38, 13], [13, 26]],
            "feature_selection": [[0, 51], [0, 39]]
        },
        "RFC": {
            "normal": [[27, 24], [0, 39]],
            "feature_selection": [[27, 24], [0, 39]]
        },
        "NN": {
            "normal": [[51, 0], [39, 0]],
            "feature_selection": [[0, 51], [0, 39]]
        },
        "XGB": {
            "normal": [[39, 12], [14, 25]],
            "feature_selection": [[39, 12], [13, 26]]
        }
    },

    # Table 37
    37: {
        "KNN": {
            "normal": [[32, 19], [13, 26]],
            "feature_selection": [[41, 10], [32, 7]]
        },
        "DTR": {
            "normal": [[39, 12], [17, 22]],
            "feature_selection": [[43, 8], [17, 22]]
        },
        "DTC": {
            "normal": [[27, 24], [0, 39]],
            "feature_selection": [[26, 25], [0, 39]]
        },
        "LR": {
            "normal": [[32, 19], [4, 35]],
            "feature_selection": [[51, 0], [39, 0]]
        },
        "SVM": {
            "normal": [[32, 19], [4, 35]],
            "feature_selection": [[51, 0], [39, 0]]
        },
        "RFC": {
            "normal": [[0, 51], [0, 39]],
            "feature_selection": [[0, 51], [0, 39]]
        },
        "NN": {
            "normal": [[42, 9], [21, 18]],
            "feature_selection": [[51, 0], [39, 0]]
        },
        "XGB": {
            "normal": [[33, 18], [2, 37]],
            "feature_selection": [[38, 13], [13, 26]]
        }
    },
}


# Thesis body table number -> appendix table number mapping
THESIS_TABLE_MAP = {
    31: {"table": 22, "page": 122, "section": "5.1.1.1"},
    32: {"table": 23, "page": 123, "section": "5.1.1.2"},
    33: {"table": 24, "page": 124, "section": "5.1.2.1"},
    34: {"table": 25, "page": 125, "section": "5.1.2.2"},
    35: {"table": 26, "page": 126, "section": "5.2.1.1"},
    36: {"table": 27, "page": 127, "section": "5.2.1.2"},
    37: {"table": 28, "page": 128, "section": "5.2.2.1"},
    38: {"table": 29, "page": 129, "section": "5.2.2.2"},
    39: {"table": 30, "page": 130, "section": "5.3.1.1"},
    40: {"table": 31, "page": 131, "section": "5.3.1.2"},
    41: {"table": 32, "page": 132, "section": "5.3.2.1"},
    42: {"table": 33, "page": 133, "section": "5.3.2.2"},
    43: {"table": 34, "page": 134, "section": "5.4.1.1"},
    44: {"table": 35, "page": 135, "section": "5.4.1.2"},
    45: {"table": 36, "page": 136, "section": "5.4.2.1"},
    46: {"table": 37, "page": 137, "section": "5.4.2.2"},
}

# All 16 scenarios mapped to their table numbers in results
SCENARIO_MAPPING = {
    "S1_FullYear_Current_Sequential_Default": 22,
    "S2_FullYear_Current_Sequential_Optimized": 23,
    "S3_FullYear_Current_Percentage_Default": 24,
    "S4_FullYear_Current_Percentage_Optimized": 25,
    "S5_Q1_Current_Sequential_Default": 26,
    "S6_Q1_Current_Sequential_Optimized": 27,
    "S7_Q1_Current_Percentage_Default": 28,
    "S8_Q1_Current_Percentage_Optimized": 29,
    "S9_FullYear_Next_Sequential_Default": 30,
    "S10_FullYear_Next_Sequential_Optimized": 31,
    "S11_FullYear_Next_Percentage_Default": 32,
    "S12_FullYear_Next_Percentage_Optimized": 33,
    "S13_Q1_Next_Sequential_Default": 34,
    "S14_Q1_Next_Sequential_Optimized": 35,
    "S15_Q1_Next_Percentage_Default": 36,
    "S16_Q1_Next_Percentage_Optimized": 37,
}

# 4 main scenarios used in the article (Full Year, Current Day)
ARTICLE_SCENARIOS = {
    "Scenario1": 22,  # Sequential, Default
    "Scenario2": 23,  # Sequential, Optimized
    "Scenario3": 24,  # Percentage, Default
    "Scenario4": 25,  # Percentage, Optimized
}

ALGORITHM_NAMES = ["KNN", "DTR", "DTC", "LR", "SVM", "RFC", "NN", "XGB"]
VARIANT_NAMES = ["normal", "feature_selection"]


def calculate_metrics(cm):
    """Calculate metrics from a confusion matrix [[TN, FP], [FN, TP]]."""
    import math
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp
    if total == 0:
        return None
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0
    return {
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'Total': total,
        'Accuracy': accuracy, 'Precision': precision, 'Recall': recall,
        'Specificity': specificity, 'F1': f1, 'MCC': mcc
    }


def get_scenario_data(table_num):
    """Return metrics for all algorithms and variants in a given table."""
    if table_num not in results:
        return None
    table_data = results[table_num]
    metrics = {}
    for algo in ALGORITHM_NAMES:
        if algo in table_data:
            for variant in VARIANT_NAMES:
                if variant in table_data[algo]:
                    label = algo if variant == "normal" else f"{algo}*"
                    m = calculate_metrics(table_data[algo][variant])
                    if m:
                        metrics[label] = m
    return metrics


def get_article_scenarios():
    """Return data for the 4 main article scenarios."""
    return {name: get_scenario_data(tbl) for name, tbl in ARTICLE_SCENARIOS.items()}


def get_all_scenarios():
    """Return data for all 16 scenarios."""
    return {name: get_scenario_data(tbl) for name, tbl in SCENARIO_MAPPING.items()}
