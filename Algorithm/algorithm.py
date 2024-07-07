import numpy as np
from Misc import Sort


def Blink_Detection(frame, Frame_info):

    return None


def List_of_Static_States(code_number):
    #0 - 36 states
    Static_States = (
        ("A_A"), 
        ("A_CV"), 
        ("A_VL"), 
        ("A_S"), 
        ("A_Off"), 
        ("C_A"),
        ("C_C"), 
        ("C_Cv"), 
        ("C_VL"), 
        ("C_S"), 
        ("C_Off"),
        ("F_A"), 
        ("F_C"), 
        ("F_Cv"), 
        ("F_VL"), 
        ("F_R"),
        ("F_S"), 
        ("F_Off"), 
        ("H_A"), 
        ("H_(RR+A)"), 
        ("H_C"), 
        ("H_Cv"), 
        ("H_VL"), 
        ("H_R"), 
        ("H_RR"), 
        ("H_S"), 
        ("H_Off"), 
        ("ID2_1"), 
        ("ID2_2"), 
        ("ID2_Off"), 
        ("ID3_1"), 
        ("ID3_2"),
        ("ID3_Off"),
        ("R_A"),     
        ("R_Off"),     
        ("No_Code"),    #Used for "Rcli + Acli"     
        ("A_M"),     
    )
    return  Static_States[code_number]

def List_of_Blink_States(code_number):
    Blink_States = (
        #State_name         #code_number
        ("A_Acli"           , 4),
        ("A_VLcli"          , 6),
        ("A_Scli"           , 7),
        ("C_Acli"           , 15),
        ("C_VLcli"          , 18),
        ("C_Scli"           , 19),
        ("F_Acli"           , 28),
        ("F_VLcli"          , 31),
        ("F_Rcli"           , 32),
        ("F_Scli"           , 33),
        ("H_(RRcli+A)"      , 37),
        ("A_Mcli"           , 40),
        ("H_(RR+Acli)"      , 43),
        ("H_Acli"           , 44),
        ("H_(RRcli+Acli)"   , 45),
        ("C_Mcli"           , 46),  #Not sure if this can be detected according the states in the dataset
        ("H_VLcli"          , 48),
        ("H_Rcli"           , 49),
        ("H_RRcli"          , 50),
        ("H_Scli"           , 51),
        ("F_(Rcli+Acli)"    , 52),
        ("F_Mcli"           , 53),  #Not sure if this can be detected according the states in the dataset
        ("H_(Rcli+Acli)"    , 61),
        ("H_Mcli"           , 62),  #Not sure if this can be detected according the states in the dataset
    )
    
    for state in Blink_States:
        if state[1] == code_number:
            return state[0]
    
    return None