##  =======================================================================================================================
##  Brief :  global enums
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  =======================================================================================================================

from utils2.objects.SettingsEnum import SettingsEnum


##  BSM prediction method enum
#
class BSMPredictionMethod (SettingsEnum) :
	AsInput   = 0
	ScaleByL6 = 1