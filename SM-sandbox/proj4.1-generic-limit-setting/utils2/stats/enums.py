##  =======================================================================================================================
##  Brief :  global objects associated with the stats module
##  Author:  Stephen Menary
##  Email :  sbmenary@gmail.com
##  =======================================================================================================================

from utils2.objects.SettingsEnum import SettingsEnum


##  Test statistic enum
#
class LimitsMethod (SettingsEnum) :
	CLsb = 1
	CLs  = 2


##  Test statistic enum
#
class TestStatistic (SettingsEnum) :
	chi2  = 1


##  Test statistic strategy enum
#
class TestStatisticStrategy (SettingsEnum) :
	assume = 1
	toys   = 2