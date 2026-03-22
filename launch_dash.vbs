Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "D:\belief-transformer\V3"
WshShell.Run """C:\Users\Andrew\miniconda3\python.exe"" ""D:\belief-transformer\V3\analysis\isolated_dash_prototype.py""", 0, False

