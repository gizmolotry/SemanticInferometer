Set WshShell = CreateObject("WScript.Shell")
Set Fso = CreateObject("Scripting.FileSystemObject")
ScriptDir = Fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = ScriptDir
WshShell.Run """C:\Users\Andrew\miniconda3\python.exe"" """ & ScriptDir & "\analysis\isolated_dash_prototype.py""", 0, False
