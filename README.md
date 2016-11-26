# ML-Pipeline
This is the Machine-Learning-Pipeline for the bachelor thesis project "Recognizing defects in Java code automatically".

## Setup
Python 3.5 with scikit-learn and SQLAlchemy is required. These libraries can easily be installed with Anaconda.

For _Windows_:

1. install Anaconda
  * Download it from https://www.continuum.io/downloads
  * Choose _Python 3.5 Windows 64-Bit Graphical Installer_

2. Setup environment
  * Enter into cmd (Windows):
  * `conda create --name ml scikit-learn sqlalchemy pymysql matplotlib`

    `activate ml`
    
    `pip install terminaltables`
    
3. Configure PyCharm: 
  * Open settings (CTRL + ALT + S)
  * _Project: ML-Pipeline_ -> _Project Interpreter_
  * Click the cog wheel next to the interpreter listbox -> _Add Local_
  * Choose the Python.exe of your new environment
    * E.g. `C:\Anaconda3\envs\ml\python.exe`
  * Now all packages like SQLAlchemy, scikit-learn, numpy etc. should be listed.
  * Be patient, PyCharm needs some time to rebuild its indexes
   -> In the lower right corner PyCharm tells you which processes are running.

--------------------------------------------------------------------------------------------------------

# ML-Pipeline
Die Machine-Learning-Pipeline für das Bachelorarbeitsprojekt "Fehler in Java Code automatisch erkennen".

## Setup
Es wird Python 3.5 mit Scikit-learn und SQLAlchemy benötigt. Am einfachsten geht die Installation mit Anaconda:

1. Anaconda installieren
  * Downloaden von https://www.continuum.io/downloads
  * _Python 3.5 Windows 64-Bit Graphical Installer_ wählen

2. Environment aufsetzen:
  * In der Kommandozeile (Windows):
  * `conda create --name ml scikit-learn sqlalchemy pymysql matplotlib`

    `activate ml`
    
    `pip install terminaltables`
    
3. PyCharm konfigurieren: 
  * Settings öffnen (CTRL + ALT + S)
  * _Project: ML-Pipeline_ -> _Project Interpreter_
  * Auf Zahnrad neben Interpreter Listbox klicken -> _Add Local_
  * Python.exe des neuen Environments auswählen
    * Z.B. `C:\Anaconda3\envs\ml\python.exe`
  * Nun sollten alle Packages wie SQLAlchemy, scikit-learn, numpy etc. aufgelistet sein.
  * Dann geduldig sein, PyCharm hat eine Weile weil es die Indizes und so neu bilden muss
    -> Rechts unten in PyCharm steht, was für Prozesse am laufen sind.
