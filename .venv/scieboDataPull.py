import os
import pysciebo
import requests
import xml.etree.ElementTree as ET

from getpass import getpass
from tqdm import tqdm

def main():
    os.environ["SCIEBO_URL"] = "https://fh-dortmund.sciebo.de/remote.php/webdav/EM2PIReLab/600_Datensaetze/610_VirtuelleFlusssimulation"
    print(os.environ.get("SCIEBO_USERNAME"))
    if os.environ.get("SCIEBO_USERNAME") is None or os.environ.get("SCIEBO_PASSWORD"):
        Username, UserPassword = getInput()
    else:
        Username = os.environ.get("SCIEBO_USERNAME", "NotSet")
        UserPassword = os.environ.get("SCIEBO_PASSWORD", "NotSet")
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, "Datasets")
    if not os.path.exists(path):
        os.makedirs(path)

    headers = {
        "Depth": "1",
        "Content-Type": "application/xml"
    }
    data = '''<?xml version="1.0" encoding="utf-8" ?>
    <d:propfind xmlns:d="DAV:">
    <d:prop><d:displayname/></d:prop>
    </d:propfind>'''

    response = requests.request(
        method="PROPFIND",
        url=os.environ["SCIEBO_URL"],
        headers=headers,
        data=data,
        auth=(Username, UserPassword)
    )

    if response.status_code != 207:
        print(f"Error while parsing File-List: {response.status_code}")
        exit()

    # XML-Antwort parsen und nur .xml-Dateien extrahieren
    ns = {"d": "DAV:"}
    tree = ET.fromstring(response.content)
    file_links = []

    for resp in tree.findall("d:response", ns):
        href = resp.find("d:href", ns).text
        if href.endswith(".csv"):
            filename = href.split("/")[-1]
            file_links.append((filename, os.environ["SCIEBO_URL"] + "/" + filename))

    print(f"{len(file_links)} CSV-Files found. Starting download...")

    # Dateien mit Fortschrittsanzeige herunterladen (nur neue)
    for filename, file_url in tqdm(file_links, desc="Download", unit="Datei"):
        local_path = os.path.join(path, filename)
        if os.path.exists(local_path):
            tqdm.write(f"🟡 Skipped: {filename}")
            continue

        r = requests.get(file_url, auth=(Username, UserPassword))
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(r.content)
            tqdm.write(f"✅ Downloaded: {filename}")
        else:
            tqdm.write(f"❌ Error at {filename}: HTTP {r.status_code}")
    
    print("All files downloaded.")
    add_to_gitignore("Datasets")

def getInput():
    # Get the input from the user
    infoString = "To downlaod the dataset from Sciebo, please enter your credentials.\n Your credentials are saved in a local environment variable on you pc only.\n Your credentials will not be saved anywhere else or used for any other context.\n You can choose to not save them, but you will have to enter them again next time.\n You will be asked if you want to save your credentials after your input.\n\n"

    print(infoString)

    Username = input("Please enter your Sciebo Username: ")
    UserPassword = getpass("Please enter your Sciebo Userpassword: ")

    boolSave = input("Do you want to save your credentials? (y/n): ")
    if boolSave == "y":
        # Save the credentials in a local environment variable
        os.environ["SCIEBO_USERNAME"] = Username
        os.environ["SCIEBO_PASSWORD"] = UserPassword
        print("Your credentials have been saved in a local environment varaible on your own pc/laptop.")
    else:
        if boolSave != "n":
            print("Invalid input. Your credentials will not be saved.")
        else:
            print("Your credentials will not be saved. You will have to enter them again next time.")
    return Username, UserPassword   

def add_to_gitignore(folder_name):
    gitignore_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.gitignore')

    # Überprüfen, ob die .gitignore-Datei existiert
    if os.path.exists(gitignore_path):
        # Wenn die Datei existiert, lese sie ein
        with open(gitignore_path, 'r') as file:
            gitignore_content = file.readlines()

        # Überprüfen, ob der Ordner bereits in der .gitignore steht
        if f".venv/{folder_name}/\n" not in gitignore_content:
            # Wenn nicht, füge ihn hinzu
            with open(gitignore_path, 'a') as file:
                file.write(f".venv/{folder_name}/\n")
            print(f"Folder {folder_name} was added to .gitignore.")
        else:
            print(f"Folder {folder_name} is already in .gitignore.")
    else:
        # Falls die .gitignore nicht existiert, erstelle sie und füge den Ordner hinzu
        with open(gitignore_path, 'w') as file:
            file.write(f"{folder_name}/\n")
        print(f".gitignore was created and Folder {folder_name} was added.")

if __name__ == "__main__":
    main()