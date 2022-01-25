# Spotify controller

## Students:
- Jovana Jevtić, SW-20-2018
- Andrija Vojnović, SW-7-2018

## Requirements for running program:

- Python 3.x
- pip package manager
- virtualenv package

## Running program:
1. Clone repository
2. ```console
   virtualenv venv
    ```
3. We need to activate virtual environment. If you are on Windows system
    ```console
    venv\Scripts\activate
    ```
   Or on Linux
    ```console
    source venv\bin\activate
    ```
4. ```console
   pip install -r requirements.txt
   ```

5. Change current directory to src
   ```console
    cd src
   ```
6. Run program
    ```console
   python webcam.py
    ```
###Spotify shortcuts

If you are using Linux system additional setting are needed.

1. Download file from [link](https://community.spotify.com/spotify/attachments/spotify/desktop_linux/296/1/spotify_control.zip)
2. Execute following command in directory where you downloaded file
    ```console
    chmod a+x spotify_control
    ```
3. Go to settings -> keyboard -> keyboard shortcuts -> custom shortcuts
   1. Add new shortcut by clicking the plus button
    
       [![Guide](https://i.stack.imgur.com/3rfRN.png)]

       The command you see in the picture above is for 'play/pause' function 
      ```console
      /home/your_username/Downloads/spotify_control playpause
      ```
      The commands that need to be added are
   
       - Next - Map to shortcut "Ctrl + Shift+ Right"
          ```console
         /home/your_username/Downloads/spotify_control next 
         ```
       - Previous - Map to shortcut "Ctrl + Shift+ Left"
         ```console
         /home/your_username/Downloads/spotify_control previous
         ```
   2. Go to settings -> keyboard -> keyboard shortcuts. You also need to change shortcuts for volume down and volume up to :
        
       - Volume up - Map to shortcut "Ctrl + Shift + U"
       - Volume down - Map to shortcut "Ctrl + Shift + D"

Enjoy in our program 
       