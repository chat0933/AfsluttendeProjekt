import zmq
import os

def receive_files(folder_path): # parameter
    context = zmq.Context()
    socket = context.socket(zmq.PULL) #
    socket.bind("tcp://*:5555")  # Listen on all interfaces

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    while True:
        # Receive file name and data
        file_info, data = socket.recv_multipart() # recives multipart message with name and file
        file_name = file_info.decode() # Decodes the filename to string

        file_path = os.path.join(folder_path, file_name) # construct the path for recived files

        # Write the data to the file without appending the extension again
        with open(file_path, 'wb') as f:
            f.write(data) # writes the recived data to file

def main():
    folder_to_save = "backup"
    receive_files(folder_to_save) #argument

if __name__ == "__main__":
    main()