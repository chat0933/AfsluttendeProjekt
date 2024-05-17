"""
OVERVEJ LIGE AT BRUG ZMQ HVOR BESKEDEN ER VARIABLEN = NÅR DER BLIVER SENDT EN FIL / TILFØJET EN NY FIL
"""
import zmq
import os

def receive_files(folder_path):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")  # Listen on all interfaces

    while True:
        # Receive file name, extension, and data
        file_info, data = socket.recv_multipart()
        file_name, file_extension = file_info.decode().split('|')

        file_path = os.path.join(folder_path, file_name)

        with open(file_path + file_extension, 'wb') as f:
            f.write(data)

def main():
    folder_to_save = "backup"
    receive_files(folder_to_save)

if __name__ == "__main__":
    main()