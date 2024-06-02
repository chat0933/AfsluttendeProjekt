import zmq
import os

def receive_files(folder_path):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")  # Listen on all interfaces

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    while True:
        # Receive file name and data
        file_info, data = socket.recv_multipart()
        file_name = file_info.decode()

        file_path = os.path.join(folder_path, file_name)

        # Write the data to the file without appending the extension again
        with open(file_path, 'wb') as f:
            f.write(data)

def main():
    folder_to_save = "backup"
    receive_files(folder_to_save)

if __name__ == "__main__":
    main()