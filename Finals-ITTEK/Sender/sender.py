import zmq
import os
from time import sleep

def send_files(folder_path, receiver_ip):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://" + receiver_ip + ":5555")  # Specify the IP address of the receiver

    while True:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file) #returns tuple with root directory, subdirectory is ignored + list of files
                with open(file_path, 'rb') as f:
                    data = f.read()
                    # Send file name and data
                    file_info = file
                    socket.send_multipart([file_info.encode(), data])

        # Delay for some time before checking for new files again
        sleep(5)  # Adjust the delay as needed

def main():
    folder_to_send = "files_to_send"
    #receiver_ip = "192.168.189.149"  # This is the servers IP-address with GUI 
    receiver_ip = "192.168.189.151"  # This is the servers IP-address without GUI
    send_files(folder_to_send, receiver_ip)

if __name__ == "__main__":
    main()
