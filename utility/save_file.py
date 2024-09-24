
import os
import shutil
import datetime

def sivefile_config(main_dir, dataset, model, exp_name):
    # Get the current time
    current_time = datetime.datetime.now()

    # Format the current time as a folder name with seconds
    folder_name = current_time.strftime('%Y-%m-%d_%H-%M-%S')

    file_name = [
        "other/whole_train_time.txt",
        "other/accuracy.txt"
    ]
    for ii in range(len(file_name)):
        with open(file_name[ii], 'r+') as file:
            file.truncate(0)

    save_file_dir = main_dir + dataset + "/" + model + "/" + exp_name + "/" + folder_name + "/"
    # Create the destination folder if it doesn't exist
    os.makedirs(save_file_dir, exist_ok=True)

    save_file_list = [
        "other/whole_train_time.txt",
        "other/accuracy.txt"
    ]
    return save_file_list, save_file_dir

def write_to_file(filename, data):
    f = open(filename, 'a')
    f.write(data)
    f.write('\n')
    f.close()

def copy_files_to_folders(file_list, destination):
    # Get the current time
    # current_time = datetime.datetime.now()

    # Format the current time as a folder name with seconds
    # folder_name = current_time.strftime('%Y-%m-%d_%H-%M-%S')

    # Construct the destination folder path
    folder_path = destination# os.path.join(destination, folder_name)

    # Create the destination folder if it doesn't exist
    # os.makedirs(folder_path, exist_ok=True)

    for file_path in file_list:
        # Copy the file to the destination folder
        shutil.copy(file_path, folder_path)

if __name__ == "__main__":
    # Define the list of files to move
    file_list = [
        "file1.txt",
        "file2.txt",
        "file3.txt"
        # Add more file paths here as needed
    ]

    # Define the destination directory
    destination = "/path/to/destination/directory"

    # Call the function to move files to folders
    copy_files_to_folders(file_list, destination)
