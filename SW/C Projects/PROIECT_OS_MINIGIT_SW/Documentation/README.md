# USER INSTRUCTIONS: DUMMY VERSIONATING SYSTEM WITH THREAT FINDING CONTROL//OS

## Command Line Usage
* In the command line, the name of the executable is called to create a process responsible for launching the versioning program.
* The usage syntax involves specifying the executable (for example, `./execProgName`) followed by arguments.
* The arguments represent the directories we want to version.

## Program Execution Results
The program will return messages directly to the screen, depending on the execution state.

### 1. Program terminated abnormally
This message appears in the following situations:
* When the number of directories we want to version is greater than `maxDirToVers(macro)` or less than 1.
* When the initial value for `maxDirToVers` is too large, a situation in which the operating system would no longer have available PIDs to create all the necessary processes through the `fork` function.

### 2. Program terminated succefully
This message appears if the conditions for abnormal termination are not met.
* The program will iterate through each directory and create a dedicated process for it.
* Within each process, algorithm K will be applied.
* The processes responsible for applying algorithm K will run in parallel.

## Algorithm K
The algorithm is applied only if the directory received from the command line simultaneously meets the following conditions: it exists in the system, it is a directory-type file, and it has not been entered again from the keyboard.
The algorithm consists of two main stages: cleaning the directory and versioning it.

### Stage 1: Cleaning the directory of malicious files
* The entire file tree from the specified directory is recursively traversed.
* If the element is a directory, the recursive traversal continues.
* If the element is a file, it is analyzed.
* If the file has permissions other than `000 (u+g+o)`, it is ignored.
* If the file has `000` permissions (which indicates possible malware), a grandchild process (child process) is created that launches the `checkIntegrity.sh` script into execution.
* The `checkIntegrity.sh` script checks three conditions (1, 2, or 3).
* If at least one of these conditions is true (`1||2||3`), the process writes the message "CORRUPT" into a pipe.
* If none is true, the process writes the message "SUCCES" into the pipe.
* The parent process (the one applying algorithm K on that directory) reads the information from the pipe.
* If the read message is "CORRUPT", the malicious file is moved to the pre-established `isolatedFiles` directory, deleted from the current directory, and the number of malicious files found is counted.
* Each grandchild process that analyzes a file with `000` permissions runs in parallel with the other traversals.
* A different pipe is used for each grandchild process to avoid overwriting data by other processes that detect malicious files.

### Stage 2: Versioning the directory
* The versioning process takes place only after the malicious files have been found and deleted.
* There is a standard directory named `localSaves` where snapshots are saved.
* In `localSaves`, each directory has its own folder containing binary data (`metadata.bin`).
* The `metadata.bin` file keeps the structure of the file tree, noting the details from the INode of each element.
* The system checks if the directory has been previously versioned.
* If the directory is NOT versioned, the system versions it.
* If the directory IS already versioned, the system keeps a reference to the old version in `localSaves` and creates a reference to the current directory (the new version).
* The two references are compared, and if differences are identified, the new version is saved.
* The system can recognize modifications such as: changing the name of the main directory, changing the name of a subdirectory, changing the name of a file, adding data to a file (increased size), or, if the size remained identical, checking the last modification date.
* A pipe is used through which a content is written to standard output, then redirected to the write end of the pipe.
* This content reports the state of the directory: whether it was already versioned and modifications were found or not, or if it was not versioned and has just been versioned.

## Returning the Results
* Each process that applied algorithm K returns to the parent process the number of malicious files it found.
* The parent process waits for the completion of all versioning processes and sums up the number of malicious files detected by each one.

## Execution Examples
* Running the `./ex` command will generate a `-1` error and display the `program terminated abnormally` message.
* Running the `./ex test1 test2` command will return `0` (success) and display the status of each directory after versioning (if the directory exists, is of DIR type, and has not been submitted for versioning again).
* At the end of a successful execution, the program will display the following information:
    * `program terminated succefully`
    * `mainProcess terminated with a total of Malitious Files found: [number of malicious files found]`
    * `totalExecTime: %f sec`

## Important Note
* To correctly view a buffer-type file (which contains non-ASCII characters), it is recommended to open it in VSCode.